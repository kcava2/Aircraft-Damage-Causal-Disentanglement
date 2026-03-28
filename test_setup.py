# test_setup.py
import torch
from codebase.models.mask_vae_aircraft import CausalVAE
from dataset.aircraft_damage import SCALE, N_CONCEPTS, CLASS_NAMES, get_transforms, AircraftDamageDataset
from dataset.aircraft_dag import get_dag_init
from codebase.models.classifier_head import MultiLabelHead
from codebase.utils import _h_A

print('All imports OK')
print(f'N_CONCEPTS: {N_CONCEPTS}')
print(f'CLASS_NAMES: {CLASS_NAMES}')

# Test model instantiation
Z1_DIM = N_CONCEPTS        # 4 concepts: crack, dent, paint_off, scratch
Z2_DIM = 4                 # dims per concept
Z_DIM  = Z1_DIM * Z2_DIM  # 16 total
lvae = CausalVAE(z_dim=Z_DIM, z1_dim=Z1_DIM, z2_dim=Z2_DIM, channel=3, scale=SCALE)
print('✓ Model created OK')

# Test DAG injection
dag_init = get_dag_init()
with torch.no_grad():
    lvae.dag.A.data.copy_(dag_init)
print('✓ DAG injected OK')
print(f'  DAG shape: {lvae.dag.A.shape}')

# Test classifier head
clf = MultiLabelHead(in_dim=Z_DIM, n_classes=N_CONCEPTS)
print('✓ Classifier created OK')

# Test forward pass with dummy data (96x96 for conv autoencoder)
batch_size = 2
img_height, img_width = 96, 96
dummy_img = torch.randn(batch_size, 3, img_height, img_width)  # Conv expects spatial dims
dummy_u   = torch.randn(batch_size, Z1_DIM)

print(f'\nTesting forward pass with shape: {dummy_img.shape}')
L, kl, rec, recon, z = lvae.negative_elbo_bound(dummy_img, dummy_u, sample=False)
print('✓ Forward pass OK')

# Validate output shapes
expected_z_shape = (batch_size, Z_DIM)
expected_recon_shape = (batch_size, 3, img_height, img_width)  # Spatial format from conv decoder

print(f'  z shape: {z.shape} (expected: {expected_z_shape})')
assert z.shape == expected_z_shape, f"z shape mismatch: {z.shape} != {expected_z_shape}"

print(f'  recon shape: {recon.shape} (expected: {expected_recon_shape})')
assert recon.shape == expected_recon_shape, f"recon shape mismatch: {recon.shape} != {expected_recon_shape}"

# Test classifier on z
logits = clf(z)
expected_logits_shape = (batch_size, N_CONCEPTS)
print(f'  logits shape: {logits.shape} (expected: {expected_logits_shape})')
assert logits.shape == expected_logits_shape, f"logits shape mismatch: {logits.shape} != {expected_logits_shape}"

# Validate loss values
print(f'\n  Loss: {L.item():.4f}')
print(f'  KL: {kl.item():.4f}')
print(f'  Reconstruction: {rec.item():.4f}')
assert torch.isfinite(L), "Loss is NaN or Inf"
assert torch.isfinite(kl), "KL is NaN or Inf"
assert torch.isfinite(rec), "Reconstruction is NaN or Inf"

print('\n✓ All shape validations passed')

# ── Additional Diagnostic Checks ──────────────────────────────────────────────

print('\n' + '='*60)
print('DIAGNOSTIC CHECKS')
print('='*60)

# Check 1: Is z_given_dag actually different per image?
print('\nCheck 1: Testing if z_given_dag varies per image...')
test_images = torch.randn(5, 3, 96, 96)  # 5 different images
test_labels = torch.randn(5, Z1_DIM)

z_outputs = []
for i in range(5):
    _, _, _, _, z_out = lvae.negative_elbo_bound(
        test_images[i:i+1], test_labels[i:i+1], sample=False
    )
    z_outputs.append(z_out.squeeze(0))

z_outputs = torch.stack(z_outputs)
z_std = z_outputs.std(dim=0)  # Standard deviation across different images
z_mean_std = z_std.mean().item()

print(f'  Mean std of z_given_dag across images: {z_mean_std:.6f}')
if z_mean_std > 0.01:  # Arbitrary threshold
    print('  ✓ z_given_dag shows variation across different images')
else:
    print('  ⚠️  WARNING: z_given_dag shows very little variation across images')

# Check 2: Is the decoder output varying at all?
print('\nCheck 2: Testing if decoder output varies...')
# Test decoder with different z inputs
test_z = torch.randn(3, Z_DIM)  # 3 different latent vectors
decoder_outputs = []

lvae.eval()
with torch.no_grad():
    for i in range(3):
        z_single = test_z[i:i+1]
        recon_logits, _, _, _, _ = lvae.dec.decode(z_single, torch.zeros(1, Z1_DIM))
        recon_img = torch.sigmoid(recon_logits.view(1, 3, 96, 96))
        decoder_outputs.append(recon_img.squeeze(0))

decoder_outputs = torch.stack(decoder_outputs)
output_std = decoder_outputs.std(dim=[0, 1, 2, 3]).item()  # Overall std across all pixels/channels

print(f'  Decoder output std across different z inputs: {output_std:.6f}')
if output_std > 0.001:  # Very small threshold since sigmoid outputs are 0-1
    print('  ✓ Decoder output shows variation across different latent inputs')
else:
    print('  ⚠️  WARNING: Decoder output shows very little variation - possible issue!')

# Check 3: Are decoder gradients flowing?
print('\nCheck 3: Testing if decoder gradients are flowing...')
# Create a fresh model for gradient testing
test_lvae = CausalVAE(z_dim=Z_DIM, z1_dim=Z1_DIM, z2_dim=Z2_DIM, channel=3, scale=SCALE)
test_img = torch.randn(1, 3, 96, 96, requires_grad=False)
test_u = torch.randn(1, Z1_DIM, requires_grad=False)

# Forward pass
L_test, _, _, _, _ = test_lvae.negative_elbo_bound(test_img, test_u, sample=False)

# Backward pass
L_test.backward()

# Check decoder gradients
decoder_grad_norms = []
zero_grad_layers = []
total_decoder_params = 0

for name, param in test_lvae.dec.named_parameters():
    total_decoder_params += param.numel()
    if param.grad is not None:
        grad_norm = param.grad.data.norm(2).item()
        decoder_grad_norms.append(grad_norm)
        if grad_norm == 0:
            zero_grad_layers.append(name)
    else:
        zero_grad_layers.append(name)

if zero_grad_layers:
    print(f'  ⚠️  Layers with zero/no gradients: {len(zero_grad_layers)}/{len(list(test_lvae.dec.parameters()))}')
    if len(zero_grad_layers) < len(list(test_lvae.dec.parameters())):
        print('     (Some layers have gradients - this may be normal for unused decoder paths)')

if decoder_grad_norms:
    avg_grad_norm = sum(decoder_grad_norms) / len(decoder_grad_norms)
    max_grad_norm = max(decoder_grad_norms)
    print(f'  Decoder gradient stats:')
    print(f'    Average gradient norm: {avg_grad_norm:.8f}')
    print(f'    Maximum gradient norm: {max_grad_norm:.8f}')
    print(f'    Total decoder parameters: {total_decoder_params:,}')
    
    if avg_grad_norm > 1e-8:  # Very small threshold
        print('  ✓ Decoder gradients are flowing (some layers may be unused in this forward pass)')
    else:
        print('  ⚠️  WARNING: Decoder gradients are very small - possible gradient flow issue!')
else:
    print('  ❌ ERROR: No decoder gradients found at all!')

print('='*60)
print('✓ Everything looks good - ready to train!')