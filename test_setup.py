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
Z_DIM, Z1_DIM, Z2_DIM = 20, 5, 4
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
print('✓ Everything looks good - ready to train!')