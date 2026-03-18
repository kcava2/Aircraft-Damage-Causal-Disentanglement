# test_setup.py
import torch
from codebase.models.mask_vae_aircraft import CausalVAE
from dataset.aircraft_damage import SCALE, N_CONCEPTS, CLASS_NAMES, get_transforms, AircraftDamageDataset
from dataset.aircraft_dag import get_dag_init
from models.classifier_head import MultiLabelHead
from utils import _h_A

print('All imports OK')
print(f'N_CONCEPTS: {N_CONCEPTS}')
print(f'CLASS_NAMES: {CLASS_NAMES}')

# Test model instantiation
lvae = CausalVAE(z_dim=20, z1_dim=5, z2_dim=4, channel=3, scale=SCALE)
print('Model created OK')

# Test DAG injection
dag_init = get_dag_init()
with torch.no_grad():
    lvae.dag.A.data.copy_(dag_init)
print('DAG injected OK')
print(f'DAG shape: {lvae.dag.A.shape}')

# Test classifier head
clf = MultiLabelHead(in_dim=20, n_classes=5)
print('Classifier created OK')

# Test forward pass with dummy data
dummy_img = torch.zeros(2, 3, 64, 64)
dummy_u   = torch.zeros(2, 5)
L, kl, rec, recon, z = lvae.negative_elbo_bound(dummy_img, dummy_u, sample=False)
print('Forward pass OK')
print(f'z shape: {z.shape}')
print(f'recon shape: {recon.shape}')

# Test classifier on z
logits = clf(z)
print(f'Classifier output shape: {logits.shape}')
print()
print('Everything looks good - ready to train.')