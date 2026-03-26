import os
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── Original repo imports ────────────────────────────────────────────────────
from codebase import utils as ut
from codebase.models.mask_vae_aircraft import CausalVAE
from codebase.utils import _h_A

# ── Our new file imports ─────────────────────────────────────────────────────
from dataset.aircraft_damage import (
    AircraftDamageDataset,
    get_transforms,
    SCALE,
    N_CONCEPTS,
    CLASS_NAMES,
)
from dataset.aircraft_dag import get_dag_init
from codebase.models.classifier_head import (
    MultiLabelHead,
    multilabel_loss,
    compute_metrics,
)

# ── Device ───────────────────────────────────────────────────────────────────
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Config ───────────────────────────────────────────────────────────────────
# Change data_root to wherever your Roboflow dataset is
DATA_ROOT    = './dataset'
IMG_SIZE     = 96  # Conv encoder/decoder designed for 96x96
Z1_DIM       = N_CONCEPTS        # 5 concepts
Z2_DIM       = 4                 # dims per concept
Z_DIM        = Z1_DIM * Z2_DIM  # 20 total
EPOCHS       = 50
BATCH_SIZE   = 64
LR           = 1e-4
CLF_WEIGHT   = 1.0
SAVE_EVERY   = 5
SAVE_DIR     = './checkpoints'

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs('./figs_aircraft', exist_ok=True)


# ── Warmup scheduler (kept from original run_pendulum.py) ────────────────────
class DeterministicWarmup(object):
    def __init__(self, n=100, t_max=1):
        self.t = 0
        self.t_max = t_max
        self.inc = 1 / n

    def __iter__(self):
        return self

    def __next__(self):
        t = self.t + self.inc
        self.t = self.t_max if t > self.t_max else t
        return self.t


# ── Training loop ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ── Datasets ─────────────────────────────────────────────────────────────────
    train_loader = DataLoader(
        AircraftDamageDataset(
            img_dir   = os.path.join(DATA_ROOT, 'train', 'images'),
            label_dir = os.path.join(DATA_ROOT, 'train', 'labels'),
            transform = get_transforms('train', IMG_SIZE),
        ),
        batch_size  = BATCH_SIZE,
        shuffle     = True,
        num_workers = 0,  # Windows-safe, avoids multiprocessing spawn issues
        pin_memory  = False,
    )

    valid_loader = DataLoader(
        AircraftDamageDataset(
            img_dir   = os.path.join(DATA_ROOT, 'valid', 'images'),
            label_dir = os.path.join(DATA_ROOT, 'valid', 'labels'),
            transform = get_transforms('valid', IMG_SIZE),
        ),
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        num_workers = 0,
        pin_memory  = False,
    )

    print(f"Train batches: {len(train_loader)}  |  Valid batches: {len(valid_loader)}")


    # ── Model ─────────────────────────────────────────────────────────────────────
    lvae = CausalVAE(
        name    = 'aircraft_causalvae',
        z_dim   = Z_DIM,     # 20
        z1_dim  = Z1_DIM,    # 5 concepts
        z2_dim  = Z2_DIM,    # 4 dims per concept
        channel = 3,         # RGB
        scale   = SCALE,     # (5,2) numpy array
        initial = True,
    ).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Inject domain-informed DAG prior
    dag_init = get_dag_init().to(next(lvae.parameters()).device)
    with torch.no_grad():
        lvae.dag.A.data.copy_(dag_init)
    print("DAG prior injected.")

    # ── Classifier head ───────────────────────────────────────────────────────────
    clf = MultiLabelHead(
        in_dim    = Z_DIM,
        n_classes = N_CONCEPTS,
        dropout   = 0.3,
    ).to(next(lvae.parameters()).device)

    # ── Optimiser ─────────────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        list(lvae.parameters()) + list(clf.parameters()),
        lr=LR, betas=(0.9, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5
    )


    # ── Helper ────────────────────────────────────────────────────────────────────
    def u_to_targets(u_norm):
        """Convert normalised [-1,1] concept vector back to binary {0,1} targets."""
        raw = u_norm * 0.5 + 0.5
        return (raw >= 0.5).float()


    best_val_f1 = 0.0

    # ── Tracking lists for plotting ───────────────────────────────────────────────
    history = {
        'epoch': [],
        'train_loss': [],
        'train_kl': [],
        'train_rec': [],
        'train_clf': [],
        'val_f1': [],
        'val_acc': [],
    }

    for epoch in range(1, EPOCHS + 1):
        lvae.train()
        clf.train()

        total_loss = total_kl = total_rec = total_clf = 0.0
        n_batches  = 0

        for imgs, u in train_loader:
            imgs = imgs.to(next(lvae.parameters()).device)
            u    = u.to(imgs.device)

            optimizer.zero_grad()

            kl_weight = min(1.0, epoch / 25.0) * 0.1 # Gradually increase KL weight over first 25 epochs

            # CausalVAE forward — returns nelbo, kl, rec, recon_image, z_given_dag
            L, kl, rec, recon_img, z_given_dag = lvae.negative_elbo_bound(
                imgs, u, sample=False, alpha=kl_weight, beta=kl_weight
            )

            # DAG acyclicity penalty (same as run_pendulum.py)
            dag_param = lvae.dag.A
            h_a = _h_A(dag_param, dag_param.size()[0])
            L   = L + 3 * h_a + 0.5 * h_a * h_a

            # Classification loss on top of latent z
            clf_logits = clf(z_given_dag)
            clf_loss   = multilabel_loss(clf_logits, u_to_targets(u))
            L          = L + CLF_WEIGHT * clf_loss

            L.backward()
            torch.nn.utils.clip_grad_norm_(
                list(lvae.parameters()) + list(clf.parameters()), 1.0
            )
            optimizer.step()

            total_loss += L.item()
            total_kl   += kl.item()
            total_rec  += rec.item()
            total_clf  += clf_loss.item()
            n_batches  += 1

        scheduler.step()

        # ── Validation ────────────────────────────────────────────────────────────
        lvae.eval()
        clf.eval()
        all_logits = []
        all_targets = []

        with torch.no_grad():
            for imgs, u in valid_loader:
                imgs = imgs.to(next(lvae.parameters()).device)
                u    = u.to(imgs.device)
                _, _, _, _, z_given_dag = lvae.negative_elbo_bound(imgs, u, sample=False)
                all_logits.append(clf(z_given_dag).cpu())
                all_targets.append(u_to_targets(u).cpu())

        all_logits  = torch.cat(all_logits)
        all_targets = torch.cat(all_targets)
        metrics     = compute_metrics(all_logits, all_targets, class_names=CLASS_NAMES)
        val_f1      = metrics["f1_macro"]
        val_acc     = metrics["exact_match_accuracy"]

        print(
            f"Epoch {epoch:3d}/{EPOCHS}  "
            f"loss={total_loss/n_batches:.4f}  "
            f"kl={total_kl/n_batches:.4f}  "
            f"rec={total_rec/n_batches:.4f}  "
            f"clf={total_clf/n_batches:.4f}  |  "
            f"val_f1={val_f1:.4f}  val_acc={val_acc:.4f}"
        )
        
        # Track metrics for plotting
        history['epoch'].append(epoch)
        history['train_loss'].append(total_loss/n_batches)
        history['train_kl'].append(total_kl/n_batches)
        history['train_rec'].append(total_rec/n_batches)
        history['train_clf'].append(total_clf/n_batches)
        history['val_f1'].append(val_f1)
        history['val_acc'].append(val_acc)
        for cls_name, m in metrics["per_class"].items():
            print(f"    {cls_name:<15} F1={m['f1']:.3f}  "
                  f"P={m['precision']:.3f}  R={m['recall']:.3f}")

        # Save reconstruction samples every SAVE_EVERY epochs
        if epoch % SAVE_EVERY == 0:
            with torch.no_grad():
                sample_imgs, sample_u = next(iter(valid_loader))
                sample_imgs = sample_imgs[:8].to(next(lvae.parameters()).device)
                sample_u    = sample_u[:8].to(sample_imgs.device)
                _, _, _, recon, _ = lvae.negative_elbo_bound(
                    sample_imgs, sample_u, sample=False
                )
                save_image(
                    sample_imgs.clamp(0, 1),
                    f'figs_aircraft/orig_epoch{epoch:04d}.png', nrow=8
                )
                save_image(
                    torch.sigmoid(recon.reshape(sample_imgs.size())).clamp(0, 1),
                    f'figs_aircraft/recon_epoch{epoch:04d}.png', nrow=8
                )

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                'epoch':   epoch,
                'lvae':    lvae.state_dict(),
                'clf':     clf.state_dict(),
                'val_f1':  val_f1,
                'val_acc': val_acc,
                'config':  {'z_dim': Z_DIM, 'z1_dim': Z1_DIM, 'z2_dim': Z2_DIM},
            }, os.path.join(SAVE_DIR, 'best_model.pt'))
            print(f"    ✓ Best model saved (val_f1={val_f1:.4f})")

            torch.save({
                'epoch': epoch,
                'lvae':  lvae.state_dict(),
                'clf':   clf.state_dict(),
            }, os.path.join(SAVE_DIR, f'checkpoint_epoch{epoch:04d}.pt'))

    print(f"\nTraining complete. Best val F1: {best_val_f1:.4f}")

    # ── Plot training history ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Loss plot
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(history['epoch'], history['train_loss'], 'b-', label='Total Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Training Loss', fontsize=12, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # KL and Reconstruction loss
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(history['epoch'], history['train_kl'], 'g-', label='KL', linewidth=2)
    ax2.plot(history['epoch'], history['train_rec'], 'r-', label='Reconstruction', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Loss', fontsize=11)
    ax2.set_title('KL & Reconstruction Loss', fontsize=12, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    # Classification loss
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(history['epoch'], history['train_clf'], 'purple', label='Classification Loss', linewidth=2)
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Loss', fontsize=11)
    ax3.set_title('Classification Loss', fontsize=12, fontweight='bold')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)

    # Validation F1 Score
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(history['epoch'], history['val_f1'], 'b-o', label='F1-Score (Macro)', linewidth=2, markersize=5)
    ax4.set_xlabel('Epoch', fontsize=11)
    ax4.set_ylabel('F1-Score', fontsize=11)
    ax4.set_title('Validation F1-Score', fontsize=12, fontweight='bold')
    ax4.set_ylim([0, 1.05])
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3)

    # Validation Accuracy
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(history['epoch'], history['val_acc'], 'g-o', label='Accuracy', linewidth=2, markersize=5)
    ax5.set_xlabel('Epoch', fontsize=11)
    ax5.set_ylabel('Accuracy', fontsize=11)
    ax5.set_title('Validation Accuracy', fontsize=12, fontweight='bold')
    ax5.set_ylim([0, 1.05])
    ax5.legend(loc='best')
    ax5.grid(True, alpha=0.3)

    plt.savefig('./figs_aircraft/training_history.png', dpi=150, bbox_inches='tight')
    print(f"✓ Training history plot saved to ./figs_aircraft/training_history.png")
    plt.close()

    # Save metrics to CSV for further analysis
    import csv
    csv_path = './figs_aircraft/training_metrics.csv'
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['epoch', 'train_loss', 'train_kl', 'train_rec', 'train_clf', 'val_f1', 'val_acc']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(history['epoch'])):
            writer.writerow({
                'epoch': history['epoch'][i],
                'train_loss': history['train_loss'][i],
                'train_kl': history['train_kl'][i],
                'train_rec': history['train_rec'][i],
                'train_clf': history['train_clf'][i],
                'val_f1': history['val_f1'][i],
                'val_acc': history['val_acc'][i],
            })
    print(f"✓ Training metrics saved to {csv_path}")
