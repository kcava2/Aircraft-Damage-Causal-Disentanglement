# inference_aircraft.py
# Place this file in the root of trustworthyAI/research/CausalVAE/
# i.e. at the same level as run_aircraft.py

import os
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision.utils import save_image
from torchvision import transforms

# ── Original repo imports ────────────────────────────────────────────────────
from codebase import utils as ut

# ── Our new file imports ─────────────────────────────────────────────────────
from codebase.models.mask_vae_aircraft import CausalVAE
from dataset.aircraft_damage import SCALE, N_CONCEPTS, CLASS_NAMES, get_transforms
from codebase.models.classifier_head import MultiLabelHead

# ── Device ───────────────────────────────────────────────────────────────────
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ─────────────────────────────────────────────────────────────────────────────
# Load checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint(checkpoint_path: str):
    """
    Load a saved checkpoint and reconstruct both the CausalVAE
    and classifier head in eval mode.
    """
    ckpt   = torch.load(checkpoint_path, map_location=device)
    config = ckpt.get('config', {'z_dim': 20, 'z1_dim': 5, 'z2_dim': 4})

    z_dim  = config.get('z_dim',  20)
    z1_dim = config.get('z1_dim',  5)
    z2_dim = config.get('z2_dim',  4)

    lvae = CausalVAE(
        name    = 'aircraft_causalvae',
        z_dim   = z_dim,
        z1_dim  = z1_dim,
        z2_dim  = z2_dim,
        channel = 3,
        scale   = SCALE,
        initial = False,   # do not re-initialise DAG — load from checkpoint
    ).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    lvae.load_state_dict(ckpt['lvae'])
    lvae.eval()

    clf = MultiLabelHead(
        in_dim    = z_dim,
        n_classes = N_CONCEPTS,
    ).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    clf.load_state_dict(ckpt['clf'])
    clf.eval()

    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}  "
          f"val_f1={ckpt.get('val_f1', 0):.4f}")
    print(f"Learned DAG:\n{lvae.dag.A.data}")

    return lvae, clf, z1_dim, z2_dim, z_dim


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def denorm(t: torch.Tensor) -> torch.Tensor:
    """Convert [-1,1] normalised tensor to [0,1] for saving."""
    return (t * 0.5 + 0.5).clamp(0, 1)


def encode_image(lvae, img_tensor: torch.Tensor, u: torch.Tensor, z1_dim: int, z2_dim: int):
    """
    Run the encoder and DAG layer to get z_given_dag.
    Returns z_given_dag shape (1, z_dim) and the mean q_m shape (1, z1_dim, z2_dim).
    """
    with torch.no_grad():
        q_m, q_v, skips = lvae.enc.encode(img_tensor)
        q_m = q_m.reshape([q_m.size(0), z1_dim, z2_dim])
        q_v = torch.ones_like(q_m)

        decode_m, decode_v = lvae.dag.calculate_dag(q_m, q_v)
        decode_m = decode_m.reshape([q_m.size(0), z1_dim, z2_dim])

        m_zm = lvae.dag.mask_z(decode_m).reshape([q_m.size(0), z1_dim, z2_dim])
        m_u  = lvae.dag.mask_u(u)

        f_z      = lvae.mask_z.mix(m_zm).reshape([q_m.size(0), z1_dim, z2_dim])
        e_tilde  = lvae.attn.attention(
            decode_m.reshape([q_m.size(0), z1_dim, z2_dim]),
            q_m.reshape([q_m.size(0), z1_dim, z2_dim])
        )[0]
        f_z1 = f_z + e_tilde

        lambdav   = 0.001
        z_given_dag = ut.conditional_sample_gaussian(
            f_z1, torch.ones_like(f_z1) * lambdav
        )
        z_given_dag = z_given_dag.reshape([q_m.size(0), -1])

    return z_given_dag, f_z1, skips


def decode_z(lvae, z_flat: torch.Tensor, u: torch.Tensor, skips=None):
    """Decode a flat z vector back to an image."""
    with torch.no_grad():
        recon, _, _, _, _ = lvae.dec.decode_sep(z_flat, u, skips)
    return recon


# ─────────────────────────────────────────────────────────────────────────────
# Classification
# ─────────────────────────────────────────────────────────────────────────────

def classify_image(
    lvae, clf,
    img_path:   str,
    z1_dim:     int,
    z2_dim:     int,
    z_dim:      int,
    threshold:  float = 0.5,
    output_dir: str   = './results',
    save_recon: bool  = True,
) -> dict:
    """
    Classify damage types in a single image.
    Returns a dict with probabilities, predictions, and causal notes.
    """
    os.makedirs(output_dir, exist_ok=True)
    transform  = get_transforms('test', 96)
    img_pil    = Image.open(img_path).convert('RGB')
    img_tensor = transform(img_pil).unsqueeze(0).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Zero concept label at inference (no ground truth available)
    u = torch.zeros(1, N_CONCEPTS, device=img_tensor.device)

    z_given_dag, f_z1, skips = encode_image(lvae, img_tensor, u, z1_dim, z2_dim)

    # Classification
    with torch.no_grad():
        logits = clf(z_given_dag)[0]
        probs  = torch.sigmoid(logits).cpu().numpy()
        preds  = (probs >= threshold).astype(int)

    damage_found = [CLASS_NAMES[i] for i, p in enumerate(preds) if p == 1]

    # Causal notes based on detected damage and DAG
    causal_notes = []
    dag_weights  = lvae.dag.A.data.cpu()

    if preds[CLASS_NAMES.index('dent')] == 1:
        if preds[CLASS_NAMES.index('scratch')] == 1:
            w = dag_weights[CLASS_NAMES.index('scratch'), CLASS_NAMES.index('dent')].item()
            causal_notes.append(
                f"Scratch likely downstream of dent (DAG edge weight: {w:.3f})"
            )
        if preds[CLASS_NAMES.index('crack')] == 1:
            w = dag_weights[CLASS_NAMES.index('crack'), CLASS_NAMES.index('dent')].item()
            causal_notes.append(
                f"Crack likely downstream of dent (DAG edge weight: {w:.3f})"
            )
    if preds[CLASS_NAMES.index('scratch')] == 1:
        if preds[CLASS_NAMES.index('paint_off')] == 1:
            w = dag_weights[CLASS_NAMES.index('paint_off'), CLASS_NAMES.index('scratch')].item()
            causal_notes.append(
                f"Paint loss likely downstream of scratch (DAG edge weight: {w:.3f})"
            )

    # Save reconstruction
    stem = Path(img_path).stem
    if save_recon:
        recon = decode_z(lvae, z_given_dag, u, skips)
        save_image(denorm(img_tensor), f'{output_dir}/{stem}_original.png')
        save_image(torch.sigmoid(recon), f'{output_dir}/{stem}_reconstruction.png')

    result = {
        'image':        str(img_path),
        'damage_found': damage_found,
        'probabilities': {CLASS_NAMES[i]: round(float(probs[i]), 4)
                          for i in range(N_CONCEPTS)},
        'predictions':   {CLASS_NAMES[i]: int(preds[i])
                          for i in range(N_CONCEPTS)},
        'causal_notes':  causal_notes,
    }

    # Print summary
    print(f"\n{'='*55}")
    print(f"  Image : {Path(img_path).name}")
    print(f"  Damage: {', '.join(damage_found) if damage_found else 'none detected'}")
    for name, prob in result['probabilities'].items():
        bar = '█' * int(prob * 20) + '░' * (20 - int(prob * 20))
        print(f"  {name:<15} {bar} {prob:.2f}")
    if causal_notes:
        print("  Causal reasoning:")
        for note in causal_notes:
            print(f"    → {note}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Counterfactual generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_counterfactuals(
    lvae, clf,
    img_path:    str,
    z1_dim:      int,
    z2_dim:      int,
    z_dim:       int,
    output_dir:  str   = './results',
    threshold:   float = 0.5,
):
    """
    For each detected damage concept, generate a counterfactual image
    showing what the panel would look like without that damage.
    Also shows whether downstream concepts reduce — confirming causality.
    """
    os.makedirs(output_dir, exist_ok=True)
    transform  = get_transforms('test', 96)
    img_pil    = Image.open(img_path).convert('RGB')
    img_tensor = transform(img_pil).unsqueeze(0).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    u          = torch.zeros(1, N_CONCEPTS, device=img_tensor.device)
    stem       = Path(img_path).stem

    z_given_dag, f_z1, skips = encode_image(lvae, img_tensor, u, z1_dim, z2_dim)

    with torch.no_grad():
        logits = clf(z_given_dag)[0]
        probs  = torch.sigmoid(logits).cpu().numpy()
        preds  = (probs >= threshold).astype(int)

    # Save original and reconstruction
    recon_orig = decode_z(lvae, z_given_dag, u, skips)
    save_image(denorm(img_tensor),        f'{output_dir}/{stem}_original.png')
    save_image(torch.sigmoid(recon_orig), f'{output_dir}/{stem}_reconstruction.png')

    cf_results = []

    # Only generate counterfactuals for detected damage types
    for concept_idx in range(N_CONCEPTS):
        if preds[concept_idx] == 0:
            continue

        concept_name = CLASS_NAMES[concept_idx]

        # Intervene: set this concept's z sub-vector to -1 (force absent)
        f_z1_cf = f_z1.clone()
        f_z1_cf[:, concept_idx, :] = -1.0

        z_cf = f_z1_cf.reshape([1, -1])

        # Decode counterfactual (reuse skips from original image encoding)
        recon_cf = decode_z(lvae, z_cf, u, skips)
        save_image(
            torch.sigmoid(recon_cf),
            f'{output_dir}/{stem}_no_{concept_name}.png'
        )

        # Check how other concept probabilities change
        with torch.no_grad():
            logits_cf   = clf(z_cf)[0]
            probs_cf    = torch.sigmoid(logits_cf).cpu().numpy()

        changes = {}
        for j, name in enumerate(CLASS_NAMES):
            delta = float(probs_cf[j]) - float(probs[j])
            if abs(delta) > 0.05:   # only report meaningful changes
                changes[name] = round(delta, 3)

        cf_result = {
            'intervention':    f'remove {concept_name}',
            'concept_removed': concept_name,
            'prob_changes':    changes,
            'saved_to':        f'{output_dir}/{stem}_no_{concept_name}.png',
        }
        cf_results.append(cf_result)

        print(f"\n  Counterfactual: remove '{concept_name}'")
        for name, delta in changes.items():
            direction = '▼' if delta < 0 else '▲'
            print(f"    {name:<15} {direction} {abs(delta):.3f}  "
                  f"{'← causal downstream' if delta < -0.05 else ''}")

    return cf_results


# ─────────────────────────────────────────────────────────────────────────────
# DAG visualisation
# ─────────────────────────────────────────────────────────────────────────────

def print_learned_dag(lvae):
    """Print the learned DAG edge weights in a readable format."""
    A = lvae.dag.A.data.cpu().numpy()
    print(f"\n{'='*55}")
    print("  Learned DAG Edge Weights")
    print(f"  (rows=children, cols=parents)")
    print(f"\n  {'':15}", end='')
    for name in CLASS_NAMES:
        print(f"{name[:6]:>8}", end='')
    print()
    for i, child in enumerate(CLASS_NAMES):
        print(f"  {child:<15}", end='')
        for j in range(N_CONCEPTS):
            val = A[i, j]
            marker = f'{val:8.3f}' if abs(val) > 0.01 else '       -'
            print(marker, end='')
        print()
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Batch processing
# ─────────────────────────────────────────────────────────────────────────────

def process_folder(
    lvae, clf,
    img_dir:     str,
    z1_dim:      int,
    z2_dim:      int,
    z_dim:       int,
    output_dir:  str   = './results',
    threshold:   float = 0.5,
    do_cf:       bool  = True,
):
    """Run classification + counterfactuals on every image in a folder."""
    import csv

    os.makedirs(output_dir, exist_ok=True)
    img_paths = sorted([
        str(p) for p in Path(img_dir).iterdir()
        if p.suffix.lower() in IMG_EXTENSIONS
    ])
    print(f"Processing {len(img_paths)} images from {img_dir}")

    all_results = []
    for img_path in img_paths:
        stem     = Path(img_path).stem
        img_out  = os.path.join(output_dir, stem)
        os.makedirs(img_out, exist_ok=True)

        result = classify_image(
            lvae, clf, img_path, z1_dim, z2_dim, z_dim,
            threshold=threshold, output_dir=img_out
        )

        if do_cf:
            cf_results = generate_counterfactuals(
                lvae, clf, img_path, z1_dim, z2_dim, z_dim,
                output_dir=img_out, threshold=threshold
            )
            result['counterfactuals'] = cf_results

        # Save per-image JSON report
        with open(os.path.join(img_out, 'report.json'), 'w') as f:
            json.dump(result, f, indent=2, default=str)

        all_results.append(result)

    # Summary CSV
    csv_path = os.path.join(output_dir, 'summary.csv')
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['image'] + CLASS_NAMES + ['damage_found', 'causal_notes']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            row = {'image': Path(r['image']).name}
            for cls in CLASS_NAMES:
                row[cls] = r['predictions'][cls]
            row['damage_found']  = ','.join(r['damage_found'])
            row['causal_notes']  = ' | '.join(r['causal_notes'])
            writer.writerow(row)

    print(f"\nSummary saved to {csv_path}")
    print(f"Results saved to {output_dir}/")
    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',  type=str, required=True,
                   help='Path to best_model.pt from training')
    p.add_argument('--image',       type=str, default=None,
                   help='Single image path')
    p.add_argument('--image_dir',   type=str, default=None,
                   help='Folder of images for batch mode')
    p.add_argument('--output_dir',  type=str, default='./results')
    p.add_argument('--threshold',   type=float, default=0.5)
    p.add_argument('--no_cf',       action='store_true',
                   help='Skip counterfactual generation')
    args = p.parse_args()

    # Load model
    lvae, clf, z1_dim, z2_dim, z_dim = load_checkpoint(args.checkpoint)

    # Print learned DAG
    print_learned_dag(lvae)

    if args.image:
        # Single image
        result = classify_image(
            lvae, clf, args.image,
            z1_dim, z2_dim, z_dim,
            threshold  = args.threshold,
            output_dir = args.output_dir,
        )
        if not args.no_cf:
            generate_counterfactuals(
                lvae, clf, args.image,
                z1_dim, z2_dim, z_dim,
                output_dir = args.output_dir,
                threshold  = args.threshold,
            )
        with open(os.path.join(args.output_dir, 'report.json'), 'w') as f:
            json.dump(result, f, indent=2, default=str)

    elif args.image_dir:
        # Batch folder
        process_folder(
            lvae, clf, args.image_dir,
            z1_dim, z2_dim, z_dim,
            output_dir = args.output_dir,
            threshold  = args.threshold,
            do_cf      = not args.no_cf,
        )

    else:
        print("Provide --image or --image_dir")
