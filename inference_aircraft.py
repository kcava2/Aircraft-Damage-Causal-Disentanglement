# inference_aircraft.py

import os
import json
import csv
import torch
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from torchvision.utils import save_image
from torchvision import transforms

from codebase import utils as ut
from codebase.models.mask_vae_aircraft import CausalVAE
from dataset.aircraft_damage import SCALE, N_CONCEPTS, CLASS_NAMES, get_transforms
from codebase.models.classifier_head import MultiLabelHead

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# ── Colours per damage class (RGB) ───────────────────────────────────────────
CLASS_COLOURS = {
    "crack":     (220,  50,  50),   # red
    "dent":      (255, 165,   0),   # orange
    "paint_off": (50,  180,  50),   # green
    "scratch":   (80,  140, 255),   # blue
}

# ─────────────────────────────────────────────────────────────────────────────
# Load checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint(checkpoint_path: str):
    ckpt   = torch.load(checkpoint_path, map_location=device)
    config = ckpt.get('config', {'z_dim': 16, 'z1_dim': 4, 'z2_dim': 4})
    z_dim  = config.get('z_dim',  16)
    z1_dim = config.get('z1_dim',  4)
    z2_dim = config.get('z2_dim',  4)

    lvae = CausalVAE(
        name    = 'aircraft_causalvae',
        z_dim   = z_dim,
        z1_dim  = z1_dim,
        z2_dim  = z2_dim,
        channel = 3,
        scale   = SCALE,
        initial = False,
    ).to(device)
    lvae.load_state_dict(ckpt['lvae'])
    lvae.eval()

    clf = MultiLabelHead(in_dim=z_dim, n_classes=N_CONCEPTS).to(device)
    clf.load_state_dict(ckpt['clf'])
    clf.eval()

    print(f"Loaded checkpoint — epoch {ckpt.get('epoch','?')}  "
          f"val_f1={ckpt.get('val_f1',0):.4f}")
    return lvae, clf, z1_dim, z2_dim, z_dim


# ─────────────────────────────────────────────────────────────────────────────
# Encode / decode helpers
# ─────────────────────────────────────────────────────────────────────────────

def encode_image(lvae, img_tensor, u, z1_dim, z2_dim):
    with torch.no_grad():
        q_m, q_v, skips = lvae.enc.encode(img_tensor)
        q_m = q_m.reshape([q_m.size(0), z1_dim, z2_dim])
        q_v = torch.ones_like(q_m)

        decode_m, decode_v = lvae.dag.calculate_dag(q_m, q_v)
        decode_m = decode_m.reshape([q_m.size(0), z1_dim, z2_dim])

        m_zm    = lvae.dag.mask_z(decode_m).reshape([q_m.size(0), z1_dim, z2_dim])
        f_z     = lvae.mask_z.mix(m_zm).reshape([q_m.size(0), z1_dim, z2_dim])
        e_tilde = lvae.attn.attention(
            decode_m.reshape([q_m.size(0), z1_dim, z2_dim]),
            q_m.reshape([q_m.size(0), z1_dim, z2_dim])
        )[0]
        f_z1 = f_z + e_tilde

        z_given_dag = ut.conditional_sample_gaussian(
            f_z1, torch.ones_like(f_z1) * 0.001
        ).reshape([q_m.size(0), -1])

    return z_given_dag, f_z1, skips


def decode_z(lvae, z_flat, skips=None):
    with torch.no_grad():
        recon = lvae.dec.unet(z_flat, skips=skips)
    return torch.sigmoid(recon)


def compute_recon_metrics(orig_t: torch.Tensor, cf_t: torch.Tensor) -> dict:
    """Compute pixel-level distance between two (1,3,H,W) tensors in [0,1]."""
    with torch.no_grad():
        mse  = torch.nn.functional.mse_loss(cf_t, orig_t).item()
        mae  = torch.nn.functional.l1_loss(cf_t, orig_t).item()
        psnr = 10 * np.log10(1.0 / (mse + 1e-10))
    return {'mse': round(mse, 6), 'mae': round(mae, 6), 'psnr': round(psnr, 3)}


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """Convert a (3,H,W) tensor in [0,1] to a PIL image."""
    arr = (t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def load_font(size=14):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        try:
            return ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
        except Exception:
            return ImageFont.load_default()


# ─────────────────────────────────────────────────────────────────────────────
# Panel 1 — Original + Reconstruction + Damage percentages
# ─────────────────────────────────────────────────────────────────────────────

def build_analysis_panel(
    orig_pil:   Image.Image,
    recon_pil:  Image.Image,
    probs:      np.ndarray,
    preds:      np.ndarray,
    causal_notes: list,
    img_name:   str,
) -> Image.Image:
    """
    Produces a single image containing:
      Left half:  original image
      Right half: reconstruction image
      Below:      damage probability bars + causal notes
    """
    IMG_W, IMG_H = 256, 256
    BAR_H        = 28
    NOTE_H       = 22
    PADDING      = 12
    TITLE_H      = 36

    n_bars  = N_CONCEPTS
    n_notes = len(causal_notes)
    bar_section_h  = n_bars * BAR_H + PADDING * 2
    note_section_h = (n_notes * NOTE_H + PADDING) if n_notes > 0 else 0
    total_h = TITLE_H + IMG_H + bar_section_h + note_section_h + PADDING
    total_w = IMG_W * 2 + PADDING * 3

    canvas = Image.new("RGB", (total_w, total_h), (30, 30, 35))
    draw   = ImageDraw.Draw(canvas)

    font_title  = load_font(15)
    font_label  = load_font(13)
    font_small  = load_font(11)
    font_note   = load_font(11)

    # ── Title ─────────────────────────────────────────────────────────────────
    draw.text((PADDING, 8), f"Aircraft Damage Analysis — {img_name}",
              fill=(220, 220, 220), font=font_title)

    # ── Images ────────────────────────────────────────────────────────────────
    y_img = TITLE_H
    orig_r  = orig_pil.resize((IMG_W, IMG_H), Image.LANCZOS)
    recon_r = recon_pil.resize((IMG_W, IMG_H), Image.LANCZOS)

    canvas.paste(orig_r,  (PADDING, y_img))
    canvas.paste(recon_r, (PADDING * 2 + IMG_W, y_img))

    # Image labels
    draw.text((PADDING + 4,            y_img + 4), "Original",
              fill=(255,255,255), font=font_small)
    draw.text((PADDING * 2 + IMG_W + 4, y_img + 4), "Reconstruction",
              fill=(255,255,255), font=font_small)

    # ── Damage probability bars ───────────────────────────────────────────────
    y_bar  = y_img + IMG_H + PADDING
    BAR_MAX_W = total_w - PADDING * 2 - 110   # leave space for label + number

    for i, cls_name in enumerate(CLASS_NAMES):
        prob    = float(probs[i])
        pred    = int(preds[i])
        colour  = CLASS_COLOURS.get(cls_name, (180, 180, 180))
        y       = y_bar + i * BAR_H

        # Background track
        draw.rectangle(
            [110 + PADDING, y + 6, 110 + PADDING + BAR_MAX_W, y + BAR_H - 4],
            fill=(60, 60, 65)
        )
        # Filled portion
        filled_w = int(prob * BAR_MAX_W)
        if filled_w > 0:
            draw.rectangle(
                [110 + PADDING, y + 6, 110 + PADDING + filled_w, y + BAR_H - 4],
                fill=colour
            )

        # Class label
        label_colour = colour if pred == 1 else (140, 140, 140)
        marker = "●" if pred == 1 else "○"
        draw.text((PADDING, y + 7), f"{marker} {cls_name}",
                  fill=label_colour, font=font_label)

        # Percentage
        draw.text(
            (110 + PADDING + BAR_MAX_W + 8, y + 7),
            f"{prob*100:.1f}%",
            fill=(200, 200, 200), font=font_label
        )

    # ── Causal notes ──────────────────────────────────────────────────────────
    if causal_notes:
        y_note = y_bar + bar_section_h
        draw.text((PADDING, y_note), "Causal Reasoning:",
                  fill=(180, 180, 100), font=font_label)
        for k, note in enumerate(causal_notes):
            draw.text((PADDING, y_note + NOTE_H + k * NOTE_H),
                      f"→ {note}",
                      fill=(160, 200, 160), font=font_note)

    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# Panel 2 — Counterfactual comparison
# ─────────────────────────────────────────────────────────────────────────────

def build_counterfactual_panel(
    orig_pil:     Image.Image,
    recon_pil:    Image.Image,
    cf_images:    dict,          # {concept_name: PIL image}
    probs_orig:   np.ndarray,
    cf_probs:     dict,          # {concept_name: np.ndarray of probs}
    img_name:     str,
) -> Image.Image:
    """
    Produces a counterfactual comparison panel.
    Row 1: original | reconstruction | no_crack | no_dent | no_paint_off | no_scratch
    Row 2: probability bars for each column
    """
    IMG_W, IMG_H  = 180, 180
    BAR_SECTION_H = N_CONCEPTS * 22 + 20
    TITLE_H       = 34
    PADDING       = 10
    LABEL_H       = 24

    all_cols   = ["original", "reconstruction"] + list(cf_images.keys())
    n_cols     = len(all_cols)
    total_w    = n_cols * (IMG_W + PADDING) + PADDING
    total_h    = TITLE_H + LABEL_H + IMG_H + BAR_SECTION_H + PADDING

    canvas = Image.new("RGB", (total_w, total_h), (30, 30, 35))
    draw   = ImageDraw.Draw(canvas)

    font_title = load_font(14)
    font_label = load_font(12)
    font_small = load_font(10)

    draw.text((PADDING, 8),
              f"Counterfactual Analysis — {img_name}",
              fill=(220, 220, 220), font=font_title)

    # Map column name → (image, probs)
    col_data = {"original": (orig_pil, probs_orig),
                "reconstruction": (recon_pil, probs_orig)}
    for cname, cpil in cf_images.items():
        col_data[cname] = (cpil, cf_probs[cname])

    for col_idx, col_name in enumerate(all_cols):
        x       = PADDING + col_idx * (IMG_W + PADDING)
        img_pil, col_probs = col_data[col_name]

        # Column label
        label = col_name.replace("_", " ")
        if col_name not in ("original", "reconstruction"):
            label = f"no {label}"
        draw.text((x, TITLE_H), label,
                  fill=(200, 200, 200), font=font_label)

        # Image
        y_img = TITLE_H + LABEL_H
        resized = img_pil.resize((IMG_W, IMG_H), Image.LANCZOS)
        canvas.paste(resized, (x, y_img))

        # Mini probability bars
        y_bars   = y_img + IMG_H + 8
        bar_maxw = IMG_W - 50

        for i, cls_name in enumerate(CLASS_NAMES):
            prob   = float(col_probs[i])
            colour = CLASS_COLOURS.get(cls_name, (180, 180, 180))
            yb     = y_bars + i * 22

            # Comparison delta if this is a counterfactual column
            if col_name not in ("original", "reconstruction"):
                delta = prob - float(probs_orig[i])
                delta_colour = (100, 220, 100) if delta < -0.05 else \
                               (220, 100, 100) if delta >  0.05 else \
                               (160, 160, 160)
                delta_str = f"{delta:+.2f}"
            else:
                delta_colour = (180, 180, 180)
                delta_str    = f"{prob:.2f}"

            # Bar track
            draw.rectangle(
                [x + 36, yb + 4, x + 36 + bar_maxw, yb + 14],
                fill=(60, 60, 65)
            )
            # Filled
            fw = int(prob * bar_maxw)
            if fw > 0:
                draw.rectangle(
                    [x + 36, yb + 4, x + 36 + fw, yb + 14],
                    fill=colour
                )
            # Label
            draw.text((x, yb + 3),
                      cls_name[:4],
                      fill=(160, 160, 160), font=font_small)
            # Delta / value
            draw.text((x + 36 + bar_maxw + 3, yb + 3),
                      delta_str,
                      fill=delta_colour, font=font_small)

    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis function
# ─────────────────────────────────────────────────────────────────────────────

def analyse_image(
    lvae, clf,
    img_path:   str,
    z1_dim:     int,
    z2_dim:     int,
    z_dim:      int,
    output_dir: str   = './results',
    threshold:  float = 0.5,
    do_cf:      bool  = True,
) -> dict:

    os.makedirs(output_dir, exist_ok=True)
    stem      = Path(img_path).stem
    transform = get_transforms('test', 96)

    img_pil    = Image.open(img_path).convert('RGB')
    img_tensor = transform(img_pil).unsqueeze(0).to(device)
    u          = torch.zeros(1, N_CONCEPTS, device=device)

    # ── Encode ────────────────────────────────────────────────────────────────
    z_given_dag, f_z1, skips = encode_image(lvae, img_tensor, u, z1_dim, z2_dim)

    # ── Classify ──────────────────────────────────────────────────────────────
    with torch.no_grad():
        logits = clf(z_given_dag)[0]
        probs  = torch.sigmoid(logits).cpu().numpy()
        preds  = (probs >= threshold).astype(int)

    damage_found = [CLASS_NAMES[i] for i, p in enumerate(preds) if p == 1]

    # ── Causal notes from learned DAG ─────────────────────────────────────────
    dag_w        = lvae.dag.A.data.cpu().numpy()
    causal_notes = []
    idx          = {n: i for i, n in enumerate(CLASS_NAMES)}

    if preds[idx['dent']] == 1:
        if preds[idx['scratch']] == 1:
            w = dag_w[idx['scratch'], idx['dent']]
            causal_notes.append(f"Scratch downstream of dent (edge weight {w:.3f})")
        if preds[idx['crack']] == 1:
            w = dag_w[idx['crack'], idx['dent']]
            causal_notes.append(f"Crack downstream of dent (edge weight {w:.3f})")
    if preds[idx['scratch']] == 1 and preds[idx['paint_off']] == 1:
        w = dag_w[idx['paint_off'], idx['scratch']]
        causal_notes.append(f"Paint loss downstream of scratch (edge weight {w:.3f})")

    # ── Decode reconstruction ─────────────────────────────────────────────────
    recon_tensor = decode_z(lvae, z_given_dag, skips)     # (1,3,H,W) in [0,1]
    orig_display = (img_tensor * 0.5 + 0.5).clamp(0, 1)

    orig_pil  = tensor_to_pil(orig_display)
    recon_pil = tensor_to_pil(recon_tensor)
    recon_metrics = compute_recon_metrics(orig_display, recon_tensor)

    # ── Panel 1: analysis panel ───────────────────────────────────────────────
    panel1 = build_analysis_panel(
        orig_pil, recon_pil, probs, preds, causal_notes,
        img_name=Path(img_path).name
    )
    panel1.save(os.path.join(output_dir, f'{stem}_analysis.png'))
    print(f"\nSaved analysis panel → {stem}_analysis.png")

    # ── Counterfactuals ───────────────────────────────────────────────────────
    cf_result_list = []
    cf_images      = {}
    cf_probs_dict  = {}

    if do_cf:
        for concept_idx in range(N_CONCEPTS):
            concept_name = CLASS_NAMES[concept_idx]

            # Intervene: force concept absent (-1 in normalised space)
            f_z1_cf = f_z1.clone()
            f_z1_cf[:, concept_idx, :] = -1.0
            z_cf    = f_z1_cf.reshape([1, -1])

            # Decode counterfactual (use encoder skips for sharper U-Net output)
            recon_cf = decode_z(lvae, z_cf, skips)
            cf_pil   = tensor_to_pil(recon_cf)
            cf_images[concept_name] = cf_pil

            # Classify counterfactual
            with torch.no_grad():
                logits_cf = clf(z_cf)[0]
                probs_cf  = torch.sigmoid(logits_cf).cpu().numpy()
            cf_probs_dict[concept_name] = probs_cf

            # Compute classification changes
            changes = {}
            for j, name in enumerate(CLASS_NAMES):
                delta = float(probs_cf[j]) - float(probs[j])
                if abs(delta) > 0.03:
                    changes[name] = round(delta, 3)

            # Compute pixel-level distance from original to counterfactual
            cf_metrics = compute_recon_metrics(orig_display, recon_cf)

            cf_result_list.append({
                'concept_removed': concept_name,
                'prob_changes':    changes,
                'recon_metrics':   cf_metrics,
            })

            # Console output
            print(f"\n  Counterfactual: remove '{concept_name}'  "
                  f"MSE={cf_metrics['mse']:.4f}  PSNR={cf_metrics['psnr']:.1f}dB  "
                  f"MAE={cf_metrics['mae']:.4f}")
            if changes:
                for name, delta in changes.items():
                    direction = '▼' if delta < 0 else '▲'
                    causal_tag = ' ← causal downstream' if delta < -0.05 else ''
                    print(f"    {name:<15} {direction} {abs(delta):.3f}{causal_tag}")
            else:
                print(f"    No significant probability changes "
                      f"(concept may be causally independent)")

        # ── Panel 2: counterfactual panel ─────────────────────────────────────
        panel2 = build_counterfactual_panel(
            orig_pil, recon_pil, cf_images,
            probs, cf_probs_dict,
            img_name=Path(img_path).name
        )
        panel2.save(os.path.join(output_dir, f'{stem}_counterfactuals.png'))
        print(f"\nSaved counterfactual panel → {stem}_counterfactuals.png")

    # ── Print classification summary ──────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Image : {Path(img_path).name}")
    print(f"  Damage: {', '.join(damage_found) if damage_found else 'none detected'}")
    for name, prob in zip(CLASS_NAMES, probs):
        bar = '█' * int(prob * 20) + '░' * (20 - int(prob * 20))
        marker = '●' if prob >= threshold else '○'
        print(f"  {marker} {name:<15} {bar} {prob*100:.1f}%")
    if causal_notes:
        print("  Causal reasoning:")
        for note in causal_notes:
            print(f"    → {note}")

    # ── JSON report ───────────────────────────────────────────────────────────
    report = {
        'image':         str(img_path),
        'damage_found':  damage_found,
        'probabilities': {CLASS_NAMES[i]: round(float(probs[i]), 4)
                          for i in range(N_CONCEPTS)},
        'predictions':   {CLASS_NAMES[i]: int(preds[i])
                          for i in range(N_CONCEPTS)},
        'causal_notes':        causal_notes,
        'reconstruction_metrics': recon_metrics,
        'counterfactuals':     cf_result_list,
        'dag_weights':   {
            f"{CLASS_NAMES[i]}_to_{CLASS_NAMES[j]}": round(float(dag_w[i, j]), 4)
            for i in range(N_CONCEPTS) for j in range(N_CONCEPTS)
            if abs(dag_w[i, j]) > 0.01
        },
    }
    with open(os.path.join(output_dir, f'{stem}_report.json'), 'w') as f:
        json.dump(report, f, indent=2, default=str)

    return report


# ─────────────────────────────────────────────────────────────────────────────
# DAG printout
# ─────────────────────────────────────────────────────────────────────────────

def print_learned_dag(lvae):
    A = lvae.dag.A.data.cpu().numpy()
    print(f"\n{'='*55}")
    print("  Learned DAG Edge Weights (rows=children, cols=parents)")
    print(f"\n  {'':15}", end='')
    for name in CLASS_NAMES:
        print(f"{name[:7]:>9}", end='')
    print()
    for i, child in enumerate(CLASS_NAMES):
        print(f"  {child:<15}", end='')
        for j in range(N_CONCEPTS):
            val    = A[i, j]
            marker = f'{val:9.3f}' if abs(val) > 0.01 else '        -'
            print(marker, end='')
        print()
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Batch folder
# ─────────────────────────────────────────────────────────────────────────────

def process_folder(lvae, clf, img_dir, z1_dim, z2_dim, z_dim,
                   output_dir='./results', threshold=0.5, do_cf=True):
    os.makedirs(output_dir, exist_ok=True)
    img_paths = sorted([
        str(p) for p in Path(img_dir).iterdir()
        if p.suffix.lower() in IMG_EXTENSIONS
    ])
    print(f"Processing {len(img_paths)} images from {img_dir}")

    all_reports = []
    for img_path in img_paths:
        stem    = Path(img_path).stem
        img_out = os.path.join(output_dir, stem)
        os.makedirs(img_out, exist_ok=True)
        try:
            report = analyse_image(
                lvae, clf, img_path, z1_dim, z2_dim, z_dim,
                output_dir=img_out, threshold=threshold, do_cf=do_cf
            )
            all_reports.append(report)
        except Exception as e:
            print(f"ERROR on {img_path}: {e}")

    # Summary CSV
    csv_path = os.path.join(output_dir, 'summary.csv')
    with open(csv_path, 'w', newline='') as f:
        fieldnames = (['image'] + CLASS_NAMES +
                      ['damage_found', 'causal_notes'])
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_reports:
            row = {'image': Path(r['image']).name}
            for cls in CLASS_NAMES:
                row[cls] = r['predictions'][cls]
            row['damage_found'] = ','.join(r['damage_found'])
            row['causal_notes'] = ' | '.join(r['causal_notes'])
            writer.writerow(row)

    print(f"\nSummary CSV → {csv_path}")
    return all_reports


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--image',      type=str, default=None)
    p.add_argument('--image_dir',  type=str, default=None)
    p.add_argument('--output_dir', type=str, default='./results')
    p.add_argument('--threshold',  type=float, default=0.5)
    p.add_argument('--no_cf',      action='store_true')
    args = p.parse_args()

    lvae, clf, z1_dim, z2_dim, z_dim = load_checkpoint(args.checkpoint)
    print_learned_dag(lvae)

    if args.image:
        analyse_image(
            lvae, clf, args.image,
            z1_dim, z2_dim, z_dim,
            output_dir = args.output_dir,
            threshold  = args.threshold,
            do_cf      = not args.no_cf,
        )
    elif args.image_dir:
        process_folder(
            lvae, clf, args.image_dir,
            z1_dim, z2_dim, z_dim,
            output_dir = args.output_dir,
            threshold  = args.threshold,
            do_cf      = not args.no_cf,
        )
    else:
        print("Provide --image or --image_dir")