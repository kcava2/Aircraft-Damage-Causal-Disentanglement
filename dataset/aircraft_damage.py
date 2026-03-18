"""
AircraftDamageDataset
=====================
Converts YOLO bounding-box annotations from the Innovation Hangar v2 dataset
into the image-level concept vectors that CausalVAE expects.

YOLO class mapping (from data.yaml):
  0 = crack
  1 = dent
  2 = missing-head
  3 = paint-off
  4 = scratch

Concept vector (5 dimensions, one per damage type):
  u = [crack_present, dent_present, missing_head_present, paint_off_present, scratch_present]
  Each value is 0.0 (absent) or 1.0 (present) in any bounding box in the image.
  This is then normalised to [-1, 1] using the scale array below.

Causal DAG we assume for this dataset:
  Impact force (latent) --> dent --> scratch --> paint_off
                        --> crack
  Fastener failure      --> missing_head
  No strong causal link between the 5 classes is provable from data alone,
  so we use a weak prior DAG and let the model refine it.
"""

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# --------------------------------------------------------------------------- #
# Class index  -> human name
# --------------------------------------------------------------------------- #
CLASS_NAMES = ["crack", "dent", "missing_head", "paint_off", "scratch"]
N_CONCEPTS   = len(CLASS_NAMES)   # 5

# --------------------------------------------------------------------------- #
# scale[i] = [mean, half_range] used to normalise concept i to [-1, 1].
# Because every concept is binary (0 or 1):
#   mean       = 0.5
#   half_range = 0.5
# So:  normalised = (raw - 0.5) / 0.5  --> maps 0 -> -1, 1 -> +1
# --------------------------------------------------------------------------- #
SCALE = np.array([[0.5, 0.5]] * N_CONCEPTS, dtype=np.float32)   # shape (5, 2)


def parse_yolo_label(label_path: str) -> np.ndarray:
    """
    Read a YOLO .txt file and return a binary presence vector of length N_CONCEPTS.
    Each row in the file: <class_id> <cx> <cy> <w> <h>
    We only need the class_id column.
    """
    presence = np.zeros(N_CONCEPTS, dtype=np.float32)
    if not os.path.exists(label_path):
        return presence                      # no label file = no damage
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            class_id = int(line.split()[0])
            if 0 <= class_id < N_CONCEPTS:
                presence[class_id] = 1.0
    return presence


class AircraftDamageDataset(Dataset):
    """
    Parameters
    ----------
    img_dir   : str  – folder containing .jpg / .png images
    label_dir : str  – folder containing matching YOLO .txt label files
                       (same stem as image, e.g. img001.jpg -> img001.txt)
    transform : optional torchvision transform (applied to the PIL image)
    split     : 'train' | 'valid' | 'test'  (informational only)
    """

    IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    def __init__(
        self,
        img_dir: str,
        label_dir: str,
        transform=None,
        split: str = "train",
    ):
        self.img_dir   = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.transform = transform
        self.split     = split
        self.scale     = SCALE                         # (5, 2)

        # Collect all image paths
        self.samples = sorted([
            p for p in self.img_dir.iterdir()
            if p.suffix.lower() in self.IMG_EXTENSIONS
        ])

        if len(self.samples) == 0:
            raise FileNotFoundError(
                f"No images found in {img_dir}. "
                "Check your path and file extensions."
            )

        print(f"[AircraftDamageDataset] {split}: {len(self.samples)} images "
              f"from {img_dir}")

    # ------------------------------------------------------------------ #

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path   = self.samples[idx]
        label_path = self.label_dir / (img_path.stem + ".txt")

        # --- image ---
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # --- label ---
        presence_raw  = parse_yolo_label(str(label_path))          # (5,) in [0,1]
        presence_norm = (presence_raw - self.scale[:, 0]) / self.scale[:, 1]
        u = torch.tensor(presence_norm, dtype=torch.float32)       # (5,)

        return image, u

    # ------------------------------------------------------------------ #

    def class_distribution(self) -> dict:
        """Utility: count how many images contain each damage type."""
        counts = np.zeros(N_CONCEPTS, dtype=int)
        for img_path in self.samples:
            label_path = self.label_dir / (img_path.stem + ".txt")
            pres = parse_yolo_label(str(label_path))
            counts += pres.astype(int)
        return {name: int(counts[i]) for i, name in enumerate(CLASS_NAMES)}


# --------------------------------------------------------------------------- #
# Default transforms
# --------------------------------------------------------------------------- #

def get_transforms(split: str = "train", img_size: int = 64):
    """
    Returns torchvision transforms appropriate for each split.
    CausalVAE uses 64x64 by default.  Increase img_size if you modify the
    encoder/decoder architecture.
    """
    if split == "train":
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5]),
        ])
