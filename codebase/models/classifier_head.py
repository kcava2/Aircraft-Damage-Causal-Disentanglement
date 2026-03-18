"""
classifier_head.py
==================
A lightweight MLP classification head that operates on CausalVAE's latent
representation z to predict which damage types are present.

Architecture
------------
CausalVAE produces z of shape (batch, n_concept, z_dim).
  - n_concept = 5   (one per damage type)
  - z_dim     = 4   (default in original code; 16 total latent dims)

We flatten z → (batch, n_concept * z_dim) then classify with an MLP.

Two heads are provided:
  1. MultiLabelHead  – sigmoid outputs, one per class (recommended: allows
                       multiple damage types per image, matches the data)
  2. MultiClassHead  – softmax, predicts the single most prominent damage type
                       (use only if you want 1-of-N classification)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLabelHead(nn.Module):
    """
    Multi-label binary classifier.
    Output: (batch, n_classes) – each in [0, 1] via sigmoid.
    Loss:   BCEWithLogitsLoss (pass raw logits, not sigmoid output).

    Parameters
    ----------
    in_dim    : int – flattened latent size, i.e. n_concept * z_dim
    n_classes : int – number of damage types (5 for this dataset)
    dropout   : float – dropout probability
    """

    def __init__(self, in_dim: int, n_classes: int = 5, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),          # raw logits, no activation
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z : (batch, n_concept, z_dim)  or  (batch, n_concept * z_dim)
        returns logits : (batch, n_classes)
        """
        x = z.view(z.size(0), -1)             # flatten
        return self.net(x)                     # logits

    def predict(self, z: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Returns binary predictions (0/1) at given threshold."""
        logits = self.forward(z)
        return (torch.sigmoid(logits) >= threshold).float()


class MultiClassHead(nn.Module):
    """
    Single-label multi-class classifier (softmax).
    Only use if each image has exactly one dominant damage type.
    Loss: CrossEntropyLoss.
    """

    def __init__(self, in_dim: int, n_classes: int = 5, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = z.view(z.size(0), -1)
        return self.net(x)                     # raw logits for CrossEntropyLoss


# --------------------------------------------------------------------------- #
# Loss helpers
# --------------------------------------------------------------------------- #

def multilabel_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    logits  : (batch, n_classes)  – raw output from MultiLabelHead
    targets : (batch, n_classes)  – binary float, 1.0 if class present
    """
    return F.binary_cross_entropy_with_logits(logits, targets)


def multiclass_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    logits  : (batch, n_classes)
    targets : (batch,) long – class index of dominant damage type
    """
    return F.cross_entropy(logits, targets)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    class_names: list = None,
) -> dict:
    """
    Compute per-class and overall accuracy / F1 for multi-label classification.

    Parameters
    ----------
    logits   : (batch, n_classes)
    targets  : (batch, n_classes) binary float
    threshold: decision boundary

    Returns a dict with keys: accuracy, f1_macro, per_class
    """
    preds   = (torch.sigmoid(logits) >= threshold).float()
    targets = targets.float()

    tp = (preds * targets).sum(dim=0)
    fp = (preds * (1 - targets)).sum(dim=0)
    fn = ((1 - preds) * targets).sum(dim=0)

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    exact_match = (preds == targets).all(dim=1).float().mean().item()

    result = {
        "exact_match_accuracy": exact_match,
        "f1_macro": f1.mean().item(),
        "per_class": {}
    }

    names = class_names or [f"class_{i}" for i in range(logits.size(1))]
    for i, name in enumerate(names):
        result["per_class"][name] = {
            "f1":        f1[i].item(),
            "precision": precision[i].item(),
            "recall":    recall[i].item(),
        }

    return result
