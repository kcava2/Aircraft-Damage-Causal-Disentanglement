"""
aircraft_dag.py
===============
Defines the prior causal DAG (adjacency matrix A) for the 5 aircraft damage
concepts:

  Index  Concept
  -----  -----------
    0    crack
    1    dent
    2    missing_head
    3    paint_off
    4    scratch

Assumed causal structure (weakly informed by domain knowledge):

  dent  ──► scratch ──► paint_off
  dent  ──► crack
  (missing_head is independent of mechanical damage types)

Adjacency matrix A where A[i, j] = 1 means concept j causes concept i
(i.e. parent → child in the DAG):

         crack  dent  miss  poff  scratch
  crack  [  0     1     0     0     0  ]   crack caused by dent
  dent   [  0     0     0     0     0  ]   dent has no parents here
  miss   [  0     0     0     0     0  ]   missing_head is independent
  poff   [  0     0     0     0     1  ]   paint_off caused by scratch
  scr    [  0     1     0     0     0  ]   scratch caused by dent

This is a PRIOR – CausalVAE will refine the edge weights during training
(the DAG layer uses continuous relaxation + acyclicity constraint).

If you want a fully data-driven DAG (no prior), set A_INIT to all zeros.
"""

import torch

# 5 x 5 adjacency matrix  (rows = children, cols = parents)
A_INIT = torch.tensor([
    [0, 1, 0, 0, 0],   # crack       <-- dent
    [0, 0, 0, 0, 0],   # dent        (root)
    [0, 0, 0, 0, 0],   # missing_head (root)
    [0, 0, 0, 0, 1],   # paint_off   <-- scratch
    [0, 1, 0, 0, 0],   # scratch     <-- dent
], dtype=torch.float32)


def get_dag_init() -> torch.Tensor:
    """Return a copy of the prior adjacency matrix."""
    return A_INIT.clone()
