"""Loads what the simulator's dataset mode wrote."""
from __future__ import annotations

import pathlib

import numpy as np
import torch

from siso_nn.model import FEATURES


def load(directory) -> tuple[torch.Tensor, torch.Tensor]:
    d = pathlib.Path(directory)
    x = np.fromfile(d / "X.bin", dtype=np.float32).reshape(-1, FEATURES)
    y = np.fromfile(d / "y.bin", dtype=np.float32)
    if len(x) != len(y):
        raise ValueError(f"{len(x)} features rows against {len(y)} targets")
    return torch.from_numpy(x), torch.from_numpy(y)


def split(x, y, holdout: float = 0.1, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(x), generator=g)
    n = int(len(x) * (1.0 - holdout))
    return (x[idx[:n]], y[idx[:n]]), (x[idx[n:]], y[idx[n:]])
