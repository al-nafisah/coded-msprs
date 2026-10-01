"""Loads what the simulator's dataset mode wrote."""
from __future__ import annotations

import pathlib

import numpy as np
import torch

from siso_nn.model import FEATURES


def load(directory) -> dict[str, torch.Tensor]:
    """Features x, bits b, and the BCJR extrinsic y."""
    d = pathlib.Path(directory)
    x = np.fromfile(d / "X.bin", dtype=np.float32).reshape(-1, FEATURES)
    data = {
        "x": x,
        "b": np.fromfile(d / "b.bin", dtype=np.uint8).astype(np.float32),
        "y": np.fromfile(d / "y.bin", dtype=np.float32),
    }
    for k, v in data.items():
        if len(v) != len(x):
            raise ValueError(f"{len(v)} rows in {k} against {len(x)} feature rows")
    return {k: torch.from_numpy(v) for k, v in data.items()}


def split(data, holdout: float = 0.1, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(data["x"]), generator=g)
    n = int(len(idx) * (1.0 - holdout))
    return ({k: v[idx[:n]] for k, v in data.items()},
            {k: v[idx[n:]] for k, v in data.items()})
