"""The student network: a windowed stand-in for the BCJR equaliser."""
from __future__ import annotations

import torch
from torch import nn

HALF_SYM = 4
HALF_BIT = 8
FEATURES = (2 * HALF_SYM + 1) + (2 * HALF_BIT + 1) + 2  # + sigma, sub-stream


class InnerSisoNet(nn.Module):
    """Maps a window of received symbols and a priori LLRs to one extrinsic LLR."""

    def __init__(self, hidden: int = 128, layers: int = 3):
        super().__init__()
        dims = [FEATURES] + [hidden] * layers
        body = []
        for a, b in zip(dims, dims[1:]):
            body += [nn.Linear(a, b), nn.ReLU()]
        self.body = nn.Sequential(*body)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x)).squeeze(-1)

    def export(self, path) -> None:
        """Flat weights the C++ side reads back."""
        with open(path, "w") as fh:
            fh.write(f"{FEATURES} {len(self.body) // 2} {self.head.in_features}\n")
            for m in list(self.body) + [self.head]:
                if isinstance(m, nn.Linear):
                    fh.write(f"{m.in_features} {m.out_features}\n")
                    fh.write(" ".join(f"{v:.9g}" for v in m.weight.detach().flatten()) + "\n")
                    fh.write(" ".join(f"{v:.9g}" for v in m.bias.detach()) + "\n")
