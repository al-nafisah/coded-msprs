"""Trains the student on the transmitted bits.

The network's output is an extrinsic LLR: the bit's log-odds given everything
in the window except its own a priori. The bits are uniform, so cross-entropy
of that output against the bit trains exactly this. Putting the own a priori
into the loss as well would starve the samples where it is already confident
of gradient. Learning from the bits rather than from the BCJR's output is what
lets the network do better than the BCJR wherever the BCJR's channel model is
wrong.

    python -m siso_nn.train --data data/siso_L3_balanced --out siso_nn/weights_L3_balanced.txt
"""
from __future__ import annotations

import argparse
import math
import pathlib

import torch
from torch import nn

from siso_nn.data import load, split
from siso_nn.model import InnerSisoNet


def mutual_information(llr, bits):
    """Averaging estimator of I(L; b), LLRs as ln P(b=0)/P(b=1)."""
    s = (1.0 - 2.0 * bits) * llr
    return 1.0 - (torch.nn.functional.softplus(-s) / math.log(2.0)).mean().item()


def run(data_dir, out_path, epochs=40, batch=4096, lr=2e-3, hidden=128, seed=0):
    torch.manual_seed(seed)
    train, val = split(load(data_dir), seed=seed)

    # Standardise on the training split only; the C++ side applies the same shift.
    mu, sd = train["x"].mean(0), train["x"].std(0).clamp_min(1e-6)
    xt, xv = (train["x"] - mu) / sd, (val["x"] - mu) / sd

    # A positive LLR means bit 0, so the target of the sigmoid is 1 - b.
    tt, tv = 1.0 - train["b"], 1.0 - val["b"]

    net = InnerSisoNet(hidden=hidden)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    bce = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        ref_bce = bce(val["y"], tv).item()
        ref_mi = mutual_information(val["y"], val["b"])
    print(f"  BCJR on the validation split: BCE {ref_bce:.4f}  extrinsic MI {ref_mi:.4f}")

    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), batch):
            j = perm[i:i + batch]
            opt.zero_grad()
            bce(net(xt[j]), tt[j]).backward()
            opt.step()
        sched.step()

        if ep % 5 == 0 or ep == epochs - 1:
            net.eval()
            with torch.no_grad():
                ext = net(xv)
                loss = bce(ext, tv).item()
                mi = mutual_information(ext, val["b"])
                agree = ((ext > 0) == (val["y"] > 0)).float().mean().item()
            print(f"  epoch {ep:3d}  BCE {loss:.4f}  extrinsic MI {mi:.4f}"
                  f"  sign agreement with BCJR {agree:.4f}")

    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    net.export(out)
    with open(out.with_suffix(".norm"), "w") as fh:
        fh.write(" ".join(f"{v:.9g}" for v in mu) + "\n")
        fh.write(" ".join(f"{v:.9g}" for v in sd) + "\n")
    print(f"  wrote {out} and {out.with_suffix('.norm')}")
    return loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--hidden", type=int, default=128)
    a = ap.parse_args()
    run(a.data, a.out, epochs=a.epochs, hidden=a.hidden)


if __name__ == "__main__":
    main()
