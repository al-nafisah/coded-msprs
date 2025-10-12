"""Trains the student against the BCJR extrinsic the simulator exported.

    python -m siso_nn.train --data data/siso --out siso_nn/weights.txt
"""
from __future__ import annotations

import argparse
import pathlib

import torch
from torch import nn

from siso_nn.data import load, split
from siso_nn.model import InnerSisoNet


def run(data_dir, out_path, epochs=40, batch=4096, lr=2e-3, hidden=128, seed=0):
    torch.manual_seed(seed)
    x, y = load(data_dir)
    (xt, yt), (xv, yv) = split(x, y, seed=seed)

    # Standardise on the training half only; the C++ side applies the same shift.
    mu, sd = xt.mean(0), xt.std(0).clamp_min(1e-6)
    xt, xv = (xt - mu) / sd, (xv - mu) / sd

    # Train on the soft bit tanh(L/2) rather than the LLR itself. The target is
    # bounded, so small LLRs carry as much weight as saturated ones, which is
    # where the decisions actually are.
    yt_s, yv_s = torch.tanh(yt / 2), torch.tanh(yv / 2)

    net = InnerSisoNet(hidden=hidden)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    loss_fn = nn.MSELoss()

    var = yv.var().item()
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), batch):
            j = perm[i:i + batch]
            opt.zero_grad()
            loss_fn(torch.tanh(net(xt[j]) / 2), yt_s[j]).backward()
            opt.step()
        sched.step()

        net.eval()
        with torch.no_grad():
            pred = net(xv)
            mse = loss_fn(pred, yv).item()
            agree = ((pred > 0) == (yv > 0)).float().mean().item()
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"  epoch {ep:3d}  val MSE {mse:8.4f}   R2 {1 - mse / var:6.3f}"
                  f"   sign agreement {agree:.4f}")

    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    net.export(out)
    with open(out.with_suffix(".norm"), "w") as fh:
        fh.write(" ".join(f"{v:.9g}" for v in mu) + "\n")
        fh.write(" ".join(f"{v:.9g}" for v in sd) + "\n")
    print(f"  wrote {out} and {out.with_suffix('.norm')}")
    return 1 - mse / var


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/siso")
    ap.add_argument("--out", default="siso_nn/weights.txt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--hidden", type=int, default=128)
    a = ap.parse_args()
    run(a.data, a.out, epochs=a.epochs, hidden=a.hidden)


if __name__ == "__main__":
    main()
