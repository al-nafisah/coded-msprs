"""The classical receiver on the same burst: the simulator's BCJR equaliser.

The burst's in-phase rail goes through `msprs --mode tdemodtest` with no a
priori, so the output is the BCJR's LLR of every coded bit, ln P(b=0)/P(b=1),
from the same C++ code the BER curves come from.
"""
from __future__ import annotations

import subprocess

import numpy as np

from ota.waveform import ROOT

MSPRS = ROOT / "build" / "bin" / "msprs"


def bcjr_llr(y: np.ndarray, sigma: float, n_bits: int, L0: int, family: str, bcjr: str = "map") -> np.ndarray:
    text = "\n".join([str(len(y))] + [f"{v:.7g}" for v in y] + ["0"] * n_bits)
    out = subprocess.run(
        [str(MSPRS), "--mode", "tdemodtest", "--L0", str(L0), "--family", family,
         "--sigma", f"{sigma:.7g}", "--bcjr", bcjr],
        input=text, capture_output=True, text=True, check=True).stdout
    llr = np.array(out.split(), float)
    if len(llr) != n_bits:
        raise RuntimeError(f"BCJR returned {len(llr)} LLRs, expected {n_bits}")
    return llr
