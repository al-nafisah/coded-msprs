"""What goes on air: coded bits, the MS-PRS symbols they make, and the burst around them.

Conventions follow the simulator: bit b maps to x = 1 - 2b, the taps are
normalised to ||h0||^2 + h1^2 = 1, and the (5,7) code ends in a zero tail.
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_taps(L0: int, family: str) -> tuple[np.ndarray, float]:
    """h0 and h1 exactly as the simulator scales them (include/MSPRS/Taps.hpp)."""
    j = json.loads((ROOT / "filters" / f"{family}.json").read_text())[str(L0)]
    eta = j["eta"]
    h0 = np.asarray(j["h0"], float) * np.sqrt(eta / 5.0)
    h1 = float(j["h1"][0]) * np.sqrt((5.0 - eta) / 5.0)
    if abs(h0 @ h0 + h1 * h1 - 1.0) > 1e-9:
        raise ValueError("tap energy is not 1")
    return h0, h1


def n_symbols(N: int, L0: int) -> int:
    """Symbols for N coded bits; the extra ones flush the FIR memory."""
    return (N + L0) // 2


def bit_sites(N: int, L0: int) -> tuple[np.ndarray, np.ndarray]:
    """Trellis step and stream (0: FIR x0, 1: single tap x1) of each bit, in the
    order the modulator consumes them: (b0, b1) per step while x0 is free, then
    b1 alone while the FIR is flushed."""
    L = n_symbols(N, L0)
    n0 = L - L0 + 1
    steps, streams = [], []
    for t in range(L):
        if t < n0:
            steps.append(t); streams.append(0)
        if not (L0 % 2 == 0 and t == L - 1):
            steps.append(t); streams.append(1)
    if len(steps) != N:
        raise ValueError(f"trellis holds {len(steps)} bits, expected {N}")
    return np.array(steps), np.array(streams)


def modulate(bits: np.ndarray, L0: int, h0: np.ndarray, h1: float) -> np.ndarray:
    """MS-PRS: s[t] = sum_j h0[j] x0[t-j] + h1 x1[t]; outside its range x0 is the
    boundary symbol +1, as in the simulator."""
    N = len(bits)
    L = n_symbols(N, L0)
    steps, streams = bit_sites(N, L0)
    x = 1.0 - 2.0 * bits
    x0 = np.ones(L)
    x1 = np.ones(L)
    x0[steps[streams == 0]] = x[streams == 0]
    x1[steps[streams == 1]] = x[streams == 1]
    s0 = np.convolve(np.concatenate([np.ones(L0 - 1), x0]), h0, mode="valid")
    return s0 + h1 * x1


def load_interleaver(path: pathlib.Path = ROOT / "rom" / "interleaver.hex") -> np.ndarray:
    """The simulator's and the HDL's permutation: the i-th bit sent is bit
    pi[i] of the coded block."""
    pi = np.array([int(v, 16) for v in path.read_text().split("\n") if v and not v.startswith("//")])
    if not np.array_equal(np.sort(pi), np.arange(len(pi))):
        raise ValueError(f"{path} is not a permutation")
    return pi


def encode_57(u: np.ndarray) -> np.ndarray:
    """Rate-1/2 K=3 (5,7) code with a two-bit zero tail, outputs interleaved
    per step, generator 5 first: the simulator's Encoder_NSC."""
    v = np.concatenate([u, [0, 0]]).astype(np.int64)
    d1 = np.concatenate([[0], v[:-1]])
    d2 = np.concatenate([[0, 0], v[:-2]])
    out = np.empty(2 * len(v), dtype=np.int64)
    out[0::2] = v ^ d2
    out[1::2] = v ^ d1 ^ d2
    return out


def rrc(rolloff: float, sps: int, span: int) -> np.ndarray:
    """Root-raised-cosine pulse, unit energy, 2*span*sps + 1 taps."""
    t = np.arange(-span * sps, span * sps + 1) / sps
    h = np.empty_like(t)
    for i, ti in enumerate(t):
        if ti == 0.0:
            h[i] = 1.0 - rolloff + 4.0 * rolloff / np.pi
        elif rolloff > 0 and abs(abs(ti) - 1.0 / (4.0 * rolloff)) < 1e-12:
            h[i] = rolloff / np.sqrt(2.0) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * rolloff))
                                             + (1 - 2 / np.pi) * np.cos(np.pi / (4 * rolloff)))
        else:
            h[i] = (np.sin(np.pi * ti * (1 - rolloff)) + 4 * rolloff * ti * np.cos(np.pi * ti * (1 + rolloff))) \
                   / (np.pi * ti * (1 - (4 * rolloff * ti) ** 2))
    return h / np.linalg.norm(h)


def m_sequence(n: int = 6) -> np.ndarray:
    """Maximal-length +-1 sequence of length 2^n - 1 (x^6 + x + 1 for n = 6)."""
    if n != 6:
        raise ValueError("only n = 6 is wired")
    reg = [1] * n
    out = []
    for _ in range(2**n - 1):
        out.append(reg[-1])
        fb = reg[-1] ^ reg[0]
        reg = [fb] + reg[:-1]
    return 1.0 - 2.0 * np.array(out, float)


@dataclass
class Waveform:
    """One burst layout: preamble (two copies of a 63-chip m-sequence, for the
    frequency offset), the data block, then silence."""
    L0: int = 3
    family: str = "balanced"
    info_bits: int = 4998
    sps: int = 8
    rolloff: float = 0.3
    span: int = 8
    guard: int = 64

    def __post_init__(self):
        self.h0, self.h1 = load_taps(self.L0, self.family)
        self.N = 2 * (self.info_bits + 2)
        self.L = n_symbols(self.N, self.L0)
        self.pn = m_sequence(6)
        self.preamble = np.concatenate([self.pn, self.pn])
        self.pulse = rrc(self.rolloff, self.sps, self.span)
        self.perm = load_interleaver()
        if len(self.perm) != self.N:
            raise ValueError("the interleaver table does not match the coded block")
        self.sites = bit_sites(self.N, self.L0)
        self.burst_symbols = len(self.preamble) + self.L + self.guard

    def block(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Information bits, the interleaved coded bits as sent, and their symbols."""
        u = rng.integers(0, 2, self.info_bits)
        sent = encode_57(u)[self.perm]
        return u, sent, modulate(sent, self.L0, self.h0, self.h1)

    def samples(self, symbols: np.ndarray) -> np.ndarray:
        """Complex baseband burst at sps samples per symbol, peak 1. MS-PRS is
        real, so it rides the in-phase rail."""
        seq = np.concatenate([self.preamble, symbols, np.zeros(self.guard)])
        up = np.zeros(len(seq) * self.sps)
        up[:: self.sps] = seq
        x = np.convolve(up, self.pulse)[: len(up)]
        return (x / np.abs(x).max()).astype(np.complex64)
