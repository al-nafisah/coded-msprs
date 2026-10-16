"""From a capture to symbols: find each burst, correct frequency, phase and gain.

This is coarse sync only: one frequency offset and one phase per burst. What
the two free-running crystals add on top of that line, the phase walk, is left
in the symbols for the receivers to deal with.

The frequency offset comes in two steps. The two halves of the preamble give
a coarse value over a wide range. MS-PRS symbols are real, so squaring a
received symbol strips the data and leaves twice the carrier phase; a line
through the phase of the squared, block-averaged data then refines it over
the whole burst without pilots. The preamble settles the sign the squaring
leaves open.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ota.waveform import Waveform


@dataclass
class Burst:
    symbols: np.ndarray   # L complex symbols, unit symbol energy, phase-corrected
    sigma: float          # noise standard deviation per real dimension
    esn0_db: float        # from the preamble
    cfo_hz: float
    start: int            # sample index of the preamble in the matched-filter output


def _xcorr(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    """c[n] = sum_m t[m] y[n + m] for every n where t fits, via FFT."""
    n = len(y) + len(t)
    m = 1 << (n - 1).bit_length()
    c = np.fft.ifft(np.fft.fft(y, m) * np.conj(np.fft.fft(t, m)))
    return c[: len(y) - len(t) + 1]


def bursts(rx: np.ndarray, w: Waveform, fs: float, threshold: float = 0.45) -> list[Burst]:
    """Every complete burst in the capture. The transmitter repeats one burst
    back to back, so bursts sit exactly one burst length apart from the
    strongest preamble; nothing in between is considered. The threshold keeps
    noise out (about 1/sqrt(P) on noise) and preambles in down to about -5 dB."""
    y = np.convolve(rx.astype(np.complex128), w.pulse)
    sps, P, half = w.sps, len(w.preamble), len(w.pn)
    T = sps / fs

    # Normalised preamble correlation: close to 1 on a preamble, about
    # 1/sqrt(P) on noise.
    tpl = np.zeros((P - 1) * sps + 1)
    tpl[::sps] = w.preamble
    ones = np.zeros_like(tpl)
    ones[::sps] = 1.0
    c = np.abs(_xcorr(y, tpl))
    e = np.sqrt(np.maximum(_xcorr(np.abs(y) ** 2, ones).real, 1e-30) * P)
    rho = c / e

    span = (P + w.L) * sps
    period = w.burst_symbols * sps
    best = int(np.argmax(rho))
    if rho[best] < threshold:
        return []
    found = []
    for n in range(best % period, len(rho), period):
        # the grid can be off by a sample where the peak is flat
        n = max(0, n - 1) + int(np.argmax(rho[max(0, n - 1) : n + 2]))
        if rho[n] >= threshold and n + span <= len(y):
            found.append(n)

    out = []
    for n in found:
        seg = y[n : n + span : sps]                 # preamble and data at symbol rate
        k = np.arange(len(seg))

        # A preamble cut off at the start of the capture still lines its second
        # half up with the template's first half and scores about 0.5; one of
        # its halves is then data. A real preamble has two matching halves.
        a1 = np.sum(w.pn * seg[:half])
        a2 = np.sum(w.pn * seg[half:P])
        if min(abs(a1), abs(a2)) < 0.5 * max(abs(a1), abs(a2)):
            continue

        # Coarse: phase between the two preamble halves, 63 symbols apart.
        f = np.angle(a2 * np.conj(a1)) / (2 * np.pi * half * T)
        seg = seg * np.exp(-2j * np.pi * f * k * T)

        # Fine: twice the residual phase, from squared data averaged per block.
        d2 = seg[P:] ** 2
        blk = 128
        nb = len(d2) // blk
        z = d2[: nb * blk].reshape(nb, blk).mean(axis=1)
        phi = np.unwrap(np.angle(z))
        tc = (np.arange(nb) * blk + (blk - 1) / 2 + P) * T
        slope = np.polyfit(tc, phi, 1)[0]
        df = slope / (4 * np.pi)
        seg = seg * np.exp(-2j * np.pi * df * k * T)
        f += df

        # Gain and phase from the whole preamble, then unit symbol energy.
        g = np.sum(w.preamble * seg[:P]) / P
        seg = seg / g
        resid = seg[:P] - w.preamble
        sigma = float(np.sqrt(np.mean(np.abs(resid) ** 2) / 2))
        out.append(Burst(symbols=seg[P:], sigma=sigma, esn0_db=float(10 * np.log10(1 / (2 * sigma**2))),
                         cfo_hz=float(f), start=n))
    return out


def matches(b: Burst, sent_symbols: np.ndarray, floor: float = 0.1) -> bool:
    """True when the burst carries this block. After a new burst goes on air the
    first capture can still hold the old one; its correlation with the new
    symbols is about 1/sqrt(L), far under the floor."""
    r = b.symbols.real
    rho = abs(r @ sent_symbols) / np.sqrt((r @ r) * (sent_symbols @ sent_symbols))
    return rho > floor
