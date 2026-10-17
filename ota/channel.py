"""A simulated bench: what the air and the two radios do to a burst.

From the transmitter's DAC words to the receiver's ADC words: a gain set by the
Es/N0 wanted at the receiver, a delay with a fractional part, a carrier
frequency offset with a phase that walks on top of it, white noise at the level
the receiver sees at 65 dB gain, and the receiver's 12-bit converter. Both
receivers in ota/ can then be tested with the radios off.

The phase walk is a Wiener process, the phase noise of two free-running
oscillators, set by its linewidth: the phase drifts by sqrt(2 pi linewidth t)
radians rms over a time t. The bench has not measured it yet.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ADC_BITS = 12
NOISE_LSB = 7.0   # ADC noise per real dimension at 65 dB receive gain, measured on the bench


@dataclass
class Link:
    esn0_db: float            # Es/N0 at the receiver's matched filter
    amp: float                # transmitter DAC LSBs per unit of symbol amplitude
    cfo_hz: float = 0.0
    linewidth_hz: float = 0.0
    delay: float = 0.0        # samples before the burst, fractional part included
    phase: float = 0.0        # carrier phase at the first sample, rad
    tail: int = 256           # samples after the burst
    noise_lsb: float = NOISE_LSB
    fs: float = 1e6

    def gain(self) -> float:
        """ADC LSBs per DAC LSB. With unit-energy pulses at both ends a symbol of
        amplitude s leaves the matched filter as gain * amp * s, over noise of
        noise_lsb per real dimension."""
        return np.sqrt(2 * self.noise_lsb**2 * 10 ** (self.esn0_db / 10)) / self.amp

    def __call__(self, dac: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """ADC words, complex int, for the DAC words of one burst."""
        n0 = int(np.floor(self.delay))
        n = n0 + len(dac) + self.tail
        m = 1 << (n + 64).bit_length()
        x = np.zeros(m, complex)
        x[n0 : n0 + len(dac)] = dac
        # The fractional part of the delay, exact for a band-limited burst.
        f = np.fft.fftfreq(m)
        x = np.fft.ifft(np.fft.fft(x) * np.exp(-2j * np.pi * f * (self.delay - n0)))[:n]

        t = np.arange(n) / self.fs
        walk = np.cumsum(rng.normal(0.0, np.sqrt(2 * np.pi * self.linewidth_hz / self.fs), n))
        x = self.gain() * x * np.exp(1j * (self.phase + 2 * np.pi * self.cfo_hz * t + walk))
        x = x + self.noise_lsb * (rng.normal(size=n) + 1j * rng.normal(size=n))

        top = (1 << (ADC_BITS - 1)) - 1
        return np.clip(np.round(x.real), -top - 1, top).astype(np.int64) + \
            1j * np.clip(np.round(x.imag), -top - 1, top).astype(np.int64)
