"""The transmitter and the receiver's front end in integer arithmetic, bit-true
to what the FPGA computes, from the information bits to the 12-bit DAC words
and from the 12-bit ADC words to the received symbols the turbo receiver takes.

Every word, its width and its rounding are listed in docs/fixed_point.md; the
widths come from the "fixed_point" section of config.json. Complex words are
pairs of int64 arrays. Shifts are arithmetic, so >> floors, and rounding is
half away from zero wherever it is written as `rshift`.

Transmit: the modulator is a table from the newest L0 x0 bits and the x1 bit to
a symbol word; the root-raised-cosine interpolator runs on symbol words and
writes the DAC word, never saturating, because its worst-case sum fits.

Receive, per burst:
  1. Matched filter on the ADC words, at 8 samples per symbol.
  2. Preamble detection, |c|^2 > (13/64) P E with c the correlation and E the
     energy under the preamble, then the largest |c|^2 within the next 65
     symbols sets the timing to the sample, and a parabola through it and its
     two neighbours to 1/2^frac_bits of a sample.
  3. The burst's preamble and data at symbol rate, interpolated linearly
     between two samples, go into a buffer, shifted right so that the energy
     the correlator saw at the peak puts their rms at most 2^buf_rms (block
     floating point).
  4. Coarse frequency: the angle between the two preamble halves.
  5. Fine frequency: data derotated by the coarse value, squared and summed
     in blocks of 128 symbols; the line through the unwrapped block angles.
  6. Phase and gain from the whole preamble, derotated by both frequencies,
     and one division per burst for the gain.
  7. Data derotated, normalised to the turbo receiver's grid, with a
     second-order loop following the phase between the preamble and the end
     of the burst on sign(Re z) Im z.
  8. The gain corrected from the whole burst: the power of the in-phase rail
     less that of the quadrature rail is the signal's, and one pass rescales
     the received words. The preamble alone leaves the gain too loose for the
     integer turbo receiver, which fails on bursts whose gain comes out low.
Angles are in turns, 2^16 per turn at the CORDIC and 2^24 in the phase
accumulator.
"""
from __future__ import annotations

import dataclasses
import json
import math
import subprocess
from dataclasses import dataclass, field

import numpy as np

from ota.waveform import ROOT, Waveform, bit_sites

MSPRS = ROOT / "build" / "bin" / "msprs"
P_BITS = 16    # CORDIC angles
NCO_BITS = 24  # phase accumulator
BLOCK = 128    # symbols per block of the fine frequency estimate
DAC_BITS = 12


def rshift(v, sh: int):
    """v / 2^sh rounded half away from zero; a negative sh shifts left."""
    v = np.asarray(v, np.int64)
    if sh <= 0:
        return v << -sh
    h = np.int64(1) << (sh - 1)
    return np.where(v >= 0, (v + h) >> sh, -((-v + h) >> sh))


def sat(v, bits: int):
    """Symmetric saturation to a signed word of `bits`."""
    t = (1 << (bits - 1)) - 1
    return np.clip(np.asarray(v, np.int64), -t, t)


def wrap(v, bits: int):
    """Two's-complement wrap to a signed word of `bits`: angle differences."""
    m = 1 << bits
    return ((np.asarray(v, np.int64) + (m >> 1)) & (m - 1)) - (m >> 1)


# ---------------------------------------------------------------------------- CORDIC

def _atan_table(n: int) -> list[int]:
    return [round(math.atan(2.0**-i) / (2 * math.pi) * (1 << P_BITS)) for i in range(n)]


def cordic_gain(n: int) -> float:
    return math.prod(math.sqrt(1 + 4.0**-i) for i in range(n))


def vector(x, y, n: int, guard: int):
    """Magnitude times cordic_gain(n), and angle in 2^16 per turn, of x + jy.
    The iterations run with `guard` extra fractional bits."""
    x = np.array(x, np.int64, ndmin=1) << guard
    y = np.array(y, np.int64, ndmin=1) << guard
    neg = x < 0
    x, y = np.where(neg, -x, x), np.where(neg, -y, y)
    a = np.where(neg, 1 << (P_BITS - 1), 0).astype(np.int64)
    for i, t in enumerate(_atan_table(n)):
        d = np.where(y >= 0, 1, -1)
        x, y = x + d * (y >> i), y - d * (x >> i)
        a = a + d * t
    return rshift(x, guard), a & ((1 << P_BITS) - 1)


def rotate(x, y, a, n: int, guard: int):
    """(x + jy) e^(j 2 pi a / 2^16) times cordic_gain(n), with `guard` extra
    fractional bits in the iterations."""
    x = np.array(x, np.int64, ndmin=1) << guard
    y = np.array(y, np.int64, ndmin=1) << guard
    a = wrap(a, P_BITS)
    big = np.abs(a) > (1 << (P_BITS - 2))
    x, y = np.where(big, -x, x), np.where(big, -y, y)
    a = np.where(big, wrap(a + (1 << (P_BITS - 1)), P_BITS), a)
    for i, t in enumerate(_atan_table(n)):
        d = np.where(a >= 0, 1, -1)
        x, y = x - d * (y >> i), y + d * (x >> i)
        a = a - d * t
    return rshift(x, guard), rshift(y, guard)


def _rotate1(x: int, y: int, a: int, n: int, guard: int, table: list[int]) -> tuple[int, int]:
    """rotate() on one sample with Python ints, for the phase loop."""
    x, y = x << guard, y << guard
    a = ((a + (1 << (P_BITS - 1))) & ((1 << P_BITS) - 1)) - (1 << (P_BITS - 1))
    if abs(a) > (1 << (P_BITS - 2)):
        x, y = -x, -y
        a = ((a + (1 << P_BITS)) & ((1 << P_BITS) - 1)) - (1 << (P_BITS - 1))
    for i in range(n):
        if a >= 0:
            x, y, a = x - (y >> i), y + (x >> i), a - table[i]
        else:
            x, y, a = x + (y >> i), y - (x >> i), a + table[i]
    return _rs(x, guard), _rs(y, guard)


# ---------------------------------------------------------------------------- formats

@dataclass
class Formats:
    """Front-end words; see docs/fixed_point.md."""
    sym_frac: int = 8       # transmit symbol words: round(2^sym_frac s)
    tap_bits: int = 12      # RRC taps round(2^tap_bits h), signed tap_bits
    dac_amp: int = 1024     # DAC LSBs per unit of symbol amplitude, a power of two
    mf_frac: int = 1        # matched filter output: ADC LSB / 2^mf_frac
    mf_bits: int = 14
    frac_bits: int = 4      # fractional timing, 1/2^frac_bits of a sample
    buf_rms: int = 6        # buffered symbols: rms at most 2^buf_rms
    buf_bits: int = 10
    cordic: int = 14        # CORDIC iterations
    guard: int = 4          # and its extra fractional bits
    gain_shift: int = 13    # gain word G: y = z G >> gain_shift
    gain_bits: int = 15     # G, unsigned
    loop_p: int = 11        # phase loop: phase += e << loop_p + nu; negative turns the loop off
    loop_i: int = 3         # nu += e << loop_i; negative turns the integrator off
    refine_shift: int = 12  # gain correction from the whole burst: y = y G2 >> refine_shift
    refine_bits: int = 14   # G2, unsigned
    y_bits: int = 8         # the turbo receiver's input word
    y_amp: float = 26.1     # its LSBs per unit of symbol amplitude

    @staticmethod
    def load(path=ROOT / "config.json") -> "Formats":
        fp = json.loads(path.read_text())["fixed_point"]
        f = Formats(**fp["front_end"])
        f.y_bits, f.y_amp = fp["turbo"]["y_bits"], fp["turbo"]["y_amp"]
        return f

    def wide(self) -> "Formats":
        """The same receiver with every front-end word wide enough to be exact
        to within a thousandth of a decibel: the reference its widths are
        measured against. The y word is the turbo receiver's and stays."""
        return dataclasses.replace(self, tap_bits=24, mf_frac=8, mf_bits=32, frac_bits=10, buf_rms=16,
                                   buf_bits=30, cordic=24, guard=6, gain_shift=24, gain_bits=40,
                                   refine_shift=20, refine_bits=40)


# ---------------------------------------------------------------------------- transmit

def rrc_taps(w: Waveform, f: Formats) -> np.ndarray:
    """The root-raised-cosine taps both ends use."""
    taps = np.round(w.pulse * (1 << f.tap_bits)).astype(np.int64)
    if np.abs(taps).max() >= 1 << (f.tap_bits - 1):
        raise ValueError("a tap does not fit its word")
    return taps


class Transmitter:
    """Information bits to DAC words."""

    def __init__(self, w: Waveform, f: Formats):
        self.w, self.f = w, f
        self.taps = rrc_taps(w, f)
        self.shift = f.tap_bits + f.sym_frac - int(math.log2(f.dac_amp))
        # The modulator table, indexed by x0[t], x0[t-1], .., x0[t-L0+1], x1[t]
        # as bits, newest first.
        L0 = w.L0
        self.table = np.empty(1 << (L0 + 1), np.int64)
        for idx in range(len(self.table)):
            b = [(idx >> (L0 - j)) & 1 for j in range(L0 + 1)]
            s = sum(w.h0[j] * (1 - 2 * b[j]) for j in range(L0)) + w.h1 * (1 - 2 * b[L0])
            self.table[idx] = round(s * (1 << f.sym_frac))
        self.one = 1 << f.sym_frac      # a preamble chip
        worst = max(np.abs(self.taps[p :: w.sps]).sum() for p in range(w.sps)) * max(np.abs(self.table).max(), self.one)
        if (worst >> self.shift) >= 1 << (DAC_BITS - 1):
            raise ValueError("the interpolator can overflow the DAC word")

    def symbols(self, sent: np.ndarray) -> np.ndarray:
        """Symbol words of the interleaved coded bits, through the table."""
        w = self.w
        L = w.L
        steps, streams = bit_sites(w.N, w.L0)
        b0 = np.zeros(L + w.L0 - 1, np.int64)       # x0 bits, with L0-1 boundary zeros in front
        b1 = np.zeros(L, np.int64)
        b0[w.L0 - 1 + steps[streams == 0]] = sent[streams == 0]
        b1[steps[streams == 1]] = sent[streams == 1]
        idx = b1.copy()
        for j in range(w.L0):
            idx |= b0[w.L0 - 1 - j : w.L0 - 1 - j + L] << (w.L0 - j)
        return self.table[idx]

    def burst(self, sym: np.ndarray) -> np.ndarray:
        """DAC words of the whole burst, the in-phase rail; quadrature is zero."""
        w = self.w
        pre = np.where(w.preamble > 0, self.one, -self.one)
        seq = np.concatenate([pre, sym, np.zeros(w.guard, np.int64)])
        up = np.zeros(len(seq) * w.sps, np.int64)
        up[:: w.sps] = seq
        acc = np.convolve(up, self.taps)[: len(up)]
        return rshift(acc, self.shift)


# ---------------------------------------------------------------------------- receive

@dataclass
class FixedBurst:
    y: np.ndarray                  # the turbo receiver's input words, one per data symbol
    start: int                     # matched-filter sample of the first preamble symbol
    frac: int                      # and the fraction of a sample after it, 1/2^frac_bits
    shift: int                     # block floating-point shift of the buffer
    inc: int                       # phase increment per symbol, 2^24 per turn
    gain: int                      # G, from the preamble
    gain2: int                     # and its correction from the whole burst
    trace: dict = field(default_factory=dict)

    def cfo_hz(self, symbol_rate: float) -> float:
        return float(wrap(self.inc, NCO_BITS)) / (1 << NCO_BITS) * symbol_rate


class Receiver:
    """ADC words to the turbo receiver's input words."""

    def __init__(self, w: Waveform, f: Formats):
        self.w, self.f, self.taps = w, f, rrc_taps(w, f)
        self.P = len(w.preamble)
        self.pre = w.preamble.astype(np.int64)
        self.pn = w.pn.astype(np.int64)
        self.n_sym = self.P + w.L
        self.nb = w.L // BLOCK
        self.wb = 2 * np.arange(self.nb, dtype=np.int64) - (self.nb - 1)
        sw2 = int((self.wb**2).sum())
        # Increments per symbol, 2^24 per turn: the coarse one from the angle
        # between halves 63 symbols apart, the fine one from the slope of the
        # block angles of the squared data, which turn twice as fast.
        self.r_coarse = round((1 << (NCO_BITS - P_BITS)) * (1 << 16) / len(w.pn))
        self.r_fine = round((1 << (NCO_BITS - P_BITS)) * (1 << 24) / (BLOCK * sw2))
        self.kc = cordic_gain(f.cordic)
        self.c_gain = round(f.y_amp * self.kc * self.P * (1 << f.gain_shift))
        self.c_refine = round(w.L * f.y_amp**2 * (1 << (2 * f.refine_shift)))

    def matched_filter(self, adc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        f = self.f
        sh = f.tap_bits - f.mf_frac
        i = sat(rshift(np.convolve(adc.real.astype(np.int64), self.taps), sh), f.mf_bits)
        q = sat(rshift(np.convolve(adc.imag.astype(np.int64), self.taps), sh), f.mf_bits)
        return i, q

    def detect(self, i: np.ndarray, q: np.ndarray, start: int = 0) -> tuple[int, int] | None:
        """First preamble at or after `start`: the sample of its first symbol and
        the fraction of a sample to add, in 1/2^frac_bits."""
        sps, P = self.w.sps, self.P
        n = len(i) - (P - 1) * sps
        if n <= start + 2:
            return None
        ci = np.zeros(n - start, np.int64)
        cq = np.zeros(n - start, np.int64)
        e = np.zeros(n - start, np.int64)
        for m in range(P):
            a = start + m * sps
            ci += self.pre[m] * i[a : a + n - start]
            cq += self.pre[m] * q[a : a + n - start]
            e += i[a : a + n - start] ** 2 + q[a : a + n - start] ** 2
        c2 = ci**2 + cq**2
        hit = np.flatnonzero(64 * c2 > 13 * P * e)
        if len(hit) == 0:
            return None
        n0 = max(1, int(hit[0]))
        # The peak can sit up to a preamble half later: aligned on one half, the
        # template already scores ~0.7 at high SNR.
        k = n0 + int(np.argmax(c2[n0 : min(len(c2) - 1, n0 + (len(self.pn) + 2) * sps)]))
        # Vertex of the parabola through the peak and its neighbours, rounded,
        # within half a sample.
        num = (int(c2[k - 1]) - int(c2[k + 1])) << self.f.frac_bits
        den = 2 * (int(c2[k - 1]) - 2 * int(c2[k]) + int(c2[k + 1]))
        h = 1 << (self.f.frac_bits - 1)
        d = 0 if den >= 0 else max(-h, min(h, _div_round(num, den)))
        return start + k, d

    def receive(self, adc: np.ndarray, keep: bool = False) -> list[FixedBurst]:
        """Every complete burst in a capture of ADC words."""
        i, q = self.matched_filter(adc)
        sps = self.w.sps
        out, start = [], 0
        while True:
            hit = self.detect(i, q, start)
            if hit is None or hit[0] + (self.n_sym - 1) * sps + 1 >= len(i):
                return out
            out.append(self.burst(i, q, *hit, keep))
            start = hit[0] + self.n_sym * sps

    def burst(self, i: np.ndarray, q: np.ndarray, start: int, frac: int, keep: bool = False) -> FixedBurst:
        f, P, half, sps = self.f, self.P, len(self.pn), self.w.sps
        tr = {}

        # 3. Symbol-rate samples between `start` and its neighbour on the side
        # of `frac`, scaled by 2^frac_bits, then the block floating-point shift
        # from the correlator's energy at the peak.
        k = start + sps * np.arange(self.n_sym)
        nb = k + (1 if frac >= 0 else -1)
        si = (i[k] << f.frac_bits) + abs(frac) * (i[nb] - i[k])
        sq = (q[k] << f.frac_bits) + abs(frac) * (q[nb] - q[k])
        e = int((i[k[:P]] ** 2 + q[k[:P]] ** 2).sum())
        sh = 0
        while e > P << (2 * (f.buf_rms + sh)):
            sh += 1
        ri = sat(rshift(si, sh + f.frac_bits), f.buf_bits)
        rq = sat(rshift(sq, sh + f.frac_bits), f.buf_bits)

        # 4. Coarse frequency.
        _, a1 = vector((self.pn * ri[:half]).sum(), (self.pn * rq[:half]).sum(), f.cordic, f.guard)
        _, a2 = vector((self.pn * ri[half:P]).sum(), (self.pn * rq[half:P]).sum(), f.cordic, f.guard)
        inc_c = int(rshift(wrap(a2 - a1, P_BITS) * self.r_coarse, 16)[0])

        # 5. Fine frequency.
        k = np.arange(P, P + self.nb * BLOCK, dtype=np.int64)
        th = (k * inc_c) & ((1 << NCO_BITS) - 1)
        xi, xq = rotate(ri[k], rq[k], -(th >> (NCO_BITS - P_BITS)), f.cordic, f.guard)
        zi = (xi**2 - xq**2).reshape(self.nb, BLOCK).sum(axis=1)
        zq = (2 * xi * xq).reshape(self.nb, BLOCK).sum(axis=1)
        _, ab = vector(zi, zq, f.cordic, f.guard)
        u = np.concatenate([[wrap(ab[0], P_BITS)], wrap(np.diff(ab), P_BITS)]).cumsum()
        inc_f = int(rshift(int((self.wb * u).sum()) * self.r_fine, 24))
        inc = (inc_c + inc_f) & ((1 << NCO_BITS) - 1)

        # 6. Phase and gain from the preamble.
        k = np.arange(self.n_sym, dtype=np.int64)
        th = (k * inc) & ((1 << NCO_BITS) - 1)
        pi_, pq = rotate(ri[:P], rq[:P], -(th[:P] >> (NCO_BITS - P_BITS)), f.cordic, f.guard)
        mag, a0 = vector((self.pre * pi_).sum(), (self.pre * pq).sum(), f.cordic, f.guard)
        mag, a0 = int(mag[0]), int(a0[0])
        g = min((self.c_gain + mag // 2) // max(mag, 1), (1 << f.gain_bits) - 1)

        # 7. Data, with the phase loop.
        table = _atan_table(f.cordic)
        top = (1 << (f.y_bits - 1)) - 1
        mask = (1 << NCO_BITS) - 1
        theta = (int(th[P]) + (a0 << (NCO_BITS - P_BITS))) & mask
        phi, nu = 0, 0
        y = np.empty(self.w.L, np.int64)
        err = np.empty(self.w.L, np.int64) if keep else None
        pr = pq = 0
        for j in range(self.w.L):
            ang = ((theta + phi) & mask) >> (NCO_BITS - P_BITS)
            cr, cq = _rotate1(int(ri[P + j]), int(rq[P + j]), -ang, f.cordic, f.guard, table)
            yr = max(-top, min(top, _rs(cr * g, f.gain_shift)))
            yq = max(-top, min(top, _rs(cq * g, f.gain_shift)))
            e_ = yq if yr >= 0 else -yq
            if f.loop_p >= 0:
                if f.loop_i >= 0:
                    nu += e_ << f.loop_i
                phi += (e_ << f.loop_p) + nu
            theta = (theta + inc) & mask
            y[j] = yr
            pr += yr * yr
            pq += yq * yq
            if keep:
                err[j] = e_

        # 8. The gain again, from the whole burst: the in-phase rail carries
        # signal and noise, the quadrature rail the same noise alone.
        g2 = 1 << f.refine_shift
        if pr > pq:
            g2 = min(math.isqrt(_div_round(self.c_refine, pr - pq)), (1 << f.refine_bits) - 1)
        y0 = y
        y = sat(rshift(y * g2, f.refine_shift), f.y_bits)
        if keep:
            tr = dict(buffer_i=ri, buffer_q=rq, energy=e, angle_half1=int(a1[0]), angle_half2=int(a2[0]),
                      inc_coarse=inc_c, block_i=zi, block_q=zq, block_angle=ab, inc_fine=inc_f,
                      preamble_mag=mag, preamble_angle=a0, loop_error=err, y_preamble_gain=y0,
                      power_i=pr, power_q=pq, gain_refined=g2)
        return FixedBurst(y=y, start=start, frac=frac, shift=sh, inc=inc, gain=g, gain2=g2, trace=tr)


def _div_round(num: int, den: int) -> int:
    """num / den rounded half away from zero."""
    q, r = divmod(abs(num), abs(den))
    q += 2 * r >= abs(den)
    return q if (num >= 0) == (den >= 0) else -q


def _rs(v: int, sh: int) -> int:
    """rshift() on one Python int."""
    if sh <= 0:
        return v << -sh
    h = 1 << (sh - 1)
    return (v + h) >> sh if v >= 0 else -((-v + h) >> sh)


# ---------------------------------------------------------------------------- turbo receiver

def turbo(y: np.ndarray, L0: int, family: str, sigma: np.ndarray | None = None, threads: int = 12,
          bcjr: str = "map") -> np.ndarray:
    """Decisions on the information bits of each burst (rows of y), from the
    simulator's turbo receiver: the integer one on y words, or the float one
    on received symbols of unit energy and their noise sigma."""
    y = np.atleast_2d(y)
    cmd = [str(MSPRS), "--mode", "decode", "--L0", str(L0), "--family", family, "--threads", str(threads)]
    if sigma is None:
        data = y.astype(np.int16).tobytes()
        cmd.append("--fixed")
    else:
        data = np.concatenate([y, np.atleast_1d(sigma)[:, None]], axis=1).astype(np.float32).tobytes()
        cmd += ["--bcjr", bcjr]
    out = subprocess.run(cmd, input=data, capture_output=True, check=True).stdout
    return np.frombuffer(out, np.uint8).reshape(len(y), -1)


# ---------------------------------------------------------------------------- vectors

def export_vectors(out, family: str = "balanced", esn0_db: float = 5.0, cfo_hz: float = 80.0,
                   linewidth_hz: float = 1.0, seed: int = 1) -> None:
    """One burst through every block, transmitter to decisions, for the HDL tests:
    each block's input and output, one integer per line, plus the tables the
    hardware holds in ROM. The turbo receiver's vectors come from the simulator
    (msprs --mode vectors) on the y words written here."""
    import pathlib
    from ota.channel import Link
    from ota.waveform import encode_57

    out = pathlib.Path(out)
    w, f = Waveform(family=family), Formats.load()
    tx = Transmitter(w, f)
    rx = Receiver(w, f)
    rng = np.random.default_rng(seed)
    u = rng.integers(0, 2, w.info_bits)
    coded = encode_57(u)
    sent = coded[w.perm]
    sym = tx.symbols(sent)
    dac = tx.burst(sym)
    link = Link(esn0_db=esn0_db, amp=f.dac_amp, cfo_hz=cfo_hz, linewidth_hz=linewidth_hz,
                delay=rng.uniform(500, 3000), phase=rng.uniform(0, 2 * np.pi))
    adc = link(dac, rng)
    mi, mq = rx.matched_filter(adc)
    b = rx.receive(adc, keep=True)
    if len(b) != 1:
        raise RuntimeError(f"found {len(b)} bursts, expected 1")
    b = b[0]

    def put(name, v):
        p = out / name
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(p, np.atleast_1d(np.asarray(v, np.int64)), fmt="%d")

    put("rom/rrc_taps.txt", tx.taps)
    put("rom/modulator.txt", tx.table)
    put("rom/preamble.txt", w.preamble)
    put("rom/cordic_atan.txt", _atan_table(f.cordic))
    put("tx/info_bits.txt", u)
    put("tx/coded_bits.txt", coded)
    put("tx/sent_bits.txt", sent)
    put("tx/symbols.txt", sym)
    put("tx/dac_i.txt", dac)
    put("rx/adc_i.txt", adc.real)
    put("rx/adc_q.txt", adc.imag)
    put("rx/mf_i.txt", mi)
    put("rx/mf_q.txt", mq)
    put("rx/timing.txt", [b.start, b.frac])
    t = b.trace
    put("rx/buffer_i.txt", t["buffer_i"])
    put("rx/buffer_q.txt", t["buffer_q"])
    put("rx/buffer_shift.txt", [t["energy"], b.shift])
    put("rx/coarse.txt", [t["angle_half1"], t["angle_half2"], t["inc_coarse"]])
    put("rx/fine_blocks.txt", np.stack([t["block_i"], t["block_q"], t["block_angle"]], axis=1).ravel())
    put("rx/fine.txt", [t["inc_fine"], b.inc])
    put("rx/preamble.txt", [t["preamble_mag"], t["preamble_angle"], b.gain])
    put("rx/loop_error.txt", t["loop_error"])
    put("rx/y_preamble_gain.txt", t["y_preamble_gain"])
    put("rx/refine.txt", [t["power_i"], t["power_q"], b.gain2])
    put("rx/y.txt", b.y)
    (out / "formats.json").write_text(json.dumps(dict(
        fixed_point=json.loads((ROOT / "config.json").read_text())["fixed_point"],
        waveform=dict(L0=w.L0, family=family, info_bits=w.info_bits, sps=w.sps, rolloff=w.rolloff, span=w.span,
                      guard=w.guard),
        link=dict(esn0_db=esn0_db, cfo_hz=cfo_hz, linewidth_hz=linewidth_hz, delay=link.delay, phase=link.phase,
                  seed=seed)), indent=2) + "\n")
    subprocess.run([str(MSPRS), "--mode", "vectors", "--fixed", "--L0", str(w.L0), "--family", family,
                    "--out", str(out / "turbo")], input="\n".join(map(str, b.y)) + "\n", text=True, check=True)
    errors = int((np.loadtxt(out / "turbo" / "pass7_bits.txt", dtype=np.int64) != u).sum())
    print(f"vectors in {out}: Es/N0 {esn0_db} dB, {errors} information-bit errors after 8 passes")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Write the HDL test vectors of one burst.")
    ap.add_argument("out")
    ap.add_argument("--family", default="balanced")
    ap.add_argument("--esn0", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    export_vectors(a.out, a.family, a.esn0, seed=a.seed)
