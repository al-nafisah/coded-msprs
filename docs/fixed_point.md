# Fixed point

Every word the FPGA computes, from the information bits to the 12-bit DAC
words and from the 12-bit ADC words to the decisions. `ota/fixed.py` computes
the transmitter and the receiver's front end, `include/MSPRS/Fixed.hpp` the
turbo receiver, and the HDL is written to match them bit for bit. The widths
are in the `fixed_point` section of `config.json`.

Words are two's complement. *Rounded* means to the nearest, ties away from
zero. *Saturated to b bits* clips to ±(2^(b−1) − 1), so the most negative
code never occurs. Right shifts of signed words are arithmetic. Angles are in
turns: 2^16 per turn at the CORDIC, 2^24 in the phase accumulator. The figures
are for `L0 = 3`.

## Transmitter

| Word | Format | Rule |
|---|---|---|
| Interleaver | 10,000 × 14-bit addresses, `rom/interleaver.hex` | the i-th bit sent is bit `pi[i]` of the coded block |
| Modulator table | 16 × 10-bit signed | `round(256 s)`, indexed by `x0[t] x0[t-1] x0[t-2] x1[t]` as bits, newest first; preamble chips ±256, guard 0 |
| RRC taps | 129 × 12-bit signed | `round(4096 h)`, `h` the unit-energy root-raised cosine (roll-off 0.3, ±8 symbols, 8 samples per symbol); the receiver uses the same taps |
| DAC word | 12-bit signed, in-phase rail | sum of taps × symbol words, rounded by 2^10: 1024 LSB per unit of amplitude. The worst case is 1163 (balanced) and 1106 (unbalanced), so it never saturates. The quadrature rail is 0 |

## Front end

At 1 MS/s:

| Word | Format | Rule |
|---|---|---|
| ADC word | 12-bit signed, I and Q | at 65 dB receive gain the noise is 7 LSB rms per rail and the strongest bench bursts peak below 250 LSB |
| Matched filter | 14-bit signed, ½ ADC LSB | sum of taps × ADC words, rounded by 2^11, saturated |
| Correlation `c`, energy `E` | exact, `c` < 2^20 per rail, `E` < 2^34 | over the 126 preamble symbols, every 8th sample: `c = Σ p_m mf[n+8m]`, `E = Σ |mf[n+8m]|²` |
| Detection | | `64 |c|² > 13 · 126 · E`, a normalised correlation above 0.45; the burst starts at the largest `|c|²` in the 520 samples from the first hit, since one preamble half alone scores 0.7 |
| Fractional timing `d` | 1/16 sample, −8 to 8 | `round(16 (C₋ − C₊) / 2(C₋ − 2C₀ + C₊))`, `C` = `|c|²` at the peak and its two neighbours; 0 unless the denominator is negative |

Per burst, at symbol rate, 5,127 symbols of preamble and data:

| Word | Format | Rule |
|---|---|---|
| Interpolated sample | 18-bit signed | `16 mf[n] + |d| (mf[n±1] − mf[n])`, the neighbour on the side of `d` |
| Buffer shift `sh` | 0 to 8 | the smallest `sh` ≥ 0 with `E` ≤ 126 · 4^(6+sh), `E` from the detection |
| Buffer word | 10-bit signed, I and Q | the interpolated sample rounded by 2^(4+sh), saturated: rms at most 64 |
| CORDIC | 14 iterations, 4 guard bits | inputs shifted left by 4 and outputs rounded back; gain 1.64676; table `round(2^16 atan(2^-i) / 2π)` |
| Coarse increment | 24-bit turns per symbol | `φ1`, `φ2` the angles of `Σ pn_m r_m` over the two preamble halves; `round(wrap16(φ2 − φ1) · 266305 / 2^16)`, that is `/63` |
| Block sums `z_b` | 39 × complex, exact, < 2^28 | `Σ x²` over 128 data symbols, `x` the buffer word rotated by `−(k · coarse mod 2^24) >> 8`, `k` counted from the first preamble symbol |
| Fine increment | 24-bit | `u_b` the block angles unwrapped by `u_b = u_(b−1) + wrap16(ψ_b − ψ_(b−1))`; `round(Σ (2b − 38) u_b · 1698 / 2^24)`, the slope of the line through them, halved |
| Increment | 24-bit, wraps | coarse + fine |
| Preamble phase `θ0` | 16-bit | angle of `g = Σ p_m rotate(r_m, −(m · inc) >> 8)` |
| Gain `G` | 15-bit unsigned | `round(44,364,069 / M)`, saturated, `M` the CORDIC magnitude of `g`; 44,364,069 = `round(26.1 · 1.64676 · 126 · 2^13)` |
| Phase loop `φ`, `ν` | 24-bit, wrap | per data symbol: rotate the buffer word by `−(θ + θ0·2^8 + φ) >> 8`, `θ` the phase accumulator; `y`, `q` = real and imaginary part × `G`, rounded by 2^13, saturated to 8 bits; `e = q` if `y` ≥ 0 else `−q`; `ν += 8e`, `φ += 2048e + ν` |
| Powers `P_y`, `P_q` | 27-bit unsigned | `Σ y²` and `Σ q²` over the 5,001 data symbols |
| Gain correction `G2` | 14-bit unsigned | `isqrt(round(57,155,465,364,111 / (P_y − P_q)))`, saturated, the constant `round(5001 · 26.1² · 2^24)`; 4096 when `P_y` ≤ `P_q` |
| `y` | 8-bit signed, 26.1 LSB per unit of amplitude | `y G2` rounded by 2^12, saturated; to the turbo receiver |

## Turbo receiver

Both SISOs run max-log-MAP on costs, so every metric is an integer and smaller
is better. The equaliser's metric is `(y − s)²` with no `1/2σ²` and no noise
estimate: the scale is set once by the 26.1 LSB per unit amplitude the front
end normalises to. One message step is then 1.19 nats at Es/N0 = 5 dB, and
grows with the SNR.

| Word | Format | Rule |
|---|---|---|
| `y` | 8-bit signed | from the front end |
| Constellation | 16 × 8-bit signed | `round(26.1 s)` per state, `b0`, `b1` |
| Branch metric | 7-bit unsigned | `(y − s)²` rounded by 2^5, saturated at 127 |
| Edge cost | 9-bit signed, −112 to 239 | branch metric + 8 (`b0 La0 + b1 La1`), `La` the decoder's message codes |
| Equaliser state metric | 7-bit unsigned | min over incoming edges of state metric + edge cost, minus the minimum over the 4 states, saturated at 127; every state but 0 starts at 127, forward and backward |
| Equaliser message | 4-bit, −7 to 7 | `ext` = (least cost with the bit 1) − (least cost with the bit 0) − 8 `La`, rounded by 2^3, saturated |
| Decoder edge cost | 5-bit signed | `Σ c_j L_j` over the edge's two coded bits, in message steps |
| Decoder state metric | 5-bit unsigned | as the equaliser's, in message steps, saturated at 31 |
| Decoder message | 4-bit, −7 to 7 | `ext` = (least cost with the coded bit 1) − (with 0) − `L`, saturated |
| Decision | 1 bit | 1 when the best path with `u = 1` costs less than the best with `u = 0` |
| Passes | 8 | the first with no a priori |

## Decisions

**BCJR arithmetic.** Max-log-MAP in both SISOs. Max-log is exact under a
common scale of the channel metric and the a priori, so the constant scale
costs nothing in floating point; on the integer grid the message step in nats
follows the noise. Against the same receiver with `y` rescaled to the true
noise at every Eb/N0, which keeps the step at 1.19 nats everywhere, the
constant scale loses at most 0.03 dB between BER 1e-2 and 1e-4 in either
family. An extrinsic scale of 0.75 on either message or both gains
nothing here: +0.005 to +0.016 dB at 1e-4.

**Word lengths.** Each word was narrowed from a wider format, one at a time,
on the same frames as the floating-point reference (records under
`results/fixed/ber`). Loss at BER 1e-4 against floating-point max-log-MAP,
balanced, with every other word wide:

| Messages | Loss |
|---|---|
| 5 bits, step 0.5 / 1 nat | 0.05 / 0.05 dB |
| 4 bits, step 0.5 / 1 / 2 nats | 0.61 / 0.10 / 0.23 dB |
| 3 bits, step 1 / 2 nats | 0.83 / 0.27 dB |
| 4 bits one way, 3 bits the other | 0.24 / 0.27 dB |

Then with 4-bit messages at a step of 1 nat, the loss each narrower word adds
to their 0.10 dB:

| Word | Added loss |
|---|---|
| `y`, 8 / 7 / 6 bits | 0.00 / 0.01 / 0.15 dB |
| Branch metric, 7 / 6 / 5 bits | none / 0.007 / 0.63 dB |
| Equaliser state metric, 7 / 6 / 5 bits | none / 0.004 / 0.22 dB |
| Decoder state metric, 8 down to 5 bits | none |

*None* means the same decisions on every frame: a metric more than about 16
nats above the best state cannot reach a message that saturates at 7 steps.
Last, the step itself, through the amplitude of `y`: 0.84 / 1.00 / 1.09 /
1.19 / 1.30 / 1.42 nats lose 0.15 / 0.10 / 0.09 / 0.08 / 0.10 / 0.18 dB. The
1.19-nat step is best in both families, 0.078 dB balanced and unbalanced, and
the narrow metrics leave it unchanged.

The loss grows up the waterfall, most in the balanced family, whose
max-log-MAP already converges 0.48 dB later than MAP at 1e-3; the integer
receiver adds 0.29 dB there, 0.13 dB in the unbalanced family. From MAP, at
1e-3 and 1e-4: 0.78 and 0.085 dB balanced, 0.14 and 0.079 dB unbalanced.
From 6 to 7 dB the integer receiver has the slow tail floating point has, and
no floor of its own.

**Memory.** Full-frame forward-backward, exact against the model. The words
held per burst:

| Memory | Size |
|---|---|
| Interleaver | 10,000 × 14 = 140 kbit |
| Messages, both directions | 2 × 10,000 × 4 = 80 kbit |
| `y` | 5,001 × 8 = 40 kbit |
| Symbol buffer | 5,127 × 2 × 10 = 103 kbit |
| Forward state metrics, shared by the two SISOs | 5,002 × 4 × 7 = 140 kbit |
| Matched-filter delay line for the correlator | about 1,600 × 28 = 45 kbit |

About 550 kbit, roughly 21 of the 60 BRAM36 once each memory is fitted to the
block widths. A sliding window would save the state metrics only.

**Gain.** The preamble's 126 symbols leave the gain with an error of 3.6% rms
at Es/N0 = 5 dB. The integer turbo receiver fails on bursts whose gain comes
out more than about 6% low: at a gain of 0.90 to 0.92, BER 4.5e-2 against
2.9e-4 for floating-point max-log-MAP on the same bursts, while a gain up to
10% high costs little. A low gain shrinks the extrinsic values of the first
passes below the 1.19-nat message step, and the loop never starts; wider
messages at the same step do not help, a finer step helps. Rather than spend
message bits, the front end measures the gain again over the whole burst:
the in-phase rail carries signal and noise, the quadrature rail the same noise
alone, so their power difference is the signal's. That leaves 0.95% rms, and
the chain at 5 dB goes from BER 1.0e-3 to 1.1e-4, against 9.8e-5 with ideal
sync.

**Interleaver.** AFF3CT's random interleaver with seed 42, as the simulator
always used, so every stored BER curve stands. It is now drawn once and kept
in `rom/interleaver.hex`, which the simulator, `ota/` and the HDL read; the
NumPy permutation `ota/waveform.py` used before is gone.

**Phase walk.** Modelled as a Wiener process of a given linewidth on top of
the carrier offset (`ota/channel.py`). The bench has not measured it yet; 1 Hz
is assumed. The second-order loop on `sign(y) q` follows it (table below);
its gains were chosen from a sweep at 2 to 12 dB and 0 to 30 Hz, where wider
loops cost nothing at low SNR.

**Speed.** 8 passes of both SISOs are 80,000 trellis steps per 41.5 ms burst,
about 50 clock cycles per step at 100 MHz.

## Measured

**Turbo receiver**, Gaussian channel, ideal synchronisation, Eb/N0 at BER
1e-3 and 1e-4 (`results/fixed/ber` and the references in `results/quant/ber`):

| | Balanced | Unbalanced |
|---|---|---|
| Floating-point MAP | 3.89 / 4.92 dB | 4.03 / 5.18 dB |
| Floating-point max-log-MAP | 4.37 / 4.93 dB | 4.04 / 5.18 dB |
| Integer | 4.66 / 5.00 dB | 4.17 / 5.26 dB |

**Front end**, the MER of the received words against the 2 Es/N0 of an ideal
receiver (`results/fixed/front_end.json`, 96 bursts per point). Over the
bench's received levels, with a carrier offset of 20 to 150 Hz and a 1 Hz phase
walk, loss in dB:

| Es/N0 | Integer | Wide twin | `sync.py` |
|---|---|---|---|
| −4 dB | 0.34 | 0.33 | 0.24 |
| 0 dB | 0.12 | 0.11 | 0.19 |
| 4 dB | 0.07 | 0.07 | 0.17 |
| 8 dB | 0.05 | 0.04 | 0.33 |
| 12 dB | 0.08 | 0.07 | 0.50 |
| 16 dB | 0.15 | 0.14 | 1.46 |
| 20 dB | 0.22 | 0.23 | 2.38 |
| 24 dB | 0.70 | 0.54 | 4.27 |

The integer words cost at most 0.01 dB up to 20 dB; above that the 8-bit `y`
and the 10-bit buffer show, where the coded link has no errors left. Bursts are
found down to about −4 dB (3 of 96 missed there, none above). `sync.py` takes
the nearest of the 8 samples per symbol and has no phase loop, which is what
it loses at high SNR. Against the phase walk at Es/N0 = 5 dB:

| Linewidth | Integer | Integer, no loop | `sync.py` |
|---|---|---|---|
| 0 Hz | 0.02 | 0.01 | 0.15 |
| 1 Hz | 0.05 | 0.75 | 0.24 |
| 3 Hz | 0.09 | 4.15 | 0.74 |
| 10 Hz | 0.19 | 7.56 | 3.05 |
| 30 Hz | 0.56 | 11.04 | 7.45 |

Without the loop the walk also leaks signal into the quadrature rail, which
biases the whole-burst gain; with it the gain comes out 0.7% high at 1 Hz and
2.8% at 30 Hz, on the side the turbo receiver tolerates.

**Whole chain**, `ota/chain.py`: integer transmitter, simulated bench,
integer front end and integer turbo receiver, next to `sync.py` and the MAP
turbo receiver on the same captures (`results/fixed/chain`). "Gaussian"
draws only a carrier phase and a timing offset per burst; "bench" adds a
carrier offset of ±20 to 150 Hz and a 1 Hz phase walk. Eb/N0 at BER 1e-4 and
the loss against floating-point MAP with ideal synchronisation:

| | Balanced | Unbalanced |
|---|---|---|
| Integer turbo receiver, ideal synchronisation | 5.00 dB, 0.09 dB | 5.26 dB, 0.08 dB |
| Integer chain, Gaussian | 5.06 dB, 0.14 dB | 5.31 dB, 0.13 dB |
| Integer chain, bench | 5.08 dB, 0.16 dB | 5.32 dB, 0.14 dB |
| `sync.py` and MAP, Gaussian | 5.06 dB, 0.14 dB | 5.29 dB, 0.12 dB |
| `sync.py` and MAP, bench | above 5e-4 at 5.8 dB | above 3e-4 at 6.0 dB |

No burst was missed. The total stays under the 0.3 dB the plan set. `sync.py`
has no phase loop and leaves the walk in the symbols; its BER falls only
slowly past 5 dB.

## Vectors

`python -m ota.fixed <dir>` sends one burst through the integer chain over
the simulated bench and writes every block's input and output, one integer
per line: `rom/` (taps, modulator table, preamble, CORDIC table), `tx/`
(information bits to DAC words), `rx/` (ADC words to `y`, with the
intermediate values of each estimate) and `turbo/` (per pass, the messages in
and out of both SISOs, their state metrics, forward then backward, and the
decisions). `formats.json` records the formats and the link.
