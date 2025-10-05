# Conventions and cache format

## Records

One JSON per `(scheme, Eb/N0)` under `results/ber/<scheme>/snr_<x>dB.json`.

| Field | Meaning |
|---|---|
| `eb_no_db` | per-information-bit Eb/N0, the common abscissa |
| `ber`, `ers_cnt`, `bits_cnt`, `n_frames` | error rate and the sample behind it |
| `ber_lo95`, `ber_ub95`, `is_upper_bound` | Clopper-Pearson interval; a zero-error point is stored as a bound, never as `ber = 0` |
| `ers_per_iter` | errors after each turbo iteration |
| `config` | parameters the point was produced under |

## Energy and rate

Every curve shares one per-information-bit axis, `Es = m Rc Eb`. Rate-2 MS-PRS
carries `m = 2` coded bits per symbol; coded runs use `Rc = 1/2`. Getting this
wrong shifts a scheme by 3 dB while each piece still looks correct on its own.

## Sign

LLR is `ln P(b=0)/P(b=1)` in the modems and decoders, so positive means bit 0.
The EXIT estimators use the opposite sign and the sweeps negate on the way in
and out. Getting that wrong does not crash; it produces an EXIT curve that
falls instead of rising.

`CP[0]` is sigma, not sigma^2. The branch metric divides by `2 sigma^2`;
dropping the 2 doubles every extrinsic LLR and wrecks turbo convergence while
still looking plausible.

## Distances

`--mode bounds` returns `d2_min` on the internal `Es = 1` normalisation and on
the `Es = 5` scale, where 4-ASK has `d2 = 4`.

## Reproducibility

EXIT results depend on `--threads`: a-priori draws are seeded per thread and
the IA grid is dealt round-robin. BER sweeps do not.
