"""The whole link in simulation, both receivers on the same captures.

The integer transmitter's DAC words go through the simulated bench
(ota/channel.py). The integer receiver is ota/fixed.py followed by the
simulator's integer turbo receiver; the float one is ota/sync.py followed by
the simulator's MAP turbo receiver. One record per receiver and Es/N0 goes
under results/fixed/chain/<scheme>/snr_<x>dB.json, with Es/N0 = Eb/N0 as in
the simulator's coded records. A burst the receiver does not find is counted
in `missed` and left out of the error rate.

--front-end measures the front ends alone: the mean squared error of the
received symbols against those sent, as a modulation error ratio (MER) next
to the 2 Es/N0 an ideal receiver reaches on the in-phase rail. It covers the
bench's range of received levels and a range of phase walks, for the integer
front end, its wide twin (Formats.wide) and ota/sync.py, and writes
results/fixed/front_end.json.

    python -m ota.chain --family balanced --scenario gaussian --esn0 4 4.2 4.4
    python -m ota.chain --front-end
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import multiprocessing as mp

import numpy as np

from ota.channel import Link
from ota.fixed import Formats, Receiver, Transmitter, turbo
from ota.sync import bursts
from ota.waveform import ROOT, Waveform

# Carrier offset drawn per burst from +-(lo, hi) Hz; the phase walk's linewidth
# on the bench is not measured yet, so "bench" assumes 1 Hz.
SCENARIOS = {
    "gaussian": dict(cfo=(0.0, 0.0), linewidth=0.0),
    "bench": dict(cfo=(20.0, 150.0), linewidth=1.0),
}


def _bursts(args):
    """One chunk of bursts through the link and both front ends."""
    family, esn0, n, seed, scenario = args
    sc = SCENARIOS[scenario]
    w = Waveform(family=family)
    f = Formats.load()
    tx = Transmitter(w, f)
    rx = Receiver(w, f)
    rng = np.random.default_rng(seed)
    u_all, y_fx, y_fl, sig, ok_fx, ok_fl = [], [], [], [], [], []
    for _ in range(n):
        u, sent, _ = w.block(rng)
        link = Link(esn0_db=esn0, amp=f.dac_amp, cfo_hz=rng.uniform(*sc["cfo"]) * rng.choice([-1, 1]),
                    linewidth_hz=sc["linewidth"], delay=rng.uniform(500, 3000), phase=rng.uniform(0, 2 * np.pi))
        adc = link(tx.burst(tx.symbols(sent)), rng)
        u_all.append(u)
        b = rx.receive(adc)
        ok_fx.append(bool(b))
        y_fx.append(b[0].y if b else np.zeros(w.L, np.int64))
        fb = bursts(adc, w, link.fs)
        ok_fl.append(bool(fb))
        y_fl.append(fb[0].symbols.real if fb else np.zeros(w.L))
        sig.append(fb[0].sigma if fb else 1.0)
    return np.array(u_all), np.array(y_fx), np.array(y_fl), np.array(sig), np.array(ok_fx), np.array(ok_fl)


def run(family: str, esn0: float, scenario: str, max_bursts: int = 6000, min_errors: int = 1000,
        chunk: int = 50, procs: int = 8, seed: int = 0) -> dict:
    """Error counts of both receivers at one Es/N0, paired burst for burst."""
    w = Waveform(family=family)
    count = {r: dict(errors=0, bits=0, frames=0, frame_errors=0, missed=0) for r in ("fixed", "float")}
    k = 0
    with mp.Pool(procs) as pool:
        while count["fixed"]["frames"] + count["fixed"]["missed"] < max_bursts and \
                (count["fixed"]["errors"] < min_errors or count["float"]["errors"] < min_errors):
            jobs = [(family, esn0, chunk, [seed, round(esn0 * 1000), k + j], scenario) for j in range(procs)]
            k += procs
            for u, y_fx, y_fl, sig, ok_fx, ok_fl in pool.map(_bursts, jobs):
                for name, ok, dec in (("fixed", ok_fx, lambda: turbo(y_fx[ok_fx], w.L0, family)),
                                      ("float", ok_fl, lambda: turbo(y_fl[ok_fl], w.L0, family, sigma=sig[ok_fl]))):
                    c = count[name]
                    c["missed"] += int((~ok).sum())
                    if ok.any():
                        e = (dec() != u[ok]).sum(axis=1)
                        c["errors"] += int(e.sum())
                        c["frame_errors"] += int((e > 0).sum())
                        c["frames"] += int(ok.sum())
                        c["bits"] += int(ok.sum()) * u.shape[1]
    return count


def _mer(args):
    """Squared symbol errors of each front end on one chunk of captures."""
    family, esn0, n, seed, cfo, linewidth = args
    w = Waveform(family=family)
    f = Formats.load()
    tx = Transmitter(w, f)
    receivers = {"fixed": Receiver(w, f), "wide": Receiver(w, f.wide()),
                 "fixed_no_loop": Receiver(w, dataclasses.replace(f, loop_p=-1))}
    rng = np.random.default_rng(seed)
    se = {k: [] for k in (*receivers, "float")}
    for _ in range(n):
        _, sent, sym = w.block(rng)
        link = Link(esn0_db=esn0, amp=f.dac_amp, cfo_hz=rng.uniform(*cfo) * rng.choice([-1, 1]),
                    linewidth_hz=linewidth, delay=rng.uniform(500, 3000), phase=rng.uniform(0, 2 * np.pi))
        adc = link(tx.burst(tx.symbols(sent)), rng)
        for k, rx in receivers.items():
            b = rx.receive(adc)
            se[k].append(float(np.mean((b[0].y / f.y_amp - sym) ** 2)) if b else None)
        fb = bursts(adc, w, link.fs)
        se["float"].append(float(np.mean((fb[0].symbols.real - sym) ** 2)) if fb else None)
    return se


def front_end(family: str = "balanced", bursts_per_point: int = 96, procs: int = 8) -> dict:
    """MER of each front end over the bench's received levels (carrier offset
    20 to 150 Hz, 1 Hz linewidth) and over phase walks at Es/N0 = 5 dB."""
    def point(esn0, cfo, linewidth, key):
        jobs = [(family, esn0, bursts_per_point // procs, [key, k], cfo, linewidth) for k in range(procs)]
        with mp.Pool(procs) as pool:
            parts = pool.map(_mer, jobs)
        row = dict(esn0_db=esn0, linewidth_hz=linewidth, ideal_db=esn0 + 10 * np.log10(2))
        for k in parts[0]:
            v = [x for q in parts for x in q[k] if x is not None]
            row[k + "_db"] = float(-10 * np.log10(np.mean(v))) if v else None
            row[k + "_missed"] = sum(x is None for q in parts for x in q[k])
        return row
    levels = [point(x, SCENARIOS["bench"]["cfo"], SCENARIOS["bench"]["linewidth"], 1000 + i)
              for i, x in enumerate((-4, -2, 0, 2, 4, 6, 8, 12, 16, 20, 24))]
    walks = [point(5.0, SCENARIOS["bench"]["cfo"], lw, 2000 + i) for i, lw in enumerate((0.0, 1.0, 3.0, 10.0, 30.0))]
    return dict(family=family, bursts_per_point=bursts_per_point, levels=levels, walks=walks,
                fixed_point=json.loads((ROOT / "config.json").read_text())["fixed_point"])


def write(family: str, esn0: float, scenario: str, count: dict, out=ROOT / "results" / "fixed" / "chain") -> None:
    f = Formats.load()
    for name, c in count.items():
        d = out / f"nsm_L3_{family}_{name}_{scenario}"
        d.mkdir(parents=True, exist_ok=True)
        rec = dict(eb_no_db=esn0, ber=c["errors"] / max(c["bits"], 1), ers_cnt=c["errors"], bits_cnt=c["bits"],
                   n_frames=c["frames"], frame_errors=c["frame_errors"], missed=c["missed"],
                   config=dict(L0=3, filter=family, receiver=name, scenario=scenario, **SCENARIOS[scenario],
                               fixed_point=json.loads((ROOT / "config.json").read_text())["fixed_point"]
                               if name == "fixed" else None, iters=7))
        (d / f"snr_{esn0:.1f}dB.json").write_text(json.dumps(rec, indent=2) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="balanced")
    ap.add_argument("--scenario", default="gaussian", choices=list(SCENARIOS))
    ap.add_argument("--esn0", type=float, nargs="+", default=[])
    ap.add_argument("--front-end", action="store_true")
    ap.add_argument("--max-bursts", type=int, default=6000)
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    if a.front_end:
        r = front_end(a.family, procs=a.procs)
        out = ROOT / "results" / "fixed" / "front_end.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(r, indent=2) + "\n")
        print(f"front end in {out}")
    for x in a.esn0:
        c = run(a.family, x, a.scenario, max_bursts=a.max_bursts, procs=a.procs)
        write(a.family, x, a.scenario, c)
        print(f"{a.family} {a.scenario} {x:.1f} dB: " + "  ".join(
            f"{n} {v['errors'] / max(v['bits'], 1):.3e} ({v['errors']} in {v['frames']}, missed {v['missed']})"
            for n, v in c.items()), flush=True)
