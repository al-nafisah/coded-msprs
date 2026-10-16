"""The two Plutos on the bench: one transmits a burst in a loop, the other listens.

Units are found by serial, from bench.local.json at the repo root (or the file
passed as `serials`), which git ignores:

    {"tx": "<serial>", "rx": "<serial>"}

A prefix of each serial is enough. USB addresses change on every plug-in;
serials do not, and the bench is cabled one way, so the roles are fixed.
"""
from __future__ import annotations

import json
import pathlib

import adi
import iio
import numpy as np

from ota.waveform import ROOT

FLOOR_DB = -89.75   # lowest transmit gain: the transmitter is effectively off


def find(serial: str) -> str:
    for uri, desc in iio.scan_contexts().items():
        if uri.startswith("usb:") and f"serial={serial}" in desc:
            return uri
    raise RuntimeError(f"no Pluto with serial {serial}... on USB")


class Bench:
    def __init__(self, carrier: float = 448e6, fs: float = 1e6, rx_gain: float = 65.0,
                 bandwidth: float = 1e6, capture: int = 2**17,
                 serials: pathlib.Path = ROOT / "bench.local.json"):
        units = json.loads(pathlib.Path(serials).read_text())
        self.tx = adi.Pluto(find(units["tx"]))
        self.rx = adi.Pluto(find(units["rx"]))
        for sdr in (self.tx, self.rx):
            sdr.sample_rate = int(fs)   # below 2.083 MS/s pyadi-iio loads the decimating FIR
            sdr.rx_lo = int(carrier)
            sdr.tx_lo = int(carrier)
            sdr.rx_rf_bandwidth = int(bandwidth)
            sdr.tx_rf_bandwidth = int(bandwidth)
            sdr.tx_hardwaregain_chan0 = FLOOR_DB
        self.rx.gain_control_mode_chan0 = "manual"
        self.rx.rx_hardwaregain_chan0 = rx_gain
        self.rx.rx_buffer_size = capture
        # One kernel buffer, so a capture starts after it is asked for. With the
        # default queue, buffers filled while the caller was busy still hold the
        # previous burst; on the bench 11 of 12 captures were stale that way.
        self.rx._rxadc.set_kernel_buffers_count(1)
        self.tx.tx_cyclic_buffer = True
        self.fs = fs

    def send(self, x: np.ndarray, atten_db: float) -> None:
        """Put burst x (peak 1) on air, repeated back to back, at this attenuation."""
        self.tx.tx_destroy_buffer()
        self.tx.tx_hardwaregain_chan0 = -float(atten_db)
        self.tx.tx(x * 2**14)

    def capture(self) -> np.ndarray:
        """One buffer, after dropping one that may still hold the previous burst."""
        self.rx.rx()
        return self.rx.rx()

    def readback(self) -> dict:
        return {name: {"lo": sdr.rx_lo, "fs": sdr.sample_rate, "tx_gain": sdr.tx_hardwaregain_chan0}
                for name, sdr in (("tx", self.tx), ("rx", self.rx))} | {
                "rx_gain": self.rx.rx_hardwaregain_chan0, "rx_mode": self.rx.gain_control_mode_chan0}

    def close(self) -> None:
        """Silence the transmitter."""
        self.tx.tx_destroy_buffer()
        self.tx.tx_hardwaregain_chan0 = FLOOR_DB

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
