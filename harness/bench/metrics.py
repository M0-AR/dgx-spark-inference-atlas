"""Measurement primitives: latency, throughput, energy, and their aggregates.

Energy follows the pre-registered policy (docs/METHODOLOGY.md, energy section):
  * Primary counter: NVML cumulative device energy, `nvmlDeviceGetTotalEnergyConsumption`
    (mJ, monotone) read at start/end of each condition; joules/token derived.
  * Diagnostics: instantaneous power sampled at 10 Hz (Watt Counts arXiv:2604.09048).
  * Idle baseline recorded before each condition; net energy = condition - idle*dt.

All functions are pure and dependency-light; NVML is optional (pynvml).
"""

from __future__ import annotations

import statistics as st
from dataclasses import dataclass, asdict
from typing import Sequence

try:  # optional
    import pynvml as _nvml  # type: ignore
    _NVML_OK = True
except Exception:  # pragma: no cover
    _NVML_OK = False


def percentile(sorted_vals: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (numpy-compatible 'linear' method)."""
    if not sorted_vals:
        raise ValueError("empty series")
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = (len(sorted_vals) - 1) * (q / 100)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return float(sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac)


def summarize_latencies(latencies_ms: Sequence[float]) -> dict[str, float]:
    xs = sorted(float(v) for v in latencies_ms)
    if not xs:
        return {}
    mean = st.fmean(xs)
    sd = st.pstdev(xs) if len(xs) > 1 else 0.0
    return {
        "n": len(xs),
        "mean_ms": mean,
        "sd_ms": sd,
        "cv": (sd / mean) if mean else float("nan"),
        "p50_ms": percentile(xs, 50),
        "p90_ms": percentile(xs, 90),
        "p99_ms": percentile(xs, 99),
        "min_ms": xs[0],
        "max_ms": xs[-1],
    }


def decode_tok_per_s(output_tokens: int, decode_seconds: float) -> float:
    if decode_seconds <= 0:
        raise ValueError("decode_seconds must be positive")
    return output_tokens / decode_seconds


@dataclass
class EnergyReading:
    name: str
    total_energy_mj_start: int | None
    total_energy_mj_end: int | None
    duration_s: float
    joules: float | None
    joules_per_token: float | None
    mean_power_w: float | None

    def to_dict(self) -> dict[str, float | int | str | None]:
        return asdict(self)


class EnergyMeter:
    """Context manager around NVML cumulative-energy counter (single GPU index)."""

    def __init__(self, gpu_index: int = 0, label: str = "", tokens: int | None = None):
        self.gpu_index = gpu_index
        self.label = label
        self.tokens = tokens
        self._start_mj: int | None = None
        self._handle = None
        self._t0 = 0.0
        self._t1 = 0.0

    def __enter__(self) -> "EnergyMeter":
        import time
        self._t0 = time.monotonic()
        if _NVML_OK:
            try:
                _nvml.nvmlInit()
                self._handle = _nvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
                self._start_mj = int(
                    _nvml.nvmlDeviceGetTotalEnergyConsumption(self._handle))
            except Exception:
                self._handle = None
                self._start_mj = None
        return self

    def __exit__(self, *exc) -> None:
        import time
        self._t1 = time.monotonic()

    def reading(self) -> EnergyReading:
        end_mj: int | None = None
        if _NVML_OK and self._handle is not None:
            try:
                end_mj = int(_nvml.nvmlDeviceGetTotalEnergyConsumption(self._handle))
            except Exception:
                end_mj = None
        dur = max(self._t1 - self._t0, 1e-9)
        joules = None
        jpt = None
        if self._start_mj is not None and end_mj is not None:
            joules = (end_mj - self._start_mj) / 1000.0
            if self.tokens:
                jpt = joules / self.tokens
        return EnergyReading(
            name=self.label,
            total_energy_mj_start=self._start_mj,
            total_energy_mj_end=end_mj,
            duration_s=dur,
            joules=joules,
            joules_per_token=jpt,
            mean_power_w=(joules / dur) if joules is not None else None,
        )


def nvml_available() -> bool:
    return _NVML_OK


def json_strictness_report(parse_ok: bool, repaired: bool, schema_ok: bool) -> dict[str, bool]:
    """Standard record for output-format analysis. 'repaired' must be recorded
    separately from success so that repair-window effects stay visible."""
    return {
        "parse_ok_no_repair": bool(parse_ok),
        "needed_repair": bool(repaired),
        "schema_ok": bool(schema_ok),
    }
