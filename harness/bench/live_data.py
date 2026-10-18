"""Live public-data fetchers used as ground truth for agentic tool-calling tasks.

Design constraints (see docs/METHODOLOGY.md):
  * No API keys, no paid endpoints, public read-only sources only.
  * Every fetch is recorded with: UTC timestamp, source URL, raw response
    digest (SHA256), parsed values. Provenance is part of the artifact.
  * Ground truth is computed independently by this module ("harness truth"),
    never by the model under test.

Sources
-------
BTC/USD : Kraken public Ticker endpoint (api.kraken.com/0/public/Ticker)
EUR/USD : Frankfurter (ECB reference rates; frankfurter.dev)

Both are keyless and stable; see docs/VERIFICATION.md for tolerance policy.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any

USER_AGENT = "dgx-spark-inference-atlas/0.1 (+reproducible research harness)"

KRAKEN_TICKER_URL = "https://api.kraken.com/0/public/Ticker?pair=XBTUSD"
FRANKFURTER_URL = "https://api.frankfurter.dev/v1/latest?base=EUR&symbols=USD"


class LiveDataError(RuntimeError):
    """Raised when a live source cannot be fetched or parsed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _fetch_json(url: str, *, timeout: float = 15.0, retries: int = 3,
                backoff_s: float = 2.0) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fetch a URL, returning (parsed_json, provenance_record)."""
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        t0 = time.monotonic()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                                       "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                status = resp.status
            elapsed_ms = round((time.monotonic() - t0) * 1000, 2)
            parsed = json.loads(raw.decode("utf-8"))
            prov = {
                "url": url,
                "fetched_at_utc": _utc_now(),
                "http_status": status,
                "elapsed_ms": elapsed_ms,
                "sha256": _sha256_bytes(raw),
                "bytes": len(raw),
                "attempt": attempt,
            }
            return parsed, prov
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
                json.JSONDecodeError) as exc:  # noqa: PERF203
            last_err = exc
            if attempt < retries:
                time.sleep(backoff_s * attempt)
    raise LiveDataError(f"fetch failed after {retries} attempts: {url}: {last_err}")


@dataclass
class MarketSnapshot:
    """One consistent view of the market cross-rate, with provenance."""

    fetched_start_utc: str = ""
    fetched_end_utc: str = ""
    btc_usd: float = 0.0
    eur_usd: float = 0.0
    btc_eur: float = 0.0
    btc_usd_source: dict[str, Any] = field(default_factory=dict)
    eur_usd_source: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def cross_is_consistent(self) -> bool:
        if self.btc_usd <= 0 or self.eur_usd <= 0:
            return False
        return abs(self.btc_eur - self.btc_usd / self.eur_usd) < 1e-9 * self.btc_usd


def fetch_btc_usd() -> tuple[float, dict[str, Any]]:
    """Latest BTC/USD trade price from Kraken (last trade of most recent tick)."""
    parsed, prov = _fetch_json(KRAKEN_TICKER_URL)
    errs = parsed.get("error") or []
    if errs:
        raise LiveDataError(f"kraken error: {errs}")
    result = parsed.get("result") or {}
    # Kraken keys: XXBTZUSD (classic) or XBTUSD; take the only entry.
    if len(result) != 1:
        raise LiveDataError(f"unexpected kraken result shape: keys={list(result)}")
    pair, tick = next(iter(result.items()))
    last_trade = tick.get("c") or []
    if not last_trade:
        raise LiveDataError(f"kraken ticker missing 'c' field: {tick}")
    price = float(last_trade[0])
    prov["pair"] = pair
    prov["field"] = "c[0] (last trade price)"
    prov["bid_ask"] = {
        "bid": float((tick.get("b") or ["nan"])[0]),
        "ask": float((tick.get("a") or ["nan"])[0]),
    }
    return price, prov


def fetch_eur_usd() -> tuple[float, dict[str, Any]]:
    """Latest ECB-based EUR->USD reference rate from Frankfurter."""
    parsed, prov = _fetch_json(FRANKFURTER_URL)
    rate = (parsed.get("rates") or {}).get("USD")
    if rate is None:
        raise LiveDataError(f"frankfurter missing USD rate: {parsed}")
    prov["date"] = parsed.get("date")
    prov["field"] = "rates.USD (ECB reference)"
    return float(rate), prov


def snapshot_market() -> MarketSnapshot:
    """Fetch both sources and compute the BTC/EUR cross rate."""
    snap = MarketSnapshot(fetched_start_utc=_utc_now())
    snap.btc_usd, snap.btc_usd_source = fetch_btc_usd()
    snap.eur_usd, snap.eur_usd_source = fetch_eur_usd()
    snap.btc_eur = snap.btc_usd / snap.eur_usd
    snap.fetched_end_utc = _utc_now()
    if not snap.cross_is_consistent:
        raise LiveDataError("cross-rate consistency check failed")
    snap.notes.append("btc_eur = btc_usd / eur_usd, computed by harness")
    if snap.btc_usd_source.get("date") == snap.eur_usd_source.get("date"):
        snap.notes.append("same calendar date on both sources")
    return snap


def tolerance_for(field_name: str) -> float:
    """Relative tolerance for grading a model-computed value against harness truth.

    Rationale: Kraken BTC/USD moves intra-minute; a model that fetched its own
    live quote seconds before/after the harness will differ by a small amount.
    ECB EUR/USD is fixed for the day, but we allow the same band for uniformity.
    """
    tolerances = {
        "btc_usd": 0.01,   # +/- 1.0%: covers a few minutes of BTC drift + spread
        "eur_usd": 0.001,  # ECB rate fixed daily; tolerate rounding conventions
        "btc_eur": 0.01,   # inherits BTC drift
    }
    if field_name not in tolerances:
        raise KeyError(f"unknown field: {field_name}")
    return tolerances[field_name]


if __name__ == "__main__":  # pragma: no cover - manual probe
    print(json.dumps(snapshot_market().to_json(), indent=2))
