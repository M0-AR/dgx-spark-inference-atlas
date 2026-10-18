#!/usr/bin/env python3
"""Save a timestamped market snapshot artifact (results/raw/market-*.json).

Run before and after long experiment conditions so drift over the session is
documented (see docs/VERIFICATION.md, tolerance policy).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from harness.bench.live_data import snapshot_market  # noqa: E402


def main() -> None:
    snap = snapshot_market()
    out_dir = REPO / "results" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = out_dir / f"market-{stamp}.json"
    out.write_text(json.dumps(snap.to_json(), indent=2))
    print(f"{out}  btc_usd={snap.btc_usd:.2f} eur_usd={snap.eur_usd:.4f} "
          f"btc_eur={snap.btc_eur:.2f}")


if __name__ == "__main__":
    main()
