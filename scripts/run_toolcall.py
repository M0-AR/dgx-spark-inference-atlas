#!/usr/bin/env python3
"""Run N live-data tool-calling episodes against an OpenAI-compatible endpoint.

Each episode executes real Kraken/Frankfurter fetches at episode time; the
served values are the episode truth (docs/VERIFICATION.md §2). Market drift
over the run is audited with before/after snapshots. Outputs:

  results/raw/toolcall-<tag>-<utc>.jsonl      full episode records + grades
  results/derived/agentic-<tag>-<utc>.json    per-episode grades + aggregate

Usage:
  python3 scripts/run_toolcall.py --endpoint http://127.0.0.1:8080/v1 \
      --model MODEL --tag e40 --episodes 30
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from harness.bench import stats as S  # noqa: E402
from harness.bench import toolcall_eval as te  # noqa: E402
from harness.bench.live_data import snapshot_market  # noqa: E402


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--max-turns", type=int, default=6)
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--api-key", default="local")
    args = ap.parse_args()
    if args.episodes < 1:
        ap.error("--episodes must be >= 1 (the aggregate divides by it)")
    if args.max_turns < 1:
        ap.error("--max-turns must be >= 1")

    tag_run = f"{args.tag}-{_utc()}"
    raw_path = REPO / "results" / "raw" / f"toolcall-{tag_run}.jsonl"
    agg_path = REPO / "results" / "derived" / f"agentic-{tag_run}.json"
    agg_path.parent.mkdir(parents=True, exist_ok=True)

    before = snapshot_market()
    client = te.OpenAIChatClient(args.endpoint, args.model, args.api_key)
    episodes: list[dict] = []
    with raw_path.open("w") as fh:
        fh.write(json.dumps({"type": "meta", "tag": args.tag,
                             "model": args.model, "endpoint": args.endpoint,
                             "snapshot_before": before.to_json()}) + "\n")
        for i in range(args.episodes):
            rec = te.run_episode(client, max_turns=args.max_turns,
                                 seed=args.seed_base + i)
            grade = te.grade_episode(rec)
            fh.write(json.dumps({"type": "episode", "index": i,
                                 "grade": grade,
                                 "record": rec.to_json()}) + "\n")
            episodes.append({"index": i, "grade": grade})
            done = sum(1 for e in episodes if e["grade"]["C_outcome"])
            print(f"episode {i + 1}/{args.episodes} "
                  f"C_outcome so far {done}/{i + 1}")
    after = snapshot_market()

    outcome = [1.0 if e["grade"]["C_outcome"] else 0.0 for e in episodes]
    ci = S.bootstrap_ci(outcome, seed=0)
    agg = {
        "meta": {
            "tag": args.tag, "run_id": tag_run, "model": args.model,
            "endpoint": args.endpoint, "episodes": args.episodes,
            "seed_base": args.seed_base, "max_turns": args.max_turns,
            "snapshot_before": before.to_json(),
            "snapshot_after": after.to_json(),
            "generated_by": "scripts/run_toolcall.py",
        },
        "episodes": episodes,
        "aggregate": {
            "n": len(episodes),
            "C_tool_rate": sum(e["grade"]["C_tool"] for e in episodes) / len(episodes),
            "C_task_rate": sum(e["grade"]["C_task"] for e in episodes) / len(episodes),
            "C_outcome_rate": sum(e["grade"]["C_outcome"] for e in episodes) / len(episodes),
            "C_outcome_ci95": ci,
            "repair_rate": sum(e["grade"]["needed_repair"] for e in episodes) / len(episodes),
            "hallucinated_total": sum(e["grade"]["hallucinated_tool_calls"] for e in episodes),
            "api_errors_total": sum(len(e["grade"]["api_errors"]) for e in episodes),
            "mde_mcnemar": S.minimum_detectable_degradation_mcnemar(len(episodes)),
        },
    }
    agg_path.write_text(json.dumps(agg, indent=2))
    print(f"wrote {raw_path}")
    print(f"wrote {agg_path}")
    print(f"C_outcome {agg['aggregate']['C_outcome_rate']:.3f} "
          f"CI {ci['lo']:.3f}..{ci['hi']:.3f}")


if __name__ == "__main__":
    main()
