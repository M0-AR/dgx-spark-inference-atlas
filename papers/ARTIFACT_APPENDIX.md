# Artifact Appendix

Following PACT-2026-AE conventions (artifact appendix, ≤30-minute getting
started, reproducible environment).

## A.1 Artifact location

Repository root. The artifact is self-contained: harness, protocols,
generated tables, raw evidence under `results/`, and the Spark replication
kit (`docs/SPARK_KIT.md`).

## A.2 Getting started (≤30 minutes, no GPU)

```bash
# 1) container floor (builds a small CPU image; runs all self-tests)
docker compose -f docker/docker-compose.yml run --rm smoke      # ~2 min
#    Expected: "19 passed" — or "18 passed, 1 skipped" when the container
#    has no network (the live-data grader skips cleanly) — then
#    results/raw/env-<utc>.json written.

# 2) inspect the live-data graders actually fetching real prices
python3 -m pytest harness/tests/test_toolcall_eval.py -q        # network needed
#    Expected: pass with network; skipped (never failed) without it. Graders
#    compare against live Kraken/Frankfurter values.

# 3) regenerate every README number from committed artifacts
python3 scripts/sync_readme.py && git diff --stat README.md
#    Expected: no unexpected diff (idempotent); results/derived/readme-sync.json
#    lists input artifacts and output hashes.
```

## A.3 Claims supported by this artifact

- Statistics harness correctness: `harness/tests/test_stats.py` (known-value
  tests for McNemar exact, the McNemar MDE planning formula, chi-square
  survival values, bootstrap determinism and seed sensitivity, Friedman,
  changepoint behaviour).
- Runner protocol correctness (offline): `harness/tests/test_runner.py`
  exercises `run_condition` end-to-end against a local mock SSE server —
  wave batching (incl. partial final batch), TTFT/total-latency semantics
  for non-streaming endpoints, warm-up bookkeeping.
- Live-data verification policy: `docs/VERIFICATION.md`, exercised by the
  test suite and by mock-episode plumbing runs.
- Protocol implementation (warm-up, isolation gate, energy meter, repair
  accounting): `harness/bench/runner.py`, `harness/bench/toolcall_eval.py`.
- Data-integrity rule (README numbers only from artifacts): enforced by
  `scripts/sync_readme.py` and its manifest.

## A.4 What needs special hardware

- The EXL3 ladder (quality/timing/energy rows in README) needs an NVIDIA GPU
  with a working exllamav3 build (`scripts/setup_gpu.sh`), and a serving
  endpoint for the tool-calling arm.
- The Spark arm needs DGX Spark hardware; the kit is executable as published
  and expects results to be dropped into `results/raw/` unchanged.

## A.5 Reuse

The harness is dataset-agnostic per episode: swap `TOOL_SCHEMAS` in
`harness/bench/toolcall_eval.py` and the tolerance map in
`harness/bench/live_data.py` to evaluate other live-data tool-calling tasks
with the same statistics. The isolation/energy runner is fully generic over
any OpenAI-compatible endpoint.
