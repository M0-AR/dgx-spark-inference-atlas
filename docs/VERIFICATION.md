# Verification — live-data policy and tolerance rules

This project's distinctive constraint: **every evaluated agent episode in the
tool-calling arm executes its tools against live public endpoints at episode
time**, and every claim that can be checked against an external registry is
checked at build time. This document states the rules.

## 1. Live sources (the truth of an episode)

| Field | Source | Endpoint | Extraction |
|-------|--------|----------|------------|
| `btc_usd` | Kraken public ticker | `https://api.kraken.com/0/public/Ticker?pair=XBTUSD` | last trade price `result[XBTUSD].c[0]` |
| `eur_usd` | Frankfurter (ECB reference rates) | `https://api.frankfurter.dev/v1/latest?base=EUR&symbols=USD` | `rates.USD` (USD per EUR, quoted field) |
| `btc_eur` | derived by the harness | `btc_usd / eur_usd` | never fetched from a third party |

Rationale: BTC leg comes from an exchange with continuous trading
(market truth); the FX leg from an institutional reference rate that is
auditable (ECB). Using them together exercises both "continuous" and
"reference" data semantics in the same episode — mirroring real agentic
pipelines, and dodging the well-known stale-quote trap of free FX APIs.

## 2. Episode-time capture (provenance)

For every episode, the tool executor records, alongside the served values:

- exact URL, response body, and SHA-256 of the body;
- UTC timestamp of the fetch;
- the harness clock time used in grading.

These records are committed inside `results/raw/run-*.jsonl`. A grader that
used a *later* snapshot than the one served to the model would be dishonest —
so grading compares against the episode-captured values, and the
before/after snapshots only quantify drift over the condition window.

## 3. Tolerance policy (pre-registered)

| Field | Tolerance | Rationale |
|-------|-----------|-----------|
| `btc_usd` | ±1.0 % | BTC can move >0.5 % within a 2-minute episode window; ±1 % separates "copied the live value" from "stale/hallucinated" without flagging honest micro-drift |
| `btc_eur` | ±1.0 % | inherits BTC tolerance (FX reference rate is fixed daily) |
| `eur_usd` | ±0.1 % | ECB reference rate changes once per working day; ±0.1 % is far above float/rounding noise and far below any daily move |

Tolerance application is absolute on the served value, and the per-field
allowed and observed deltas are written into the grading record, so a
marginal pass can be audited later.

## 4. What the graders verify (see `toolcall_eval.py`)

- **C_tool** — every `tool_calls` entry names a tool in the advertised schema
  (hallucinated names fail; count recorded), arguments parse as JSON objects.
- **C_task** — final answer is strict JSON with exactly the required keys and
  numeric values within tolerance of the episode-captured truth; the derived
  `btc_eur` must be *consistent with the model's own two quoted legs* (we
  check both: internal consistency and truth tolerance — these are separate
  cell columns).
- **C_outcome** — C_tool ∧ C_task ∧ output required no repair. Repair paths
  (`fenced JSON`, `reasoning prefix`) are attempted in a fixed documented
  order and flagged; a repaired success is a success for C_task but never
  for C_outcome.

## 5. Offline behaviour

Network failure must never silently become a result. `live_data.py` retries
(3 attempts, backoff) and then raises `LiveDataError`; tests that need live
data *skip* (not pass); the runner aborts the condition and records the
error. `scripts/snapshot_market.py` gives the drift audit trail; two
snapshots 3 s apart must be identical on the FX leg (a self-check we run
before big conditions).

## 6. External-claim verification rules (README tables)

Rules for any number that is not ours:

1. Must carry a live URL that was actually fetched during the build
   (committed snapshot or the fetcher's raw record).
2. Must be labelled `[external]` with the date observed.
3. Contradictory external claims (e.g., TensorFold vs patched-vLLM speed)
   are reported *as a contradiction* with both sources — never averaged,
   never resolved by preference.
4. Any external claim the atlas can execute (commands, versions) is marked
   `replication kit available` and points to `docs/SPARK_KIT.md`.

## 7. Self-checks run before results are accepted

- `python3 scripts/snapshot_market.py` twice → FX identical, BTC within drift;
- `python3 scripts/collect_env.py` → digests of harness + GPU + driver;
- `python3 -m pytest harness/tests -q` → all green, including live-grader
  test (skips offline);
- `python3 scripts/sync_readme.py` → table regeneration hash recorded in
  `results/derived/readme-sync.json` (if the hash changes without new raw
  artifacts, CI treats it as tampering).
