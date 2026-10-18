# The DGX Spark Inference Atlas

**Quantized serving of agentic workloads on GB10-class hardware: degradation
patterns, energy, and reproducibility — with every number in this README
generated from committed run artifacts.**

This repository is a pre-registered measurement study, not a leaderboard. It
asks whether aggressive quantization (EXL3 trellis 2–4 bpw) preserves
*agentic* tool-calling capability, not just perplexity; where failures
concentrate; what it costs in joules; and how the existing DGX Spark serving
claims (TensorFold vs vLLM, EXL3 vs NVFP4) survive replication under a
fixed protocol.

- **Pre-registration:** all hypotheses, conditions, tolerances and analysis
  choices were fixed *before* data collection — [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).
- **Live-data grounding:** every evaluated agent episode calls *real*
  Kraken/Frankfurter endpoints at episode time; the served values are the
  episode's truth — [`docs/VERIFICATION.md`](docs/VERIFICATION.md).
- **Mechanical honesty:** no README number is hand-typed; the marked blocks
  below are written only by [`scripts/sync_readme.py`](scripts/sync_readme.py)
  from `results/` artifacts.
- **Executable anywhere, replicable on Spark:** `docker compose` smoke path
  runs on any machine; [`docs/SPARK_KIT.md`](docs/SPARK_KIT.md) is the full
  GB10 replication protocol for the community speed dispute.

---

## 1. What is verified *in this repository* (status)

| Milestone | Evidence (committed artifact) | Status |
|---|---|---|
| Harness statistics validated (known-value tests) | `harness/tests/test_stats.py` — 12 tests | ✅ passing |
| Live-data graders validated end-to-end | `harness/tests/test_toolcall_eval.py`, mock episode run in `results/raw/toolcall-mock-smoke-*` (excluded from tables by rule) | ✅ passing |
| Live market fetch + tolerance policy exercised | `results/raw/market-20261018T0855*.json` | ✅ pinned |
| Environment capture (host + container) | `results/raw/env-*.json` (two: host run, docker smoke) | ✅ |
| Docker Compose reproducibility floor | `docker compose run --rm smoke` → 13/13 tests in-container | ✅ |
| Runner protocol (warm-up, isolation, energy) | `harness/bench/runner.py`, gates documented | ✅ implemented |
| EXL3 bpw ladder on local GPU (RTX 3090) | see generated tables below | ⏳ in progress |
| Quality arm (KLD vs FP16) | `results/derived/quality-*.json` when present | ⏳ pending |
| Spark arm (GB10) | [`docs/SPARK_KIT.md`](docs/SPARK_KIT.md) | 📦 kit published |

## 2. The strategy under test

The atlas verifies the serving strategy pinned in community discussions
through 2026: EXL3 quants of GLM-5.3-Flash / DeepSeek-v4.1-Flash / Qwen3.8
served with MTP or DFlash2 speculative decoding on one or two DGX Spars, and
the three-way dispute over TensorFold vs patched-vLLM vs SGLang throughput.
The *claims* are catalogued and quarantined as external in
[`docs/RELATED_WORK.md`](docs/RELATED_WORK.md) §5 — including the direct
contradiction (`TensorFold 1.6–3× vLLM` vs `patched vLLM faster than
TensorFold v0.6`) that the Spark Kit exists to adjudicate. **None of those
numbers appear in our tables; only numbers we measured do.**

## 3. Method in one paragraph

Conditions run one at a time (sequential isolation) with drained warm-up per
the IETF benchmarking-methodology draft (≥100 requests or ≥10k output
tokens); streaming TTFT and decode rate are recorded per request; NVML
cumulative energy is read per condition; agentic episodes execute live tools
and are graded deterministic-first into C_tool / C_task / C_outcome cells
(Tool-Veritas structure) with repair attempts flagged, never forgiven.
Paired comparisons vs FP16 use McNemar exact; equivalence requires TOST;
the bpw ladder gets permutation-tested changepoint analysis; the whole family
is Holm-Bonferroni corrected. Full details and limitations:
[`docs/STATISTICS.md`](docs/STATISTICS.md).

## 4. Results — agentic arm (live-data tool calling)

Graded on live Kraken BTC/USD + Frankfurter EUR/USD at episode time;
tolerances ±1.0 % (BTC) / ±0.1 % (EUR). A repaired-but-correct answer scores
C_task but not C_outcome.

<!-- BEGIN:gen:agentic -->
| condition | bpw | episodes | C_tool | C_task | C_outcome (95% CI) | repair-needed | vs FP16 (McNemar exact) |
|---|---|---|---|---|---|---|---|
| _no episodes recorded yet_ | | | | | | | |
<!-- END:gen:agentic -->

Changepoint note (pre-registered pattern-location analysis, α=0.01):

<!-- BEGIN:gen:changepoints -->
_Changepoint analysis requires ≥4 ladder points (fp16 + 3 quantized); not yet available._
<!-- END:gen:changepoints -->

## 5. Results — timing & energy arm

Protocol: fixed-shape prompts (published in `runner.py`), 256-token
generations, temperature 0, 5 repetitions, f16/fp16 reference included.
`CV flag ⚠` marks conditions violating the pre-registered isolation gate
(CV ≤ 0.10), whose comparisons are treated as suspect.

<!-- BEGIN:gen:timing -->
| condition | warm-up | p50 ms | p90 ms | decode tok/s (p50) | CV flag | J/token | errors |
|---|---|---|---|---|---|---|---|
| _no timing runs recorded yet_ | | | | | | | |
<!-- END:gen:timing -->

## 6. Provenance ledger

Every table above is mechanically derived from these artifacts; hashes and
the exact generation manifest live in `results/derived/readme-sync.json`.

<!-- BEGIN:gen:provenance -->
| artifact class | count | latest |
|---|---|---|
| market snapshots | 2 | market-20261018T085533Z.json |
| environment captures | 3 | env-20261018T094229Z.json |
| runner runs (timing) | 0 | — |
| tool-call runs | 0 | — |
<!-- END:gen:provenance -->

## 7. Reproduce

```bash
# 1) harness self-tests (statistics + live-data graders; live parts skip offline)
python3 -m pytest harness/tests -q

# 2) container reproducibility floor (works without a GPU)
docker compose -f docker/docker-compose.yml run --rm smoke

# 3) capture environment + live snapshot artifacts
python3 scripts/collect_env.py && python3 scripts/snapshot_market.py

# 4) serve any OpenAI-compatible endpoint, then:
python3 -m harness.bench.runner --endpoint http://127.0.0.1:8080/v1 \
    --model MODEL --tag e30 --requests 8 --reps 5 --concurrency 1,4
python3 scripts/run_toolcall.py --endpoint http://127.0.0.1:8080/v1 \
    --model MODEL --tag e30 --episodes 30

# 5) regenerate every number in this README from the artifacts
python3 scripts/sync_readme.py
```

GPU toolchain (torch + exllamav3): `bash scripts/setup_gpu.sh`.
DGX Spark (GB10) full protocol: [`docs/SPARK_KIT.md`](docs/SPARK_KIT.md).

## 8. Deviations log

Per METHODOLOGY §7, deviations are recorded, not papered over. Current log:

- **2026-10-18 — local arm runs on RTX 3090 (sm_86), not GB10 (sm_121).**
  Transfer claims are stated as method-level only; Spark numbers are
  external and marked as such. The Spark Kit exists precisely for this gap.
- **2026-10-18 — `torch` wheel download initially failed** (interrupted
  cudnn fetch); setup retried with resumable downloads and recorded.
- **2026-10-18 — mock-endpoint run exists in `results/raw/toolcall-mock-smoke-*`**
  and is excluded from all tables by rule (`model == "mock"`).

## 9. Repository map

```
docs/      METHODOLOGY (pre-registration) · STATISTICS · VERIFICATION · RELATED_WORK · SPARK_KIT
harness/   bench/ (live_data, stats, metrics, toolcall_eval, runner) · tests/
scripts/   collect_env · snapshot_market · run_toolcall · sync_readme · setup_gpu
docker/    Dockerfile · docker-compose.yml (smoke + GPU serving profiles)
results/   raw/ (immutable evidence) · derived/ (tables, macros, manifests)
papers/    LaTeX paper skeleton consuming generated macros (PACT-2026-AE-style artifact appendix)
```

## 10. Limitations (short form)

Small local models (see METHODOLOGY §3.1) bound absolute quality claims; the
ladder shape and method transfer, the absolute numbers do not. N=30 episodes
per condition detects large collapses only (power analysis in METHODOLOGY §6);
`CV flag ⚠` conditions are excluded from ladder fitting. NVML energy is
whole-device and protocol-relative (STATISTICS §7). External claims remain
external until replicated with the Spark Kit.

## Citation

```bibtex
@misc{dgxsparkinferenceatlas2026,
  title  = {The DGX Spark Inference Atlas: quantized serving of agentic
            workloads on GB10-class hardware},
  year   = {2026},
  note   = {Pre-registered measurement study; artifacts under
            results/ in this repository}
}
```
