# Methodology — Pre-Registration

**Status:** PRE-REGISTERED 2026-10-18, before any model experiment in this repo
was executed. All hypotheses, conditions, tolerances and analysis choices below
were fixed before data collection. Deviations are recorded in
`results/README.md` (Deviations log) when they occur, never silently patched.

**Study title (working):** *The DGX Spark Inference Atlas: quantized serving of
agentic workloads — degradation patterns, energy, and reproducibility on GB10
class hardware.*

---

## 1. Research questions

| # | Question | Primary artifact |
|---|----------|------------------|
| RQ1 | Does agentic tool-calling capability degrade *faster* than perplexity/KL-div suggests across a quantization bit-width ladder? | `results/derived/summary-*` (C_outcome vs KLD ranks) |
| RQ2 | Where do failures concentrate: *whether-to-call* / argument correctness / value faithfulness / output format? | grader breakdown fields |
| RQ3 | Does the calibration corpus (wikitext vs agentic transcripts) change agentic outcomes at fixed bpw? | calibration ablation conditions |
| RQ4 | Is decode-phase energy per token monotone in bit width, or is there a non-monotone "sweet spot"? | NVML joule/token series |
| RQ5 | Can the published Spark-class claims (TensorFold vs vLLM disputes) be replicated or falsified with our protocol on *available* hardware, and what is the smallest honest transfer statement to GB10? | README replication matrix + external kit |

## 2. Hypotheses (fixed before data collection)

- **H1 (format–capability gap).** Rank order of conditions by live-data
  tool-calling outcome rate (C_outcome) differs from rank order by held-out
  KLD. Pre-registered prediction: at the lowest bpw, C_outcome falls by more
  than the KLD increase predicts (nonlinear collapse), consistent with
  "margin shrinkage" (arXiv:2608.06564: margins multiplied by ~0.86 at 4 bp,
  0.33 at 3 bp median) and two-mode failure theory (ACL 2026 Findings:
  signal degradation vs computation collapse).
- **H2 (commitment cliff).** Failures at the lowest bpw concentrate in
  *whether-to-call* and *strict-output* decisions rather than in the arithmetic
  cross-rate step. Mechanism anchor: arXiv:2608.06564 finds whether-to-call
  more sensitive than which-tool under aggressive rounding.
- **H3 (calibration domain).** At fixed bpw, EXL3 calibrated on agentic
  transcripts yields higher C_outcome than EXL3 calibrated on wikitext text,
  holding evaluation constant. Anchor: documented calibration-overfit caveat in
  ExLlamaV3's EXL3 guidance (calibration and evaluation must not share corpus).
- **H4 (energy floor).** Decode joules/token is non-monotone in bpw on a
  bandwidth-bound device: it falls from FP16→4 bit, may flatten 4→3, and
  rises again at 2 bit if kernel inefficiency dominates the bandwidth gain.
- **H5 (protocol).** Under the sequential-isolation protocol (5 repetitions,
  drained warm-up, idle probes), per-condition CV of decode throughput on the
  local RTX 3090 is ≤ 0.10; conditions that violate this are flagged and their
  comparisons marked suspect rather than silently averaged.

## 3. Conditions

### 3.1 Local arm (executed in this repository, RTX 3090 24 GB, sm_86)

Model: `Qwen/Qwen2.5-1.5B-Instruct` at a pinned revision (small enough that
the full protocol — including 5 repetitions and warm-up — terminates within a
session; the method, not this model's absolute quality, is under test).

| ID | Condition | Details |
|----|-----------|---------|
| FP16 | unquantized reference | served via the same serving stack |
| E40 | EXL3 4.0 bpw | mul1 codebook, 6-bit head (ExLlamaV3 defaults) |
| E30 | EXL3 3.0 bpw | same settings |
| E20 | EXL3 2.0 bpw | same settings |
| E30-W | EXL3 3.0 bpw, wikitext calibration | tests H3 |
| E30-A | EXL3 3.0 bpw, agentic-transcript calibration | tests H3 |

Concurrency: C=1 and C=4 for FP16, E40, E30 (timing grid); quality grid
(KLD/PPL/tool-calls) at all conditions. R=5 repetitions per cell,
N=8 requests per repetition (interim; the pre-registered minimum for the
quality cells is N≥30 tool-call episodes per condition, accumulated across
repetitions and staggered live snapshots).

If exllamav3 cannot be built on sm_86, the local arm is marked **blocked**,
with logs committed; no substitute numbers are fabricated. A documented
fallback ladder (llama.cpp GGUF Q8/Q4/Q2 on the same task set) may be added
as a secondary arm, clearly labelled as a different quantization family.

### 3.2 Spark arm (external replication kit, not executed here)

Published as `docs/SPARK_KIT.md` + docker profiles: exact commands to repeat
everything on DGX Spark (GB10) including TensorFold and vLLM arms, so the
community can adjudicate the disagreement between
`MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold` (TensorFold faster)
and `sudoingX/dgx-spark-deepseek-v41` (patched vLLM faster on the same two
boxes). Our README reports their claims as *unreplicated external claims*
until any party runs this kit; we do not merge them into measured tables.

## 4. Measurement protocol

1. **Warm-up (mandatory).** Per condition: ≥100 requests or ≥10k output
   tokens of drained pre-runs before the measured window; violation is
   recorded as `WARMUP DEVIATION` in the condition notes (IETF
   draft-gaikwad-llm-benchmarking-methodology-01 requires warm-up and
   probe-stability verification <10%).
2. **Sequential isolation.** One condition at a time; no concurrent
   conditions; a 10s drain + idle-latency probe separates conditions
   (arXiv:2610.09778: isolation cut timing CV from 15.2% to 2.2%).
3. **Fixed-shape prompts.** Deterministic synthetic prompts (published in
   `runner.py`), 64 words, distinct salt per request; generation capped at
   256 tokens, temperature 0, seed recorded per request.
4. **Timing.** Streaming responses; TTFT = first content delta; decode rate
   = output_tokens / (total − TTFT). p50/p90/p99 + CV recorded per condition
   from per-request records (raw JSONL is committed).
5. **Energy.** NVML `nvmlDeviceGetTotalEnergyConsumption` (mJ) read at
   condition boundaries; joules/token derived; idle probe captured before each
   condition (Watt Counts arXiv:2604.09048 protocol; see STATISTICS.md for
   known NVML caveats and our fallbacks).
6. **Live-data integrity.** Every tool-calling episode executes against live
   Kraken/Frankfurter endpoints at episode time; the served values are
   recorded as the episode's truth. Harness snapshots are taken before and
   after each condition (drift documentation). Grading tolerances: ±1.0%
   (BTC fields), ±0.1% (EUR/USD) — see `docs/VERIFICATION.md`.
7. **Grading.** Deterministic-first (Tool-Veritas, arXiv:2607.02577):
   C_tool / C_task / C_outcome cells as implemented in
   `harness/bench/toolcall_eval.py`; repair-needed outputs never count as
   clean outcomes; grader unit tests run in CI and against live data.

## 5. Analysis plan (fixed)

- Primary quality metric: **KLD** (quantized ‖ FP16) on held-out corpora
  (never the calibration corpus), plus PPL as a secondary, always reported
  beside KLD (ExLlamaV3 guidance; "churn beside net delta",
  arXiv:2608.15046).
- Agentic metric: C_outcome rate per condition with bootstrap 95% CI
  (10k resamples, seed 0). Paired comparisons vs FP16 use **McNemar exact**
  on the same episodes (arXiv:2602.10144).
- Ladder shape: on the bpw axis, changepoints in KLD and C_outcome series are
  located with permutation-tested binary segmentation, α=0.01 (see
  `harness/bench/stats.py`). Reported as *pattern locations*, not causal
  claims.
- Multiplicity: H1–H5 family-wise Holm-Bonferroni at α=0.05. Effect sizes:
  Cliff's δ with Romano thresholds.
- Equivalence claims (e.g., "4 bpw preserves C_outcome") require **TOST**
  within a pre-declared margin (±5 percentage points on C_outcome) — absence
  of significance is never reported as equivalence.
- Energy: paired per-repetition joules/token, bootstrap CI; monotonicity
  tested by isotonic-vs-unrestricted comparison (reported descriptively).
- **No optional stopping:** N and R are fixed; if a condition fails
  mid-protocol the failure is reported with logs.

## 6. Statistical power notes

With N=30 paired episodes per condition, McNemar detects an accuracy
degradation of ≈ (z₀.₀₂₅+z₀.₈)·√(2/30) ≈ 0.31 in discordance-share terms at
α=0.05, power 0.8 — i.e., this study can *detect* large agentic collapses,
and for smaller effects we report CIs and say "not resolved at this N".
This limitation is stated in the README; no claim of "no degradation" below
that resolution is made without TOST at the declared margin.

## 7. Deviations and honesty rules

1. Any deviation (warm-up shortfall, missing NVML, failed build, stalled
   endpoint) is recorded in the condition notes and summarized in
   `results/README.md`.
2. No number appears in the README that is not derivable from a committed
   artifact (enforced by `scripts/sync_readme.py`, which fills the
   machine-generated table blocks).
3. External claims (DGX Spark community numbers) are always labelled with
   their source and marked `unreplicated`; they are excluded from our
   measured tables.
4. Negative results (e.g., no detectable difference) are reported with the
   same prominence as positive ones.

---

*Related work and how this study positions against it: `docs/RELATED_WORK.md`.
Statistical implementation details and validation: `docs/STATISTICS.md`.
Verification and tolerance policy: `docs/VERIFICATION.md`.*
