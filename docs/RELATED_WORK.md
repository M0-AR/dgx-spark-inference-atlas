# Related Work and Ecosystem Map (2026)

Every URL below was fetched during the build of this atlas; dates are the
observation dates. Community claims are labelled `[external]` and are never
merged into measured tables (see VERIFICATION §6).

---

## 1. Benchmarking methodology

- **IETF draft-gaikwad-llm-benchmarking-methodology-01** — mandates warm-up
  (≥100 requests or ≥10k output tokens), probe-stability verification (<10 %
  variance) before measurement, drain between conditions, and publishes the
  metric hierarchy. Our runner implements its warm-up and drain clauses
  verbatim and records deviations.
- **Sequential isolation protocol** (arXiv:2610.09778) — pre-measurement
  ISF/ISD phases; reported timing CV drop from 15.2 % to 2.2 % across 5
  repetitions. Our H5 gate (CV ≤ 0.10) is a conservative version of this.
- **Tool-Veritas** (arXiv:2607.02577) — deterministic-first grading of
  agentic tool use across three cells (tool selection, task completion,
  outcome correctness). Our grader structure (C_tool / C_task / C_outcome,
  repair recorded separately) follows it.

## 2. Quantization degradation science (the hypotheses' anchors)

- **"Which Decisions Low-Bit Quantization Breaks"** (arXiv:2608.06564 v5) —
  margin shrinkage: decision margins scale as m′ ≈ c·m + b (~0.86 at 4 bit,
  ~0.33 at 3 bit median); *whether-to-call* decisions collapse before
  *which-tool*; GPTQ/GGUF-style methods flip fewer decisions than RTN
  (median held-out flip rates 1.0 pp). H1/H2 predictions come from here.
- **Signal-vs-computation two-mode failure** (arXiv:2608.08188) — separates
  sub-4-bit failures into signal degradation vs computation collapse; we
  report KLD and C_outcome side by side to expose exactly this split.
- **Divergence-metric disagreement** (arXiv:2609.07664) — TVD/JSD rank
  orders can diverge from each other and from downstream behaviour; motivates
  our "KLD beside C_outcome, never KLD alone" rule.
- **"Calibration is the Bottleneck"** (arXiv:2609.00949) — action-class
  diagnostic (TOOL_CALL / ASK / REFUSE / CONFIRM) and the Acc ≤ GAR bound;
  background for H3 (calibration corpus effects on agentic behaviour).
- **Churn beside net delta** (arXiv:2608.15046) — format churn and
  correctness churn must be reported next to net score changes; our repair
  flags and per-cell breakdowns implement this.
- **Emergent-abilities in quantized LLMs** (LREC 2024,
  aclanthology 2024.lrec-main.461) — 4-bit generally preserves, 2-bit
  severely degrades emergent-style tasks; earliest anchor of our ladder's
  expected shape.
- **McNemar for LLM evaluation** (arXiv:2602.10144) — our primary paired
  test for C_outcome degradation.

## 3. Energy and efficiency measurement

- **Watt Counts** (arXiv:2604.09048) — NVML cumulative-energy protocol,
  idle subtraction, and the warning that joule/token claims are only
  comparable within a fixed protocol. Our `EnergyMeter` follows it and
  documents driver granularity caveats (STATISTICS §7).
- **NVIDIA DGX Spark hardware facts** — 128 GB unified memory, 273 GB/s
  LPDDR5x, 140 W TDP (240 W PSU), ConnectX-7 200 Gb/s, 20-core Arm
  (10 Cortex-X925 + 10 A725), ~1 PFLOP FP4 sparse. Bandwidth, not FLOPs,
  is the decode bottleneck — the axis H4 probes locally.
- **Multi-Spark scaling (Openzeka whitepaper)** — TP1→TP2 = 1.79× (89 %),
  TP2→TP4 = 1.47×; decode all-reduce is latency-bound (one-shot 16.9 µs vs
  NCCL 90.6 µs); link speed affects TTFT, not steady-state decode.

## 4. Serving stacks and EXL3

- **ExLlamaV3 / EXL3** (github.com/turboderp-org/exllamav3) — trellis-coded
  quantization (codebooks ≤ 2^16 entries; 1–8 bpw plus Q8 head); inference
  guidance explicitly recommends KLD-over-full-vocab (not dev-set PPL) and
  warns that calibration and evaluation must not share a corpus. The
  community recently produced EXL3 quantizations **on** a DGX Spark
  (JANGQ-AI), which required small ARM CPU stubs for aarch64/GB10 and
  `expandable_segments` to avoid OOM during conversion `[external]`.
- **TensorFold** (github.com/ashhart/TensorFold; docs/api.md) — unified
  MLX/CUDA runtime; CUDA backend serves EXL3, NVFP4 and MLX-4bit checkpoints
  with Triton/CUDA kernels + PyTorch loaders; families include Qwen3.8-27B,
  Nemotron 3.5 Lightning, GLM-5.3-Flash, Qwen3.6-35B-A3B; `tensorfold serve`
  exposes OpenAI + Anthropic-compatible APIs; MTP/DFlash2 draft support;
  documented known issues: TP=2 uptime degradation, stream interference.
  sm_86 (RTX 3090) support was *not* confirmed during research — treated as
  unverified here.

## 5. DGX Spark community ecosystem and the speed dispute

Observed 2026-10-18 `[external]` — the atlas's SPARK_KIT exists so these are
replicable rather than argued about:

- **MiaAI-Lab** — DeepSeek-v4-Flash on one DGX Spark: EXL3 3.0 bpw via
  `sparkinfer` (formerly b12x), DSpark K5 speculative decoding,
  nvfp4_ds_mla compressed KV, CUDA graphs [6,12,24], 439,622-token KV pool,
  long-prefill fairness work. Also GLM-5.3-Flash-EXL3 on 2 Sparks
  (TensorFold 4 bpw weights, DFlash2 with 7 draft tokens, 850k ctx).
- **vcruz305** — GLM-5.3-Flash-EXL3-K2 on SGLang: 18.57 tok/s single-stream
  decode (target ≥25 not reached); MoE kernel occupancy capped ≈67 %;
  55.5 % of theoretical peak bandwidth; quantified upside ≈21.2 tok/s;
  a K-pool tail bug (silent corruption on long generations) fixed
  2026-08-30, validated at 65,536 context.
- **LLMKube** — DeepSeek-V4.1-Flash EXL3 SAGE 1.59 bpw on a single Spark:
  13.2 tok/s warm decode; ATS copy-vs-alias fix (36 s → 8 s load;
  `EXL3_ATS_COPY`).
- **Reederey87** — production kit: EXL3 + DFlash2, 1M ctx, TP=2. Ships a
  reproducibility warning we adopt as a rule: a drafter repo republished
  three *different* weight files under the same name, so everything here is
  pinned to commits.
- **r0b0tlab** — GLM-5.3-Flash EXL3 2.25 bpw, sm121 vLLM plugin with
  DFlash2, "lever-by-lever" measurement log in
  `profiles/single-gb10-optimize.md`.
- **The dispute** — TensorFold's own materials claim **1.6–3× vLLM decode
  with MTP drafting on both sides** (summarized by third parties; measured by
  Chronara on GB10 and Apple silicon). Against that: `sudoingX/dgx-spark-deepseek-v41`
  claims a patched vLLM decodes faster than TensorFold v0.6 **on the same two
  boxes**, and `tonyd2wild/GLM-5.3-Flash-TensorFold-DGX-Spark` reports vLLM
  drafting still leads on math prompts. These three claims cannot all be
  true; the atlas reports them as a contradiction with a replication kit.
- **Community benchmarking hosts** — sparkbench.dev (powered by
  `shawnmarck/sparkbench` CLI), `Weschera/spark-bench` ("v7.0 Uncapped",
  80 scenarios), `vondraysanford/SparkBench`, LMSYS's DGX Spark review, and
  a single-Spark Qwen3.8-27B report of 65 tok/s (~85 % of bandwidth ceiling,
  ai-muninn.com). Useful priors; none pre-registered or isolation-controlled.

## 6. Artifact-evaluation norms (paper packaging)

- **PACT 2026 Artifact Evaluation** — artifact appendix, 30-minute
  getting-started guide, reproducible docker environment. `papers/` follows
  this layout (see `papers/ARTIFACT_APPENDIX.md`).

## 7. The gap this atlas fills

Prior community work is (a) single-stack hero runs without pre-registration,
(b) contradictory rather than adjudicated, and (c) rarely grounded in live
external data at evaluation time. This atlas contributes: a pre-registered,
isolation-controlled protocol with live-data-grounded agentic grading; a
local executable arm plus a Spark replication kit; and an honesty rule set
(external claims quarantined, TOST for "preserves", deviations logged) —
so the resulting README is a paper-shaped artifact, not a leaderboard.
