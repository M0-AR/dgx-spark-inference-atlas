# Spark Kit — external replication protocol (DGX Spark / GB10)

This atlas cannot execute the GB10 arm locally (its local arm runs on an
RTX 3090 with sm_86; the Spark is sm_121). To keep the study honest *and*
useful, everything needed to replicate or falsify the community speed claims
on real DGX Spark hardware (one or two units) is specified here. Results
produced with this kit can be dropped into `results/raw/` unchanged — the
schemas match.

**Ground rules**

1. **Pin everything.** Model revisions, drafter weights, container digests,
   framework commits. There is a documented case of a drafter repo
   republishing three different weight files under the same name
   (Reederey87) — treat every drafter as guilty until pinned.
2. **Warm-up: ≥100 requests AND ≥10k output tokens** before any measured
   window — both floors (the IETF methodology draft's "100 requests or
   10,000 output tokens, whichever is greater" policy is satisfied
   conservatively; the runner enforces it and logs deviations).
3. **One condition at a time**, 10 s drain + idle probe between conditions.
4. Report TTFT and decode separately; never average them into one number.
5. Publish raw JSONL + env capture; do not publish derived numbers without
   the raw records.

## 0. Environment capture (run first, on each Spark)

```bash
python3 scripts/collect_env.py     # -> results/raw/env-<utc>.json
nvidia-smi -q > results/raw/nvidia-smi-$(date -u +%Y%m%dT%H%M%SZ).txt
```

The env capture records the unified-memory size, driver, kernel, and the
harness digests, so two Sparkers' results are comparable at the artifact
level.

## 1. TensorFold arm (the claimed 1.6–3×)

```bash
# 1) install per TensorFold docs (pin the release, do not track master)
tensorfold --version
python -m tensorfold.cuda.exl3.inspect MODEL_DIR   # verify checkpoint format

# 2) serve a GLM-5.3-Flash EXL3 checkpoint (example family from MiaAI-Lab kit)
tensorfold serve --model GLM-5.3-Flash-EXL3-4bpw-TensorFold \
    --draft-mode dflash2 --draft-tokens 7 --host 127.0.0.1 --port 8080

# 3) measure with THIS runner (same protocol as the local arm)
python3 -m harness.bench.runner --endpoint http://127.0.0.1:8080/v1 \
    --model GLM-5.3-Flash-EXL3-4bpw --tag tf-dflash2 \
    --requests 8 --reps 5 --concurrency 1,4 --max-tokens 256
```

## 2. vLLM arm (the disputed side)

Use the `sudoingX/dgx-spark-deepseek-v41` patched vLLM **at a pinned
commit**, and the same model + same prompt protocol:

```bash
python3 -m vllm.entrypoints.openai.api_server --model MODEL_DIR \
    --port 8080 --max-model-len 32768          # plus that repo's patches
python3 -m harness.bench.runner --endpoint http://127.0.0.1:8080/v1 \
    --model MODEL_DIR --tag vllm-patched \
    --requests 8 --reps 5 --concurrency 1,4 --max-tokens 256
```

**The comparison only counts with MTP drafting enabled on BOTH sides**
(TensorFold's claim is phrased "when both run MTP drafts"), identical
max-tokens, identical prompt set (the runner's synthetic set is published in
`runner.py`), and one condition per box at a time.

## 3. EXL3 / SGLang arm (the vcruz305 line of work)

```bash
# GLM-5.3-Flash-EXL3-K2 on SGLang; pin commit; apply the K-pool tail fix
# (post-2026-08-30) — older builds silently corrupt long generations.
python -m sglang.launch_server --model MODEL_DIR --port 8080
python3 -m harness.bench.runner --endpoint http://127.0.0.1:8080/v1 \
    --model GLM-5.3-Flash-EXL3-K2 --tag sglang-k2 \
    --requests 8 --reps 5 --concurrency 1,4 --max-tokens 256
```

## 4. Quality (EXL3 arm) — KLD against the unquantized reference

On the Spark (or any box with enough memory), compute per the ExLlamaV3
guidance: KLD over the full vocabulary on an **evaluation** corpus that is
**not** the calibration corpus; report KLD as primary and PPL as secondary.

## 5. Agentic live-data arm on Spark

```bash
python3 scripts/snapshot_market.py     # before
python3 scripts/run_toolcall.py --endpoint http://127.0.0.1:8080/v1 \
    --model MODEL --tag <tag> --episodes 30   # after `make test` passes
python3 scripts/snapshot_market.py     # after (drift audit)
```

Grading happens offline after the fact from the recorded episode truth
(VERIFICATION §2) — no network needed at grade time.

## 6. What to publish back

- `results/raw/toolcall-*.jsonl` (episodes+grades), `results/raw/run-*.jsonl`
  (timing), `results/raw/market-*.json`, `env-*.json`
- the exact container digests / git SHAs used
- a sentence per condition: warm-up compliance, CV flag, errors

Drop them in a PR; `scripts/sync_readme.py` regenerates the tables and the
contradiction ledger is updated only where the data actually resolves it.
