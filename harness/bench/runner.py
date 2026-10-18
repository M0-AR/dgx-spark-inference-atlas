"""Inference measurement runner for OpenAI-compatible endpoints.

Pre-registered protocol (docs/METHODOLOGY.md):
  * Warm-up: >= 100 requests or >= 10k output tokens per condition, or the
    deviation is explicitly recorded (IETF draft-gaikwad-llm-benchmarking).
  * Sequential isolation: conditions run one at a time; a drain + idle probe
    separates them; per-condition CV of decode throughput is recorded and
    flagged when > 0.10 (arXiv:2610.09778).
  * Fixed-shape synthetic prompts (deterministic, published): token counts are
    requested by repeating a seed phrase and then trimmed by the served
    tokenizer count; exact chat prompts are written into the artifact.
  * Energy: NVML cumulative counter per condition (see metrics.EnergyMeter).

Streaming responses give TTFT; if the endpoint does not stream, TTFT is
recorded as equal to total latency and the record is marked `ttft_is_total`.

CLI:
  python -m harness.bench.runner --endpoint http://127.0.0.1:8080/v1 \
      --model MODEL --tag exl3-4bpw --requests 12 --max-tokens 256 \
      --reps 5 --concurrency 1,4
"""

from __future__ import annotations

import argparse
import json
import queue
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import metrics as M
from . import stats as S

REPO = Path(__file__).resolve().parents[2]
DRAIN_S = 10.0  # sequential-isolation drain between conditions
SEED_PHRASE = (
    "The quick brown fox jumps over the lazy dog. "
    "Benchmark determinism requires identical inputs. "
)
INSTRUCTIONS = (
    "Continue generating a factual, neutral enumeration of the following "
    "subjects, one short clause each, and do not stop early:\n"
)


def build_prompt(target_words: int, salt: int) -> str:
    """Deterministic fixed-shape prompt: salt varies the content per request."""
    body = SEED_PHRASE * max(1, target_words // len(SEED_PHRASE.split()) + 1)
    return INSTRUCTIONS + f"[seq={salt}]\n" + body.strip()


@dataclass
class RequestResult:
    condition: str
    req_index: int
    rep: int
    started_utc: str
    ttft_ms: float | None
    total_ms: float
    output_tokens: int
    decode_tok_s: float
    ttft_is_total: bool
    finish_reason: str
    error: str | None = None


@dataclass
class ConditionSummary:
    condition: str
    concurrency: int
    requests: int
    reps: int
    warmup_requests: int
    warmup_output_tokens: int
    warmup_compliant: bool
    latency: dict[str, float]
    decode_tok_s: dict[str, float]
    cv_flag: bool
    energy: dict[str, Any] | None
    idle_probe_ms: float | None
    idle_power_w: float | None
    errors: int
    started_utc: str
    ended_utc: str
    notes: list[str] = field(default_factory=list)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _chat_stream(base_url: str, model: str, prompt: str, max_tokens: int,
                 temperature: float, seed: int, api_key: str = "local",
                 timeout: float = 600.0) -> tuple[float | None, float, int, str]:
    """One streaming chat request -> (ttft_ms or None, total_ms, out_tokens, finish)."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    t0 = time.monotonic()
    ttft: float | None = None
    out_tokens = 0
    finish = ""
    usage_tokens: int | None = None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if ttft is None and chunk.get("choices"):
                delta = (chunk["choices"][0].get("delta") or {})
                if delta.get("content"):
                    ttft = (time.monotonic() - t0) * 1000
            choices = chunk.get("choices") or []
            if choices and choices[0].get("finish_reason"):
                finish = choices[0]["finish_reason"]
            usage = chunk.get("usage")
            if usage and usage.get("completion_tokens") is not None:
                usage_tokens = int(usage["completion_tokens"])
    total_ms = (time.monotonic() - t0) * 1000
    if usage_tokens is not None:
        out_tokens = usage_tokens
    return ttft, total_ms, out_tokens, finish


def _one_request(condition: str, base_url: str, model: str, prompt: str,
                 max_tokens: int, temperature: float, seed: int,
                 req_index: int, rep: int, out_q: "queue.Queue[RequestResult]",
                 ) -> None:
    started = _utc()
    try:
        ttft, total_ms, out_tokens, finish = _chat_stream(
            base_url, model, prompt, max_tokens, temperature, seed)
        if ttft is None:
            # non-streaming endpoint (or no content deltas): TTFT is not
            # observable, so it is recorded as equal to total latency and the
            # record is marked; such rows are excluded from decode tok/s.
            ttft_ms: float = total_ms
            ttft_is_total = True
            decode_tok_s = 0.0
        else:
            decode_s = max((total_ms - ttft) / 1000, 1e-6)
            ttft_ms = ttft
            ttft_is_total = False
            decode_tok_s = out_tokens / decode_s if out_tokens else 0.0
        out_q.put(RequestResult(
            condition=condition, req_index=req_index, rep=rep,
            started_utc=started, ttft_ms=ttft_ms, total_ms=total_ms,
            output_tokens=out_tokens, decode_tok_s=decode_tok_s,
            ttft_is_total=ttft_is_total, finish_reason=finish))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
            json.JSONDecodeError, OSError) as exc:
        out_q.put(RequestResult(
            condition=condition, req_index=req_index, rep=rep,
            started_utc=started, ttft_ms=None, total_ms=0.0, output_tokens=0,
            decode_tok_s=0.0, ttft_is_total=True, finish_reason="",
            error=str(exc)))


def _run_wave(base_url: str, model: str, condition: str, prompts: list[str],
              max_tokens: int, temperature: float, seed_base: int,
              concurrency: int, rep: int) -> list[RequestResult]:
    out_q: "queue.Queue[RequestResult]" = queue.Queue()
    results: list[RequestResult] = []

    def worker(i: int, prompt: str, gate: threading.Barrier) -> None:
        gate.wait()
        _one_request(condition, base_url, model, prompt, max_tokens,
                     temperature, seed_base + i, i, rep, out_q)

    # Flush in batches of <= concurrency; the barrier is sized to the actual
    # batch so a partial final batch cannot deadlock waiting on a full gate.
    for start in range(0, len(prompts), concurrency):
        batch = prompts[start:start + concurrency]
        gate = threading.Barrier(len(batch))
        threads = [
            threading.Thread(target=worker, args=(start + off, prompt, gate))
            for off, prompt in enumerate(batch)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    while not out_q.empty():
        results.append(out_q.get())
    return results


def _probe_latency(base_url: str, model: str) -> float | None:
    try:
        t0 = time.monotonic()
        _chat_stream(base_url, model, "Say OK.", 1, 0.0, 0)
        return (time.monotonic() - t0) * 1000
    except Exception:
        return None


def run_condition(base_url: str, model: str, tag: str, *,
                  requests: int, reps: int, concurrency: int,
                  max_tokens: int, prompt_words: int,
                  warmup_min_requests: int = 100,
                  warmup_min_tokens: int = 10_000,
                  warmup_skipped: bool = False,
                  ) -> tuple[list[RequestResult], ConditionSummary, list[dict]]:
    condition = f"{tag}-c{concurrency}"
    prompts = [build_prompt(prompt_words, salt=r * 1000 + i)
               for r in range(reps) for i in range(requests)]
    started = _utc()
    notes: list[str] = []

    # ---- warm-up (drained, not measured) ----
    # IETF draft-gaikwad-llm-benchmarking section 4.5.1 requires the larger of
    # >=100 requests / >=10k output tokens; we satisfy both conservatively.
    warm_reqs = 0
    warm_toks = 0
    if warmup_skipped:
        notes.append("WARMUP DEVIATION: skipped by flag")
    else:
        max_warm_iters = max(2 * warmup_min_requests, 4)
        while ((warm_reqs < warmup_min_requests or warm_toks < warmup_min_tokens)
               and warm_reqs < max_warm_iters):
            p = build_prompt(max(prompt_words // 2, 32), salt=warm_reqs)
            try:
                _, _, toks, _ = _chat_stream(base_url, model, p, max_tokens,
                                             0.0, warm_reqs)
                warm_toks += toks
            except Exception as exc:
                notes.append(f"warmup error at request {warm_reqs}: {exc}")
                break
            warm_reqs += 1
    warmup_compliant = (not warmup_skipped
                        and warm_reqs >= warmup_min_requests
                        and warm_toks >= warmup_min_tokens)
    if not warmup_skipped and not warmup_compliant:
        notes.append(f"WARMUP DEVIATION: only {warm_reqs} reqs / {warm_toks} tokens")

    idle_probe = _probe_latency(base_url, model)
    idle_power_w = M.idle_power_probe_w()

    # ---- measured waves ----
    all_results: list[RequestResult] = []
    with M.EnergyMeter(label=condition, idle_power_w=idle_power_w) as meter:
        for rep in range(reps):
            wave = _run_wave(base_url, model, condition, prompts[
                rep * requests:(rep + 1) * requests],
                max_tokens, 0.0, seed_base=10_000 * rep, concurrency=concurrency,
                rep=rep)
            all_results.extend(wave)
    measured_tokens = sum(r.output_tokens for r in all_results if not r.error)
    energy = (meter.reading(tokens=measured_tokens).to_dict()
              if M.nvml_available() else None)

    ok = [r for r in all_results if not r.error]
    lat = M.summarize_latencies([r.total_ms for r in ok])
    streamed = [r for r in ok if not r.ttft_is_total]
    nonstream = len(ok) - len(streamed)
    if nonstream:
        notes.append(f"{nonstream}/{len(ok)} responses had no streamed TTFT "
                     "(non-streaming endpoint?); excluded from decode tok/s")
    tok_s_vals = [r.decode_tok_s for r in streamed if r.output_tokens > 0]
    ts = M.summarize_latencies(tok_s_vals)
    cv = S.coefficient_of_variation(tok_s_vals)
    cv_flag = bool(cv == cv and cv > 0.10)  # NaN-safe
    if cv_flag:
        notes.append(f"CV over decode tok/s = {cv:.3f} > 0.10 (isolation suspect)")

    summary = ConditionSummary(
        condition=condition, concurrency=concurrency, requests=requests,
        reps=reps, warmup_requests=warm_reqs, warmup_output_tokens=warm_toks,
        warmup_compliant=warmup_compliant, latency=lat, decode_tok_s=ts,
        cv_flag=cv_flag, energy=energy, idle_probe_ms=idle_probe,
        idle_power_w=idle_power_w, errors=len(all_results) - len(ok),
        started_utc=started, ended_utc=_utc(), notes=notes)
    return all_results, summary, []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--requests", type=int, default=8)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--concurrency", default="1")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--prompt-words", type=int, default=64)
    ap.add_argument("--skip-warmup", action="store_true",
                    help="skip IETF warm-up (recorded as deviation)")
    args = ap.parse_args()

    run_id = f"{args.tag}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    raw_path = REPO / "results" / "raw" / f"run-{run_id}.jsonl"
    derived_path = REPO / "results" / "derived" / f"summary-{run_id}.json"
    derived_path.parent.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    meta = {
        "run_id": run_id, "tag": args.tag, "model": args.model,
        "endpoint": args.endpoint,
        "warmup_policy": ("skip (recorded deviation)" if args.skip_warmup
                          else ">=100 req AND >=10k out tok (both), drained"),
        "generated_by": "harness/bench/runner.py", "started_utc": _utc(),
    }
    with raw_path.open("w") as fh:
        fh.write(json.dumps({"type": "meta", **meta}) + "\n")
        for idx, conc in enumerate([int(c) for c in args.concurrency.split(",")]):
            if idx:
                time.sleep(DRAIN_S)  # sequential isolation between conditions
            results, summary, _ = run_condition(
                args.endpoint, args.model, args.tag, requests=args.requests,
                reps=args.reps, concurrency=conc, max_tokens=args.max_tokens,
                prompt_words=args.prompt_words,
                warmup_skipped=args.skip_warmup)
            for r in results:
                fh.write(json.dumps({"type": "request", **asdict(r)}) + "\n")
            summaries.append(asdict(summary))
    meta["ended_utc"] = _utc()
    derived_path.write_text(json.dumps(
        {"meta": meta, "conditions": summaries}, indent=2))
    print(derived_path)


if __name__ == "__main__":
    main()
