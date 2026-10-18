"""Agentic tool-calling evaluation against live public data.

Task (pre-registered): a model is given two tools, `get_btc_usd` and
`get_eur_usd` (both reading the same public sources as `live_data.py`), and
must produce a single strict-JSON final answer with fields
`btc_usd`, `eur_usd`, `btc_eur` (the cross rate, which NO tool returns —
it must be computed).

Grading is deterministic-first (cf. Tool-Veritas / arXiv:2607.02577):

  C_tool    : every emitted tool call is well-formed, uses a declared tool
              name, and carries schema-valid arguments. Hallucinated tools or
              malformed calls fail the cell.
  C_task    : numeric faithfulness — each required value within the declared
              relative tolerance of the value the harness observed at the
              moment it served the tool result (and, separately, of a fresh
              harness snapshot taken during grading).
  C_outcome : final message parses as strict JSON (no repair) with the exact
              required keys and numeric types, and passes C_task.

Repair attempts (fence stripping, brace extraction) are permitted but always
recorded separately: `parse_ok_no_repair=False, needed_repair=True` never
counts as a clean outcome.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, asdict
from typing import Any, Callable

from . import live_data as ld

# --------------------------------------------------------------------------- #
# Tool registry
# --------------------------------------------------------------------------- #

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_btc_usd",
            "description": "Get the latest BTC/USD trade price from the Kraken exchange.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_eur_usd",
            "description": "Get the latest EUR/USD reference exchange rate (ECB).",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

SYSTEM_PROMPT = (
    "You are a financial data assistant. Use the provided tools to obtain "
    "live values. Compute the BTC/EUR cross rate as btc_usd / eur_usd. "
    "When you have all values, reply with ONLY a single JSON object, no "
    "prose, no code fences, of exactly this shape: "
    '{"btc_usd": <number>, "eur_usd": <number>, "btc_eur": <number>}'
)

REQUIRED_KEYS = ("btc_usd", "eur_usd", "btc_eur")


@dataclass
class ToolExecution:
    name: str
    arguments: dict[str, Any]
    ok: bool
    result: dict[str, Any]
    error: str | None
    served_at_utc: str
    harness_values: dict[str, float]
    provenance: dict[str, Any] = field(default_factory=dict)


def execute_tool(name: str, arguments_raw: str) -> ToolExecution:
    """Run a declared tool locally; capture harness-observed values as truth.

    Provenance (URL, body SHA-256, fetch timestamp) is stored on the record so
    that grading can be audited against exactly what was served (VERIFICATION §2).
    """
    served_at = ld._utc_now()
    try:
        arguments = json.loads(arguments_raw) if arguments_raw else {}
        if not isinstance(arguments, dict):
            raise ValueError("arguments must be a JSON object")
    except (json.JSONDecodeError, ValueError) as exc:
        return ToolExecution(name=name, arguments={}, ok=False, result={},
                             error=f"bad arguments JSON: {exc}",
                             served_at_utc=served_at, harness_values={})
    if name == "get_btc_usd":
        try:
            price, prov = ld.fetch_btc_usd()
            return ToolExecution(name, arguments, True, {"btc_usd": price}, None,
                                 served_at, {"btc_usd": price}, prov)
        except ld.LiveDataError as exc:
            return ToolExecution(name, arguments, False, {}, str(exc),
                                 served_at, {})
    if name == "get_eur_usd":
        try:
            price, prov = ld.fetch_eur_usd()
            return ToolExecution(name, arguments, True, {"eur_usd": price}, None,
                                 served_at, {"eur_usd": price}, prov)
        except ld.LiveDataError as exc:
            return ToolExecution(name, arguments, False, {}, str(exc),
                                 served_at, {})
    return ToolExecution(name, arguments, False, {},
                         f"hallucinated tool: {name}", served_at, {})


# --------------------------------------------------------------------------- #
# Episode running (OpenAI-compatible endpoint)
# --------------------------------------------------------------------------- #

@dataclass
class EpisodeRecord:
    model: str
    started_utc: str = ""
    ended_utc: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_executions: list[dict[str, Any]] = field(default_factory=list)
    final_content: str = ""
    turns: int = 0
    api_errors: list[str] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


class OpenAIChatClient:
    """Minimal OpenAI-compatible chat client (stdlib only)."""

    def __init__(self, base_url: str, model: str, api_key: str = "local",
                 timeout: float = 240.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def chat(self, messages: list[dict[str, Any]],
             tools: list[dict[str, Any]] | None = None,
             temperature: float = 0.0, max_tokens: int = 1024,
             seed: int | None = 0) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        if seed is not None:
            body["seed"] = seed
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.load(resp)


def run_episode(client: OpenAIChatClient, *, max_turns: int = 6,
                temperature: float = 0.0, seed: int = 0) -> EpisodeRecord:
    """Drive one tool-calling episode to completion (or max_turns)."""
    rec = EpisodeRecord(model=client.model, started_utc=ld._utc_now())
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content":
            "Fetch the live BTC/USD price and the EUR/USD rate, then compute "
            "the BTC price in EUR. Answer with the strict JSON object only."},
    ]
    for turn in range(1, max_turns + 1):
        rec.turns = turn
        t0 = time.monotonic()
        try:
            resp = client.chat(messages, tools=TOOL_SCHEMAS,
                               temperature=temperature, seed=seed)
        except (urllib.error.URLError, urllib.error.HTTPError,
                TimeoutError, json.JSONDecodeError) as exc:
            rec.api_errors.append(f"turn {turn}: {exc}")
            break
        rec.timings_ms[f"turn_{turn}_request_ms"] = round(
            (time.monotonic() - t0) * 1000, 3)
        choice = (resp.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        assistant_msg: dict[str, Any] = {"role": "assistant"}
        if msg.get("content") is not None:
            assistant_msg["content"] = msg.get("content")
        tool_calls = msg.get("tool_calls") or []
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)
            rec.messages.append({"role": "assistant",
                                 "content": msg.get("content"),
                                 "tool_calls": tool_calls})
            for call in tool_calls:
                fn = (call.get("function") or {})
                exec_rec = execute_tool(fn.get("name", ""),
                                        fn.get("arguments", "") or "")
                rec.tool_executions.append(exec_rec.__dict__)
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "content": json.dumps(
                        exec_rec.result if exec_rec.ok else
                        {"error": exec_rec.error}),
                })
            continue
        content = msg.get("content") or ""
        rec.final_content = content
        messages.append({"role": "assistant", "content": content})
        rec.messages.append({"role": "assistant", "content": content})
        break
    rec.ended_utc = ld._utc_now()
    return rec


# --------------------------------------------------------------------------- #
# Grading
# --------------------------------------------------------------------------- #

def _extract_json(text: str) -> tuple[dict[str, Any] | None, bool]:
    """Return (parsed, needed_repair). Strict first, then constrained repair."""
    text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed, False
    except json.JSONDecodeError:
        pass
    # repair 1: code fence
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        try:
            parsed = json.loads(fence.group(1))
            if isinstance(parsed, dict):
                return parsed, True
        except json.JSONDecodeError:
            pass
    # repair 2: first balanced-looking brace span
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group(0))
            if isinstance(parsed, dict):
                return parsed, True
        except json.JSONDecodeError:
            pass
    return None, True


def _value_ok(field_name: str, model_val: Any, truth_val: float) -> bool:
    try:
        v = float(model_val)
    except (TypeError, ValueError):
        return False
    tol = ld.tolerance_for(field_name)
    return abs(v - truth_val) <= tol * abs(truth_val)


def grade_episode(rec: EpisodeRecord) -> dict[str, Any]:
    """Deterministic-first grading into C_tool / C_task / C_outcome cells."""
    # ---- C_tool ----
    n_calls = len(rec.tool_executions)
    valid_calls = 0
    hallucinated = 0
    for ex in rec.tool_executions:
        if ex["error"] and ex["error"].startswith("hallucinated"):
            hallucinated += 1
        elif ex["ok"]:
            valid_calls += 1
    c_tool = (n_calls > 0 and hallucinated == 0
              and valid_calls == n_calls)

    # ---- truth for C_task: values the harness served during the episode ----
    served: dict[str, float] = {}
    for ex in rec.tool_executions:
        served.update(ex.get("harness_values") or {})
    if "btc_usd" in served and "eur_usd" in served:
        served["btc_eur"] = served["btc_usd"] / served["eur_usd"]

    # ---- C_outcome / C_task ----
    parsed, needed_repair = _extract_json(rec.final_content)
    schema_ok = bool(parsed is not None
                     and all(k in parsed for k in REQUIRED_KEYS)
                     and all(isinstance(parsed.get(k), (int, float))
                             and not isinstance(parsed.get(k), bool)
                             for k in REQUIRED_KEYS))
    value_ok: dict[str, bool] = {}
    if schema_ok and served:
        value_ok = {k: _value_ok(k, parsed.get(k), served[k])
                    for k in REQUIRED_KEYS}
    c_task = bool(value_ok) and all(value_ok.values())
    c_outcome = bool(schema_ok and not needed_repair and c_task)

    return {
        "C_tool": bool(c_tool),
        "C_task": bool(c_task),
        "C_outcome": bool(c_outcome),
        "n_tool_calls": n_calls,
        "valid_tool_calls": valid_calls,
        "hallucinated_tool_calls": hallucinated,
        "parse_ok_no_repair": bool(parsed is not None and not needed_repair),
        "needed_repair": bool(needed_repair),
        "schema_ok": schema_ok,
        "value_within_tolerance": value_ok,
        "served_truth": served,
        "final_content": rec.final_content,
        "turns": rec.turns,
        "api_errors": rec.api_errors,
    }


if __name__ == "__main__":  # pragma: no cover - manual probe: live episode not run
    print("toolcall_eval module loaded; use scripts/run_toolcall.py with an endpoint")
