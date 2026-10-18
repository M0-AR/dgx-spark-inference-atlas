"""Grader tests: scripted agents (perfect / degraded) against LIVE data.

These tests hit the real Kraken and Frankfurter endpoints (public, keyless,
read-only). If the network is unavailable they skip rather than fail, so CI
offline stays green; the live path is exercised whenever the network exists.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from harness.bench import toolcall_eval as te  # noqa: E402


class ScriptedClient:
    """Agent that follows a fixed plan; mimics the OpenAI chat response shape."""

    model = "scripted"

    def __init__(self, plan: list[dict]):
        self.plan = plan
        self.turn = 0

    def chat(self, messages, tools=None, temperature=0.0, max_tokens=1024,
             seed=None):
        step = self.plan[min(self.turn, len(self.plan) - 1)]
        self.turn += 1
        return {"choices": [{"message": step}]}


def _served_from(messages):
    served = {}
    for m in messages:
        if m.get("role") == "tool":
            served.update(json.loads(m["content"]))
    return served


def _tool_call(call_id, name):
    return {"choices": [{"message": {
        "role": "assistant", "content": None,
        "tool_calls": [{"id": call_id, "function": {
            "name": name, "arguments": "{}"}}]}}]}


class PerfectClient(ScriptedClient):
    """Fetches both tools, then emits exact strict JSON from served values."""

    def __init__(self):
        super().__init__([])

    def chat(self, messages, tools=None, **kw):
        self.turn += 1
        if self.turn == 1:
            return _tool_call("t1", "get_btc_usd")
        if self.turn == 2:
            return _tool_call("t2", "get_eur_usd")
        served = _served_from(messages)
        body = {"btc_usd": served["btc_usd"], "eur_usd": served["eur_usd"],
                "btc_eur": round(served["btc_usd"] / served["eur_usd"], 2)}
        return {"choices": [{"message": {"role": "assistant",
                                         "content": json.dumps(body)}}]}


class FencedClient(PerfectClient):
    """Same values, but wrapped in a code fence: repair needed, outcome no."""

    def chat(self, messages, tools=None, **kw):
        self.turn += 1
        if self.turn == 1:
            return _tool_call("t1", "get_btc_usd")
        if self.turn == 2:
            return _tool_call("t2", "get_eur_usd")
        served = _served_from(messages)
        body = {"btc_usd": served["btc_usd"], "eur_usd": served["eur_usd"],
                "btc_eur": served["btc_usd"] / served["eur_usd"]}
        text = "```json\n" + json.dumps(body) + "\n```"
        return {"choices": [{"message": {"role": "assistant",
                                         "content": text}}]}


def _live_snapshot_or_skip():
    from harness.bench import live_data as ld
    try:
        return ld.snapshot_market()
    except ld.LiveDataError as exc:
        pytest.skip(f"live data unavailable: {exc}")


def test_graders_with_live_data():
    """Single consolidated live test: run a perfect and two degraded plans
    through the real episode runner (live tool execution) and grade them.

    Skips (pytest.skip) when the live endpoints are unreachable, so the
    offline suite reports it as skipped, not passed.
    """
    _live_snapshot_or_skip()

    rec = te.run_episode(PerfectClient())
    g = te.grade_episode(rec)
    assert g["C_tool"] and g["C_task"] and g["C_outcome"], g
    assert g["parse_ok_no_repair"] and not g["needed_repair"]
    assert g["internally_consistent"] is True

    # Degraded agent A: hallucinated tool.
    class HallucinatingClient(ScriptedClient):
        def __init__(self):
            super().__init__([
                {"role": "assistant", "content": None,
                 "tool_calls": [{"id": "t1", "function": {
                     "name": "get_btc_usd_but_imaginary", "arguments": "{}"}}]},
                {"role": "assistant",
                 "content": '{"btc_usd": 1, "eur_usd": 1, "btc_eur": 1}'},
            ])

    g2 = te.grade_episode(te.run_episode(HallucinatingClient()))
    assert not g2["C_tool"] and g2["hallucinated_tool_calls"] == 1
    assert not g2["C_task"]  # values are junk vs live truth

    # Degraded agent B: right values, fenced JSON (repair-needed).
    g3 = te.grade_episode(te.run_episode(FencedClient()))
    assert g3["C_task"] and not g3["C_outcome"] and g3["needed_repair"]
    assert g3["internally_consistent"] is True


if __name__ == "__main__":  # allow direct run without pytest
    test_graders_with_live_data()
    print("live grader tests passed")
