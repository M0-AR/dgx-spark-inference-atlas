"""End-to-end runner tests against a local mock OpenAI-compatible server.

Regression coverage:
  * cv is computed via harness.bench.stats.coefficient_of_variation (the
    runner previously called a non-existent metrics attribute after long runs).
  * a partial final batch must not deadlock the wave barrier
    (requests=3 at concurrency=2 exercises the 2+1 split).
  * non-streaming endpoints: TTFT recorded as total latency with
    ttft_is_total=True and excluded from decode tok/s, plus a note.
  * warm-up bookkeeping (both floors; compliant flag).

All tests are offline: the mock server is a stdlib ThreadingHTTPServer.
"""

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from harness.bench import runner  # noqa: E402


def _sse(obj) -> bytes:
    return f"data: {json.dumps(obj)}\n\n".encode()


class _Handler(BaseHTTPRequestHandler):
    mode = "stream"

    def do_POST(self):  # noqa: N802 (http.server API)
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        if self.mode == "stream":
            for i in range(4):
                self.wfile.write(_sse(
                    {"choices": [{"delta": {"content": f"tok{i} "}}]}))
            self.wfile.write(_sse(
                {"choices": [{"finish_reason": "stop"}],
                 "usage": {"completion_tokens": 4}}))
        else:  # "no_content": finish + usage, but never a content delta
            self.wfile.write(_sse(
                {"choices": [{"finish_reason": "stop"}],
                 "usage": {"completion_tokens": 4}}))
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def log_message(self, *args):  # silence request logging
        pass


def _serve(mode: str) -> ThreadingHTTPServer:
    handler = type("Handler", (_Handler,), {"mode": mode})
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _run_with_deadlock_guard(**kwargs):
    """Run run_condition in a helper thread; fail (not hang) on deadlock."""
    box: dict = {}

    def go():
        box["out"] = runner.run_condition(**kwargs)

    t = threading.Thread(target=go, daemon=True)
    t.start()
    t.join(timeout=60)
    assert not t.is_alive(), "run_condition deadlocked (barrier regression)"
    return box["out"]


def test_run_condition_stream_end_to_end():
    srv = _serve("stream")
    url = f"http://127.0.0.1:{srv.server_address[1]}/v1"
    try:
        results, summary, _ = _run_with_deadlock_guard(
            base_url=url, model="mock-model", tag="t", requests=3, reps=2,
            concurrency=2, max_tokens=8, prompt_words=8,
            warmup_min_requests=2, warmup_min_tokens=2)
    finally:
        srv.shutdown()
        srv.server_close()

    assert len(results) == 6
    assert all(r.error is None for r in results)
    assert all(not r.ttft_is_total for r in results)
    assert all(r.ttft_ms is not None for r in results)
    assert summary.warmup_requests >= 2 and summary.warmup_output_tokens >= 2
    assert summary.warmup_compliant is True
    assert summary.latency["n"] == 6
    assert summary.decode_tok_s["n"] == 6
    assert isinstance(summary.cv_flag, bool)  # computed without AttributeError
    assert summary.errors == 0


def test_run_condition_non_streaming_marks_ttft_total():
    srv = _serve("no_content")
    url = f"http://127.0.0.1:{srv.server_address[1]}/v1"
    try:
        results, summary, _ = _run_with_deadlock_guard(
            base_url=url, model="mock-model", tag="t", requests=2, reps=1,
            concurrency=2, max_tokens=8, prompt_words=8,
            warmup_min_requests=1, warmup_min_tokens=1)
    finally:
        srv.shutdown()
        srv.server_close()

    assert len(results) == 2
    assert all(r.error is None for r in results)
    assert all(r.ttft_is_total for r in results)
    assert all(r.ttft_ms == r.total_ms for r in results)
    assert all(r.decode_tok_s == 0.0 for r in results)
    assert summary.decode_tok_s == {}  # marked rows excluded from decode tok/s
    assert any("no streamed TTFT" in n for n in summary.notes)
    assert summary.warmup_compliant is True
