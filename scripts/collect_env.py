#!/usr/bin/env python3
"""Capture the environment of the machine a run was performed on.

Writes results/raw/env-<utc>.json with: host facts, CPU/RAM, GPU inventory
(via nvidia-smi if present), toolchain versions, and SHA256 digests of the
harness source tree so results can be tied to exact code.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def sh(cmd: list[str], timeout: float = 20.0) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (out.stdout or out.stderr).strip()
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        return f"<unavailable: {exc}>"


def _memory_info() -> str:
    """RAM totals via free(1), falling back to /proc/meminfo when free is
    missing (slim containers, e.g. the docker smoke image)."""
    out = sh(["free", "-b"])
    if not out.startswith("<unavailable"):
        return out
    try:
        info: dict[str, str] = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            info[key.strip()] = rest.strip().split()[0] if rest.strip() else "?"
        return (f"<free unavailable; /proc/meminfo: "
                f"MemTotal={info.get('MemTotal', '?')}kB "
                f"MemFree={info.get('MemFree', '?')}kB "
                f"MemAvailable={info.get('MemAvailable', '?')}kB>")
    except OSError as exc:
        return f"<unavailable: {out}; /proc/meminfo: {exc}>"


def digest_tree(root: Path) -> dict[str, str]:
    digests: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            rel = str(path.relative_to(REPO))
            digests[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digests


def main() -> None:
    env = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "hostname": platform.node(),
            "python": sys.version,
        },
        "cpu": {
            "model": next((l.split(":", 1)[1].strip()
                           for l in sh(["cat", "/proc/cpuinfo"]).splitlines()
                           if l.startswith("model name")), "unknown"),
            "logical_cores": sh(["nproc"]),
        },
        "memory": _memory_info(),
        "gpu": sh(["nvidia-smi",
                   "--query-gpu=index,name,driver_version,memory.total,"
                   "compute_cap,power.limit,power.max_limit",
                   "--format=csv"]),
        "cuda_runtime": sh(["nvcc", "--version"]),
        "docker": {"version": sh(["docker", "--version"]),
                   "compose": sh(["docker", "compose", "version"])},
        "disk": sh(["df", "-h", str(REPO)]),
        "uv_or_pip": {"pip": sh([sys.executable, "-m", "pip", "--version"])},
        "harness_digests_sha256": digest_tree(REPO / "harness"),
    }
    out_dir = REPO / "results" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = out_dir / f"env-{stamp}.json"
    out.write_text(json.dumps(env, indent=2))
    print(out)


if __name__ == "__main__":
    main()
