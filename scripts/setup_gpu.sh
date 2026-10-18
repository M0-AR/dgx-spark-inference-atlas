#!/usr/bin/env bash
# GPU toolchain setup for the atlas study (LOCAL arm: any NVIDIA GPU).
#
# Installs into <repo>/.venv:
#   * torch (CUDA wheel)          — exact version recorded to results/raw/env-*.json
#   * huggingface_hub             — model downloads (pinned revisions)
#   * exllamav3 (ExLlamaV3)       — EXL3 quantization + inference (pip/compile)
#
# Idempotent: safe to re-run. Logs everything; versions are captured by
# scripts/collect_env.py afterwards. Run:
#   bash scripts/setup_gpu.sh 2>&1 | tee /tmp/atlas-gpu-setup.log
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$REPO/.venv"
# Default: plain PyPI (CUDA-enabled wheels on linux); override with e.g.
# TORCH_INDEX=https://download.pytorch.org/whl/cu126
TORCH_INDEX="${TORCH_INDEX:-}"
PIP_NET=(--retries 10 --timeout 180 --resume-retries 20)
EXLLAMAV3_SPEC="${EXLLAMAV3_SPEC:-exllamav3}"   # e.g. pin: git+https://...@v1.5.0

echo "== repo: $REPO"
echo "== python: $(python3 --version)"

if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
fi
PIP="$VENV/bin/pip"
PY="$VENV/bin/python"

"$PY" -m pip install --quiet --upgrade pip wheel setuptools

echo "== installing torch ${TORCH_INDEX:+(from $TORCH_INDEX)}"
if [ -n "$TORCH_INDEX" ]; then
  "$PIP" install --quiet "${PIP_NET[@]}" torch --index-url "$TORCH_INDEX"
else
  "$PIP" install --quiet "${PIP_NET[@]}" torch
fi

echo "== installing huggingface_hub"
"$PIP" install --quiet "${PIP_NET[@]}" "huggingface_hub>=0.34"

echo "== installing $EXLLAMAV3_SPEC (may compile CUDA extensions; long)"
"$PIP" install --quiet "${PIP_NET[@]}" "$EXLLAMAV3_SPEC"

echo "== done. versions:"
"$PY" - <<'EOF'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
      "cap", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else "")
try:
    import exllamav3
    print("exllamav3", getattr(exllamav3, "__version__", "unknown"))
except Exception as exc:
    print("exllamav3 import failed:", exc)
EOF
