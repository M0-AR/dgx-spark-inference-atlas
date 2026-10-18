# CPU-image for the atlas harness (tests, env capture, grading, snapshots).
# GPU serving images are referenced by digest via compose profiles instead of
# being built here (see docker-compose.yml and docs/SPARK_KIT.md).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /atlas

# Harness deps only (torch/exllamav3 are GPU-arm concerns and are NOT needed
# for stats, grading, env capture or the smoke test).
RUN pip install --quiet "numpy>=2.0" "pytest>=8"

COPY . /atlas

# Default: run the harness self-tests, then capture the container environment
# as a first-class artifact. `make smoke` maps this to a one-shot run.
CMD ["sh", "-lc", "python -m pytest harness/tests -q && python scripts/collect_env.py"]
