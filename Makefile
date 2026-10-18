# DGX Spark Inference Atlas — task runner
PY ?= python3

.PHONY: setup-gpu test env market smoke readme-sync clean

setup-gpu:            ## install torch + exllamav3 into .venv
	bash scripts/setup_gpu.sh

test:                 ## run harness self-tests (stats + graders)
	$(PY) -m pytest harness/tests -q

env:                  ## capture environment to results/raw/env-*.json
	$(PY) scripts/collect_env.py

market:               ## save a live market snapshot artifact
	$(PY) scripts/snapshot_market.py

smoke:                ## docker-compose CPU smoke test (no GPU required)
	docker compose -f docker/docker-compose.yml run --rm smoke

readme-sync:          ## mechanically refresh README tables from results/
	$(PY) scripts/sync_readme.py

clean:
	rm -rf .pytest_cache harness/**/__pycache__ 2>/dev/null || true
