PYTHON ?= python3

.PHONY: install test run run-realdata-lock dashboard export-site demo lint

install:
	$(PYTHON) -m pip install -e '.[dev]'

test:
	$(PYTHON) -m pytest

run:
	$(PYTHON) scripts/run_season_pipeline.py --config configs/mclaren_2025.yaml

run-realdata-lock:
	$(PYTHON) scripts/run_realdata_locked.py --config configs/mclaren_2025.yaml

dashboard:
	$(PYTHON) scripts/run_locked_dashboard.py --port 8765

export-site:
	$(PYTHON) scripts/export_static_site.py --out-dir site

demo:
	$(PYTHON) scripts/run_demo.py

lint:
	ruff check src tests scripts
