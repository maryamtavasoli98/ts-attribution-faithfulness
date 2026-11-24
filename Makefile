# Reproduce everything: make venv data train attr figures readme
PY ?= .venv/bin/python

.PHONY: venv data train attr figures readme all test lint

venv:
	python3 -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install -e ".[dev]"

data:            ## download ETTh1.csv (2.6 MB) and write dataset.json
	$(PY) scripts/prepare_data.py

train:           ## 3 forecasters x 3 seeds (about 3 minutes on an M3 Pro)
	$(PY) scripts/train.py

attr:            ## 6 attributions + deletion/insertion + sanity + stability (about 15 minutes)
	$(PY) scripts/run_attr.py

figures:
	$(PY) scripts/make_figures.py

readme:          ## fill the README tables from results/metrics
	$(PY) scripts/update_readme.py

all: data train attr figures readme

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check . && $(PY) -m ruff format --check .
