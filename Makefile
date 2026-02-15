.SHELLFLAGS := -eu -o pipefail -c
PYTHON ?= python3

.PHONY: install dev run run-demo ci

install:
	$(PYTHON) -m pip install -r requirements.txt

dev: install
	$(PYTHON) -m pip install -r requirements-dev.txt

run:
	$(PYTHON) main.py

run-demo:
	DOCUAGENT_DEMO_MODE=1 $(PYTHON) main.py

ci: dev
	$(PYTHON) -m compileall -q main.py
	ruff check --select F .
	pytest -q
