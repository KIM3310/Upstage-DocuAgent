.SHELLFLAGS := -eu -o pipefail -c
PYTHON ?= python3

.PHONY: install run ci

install:
	$(PYTHON) -m pip install -r requirements.txt

run:
	$(PYTHON) main.py

ci: install
	$(PYTHON) -m compileall .
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py'
