PYTHON ?= python

.PHONY: test a0 paper-scale figures release-audit

test:
	PYTHONPATH=src $(PYTHON) experiments/a0/self_test_frozen.py
	cd experiments/paper_scale && $(PYTHON) scripts/self_test.py

a0:
	mkdir -p experiments/a0/run
	PYTHONPATH=src $(PYTHON) experiments/a0/run.py --run-dir experiments/a0/run

paper-scale:
	cd experiments/paper_scale && $(PYTHON) run_all.py

figures:
	$(PYTHON) figures/generate_quantitative_figures.py

release-audit:
	$(PYTHON) tools/release_audit.py
