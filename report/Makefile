# Makefile — reproduce CSDA 2013 experiments + build the LaTeX report.
#
# Usage:
#   make                # tables from the committed CSVs + PDF (no simulation)
#   make pair           # Experiments 1-3, general PMC (pair margins): ~45 min
#   make multistart     # Experiment 3 with multistart ICE: ~3.5 h
#   make state          # earlier one-margin-per-state benchmark: ~1.5 h
#   make pdf            # same as `make`
#   make clean          # delete LaTeX intermediates (results are kept)
#   make distclean      # also delete the PDF
#
# Recipes cd to the repository root first, so PYTHON is relative to *that*,
# not to report/. PYTHONPATH=. makes ``pmcprg`` importable without an editable
# install — running ``python report/script.py`` puts report/ on sys.path, not
# the working directory (same reason the test suite anchors sys.path in
# pmcprg/tests/conftest.py).
PYTHON  ?= .venv/bin/python
SCRIPT  := reproduce_csda2013.py
TABLES  := csda2013_tables.py
TEXMAIN := csda2013_reproduction
LATEXMK ?= latexmk
JOBS    ?= 5

.PHONY: all pair multistart state pdf clean distclean

all: pdf

# --- experiment runs (write results/ and tables/) --------------------------

pair:
	cd .. && PYTHONPATH=. $(PYTHON) report/$(SCRIPT) --exp1 --exp2 --reps 30 --margins pair --jobs $(JOBS)
	cd .. && PYTHONPATH=. $(PYTHON) report/$(SCRIPT) --exp3 --runs 100 --margins pair --jobs $(JOBS)

multistart:
	cd .. && PYTHONPATH=. $(PYTHON) report/$(SCRIPT) --exp3 --runs 30 --margins pair --exp3-starts multi --jobs $(JOBS)

state:
	cd .. && PYTHONPATH=. $(PYTHON) report/$(SCRIPT) --exp1 --exp2 --reps 300 --margins state --jobs $(JOBS)
	cd .. && PYTHONPATH=. $(PYTHON) report/$(SCRIPT) --exp3 --runs 100 --margins state --jobs $(JOBS)

# --- report tables + PDF ---------------------------------------------------

pdf:
	cd .. && PYTHONPATH=. $(PYTHON) report/$(TABLES)
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error $(TEXMAIN).tex

# --- cleanup ---------------------------------------------------------------

clean:
	$(LATEXMK) -c $(TEXMAIN) 2>/dev/null || true

distclean: clean
	rm -f $(TEXMAIN).pdf
