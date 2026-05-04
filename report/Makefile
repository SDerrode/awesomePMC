# Makefile — reproduce CSDA 2013 experiments + build the LaTeX report.
#
# Usage:
#   make                # quick reproduction + PDF
#   make full           # paper-sized (300 reps, 10 ICE runs) + PDF
#   make tables         # quick reproduction only — no PDF
#   make pdf            # PDF only — assumes tables/*.tex already built
#   make clean          # delete results/ + tables/ + LaTeX intermediates
#   make distclean      # also delete the PDF

PYTHON  ?= ../.venv/bin/python
SCRIPT  := reproduce_csda2013.py
TEXMAIN := csda2013_reproduction
LATEXMK ?= latexmk

.PHONY: all quick full tables pdf clean distclean

all: quick

# --- experiment runs -------------------------------------------------------

quick: tables pdf

full: SCRIPT_FLAGS := --full
full: tables pdf

tables:
	cd .. && $(PYTHON) report/$(SCRIPT) $(SCRIPT_FLAGS)

# --- PDF build -------------------------------------------------------------

pdf: $(TEXMAIN).pdf

$(TEXMAIN).pdf: $(TEXMAIN).tex tables/*.tex
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error $(TEXMAIN).tex

# --- cleanup ---------------------------------------------------------------

clean:
	rm -rf results/*.csv tables/*.tex
	$(LATEXMK) -c $(TEXMAIN) 2>/dev/null || true

distclean: clean
	rm -f $(TEXMAIN).pdf
