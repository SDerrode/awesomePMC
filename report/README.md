# CSDA 2013 reproduction

Reproduces the experiments of Derrode & Pieczynski (CSDA, 2013),
*"Unsupervised data classification using pairwise Markov chains with automatic
copulas selection"*, sections 3.2, 3.3, 4.3, using the `copulasformm` package.
The radar-image segmentation of section 5 is intentionally **not** reproduced.

## Layout

```
report/
├── README.md                       this file
├── Makefile                        one-shot build (tables + PDF)
├── reproduce_csda2013.py           main script (CLI)
├── csda2013_reproduction.tex       LaTeX report
├── results/                        (generated) per-table CSVs + log
└── tables/                         (generated) per-table .tex files
```

## Run

```bash
# Quick (default — ~10 min on a recent laptop):
#   30 reps for §3.2/3.3, 5 ICE runs for §4.3.
make            # equivalent to: make quick

# Paper-sized (~3 h):
#   300 reps and 10 ICE runs.
make full

# Run only one experiment.
python reproduce_csda2013.py --exp1 --quick

# Compile the PDF only (after tables/ is built):
make pdf
```

The script writes:

* `results/exp{1,2,3}_*.csv`  — one CSV per table (humanly readable);
* `tables/exp{1,2,3}_*.tex`   — booktabs-compatible LaTeX tables, included
  by the report's main file.

## Caveats

* The Gamma margin parameters in CSDA Table 1 use a `(λ, α, θ)` convention
  whose published values do not match the stated means. We instead generate
  Gamma margins matching the moments of the corresponding Gaussians, with
  fixed shape `α₀ = 8`, `α₁ = 3`. Standard deviations therefore match the
  Gaussian case as the paper claims.
* All experiments are deterministic given the fixed seeds. Running the
  script twice with the same flags produces byte-identical CSV files.
* The pmp module (`prg.pmc.pmm`) was added in version 0.5.0 specifically
  for §3.3.
