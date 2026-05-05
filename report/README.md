# CSDA 2013 reproduction

Reproduces the experiments of Derrode & Pieczynski (**A16** —
[doi:10.1016/j.csda.2013.01.027](https://doi.org/10.1016/j.csda.2013.01.027)),
*"Unsupervised data classification using pairwise Markov chains with
automatic copulas selection"* (Comput. Stat. Data Anal. 63, 2013),
sections 3.2, 3.3, 4.3, using the `copulasformm` package. The
radar-image segmentation of section 5 is intentionally **not**
reproduced.

The companion paper **A23** (Signal Process. 128, 2016 — GICE for
margin family selection) is exercised through the unit-test suite
(`prg/tests/test_gice_margins.py`) on the bundled fixture
`prg/pmc/models/sp2016_gice_k2.toml`. See
[`prg/pmc/README.md`](../prg/pmc/README.md) for the feature ↔ paper
map and BibTeX entries for both references.

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
# Quick (default — ~5 min on a recent 10-core laptop with parallel default):
#   30 reps for §3.2/3.3, 5 ICE runs for §4.3.
make            # equivalent to: make quick

# Paper-sized (~30-50 min wall time on the same machine, parallel):
#   300 reps and 10 ICE runs.
make full

# Run only one experiment.
python reproduce_csda2013.py --exp1 --quick

# Compile the PDF only (after tables/ is built):
make pdf
```

The reproductions are embarrassingly parallel: by default the script
uses **half the available CPU cores** (so the laptop stays usable).
Override with `--jobs N`; pass `--jobs 1` for sequential, byte-for-byte
deterministic log output. Output CSVs are bit-identical regardless of
job count — work units are dispatched in input order via
`ProcessPoolExecutor.map`.

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
