# CSDA 2013 reproduction

Reproduces the experiments of Derrode & Pieczynski (**DerrodePieczynski_CSDA2013** —
[doi:10.1016/j.csda.2013.01.027](https://doi.org/10.1016/j.csda.2013.01.027)),
*"Unsupervised data classification using pairwise Markov chains with
automatic copulas selection"* (Comput. Stat. Data Anal. 63, 2013),
sections 3.2, 3.3, 4.3, using the `awesomePMC` package. The
radar-image segmentation of section 5 is intentionally **not**
reproduced.

The companion paper **DerrodePieczynski_SP2016** (Signal Process. 128, 2016 — GICE for
margin family selection) is exercised through the unit-test suite
(`pmcprg/tests/test_gice_margins.py`) on the bundled fixture
`pmcprg/pmc/models/sp2016_gice_k2.toml`. See
[`pmcprg/pmc/README.md`](../pmcprg/pmc/README.md) for the feature ↔ paper
map and BibTeX entries for both references.

## What the reproduction established

* **The model of the paper is the general PMC** with margins `f_ij` indexed by
  the pair of states (DerrodePieczynski_CSDA2013 Eqs. 12–14). One margin per
  state (`--margins state`, the package's model in versions 0.5–0.8) is, by
  DerrodePieczynski_CSDA2013's §2.1 Proposition, a hidden Markov chain; its
  error rates were 7–19 points above the paper's.
* **Table 1's Gaussian margins are `N(μ, variance)`**: only that reading makes
  the printed Gamma parameters consistent (`--table1-sigma`, default
  `variance` with `--margins pair`).
* **Tables 2–5 are printed with rows = tested copula, columns = true copula**;
  our LaTeX tables follow that orientation (the CSVs keep `sim/est`).
* **Experiment 3 needs several ICE starts** (`--exp3-starts multi`): a single
  start ends in a lower-likelihood basin.
* Open gaps: Table 3(a) (+4.7 points on the diagonal) and the supervised error
  with Gaussian margins, and the PMM Tables 4(b) and 5(b).

## Layout

```
report/
├── README.md                       this file
├── Makefile                        runs and PDF build
├── reproduce_csda2013.py           experiments (CLI)
├── csda2013_tables.py              report tables from the CSVs (no simulation)
├── csda2013_reproduction.tex       LaTeX report
├── data/csda2013_tables2to5.csv    the paper's Tables 2–5, as printed
├── missing_benchmark/              classification/imputation with gaps, known
│                                   models (its own README; outputs in
│                                   results/, tables/ and figures/missing_benchmark/)
├── real_series/                    A4: unsupervised estimation, imputation and
│                                   classification on real series with gaps
│                                   (tsNH4, Beijing PM2.5, PAMAP2; own README)
├── results/                        CSVs: one margin per state (earlier benchmark)
│   ├── margins_pair/               general PMC, single ICE start
│   └── exp3_multistart/margins_pair/  general PMC, multistart ICE
└── tables/                         LaTeX tables (same sub-directories)
    └── synthesis/                  comparison tables of the report
```

## Run

```bash
make              # report tables from the committed CSVs + PDF (no simulation)
make pair         # Experiments 1-3, general PMC (~45 min with 5 processes)
make multistart   # Experiment 3 with multistart ICE (~3.5 h)
make state        # earlier one-margin-per-state benchmark (~1.5 h)

# One experiment, from the repository root (running the script puts report/
# on sys.path, not the cwd, so `pmcprg` needs PYTHONPATH unless installed):
PYTHONPATH=. .venv/bin/python report/reproduce_csda2013.py --exp1 --reps 30 --margins pair
```

The reproductions are embarrassingly parallel: by default the script uses
**half the available CPU cores**; override with `--jobs N` (`--jobs 1` for
sequential, byte-for-byte deterministic log output). Output CSVs are
bit-identical regardless of job count.

Experiment 3 writes `exp3_<config>__<criterion>__runs.csv` — one row per
(run, cell) with the seed and the hit flag (and, with `--exp3-starts multi`,
the winning start). Runs are *paired* across criteria (seed `3000+r`), so these
files support McNemar tests.

## Provenance of the committed results

| Directory | Invocation | Notes |
|---|---|---|
| `results/margins_pair/` | `--exp1 --exp2 --reps 30`, `--exp3 --runs 100`, `--margins pair` (2026-09-14) | The report's main results. |
| `results/exp3_multistart/margins_pair/` | `--exp3 --runs 30 --margins pair --exp3-starts multi` (2026-09-14) | Multistart table of §7 of the report. |
| `results/` (top level) | `--margins state`: Exp. 1–2 `--full` (2026-05-05), Exp. 3 `--runs 100` regenerated 2026-09-14 | Earlier benchmark, §8.3 of the report; `prior_width_isolation*.csv` from `prior_width_isolation.py`. |

The report's discussion quotes only numbers recomputed from these CSVs;
`report/out/csda_compare/` (git-ignored) holds the diagnostic scripts behind
§2.2 and §7.2.

## Caveats

* All experiments are deterministic given the fixed seeds.
* The paper uses 300 replications per cell and 10 ICE runs; we use 30 and
  100 (single start) or 30 (multistart).
* The package's ICE estimates copulas by ξ-weighted pseudo-likelihood; the
  paper (§4.2, L = 1) uses one posterior draw. The margins are known in our
  Experiment 3; the paper re-estimates their parameters.
* The PMM module (`pmcprg.pmc.pmm`) was added in version 0.5.0 for §3.3.
