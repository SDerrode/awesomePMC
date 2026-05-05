# `prg.pmc` — Pairwise Markov Chains with copula-based transitions

This sub-package implements the unsupervised classification methods
described in two papers by **S. Derrode and W. Pieczynski**. Reading
either paper alone gives a self-contained picture of one half of what
this code does; together they cover everything except the GUI.

> **A16** — *Unsupervised data classification using pairwise Markov chains
> with automatic copulas selection* (CSDA 2013).
> [doi:10.1016/j.csda.2013.01.027](https://doi.org/10.1016/j.csda.2013.01.027)
>
> **A23** — *Unsupervised classification using hidden Markov chain with
> unknown noise copulas and margins* (Signal Process. 2016).
> [doi:10.1016/j.sigpro.2016.03.008](https://doi.org/10.1016/j.sigpro.2016.03.008)

PDFs of both papers are shipped in `docs/CSDA_2013.pdf` and
`docs/SP_2016.pdf` for offline reference.

## What comes from where

| Feature                                               | Source           |
| ----------------------------------------------------- | ---------------- |
| The PMC model family and SR-PMC reversibility         | A16, §2          |
| Variants HMC-IN, HMC-IN2, HMC-DN, PMC-IN, PMC         | A16 (HMC family) + general PMC factorisation |
| Forward / backward / smoothing / MPM classification   | A16, §3          |
| **ICE** — unsupervised parameter estimation           | A16, §4          |
| Per-pair copula family selection (`candidates` field) | A16, §4.3        |
| Selection criteria: MLE, AIC, BIC, **Huard (Eq. 20)**, CvM | A16, Eq. 20 + classic IC |
| **GICE** — automatic margin family selection from a candidate set | A23, §3          |
| Margin selection rules: MLE, **Kolmogorov (Ex. 3.1)**, AIC, BIC | A23, §3, Example 3.1 |
| `betaprime` / `gamma` / `invgamma` / `norm` heuristics in `ice._INIT_PARAM_HEURISTICS` | A23, §5.1 fixture |
| Bundled SP-2016 fixture (`models/sp2016_gice_k2.toml`) | A23, §5.1        |

The Pearson-system shortcut of A23 §3 is **not** implemented — we use a
numerical L-BFGS-B fit for every margin candidate instead. Otherwise the
methodology of both papers is reproduced faithfully; see
[`report/`](../../report/) for an end-to-end reproduction of CSDA 2013
Tables 2-7.

## Citing this package

Please cite **both** papers if you use this code in published work.
BibTeX:

```bibtex
@ARTICLE{A16,
  author    = {S. Derrode and W. Pieczynski},
  title     = {Unsupervised data classification using pairwise {M}arkov chains with automatic copulas selection},
  journal   = {Comput. Stat. Data Anal.},
  volume    = {63},
  pages     = {81--98},
  year      = {2013},
  doi       = {10.1016/j.csda.2013.01.027},
}

@ARTICLE{A23,
  author    = {S. Derrode and W. Pieczynski},
  title     = {Unsupervised classification using hidden {M}arkov chain with unknown noise copulas and margins},
  journal   = {Signal Process.},
  volume    = {128},
  pages     = {8--17},
  year      = {2016},
  doi       = {10.1016/j.sigpro.2016.03.008},
}
```

## Public API

```python
from prg.pmc import (
    PMCModel,        # load / save / validate a model from TOML
    Variant,         # enum: HMC-IN, HMC-IN2, HMC-DN, PMC-IN, PMC
    simulate,        # generate (X, Y) from a PMCModel
    classify,        # supervised MPM classification (A16, §3)
    ice,             # unsupervised ICE estimation (A16, §4 + A23, §3)
    forward, backward, smooth, mpm, error_rate,
)
```

The interactive front-ends are exposed as console scripts:

```bash
pmc-gui                                  # PyQt6 explorer
pmc cli classify --model <toml> ...      # batch CLI (see prg/pmc/cli.py)
```

A representative workflow is documented in the top-level
[`README.md`](../../README.md).

## File-by-file

| File           | Role                                                        |
| -------------- | ----------------------------------------------------------- |
| `model.py`     | TOML schema + `PMCModel` validation. Implements the SR-PMC contract from A16 §2.                                         |
| `simulate.py`  | Forward sampling of `(X, Y)` under any of the 5 variants.   |
| `inference.py` | Forward/backward, smoothing, MPM (A16 §3).                  |
| `pmm.py`       | i.i.d. baseline used by A16 §3.3 / Tables 4-5.              |
| `ice.py`       | ICE (A16 §4) plus GICE margin selection (A23 §3). Selection criteria: MLE, AIC, BIC, Huard (A16 Eq. 20), CvM, Kolmogorov (A23 Ex. 3.1). |
| `models/`      | Shipped fixtures: paper-derived (`sp2016_gice_k2.toml` — A23 §5.1) and exploratory (`pmc_gauss_k2.toml`, `hmc_*.toml`).  |
| `gui/`         | PyQt6 front-end (no paper analogue — pure tooling).          |
| `cli.py`       | Batch CLI for headless workflows.                            |

## Reproduction

[`report/reproduce_csda2013.py`](../../report/reproduce_csda2013.py)
reproduces the three experimental sections of A16:

* §3.2 — PMC supervised, impact of copula shape (Tables 2, 3)
* §3.3 — PMM (i.i.d.) baseline (Tables 4, 5)
* §4.3 — Unsupervised ICE-based copula selection (Tables 6, 7)

The script is parallel by default (`--jobs cpu_count // 2`) and emits
progress bars + ETA. See [`report/README.md`](../../report/README.md).

A23 reproduction (§5.1, Pearson-mix margins) is exercised through the
unit-test suite (`prg/tests/test_gice_margins.py`) on the bundled
fixture; a dedicated `report/reproduce_sp2016.py` is on the roadmap.
