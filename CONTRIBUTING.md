# Contributing to awesomePMC

Thank you for your interest in `awesomePMC` (distribution `awesomepmc` on
PyPI, import package `pmcprg`). This document covers the development setup,
the tests, the numerical-rigour norms the codebase follows, how CI runs, and
how to report an issue.

## Development install

```bash
git clone https://github.com/SDerrode/awesomePMC.git
cd awesomePMC
pip install -e ".[dev,gui]"
```

`[dev]` pulls in pytest (with pytest-xdist and pytest-cov), ruff, the
notebook runners, and the `[image]` and `[ml]` extras so that the whole test
suite can run; `[gui]` adds PyQt6 for the graphical interface and its tests.

## Running the tests

```bash
# Fast suite — excludes the Monte-Carlo / multistart / notebook tests
# marked `slow` (458 of ~7 100); a few minutes with pytest-xdist.
pytest -n auto -m "not slow"

# Full suite, including the slow tests (~15-20 min on 4 workers).
pytest -n auto

# Lint — no autofix, fails on any warning (see the explicit rule
# selection in pyproject.toml's [tool.ruff.lint]).
ruff check pmcprg
```

On a headless machine, set `QT_QPA_PLATFORM=offscreen` so the PyQt6 tests
(`test_gui_*.py`) do not need a display; the same variable is set by CI.
Setting `OMP_NUM_THREADS=1` (and `OPENBLAS_NUM_THREADS` / `MKL_NUM_THREADS`)
avoids oversubscribing the CPU when running with `-n auto`.

## Numerical-rigour norms

This is a numerical library whose correctness is established by comparison
against independent references, not by inspection. Contributions are
expected to follow the same discipline as the existing tests
(`AUDIT.md`, `AUDIT_COPULES.md`, and the docstrings of `pmcprg/tests/`):

- **Every tolerance is justified by a measured error**, never picked to make
  a test pass. A tolerance in a test is a number someone computed — against
  `mpmath`, a textbook closed form, or an independent package — and the
  test's docstring or a comment records what was measured and against what.
- **`mpmath` is the reference of last resort** for high-precision oracles
  (arbitrary-precision Φ, closed-form CDFs/densities evaluated to more digits
  than `float64` carries, root-finding at tight tolerances) when no
  independent package implements a family.
- **The parity reference tables** under `pmcprg/tests/data/parity/` are
  regenerated only with the generators in `scripts/parity/` (see its
  `README.md`), never hand-edited. Each table records its own provenance
  (package versions, platform, generation date, the exact command) and the
  convention checks measured on it; review the diff after regenerating.
- **Deterministic seeds only.** Every Monte-Carlo test seeds its generator
  with an integer literal or `zlib.crc32(repr(...).encode())` of a
  descriptive string — never Python's built-in `hash()`, which is salted
  per process and would make a "fixed" seed silently vary from run to run.
- **Golden and bit-identity tests are never regenerated without a documented
  reason.** If a change moves a golden value or a bit-exact reference, the
  commit says why (a real behaviour change, not a tolerance chase), and the
  reason belongs in the audit trail, not just the diff.
- **Bit-exact comparisons only hold on macOS/arm64.** A handful of tests
  compare floating-point results bit for bit against values written on that
  platform (guarded by `sys.platform == "darwin" and platform.machine() ==
  "arm64"`, see e.g. `test_parity_bb.py`, `test_parity_extra.py`,
  `test_tawn3.py`, `test_fr4_ice_lystig_hughes.py`); on other platforms
  (Linux CI, other BLAS/LAPACK builds) those tests fall back to a measured
  numerical tolerance instead of failing spuriously.

If you are changing a numerical routine (a copula's pdf/cdf/h-function, an
estimator, a test statistic), run the relevant parity/golden tests locally
and re-measure the discrepancy rather than only widening a tolerance until
green.

## Continuous integration

CI is defined in [`.github/workflows/ci.yml`](.github/workflows/ci.yml) and
runs on GitHub only (GitLab, the private working repository, has no
runner). On every push and pull request it runs, on Python 3.11 and 3.14:

- `lint` — `ruff check pmcprg`;
- `smoke-install` — builds the sdist and wheel, `twine check --strict`,
  installs the wheel with no extras and checks the imports and the `pmc`
  console script;
- `test` — the **fast** suite (`-m "not slow"`) with `pytest -n auto`;
- `min-versions` — Python 3.11 with every runtime dependency (and the `ml`
  extra) pinned to the lower bound declared in `pyproject.toml`, fast suite.

The **full** suite (slow tests included) runs on a release tag (`v*`),
weekly (Monday 03:23 UTC), and on a manual `workflow_dispatch` run, on
Python 3.11 through 3.14, with coverage collected on 3.14. A separate `docs`
job builds the API documentation site in strict mode (see below) so a
broken docstring reference or a dead internal link fails CI instead of
reaching a release; it does not deploy anything.

## Reporting an issue

Please use the
[GitHub issue tracker](https://github.com/SDerrode/awesomePMC/issues) of the
public repository. Include the `awesomepmc` version (`pmcprg.__version__`),
your Python/OS/CPU architecture, and, for a numerical discrepancy, the
smallest reproducing example (a TOML model and/or a short script) rather
than a description of the symptom alone — this codebase is validated by
reproducible numerical comparisons, and a bug report is most useful in the
same form.

## Pull requests and the GitHub mirror

The public GitHub repository (`github.com/SDerrode/awesomePMC`) is an
**exported mirror** of a private working repository: each commit on
`main` here is produced by `scripts/publish_github.py`, which strips a
short list of private paths (audit notes) from the working repository's
tree. Pull requests against this repository are welcome; because of the
export process they cannot be merged directly on GitHub — they are applied
on the working repository and reach `main` here through the next export.
Expect the resulting commit to carry the same change under a different SHA,
and feel free to open an issue first for anything larger than a small fix,
so the approach can be agreed on before you invest time in a patch.
