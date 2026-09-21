# Installation

## The package

```bash
pip install awesomepmc
```

Optional extras:

| Extra | What it adds |
|---|---|
| `[gui]` | PyQt6, for the graphical interface (`pmc gui`) |
| `[ml]` | scikit-learn, for the k-means warm start of ICE/SEM (`init = "kmeans"`) |
| `[image]` | Pillow, to read and write images |
| `[dev]` | the test suite's own dependencies (pytest, ruff, …) |
| `[docs]` | this documentation site's dependencies (MkDocs, Material, mkdocstrings) |

See the [README](https://github.com/SDerrode/awesomePMC/blob/main/README.md#installation)
for the full requirements table and the development install.

## This site

The API reference is built locally with [MkDocs](https://www.mkdocs.org/)
and [mkdocstrings](https://mkdocstrings.github.io/); it is **not deployed**
anywhere (no GitHub Pages, no public URL) — build it yourself from a clone:

```bash
git clone https://github.com/SDerrode/awesomePMC.git
cd awesomePMC
pip install -e ".[docs]"

mkdocs serve            # live-reloading local server, http://127.0.0.1:8000
mkdocs build --strict   # static site under site/; fails on any broken reference
```

`mkdocstrings` reads the docstrings straight from `pmcprg/` (NumPy style),
so the site always matches the checked-out source — there is nothing to
regenerate by hand, and no separate build step is required before a commit.
