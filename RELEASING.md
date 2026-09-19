# Releasing awesomePMC

The distribution is `awesomepmc` on PyPI; the import package is `pmcprg`.
Two repositories are involved:

| Repository | Role |
|---|---|
| `gitlab.ec-lyon.fr/sderrode/awesomePMC` (remote `origin`, private) | working repository, full history |
| `github.com/SDerrode/awesomePMC` (remote `github`, public) | exported tree, CI, releases, PyPI publication |

Publication runs only on GitHub, in `.github/workflows/publish.yml`, with
[PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/): no API
token is stored anywhere. Pushing a tag publishes nothing; a **published GitHub
Release** uploads to PyPI, and a manual run of the workflow uploads to TestPyPI.
Before any upload, the workflow checks that the tag is exactly `v` +
`pmcprg.__version__` (e.g. `v1.0.0` for `__version__ = "1.0.0"`).

## One-time setup

1. **PyPI and TestPyPI accounts** (separate accounts, each with 2FA).
2. **Trusted publishers.** While the project does not exist yet, add a
   *pending publisher* (it becomes a normal one on the first upload):

   | Field | PyPI — <https://pypi.org/manage/account/publishing/> | TestPyPI — <https://test.pypi.org/manage/account/publishing/> |
   |---|---|---|
   | PyPI project name | `awesomepmc` | `awesomepmc` |
   | Owner | `SDerrode` | `SDerrode` |
   | Repository name | `awesomePMC` | `awesomePMC` |
   | Workflow name | `publish.yml` | `publish.yml` |
   | Environment name | `pypi` | `testpypi` |

   The environment name must match exactly, otherwise the upload is refused.
3. **GitHub environments** — repository *Settings → Environments*: create
   `pypi` and `testpypi`. Recommended for `pypi`: *Required reviewers* = yourself
   (each PyPI upload then waits for an approval) and a deployment tag rule `v*`.
4. The workflow must be on the default branch (`main`) of the GitHub repository
   for *Actions → Publish to PyPI → Run workflow* to appear.

## Release checklist (version `X.Y.Z`)

On the GitLab working copy (`main`):

1. **Version** — set `__version__ = "X.Y.Z"` in `pmcprg/__init__.py` (the only
   place: `pyproject.toml` reads it at build time).
2. **CHANGELOG.md** — add the `X.Y.Z` section with the release date.
3. **CITATION.cff** — set `version: X.Y.Z` and `date-released: "YYYY-MM-DD"`
   (and the date in its header comment).
4. **Full test suite** (slow tests included), GUI extra installed. CI runs
   only the fast subset on a push; the full suite runs on the release tag
   (step 8) and weekly, but a local full run before tagging saves a red tag:

   ```bash
   pip install -e ".[dev,gui]"
   QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 pytest -q -n auto   # ~20 min on 4 workers
   ruff check pmcprg
   ```

5. **Build and check the distributions** locally:

   ```bash
   rm -rf dist && python -m build && twine check --strict dist/*
   tar tzf dist/awesomepmc-X.Y.Z.tar.gz   # only pmcprg/ (no tests), README.md, LICENSE, pyproject.toml, PKG-INFO, .gitignore
   ```

6. Commit (`chore(release): X.Y.Z`), push to `origin`, wait for GitLab CI.

Export and publish:

7. **Export the public tree** (`scripts/publish_github.py` removes the private
   files and never pushes):

   ```bash
   python scripts/publish_github.py            # dry run: review the file list and warnings
   python scripts/publish_github.py --apply    # advances the local branch `public`
   git push github public:main
   ```

   Wait for the GitHub CI (`CI` workflow) to be green on `main`: lint, smoke,
   min-versions and the **fast** suite on Python 3.11 and 3.14 — a few minutes.
8. **Tag the exported commit** (the GitHub tag points to the `public` commit,
   not to the GitLab one):

   ```bash
   git tag -a vX.Y.Z public -m "awesomePMC X.Y.Z"
   git push github vX.Y.Z
   git push origin "$(git rev-parse main):refs/tags/vX.Y.Z"   # same tag name on GitLab, on the source commit
   ```

   Never `git push origin --tags`: the local tag points to the public history.
9. **Draft the GitHub Release** — *Releases → Draft a new release*, tag
   `vX.Y.Z`, title `awesomePMC X.Y.Z`, notes from the CHANGELOG section. Save as
   **draft** (do not publish yet).
10. **TestPyPI dry run** — *Actions → Publish to PyPI → Run workflow*, *Use
    workflow from* = tag `vX.Y.Z`, target `testpypi`. Then, in a fresh venv:

    ```bash
    pip install --index-url https://test.pypi.org/simple/ \
                --extra-index-url https://pypi.org/simple/ awesomepmc==X.Y.Z
    pmc --help
    ```

    A version can be uploaded only once to an index: to repeat a dry run, use a
    pre-release version (`X.Y.Zrc1`, tag `vX.Y.Zrc1`).
11. **Wait for the CI run of the tag** (*Actions → CI*, the run named after
    `vX.Y.Z`): pushing a `v*` tag is what triggers the **full** suite — slow
    tests included, Python 3.11 to 3.14, 4 workers each — and a green run
    there is the release gate. `publish.yml` does not wait for it, so check
    it yourself. (A push to `main` only runs the fast suite.)
12. **Publish the Release** — the `release: published` event runs the workflow:
    guard (tag = version) → build → upload to PyPI (after the `pypi` environment
    approval, if configured).
    If the release event run failed for an external reason, re-run it, or run the
    workflow manually from the tag with target `pypi`.

Post-release checks:

13. In a fresh virtual environment, away from the repository:

    ```bash
    python -m venv /tmp/awesomepmc-check && . /tmp/awesomepmc-check/bin/activate
    pip install awesomepmc==X.Y.Z
    pmc --help
    python -c "import pmcprg, pmcprg.pmc, pmcprg.copulas, pmcprg.missing; print(pmcprg.__version__)"
    ```

14. Check the project page <https://pypi.org/project/awesomepmc/> (README
    rendering, links, classifiers) and the GitHub *Cite this repository* box.
