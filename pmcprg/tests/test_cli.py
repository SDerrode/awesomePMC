"""
test_cli.py — smoke tests for the ``pmc`` console script.

These spawn a real subprocess for each command (``simulate``, ``classify``,
``estimate``) so that argparse wiring, the ``[project.scripts]`` entry point,
and the end-to-end CSV/TOML I/O are covered.

Each test runs in a temp directory; no artefact leaks into the repo root.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT  = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "pmcprg" / "pmc" / "models"


def _run(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    """Run ``python -m pmcprg.pmc <args>`` from ``cwd``; return the completed process.

    The subprocess runs in a temp ``cwd`` (so artefacts don't leak into the
    repo) but must still import ``pmcprg`` — which is *not* guaranteed by an
    editable install (a stale/broken ``.pth`` would shadow the source tree).
    We therefore prepend ``REPO_ROOT`` to ``PYTHONPATH`` so ``-m pmcprg.pmc``
    resolves from this checkout regardless of install state, while still
    exercising the real argparse wiring and entry point.
    """
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(REPO_ROOT), env.get("PYTHONPATH", "")) if p
    )
    return subprocess.run(
        [sys.executable, "-m", "pmcprg.pmc", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )


# ---------------------------------------------------------------------------
# --help  /  no-op behaviour
# ---------------------------------------------------------------------------

def test_cli_help(tmp_path: Path):
    """``pmc --help`` exits 0 and lists the four sub-commands."""
    res = _run("--help", cwd=tmp_path)
    assert res.returncode == 0, res.stderr
    for cmd in ("simulate", "classify", "estimate", "gui"):
        assert cmd in res.stdout


def test_cli_no_command_exits_with_error(tmp_path: Path):
    """``pmc`` without a sub-command must fail (sub.required = True)."""
    res = _run(cwd=tmp_path)
    assert res.returncode != 0


# ---------------------------------------------------------------------------
# simulate
# ---------------------------------------------------------------------------

def test_cli_simulate(tmp_path: Path):
    """``pmc simulate`` writes a (n, X, Y) CSV with N rows + header."""
    out = tmp_path / "sim.csv"
    res = _run(
        "simulate",
        "-m", str(MODELS_DIR / "hmc_in_gauss_k2.toml"),
        "--N", "200",
        "--seed", "0",
        "--out", str(out),
        cwd=tmp_path,
    )
    assert res.returncode == 0, res.stderr
    assert out.exists()

    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 200
    assert set(rows[0].keys()) == {"n", "X", "Y"}
    assert int(rows[0]["X"]) in (0, 1)


def test_cli_simulate_reproducible(tmp_path: Path):
    """Same seed → identical CSV byte-for-byte."""
    out_a = tmp_path / "a.csv"
    out_b = tmp_path / "b.csv"
    for out in (out_a, out_b):
        _run(
            "simulate",
            "-m", str(MODELS_DIR / "pmc_in_gauss_k2.toml"),
            "--N", "100",
            "--seed", "42",
            "--out", str(out),
            cwd=tmp_path,
        )
    assert out_a.read_bytes() == out_b.read_bytes()


# ---------------------------------------------------------------------------
# classify
# ---------------------------------------------------------------------------

def test_cli_classify_roundtrip(tmp_path: Path):
    """simulate → classify on the same model gives a low error rate."""
    sim_csv = tmp_path / "sim.csv"
    cls_csv = tmp_path / "cls.csv"

    _run(
        "simulate",
        "-m", str(MODELS_DIR / "pmc_gauss_k2.toml"),
        "--N", "300", "--seed", "0",
        "--out", str(sim_csv),
        cwd=tmp_path,
    )
    res = _run(
        "classify",
        "-m", str(MODELS_DIR / "pmc_gauss_k2.toml"),
        "-d", str(sim_csv),
        "--out", str(cls_csv),
        cwd=tmp_path,
    )
    assert res.returncode == 0, res.stderr
    assert cls_csv.exists()
    assert "Log-lik" in res.stdout
    assert "Error rate" in res.stdout

    rows = list(csv.DictReader(cls_csv.open()))
    assert len(rows) == 300
    # K=2 → two gamma columns
    assert {"n", "X_hat", "gamma_0", "gamma_1"} <= set(rows[0].keys())


def test_cli_classify_missing_data_file(tmp_path: Path):
    """A non-existent --data path produces a clear error and non-zero exit."""
    res = _run(
        "classify",
        "-m", str(MODELS_DIR / "pmc_gauss_k2.toml"),
        "-d", str(tmp_path / "nope.csv"),
        "--out", str(tmp_path / "out.csv"),
        cwd=tmp_path,
    )
    assert res.returncode == 1
    assert "data file not found" in res.stderr


# ---------------------------------------------------------------------------
# estimate
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_cli_estimate(tmp_path: Path):
    """``pmc estimate`` runs ICE for a few iterations and writes a fitted TOML."""
    sim_csv = tmp_path / "sim.csv"
    out_toml = tmp_path / "fit.toml"

    _run(
        "simulate",
        "-m", str(MODELS_DIR / "pmc_gauss_k2.toml"),
        "--N", "500", "--seed", "0",
        "--out", str(sim_csv),
        cwd=tmp_path,
    )
    res = _run(
        "estimate",
        "-m", str(MODELS_DIR / "pmc_gauss_k2.toml"),
        "-d", str(sim_csv),
        "--out", str(out_toml),
        "--max-iter", "3",
        cwd=tmp_path,
    )
    assert res.returncode == 0, res.stderr
    assert out_toml.exists()
    assert "Iterations" in res.stdout
    assert "Final log-lik" in res.stdout


# ---------------------------------------------------------------------------
# General PMC — pair-indexed margins f_ij (A16 Eqs. 12–14)
# ---------------------------------------------------------------------------

PAIR_MODEL = MODELS_DIR / "pmc_pair_gauss_k2.toml"


def test_cli_simulate_and_classify_a_pair_model(tmp_path: Path):
    """simulate → classify on a model with pair margins f_ij works end to end."""
    sim_csv = tmp_path / "sim.csv"
    cls_csv = tmp_path / "cls.csv"

    res = _run("simulate", "-m", str(PAIR_MODEL), "--N", "200", "--seed", "0",
               "--out", str(sim_csv), cwd=tmp_path)
    assert res.returncode == 0, res.stderr
    assert "(PMC, K=2, pair margins f_ij)" in res.stdout
    rows = list(csv.DictReader(sim_csv.open()))
    assert len(rows) == 200 and {int(r["X"]) for r in rows} <= {0, 1}

    res = _run("classify", "-m", str(PAIR_MODEL), "-d", str(sim_csv),
               "--out", str(cls_csv), cwd=tmp_path)
    assert res.returncode == 0, res.stderr
    assert "pair margins f_ij" in res.stdout
    assert "Log-lik" in res.stdout and "Error rate" in res.stdout
    assert len(list(csv.DictReader(cls_csv.open()))) == 200


def test_cli_summary_line_of_a_state_model_is_unchanged(tmp_path: Path):
    res = _run("simulate", "-m", str(MODELS_DIR / "pmc_gauss_k2.toml"),
               "--N", "50", "--seed", "0", "--out", str(tmp_path / "s.csv"),
               cwd=tmp_path)
    assert res.returncode == 0, res.stderr
    assert "(PMC, K=2)\n" in res.stdout
    assert "pair margins" not in res.stdout


def test_cli_estimate_a_pair_model_fails_cleanly_or_fits(tmp_path: Path):
    """``estimate`` on pair margins: a fitted pair model, or a one-line error.

    Estimation on pair margins is wave 2A (ice.py / sem.py). Until it lands
    the estimator raises; the CLI must then name the model structure and exit
    with status 2 — never dump a traceback.
    """
    from pmcprg.pmc.model import PMCModel

    sim_csv = tmp_path / "sim.csv"
    _run("simulate", "-m", str(PAIR_MODEL), "--N", "200", "--seed", "0",
         "--out", str(sim_csv), cwd=tmp_path)

    failed: list[str] = []
    for algorithm in ("ice", "sem"):
        out_toml = tmp_path / f"fit_{algorithm}.toml"
        res = _run("estimate", "-m", str(PAIR_MODEL), "-d", str(sim_csv),
                   "--out", str(out_toml), "--max-iter", "2",
                   "--algorithm", algorithm, cwd=tmp_path)
        assert "Traceback" not in res.stderr, res.stderr
        if res.returncode != 0:
            assert res.returncode == 2, res.stderr
            assert "pair margins f_ij" in res.stderr
            assert f"{algorithm.upper()} failed" in res.stderr
            assert not out_toml.exists()
            failed.append(algorithm)
            continue
        assert PMCModel(out_toml).margin_structure == "pair"
    if failed:
        pytest.skip(f"estimation on pair margins pending wave 2A (ice.py/sem.py) "
                    f"for {failed}; the CLI failure path was checked")


def test_cli_estimate_accepts_missing_values_with_a_warning(tmp_path: Path):
    """``estimate`` on a CSV with missing cells: exit 0, a one-line WARNING with the count, a fitted model.

    ICE and SEM estimate from the observed-data likelihood (missing values
    integrated out), for any model, state or pair; ``classify`` handles the
    same file.
    """
    from pmcprg.pmc.model import PMCModel
    sim_csv = tmp_path / "sim.csv"
    _run("simulate", "-m", str(PAIR_MODEL), "--N", "120", "--seed", "0",
         "--out", str(sim_csv), cwd=tmp_path)
    lines = sim_csv.read_text().splitlines()
    header = lines[0].split(",")
    iy = header.index("Y")
    for k in (5, 6, 40):
        cells = lines[k].split(",")
        cells[iy] = ""
        lines[k] = ",".join(cells)
    gap_csv = tmp_path / "gaps.csv"
    gap_csv.write_text("\n".join(lines) + "\n")
    for model, algorithm in ((PAIR_MODEL, "ice"), (MODELS_DIR / "pmc_gauss_k2.toml", "ice"),
                             (MODELS_DIR / "hmc_in_gauss_k2.toml", "sem")):
        out_toml = tmp_path / f"fit_{model.stem}_{algorithm}.toml"
        res = _run("estimate", "-m", str(model), "-d", str(gap_csv),
                   "--out", str(out_toml), "--max-iter", "2",
                   "--algorithm", algorithm, cwd=tmp_path)
        assert res.returncode == 0, res.stderr
        assert "Traceback" not in res.stderr
        warning = [ln for ln in res.stderr.splitlines() if ln.startswith("WARNING:")]
        assert len(warning) == 1, res.stderr
        assert "3 missing value(s) out of 120" in warning[0]
        assert out_toml.exists()
        fitted = PMCModel(out_toml)
        assert fitted.margin_structure == PMCModel(model).margin_structure
        assert "Final log-lik" in res.stdout
    res = _run("classify", "-m", str(PAIR_MODEL), "-d", str(gap_csv),
               "--out", str(tmp_path / "cls.csv"), cwd=tmp_path)
    assert res.returncode == 0, res.stderr
    assert len((tmp_path / "cls.csv").read_text().splitlines()) == 121


@pytest.mark.parametrize(("argv0", "prog"), [
    ("/usr/local/bin/pmc", "pmc"),
    (str(REPO_ROOT / "pmcprg" / "pmc" / "__main__.py"), "python -m pmcprg.pmc"),
])
def test_usage_names_the_invoked_command(monkeypatch, argv0, prog):
    from pmcprg.pmc.cli import build_parser

    monkeypatch.setattr(sys, "argv", [argv0, "--help"])
    assert build_parser().format_usage().startswith(f"usage: {prog} ")
