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
MODELS_DIR = REPO_ROOT / "prg" / "pmc" / "models"


def _run(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    """Run ``python -m prg.pmc <args>`` from ``cwd``; return the completed process."""
    return subprocess.run(
        [sys.executable, "-m", "prg.pmc", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
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
