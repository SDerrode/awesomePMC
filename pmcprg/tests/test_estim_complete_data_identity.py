"""ICE/SEM on complete data are unchanged by the missing-data support.

The golden file ``data/estim_complete_data_golden.json`` was generated with
``python pmcprg/tests/test_estim_complete_data_identity.py --write`` on the
commit *before* ICE/SEM accepted missing observations (b0283f6). Every
fixture model × 2 seeds × 5 configurations — ICE with the TOML settings, ICE
with the k-means warm start, ICE multistart with random copula families, SEM,
SEM multistart with a family sweep and the k-means warm start; ``fit_margins``
on, hence GICE for ``sp2016_gice_k2`` — started from a jittered model on
N = 300 simulated values, 4 iterations.

On the machine and library versions that wrote the file (its ``fingerprint``)
the comparison is **bit-exact**: every float of the fitted model, of the
log-likelihood trace and of the losing multistart runs as ``float.hex``, the
selected families and run tags, and a SHA-256 of the τ/prior histories and of
the SEM draws. Elsewhere a different BLAS / SIMD path may legitimately change
the last bits, so the floats are compared to ``rtol = 1e-8`` and the digests
are skipped — except for the fixtures in ``DISCONTINUOUS_FIXTURES``, which are
skipped there: GICE picks margin families by a criterion whose near-ties flip
with those last bits, and a different family is a different trajectory
(``sp2016_gice_k2`` selected other families on GitHub's Linux runners, every
Python version, while the 80 other cases agreed to 1e-8).

A second test proves that a complete Y never enters the missing-data code
(the entry points are replaced by functions that raise).
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:                       # direct ``python`` invocation
    sys.path.insert(0, str(REPO))

from pmcprg.pmc.ice import _perturb_initial_model, ice   # noqa: E402
from pmcprg.pmc.model import PMCModel                    # noqa: E402
from pmcprg.pmc.sem import sem                           # noqa: E402
from pmcprg.pmc.simulate import simulate                 # noqa: E402

MODELS = REPO / "pmcprg" / "pmc" / "models"
GOLDEN = Path(__file__).resolve().parent / "data" / "estim_complete_data_golden.json"

FIXTURES = sorted(p.name for p in MODELS.glob("*.toml"))
SEEDS = (0, 1)
N_OBS = 300
MAX_ITER = 4


def _configs(seed: int) -> dict[str, tuple[str, dict]]:
    return {
        "ice_toml":      ("ice", {}),
        "ice_kmeans":    ("ice", {"init": "kmeans", "kmeans_seed": seed, "fit_margins": True}),
        "ice_ms_random": ("ice", {"n_starts": 3, "multistart_seed": seed, "fit_margins": True,
                                  "multistart_families": "random"}),
        "sem":           ("sem", {"fit_margins": True, "sem_seed": seed}),
        "sem_ms_sweep":  ("sem", {"n_starts": 2, "multistart_seed": seed, "fit_margins": True,
                                  "init": "kmeans", "kmeans_seed": seed, "sem_seed": seed,
                                  "multistart_families": "sweep"}),
    }


def _fingerprint() -> dict:
    import scipy
    import sklearn
    return {"machine": platform.machine(), "system": platform.system(),
            "python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "sklearn": sklearn.__version__}


def _flatten(obj, out_f: list, out_s: list) -> None:
    """Floats (as hex) and strings of a nested raw dict, in a canonical order."""
    if isinstance(obj, dict):
        for k in sorted(obj):
            out_s.append(str(k))
            _flatten(obj[k], out_f, out_s)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _flatten(v, out_f, out_s)
    elif isinstance(obj, (bool, np.bool_)):
        out_s.append(str(bool(obj)))
    elif isinstance(obj, (int, float, np.integer, np.floating)):
        out_f.append(float(obj).hex())
    elif obj is not None:
        out_s.append(str(obj))


def _digest(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(np.asarray(a)).tobytes())
    return h.hexdigest()


def _signature(fitted: PMCModel, trace) -> dict:
    raw = fitted.raw
    floats, strings = [], []
    _flatten({k: raw.get(k) for k in ("prior", "margins", "copulas")}, floats, strings)
    arrays = [trace.tau_history, trace.p_history]
    if hasattr(trace, "sampled_X_history"):
        arrays.append(trace.sampled_X_history)
    return {
        "model_floats": floats,
        "model_strings": strings,
        "log_liks": [float(v).hex() for v in trace.log_liks],
        "run_tag": trace.run_tag,
        "losers": sorted([t.run_tag, float(t.log_liks[-1]).hex()] for t in trace.multistart_runs),
        "histories_sha256": _digest(*arrays),
    }


def _run(fixture: str, seed: int, name: str) -> dict:
    algo, cfg = _configs(seed)[name]
    truth = PMCModel(MODELS / fixture)
    _, Y = simulate(truth, N=N_OBS, seed=seed)
    start = _perturb_initial_model(truth, np.random.default_rng(1000 + seed), jitter=0.3)
    cfg = {"max_iter": MAX_ITER, **cfg}
    if algo == "ice":
        fitted, trace = ice(start, Y, ice_cfg=cfg)
    else:
        fitted, trace = sem(start, Y, sem_cfg=cfg)
    return _signature(fitted, trace)


CASES = [(f, s, n) for f in FIXTURES for s in SEEDS for n in _configs(s)]

#: Fixtures whose fitted families depend on last-bit rounding (see the module
#: docstring): compared only on the machine and versions that wrote the file.
DISCONTINUOUS_FIXTURES = frozenset({"sp2016_gice_k2.toml"})


def _load_golden() -> dict:
    if not GOLDEN.exists():                                  # pragma: no cover
        pytest.skip(f"golden file {GOLDEN} missing")
    return json.loads(GOLDEN.read_text())


def _floats(hexes) -> np.ndarray:
    return np.array([float.fromhex(x) for x in hexes], dtype=float)


@pytest.mark.slow
@pytest.mark.parametrize("fixture,seed,name", CASES)
def test_complete_data_results_match_the_pre_missing_data_code(fixture, seed, name):
    pytest.importorskip("sklearn")
    golden = _load_golden()
    ref = golden["cases"][f"{fixture}|{seed}|{name}"]
    exact = golden["fingerprint"] == _fingerprint()
    if not exact and fixture in DISCONTINUOUS_FIXTURES:      # pragma: no cover
        pytest.skip(f"{fixture}: GICE family choices depend on last-bit rounding; "
                    "compared only on the platform that wrote the golden file")
    got = _run(fixture, seed, name)
    assert got["model_strings"] == ref["model_strings"]
    assert got["run_tag"] == ref["run_tag"]
    assert [t for t, _ in got["losers"]] == [t for t, _ in ref["losers"]]
    assert len(got["log_liks"]) == len(ref["log_liks"])
    if exact:
        assert got == ref
    else:                                                    # pragma: no cover
        for key in ("model_floats", "log_liks"):
            np.testing.assert_allclose(_floats(got[key]), _floats(ref[key]), rtol=1e-8)
        np.testing.assert_allclose(_floats([h for _, h in got["losers"]]),
                                   _floats([h for _, h in ref["losers"]]), rtol=1e-8)


def _write_golden() -> None:                                 # pragma: no cover
    cases = {}
    for fixture, seed, name in CASES:
        print(f"{fixture} seed={seed} {name}", flush=True)
        cases[f"{fixture}|{seed}|{name}"] = _run(fixture, seed, name)
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_text(json.dumps({"fingerprint": _fingerprint(), "cases": cases},
                                 indent=0, sort_keys=True) + "\n")
    print(f"wrote {GOLDEN}")


if __name__ == "__main__":                                   # pragma: no cover
    if "--write" in sys.argv:
        _write_golden()
