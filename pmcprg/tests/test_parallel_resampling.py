"""n_jobs on the six resampling loops (audit FR-14) and the pool behind it.

What is pinned here:

* **The default path is the historical loop, to the bit.** Each function is
  compared with a verbatim copy of its loop as it stood before FR-14.
* **Bit-identity across n_jobs** where it is claimed — ``gof_test``,
  ``bootstrap_ci`` and the three Gaussian-surrogate screens draw every
  replicate's random numbers in the parent, in replicate order, so any
  ``n_jobs`` reproduces ``n_jobs=None``.
* **n_jobs-independence** where the historical stream cannot be split —
  ``parametric_bootstrap`` with an int ``n_jobs`` uses per-replicate
  ``SeedSequence`` streams: identical for 1, 2 and 3 processes, and replicate
  ``b`` is a pure function of ``(seed, b)``.
* **The pool itself** (:mod:`pmcprg._parallel`): results in task order, a
  worker exception re-raised with its replicate named, warnings and log
  records re-emitted in the parent, one BLAS thread per worker, no nested
  pool, ``n_jobs`` validation.

Every pool here uses 2–3 workers and a handful of replicates; the
helper-level ones replicate functions from ``_parallel_workers`` (imports
nothing heavy) and start in a fraction of a second.
"""
from __future__ import annotations

import logging
import os
import warnings

import numpy as np
import pytest

import _parallel_workers as W
from pmcprg import _parallel
from pmcprg.copulas import CopulaClayton, CopulaGaussian
from pmcprg.copulas._fit import _cvm_statistic
from pmcprg.diagnostics import (exchangeability_test, parametric_bootstrap,
                                radial_symmetry_test, rosenblatt_gof_test)
from pmcprg.diagnostics.exchangeability import (
    _fit_gaussian_surrogate as _exch_surrogate,
    _pseudo_obs as _exch_pseudo_obs,
    exchangeability_statistic,
)
from pmcprg.diagnostics.radial_symmetry import (
    _fit_gaussian_surrogate as _radial_surrogate,
    _pseudo_obs as _radial_pseudo_obs,
    radial_symmetry_statistic,
)
from pmcprg.diagnostics.rosenblatt import (
    _pseudo_obs as _rosen_pseudo_obs,
    rosenblatt_statistic,
    rosenblatt_transform,
)

B_GOF, B_CI, B_SCREEN, B_PB = 12, 30, 16, 10


@pytest.fixture(scope="module")
def clayton_fit():
    data = CopulaClayton(tau_k=0.4).sample(120, seed=3)
    return CopulaClayton.fit(data, method="tau")


@pytest.fixture(scope="module")
def pairs():
    return CopulaClayton(tau_k=0.5).sample(80, seed=9).T


@pytest.fixture(scope="module")
def series():
    return np.random.default_rng(1).standard_normal(60)


def _pb(series, n_jobs, statistic=W.stat_mean_draw, seed=6):
    return parametric_bootstrap(None, series, statistic, B=B_PB, seed=seed,
                                simulate_fn=W.fake_simulate, n_jobs=n_jobs)


# ---------------------------------------------------------------------------
# The historical loops, verbatim (before FR-14), as references
# ---------------------------------------------------------------------------

def _historical_gof_stats(r, B, seed):
    rng = np.random.default_rng(seed)
    cls = r.copula.__class__
    stats = np.full(B, np.nan)
    for b in range(B):
        sample = r.copula.sample(n=r.n_obs, seed=int(rng.integers(0, 2**31 - 1)))
        try:
            r_b = cls.fit(sample, method=r.method)
            stats[b] = _cvm_statistic(r_b.uv, r_b.copula)
        except Exception:
            pass
    return stats[~np.isnan(stats)]


def _historical_ci(r, B, alpha, seed):
    rng = np.random.default_rng(seed)
    cls = r.copula.__class__
    tau_boots = np.full(B, np.nan)
    for b in range(B):
        idx = rng.integers(0, r.n_obs, size=r.n_obs)
        try:
            tau_boots[b] = cls.fit(r.uv[idx], method=r.method).tau_k
        except Exception:
            pass
    valid = ~np.isnan(tau_boots)
    return (float(np.quantile(tau_boots[valid], alpha / 2.0)),
            float(np.quantile(tau_boots[valid], 1.0 - alpha / 2.0)))


def _historical_surrogate_draws(x, y, B, seed, surrogate_fn, pseudo_obs, statistic):
    from scipy.stats import kendalltau
    tau_hat, _ = kendalltau(x, y)
    surrogate = surrogate_fn(float(tau_hat) if np.isfinite(tau_hat) else 0.0)
    rng = np.random.default_rng(seed)
    raw = []
    for _ in range(int(B)):
        xb, yb = surrogate.sample(len(x), seed=int(rng.integers(0, 2**31 - 1))).T
        ub, vb = pseudo_obs(xb, yb)
        tb = statistic(ub, vb)
        if np.isfinite(tb):
            raw.append(tb)
    return np.asarray(raw, dtype=float)


def _historical_rosenblatt_draws(x, y, family_cls, B, seed, method="tau"):
    fit = family_cls.fit(np.column_stack([x, y]), method=method)
    rng = np.random.default_rng(seed)
    raw = []
    for _ in range(int(B)):
        try:
            xb, yb = fit.copula.sample(len(x), seed=int(rng.integers(0, 2**31 - 1))).T
            fit_b = family_cls.fit(np.column_stack([xb, yb]), method=method)
            ub, vb = _rosen_pseudo_obs(xb, yb)
            Ub, Vb = rosenblatt_transform(ub, vb, fit_b.copula)
            tb = rosenblatt_statistic(Ub, Vb)
        except Exception:
            continue
        if np.isfinite(tb):
            raw.append(tb)
    return np.asarray(raw, dtype=float)


def _historical_pb(series, B, seed, statistic=W.stat_mean_draw):
    rng = np.random.default_rng(seed)
    observed = dict(statistic(None, series, rng))
    null = {k: [] for k in observed}
    for _ in range(B):
        _, Y_b = W.fake_simulate(None, N=len(series), seed=int(rng.integers(0, 2**31 - 1)))
        for key, value in statistic(None, Y_b, rng).items():
            if key in null and np.isfinite(value):
                null[key].append(float(value))
    return observed, null


def _p(draws, stat):
    draws = np.asarray(draws, dtype=float)
    return float((1 + np.sum(draws >= stat)) / (draws.size + 1))


# ---------------------------------------------------------------------------
# Default path: unchanged, to the bit
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_jobs", [None, 1])
def test_gof_test_in_process_is_the_historical_loop(clayton_fit, n_jobs):
    g = clayton_fit.gof_test(B=B_GOF, seed=4, n_jobs=n_jobs)
    ref = _historical_gof_stats(clayton_fit, B_GOF, 4)
    assert np.array_equal(g.bootstrap_stats, ref)
    assert g.p_value == _p(ref, g.statistic) and g.n_valid_bootstrap == ref.size


@pytest.mark.parametrize("n_jobs", [None, 1])
def test_bootstrap_ci_in_process_is_the_historical_loop(clayton_fit, n_jobs):
    assert (clayton_fit.bootstrap_ci(B=B_CI, seed=5, n_jobs=n_jobs)
            == _historical_ci(clayton_fit, B_CI, 0.05, 5))


@pytest.mark.parametrize("n_jobs", [None, 1])
@pytest.mark.parametrize("test_fn, surrogate_fn, pseudo_obs, statistic", [
    (exchangeability_test, _exch_surrogate, _exch_pseudo_obs, exchangeability_statistic),
    (radial_symmetry_test, _radial_surrogate, _radial_pseudo_obs, radial_symmetry_statistic),
], ids=["exchangeability", "radial_symmetry"])
def test_surrogate_screens_in_process_are_the_historical_loop(
        pairs, n_jobs, test_fn, surrogate_fn, pseudo_obs, statistic):
    x, y = pairs
    res = test_fn(x, y, B=B_SCREEN, seed=2, n_jobs=n_jobs)
    ref = _historical_surrogate_draws(x, y, B_SCREEN, 2, surrogate_fn, pseudo_obs, statistic)
    assert res.n_valid == ref.size and res.p_value == _p(ref, res.statistic)


@pytest.mark.parametrize("n_jobs", [None, 1])
def test_rosenblatt_in_process_is_the_historical_loop(pairs, n_jobs):
    x, y = pairs
    res = rosenblatt_gof_test(x, y, CopulaClayton, B=B_SCREEN, seed=2, n_jobs=n_jobs)
    ref = _historical_rosenblatt_draws(x, y, CopulaClayton, B_SCREEN, 2)
    assert res.n_valid == ref.size and res.p_value == _p(ref, res.statistic)


def test_parametric_bootstrap_default_is_the_shared_stream(series):
    res = _pb(series, None)
    observed, null = _historical_pb(series, B_PB, 6)
    for key, r in res.items():
        assert r.observed == observed[key]
        assert r.n_valid == len(null[key]) and r.p_value == _p(null[key], observed[key])


# ---------------------------------------------------------------------------
# parametric_bootstrap: per-replicate streams
# ---------------------------------------------------------------------------

def test_parametric_bootstrap_streams_are_a_function_of_seed_and_b(series):
    """Replicate b under n_jobs=int is ``default_rng(SeedSequence(seed).spawn(B)[b])``,
    whatever ran before it; the observed pass is the default path's."""
    res = _pb(series, 1)
    default = _pb(series, None)
    children = np.random.SeedSequence(6).spawn(B_PB)
    null = {"mean": [], "draw": []}
    for child in reversed(children):                      # order must not matter
        rng = np.random.default_rng(child)
        _, Y_b = W.fake_simulate(None, N=len(series), seed=int(rng.integers(0, 2**31 - 1)))
        for key, value in W.stat_mean_draw(None, Y_b, rng).items():
            null[key].append(value)
    for key, r in res.items():
        assert r.observed == default[key].observed
        assert r.n_valid == B_PB and r.p_value == _p(null[key], r.observed)
    # a different stream from the default one (same law, other numbers)
    assert res != default


@pytest.mark.parametrize("n_jobs", [2, 3])
def test_parametric_bootstrap_identical_for_every_n_jobs(series, n_jobs):
    assert _pb(series, n_jobs) == _pb(series, 1)


# ---------------------------------------------------------------------------
# Bit-identity across n_jobs with a real pool
# ---------------------------------------------------------------------------

def test_gof_test_bit_identical_with_a_pool(clayton_fit):
    g0 = clayton_fit.gof_test(B=B_GOF, seed=4)
    g2 = clayton_fit.gof_test(B=B_GOF, seed=4, n_jobs=2)
    assert np.array_equal(g2.bootstrap_stats, g0.bootstrap_stats)
    assert (g2.statistic, g2.p_value, g2.n_valid_bootstrap) == \
        (g0.statistic, g0.p_value, g0.n_valid_bootstrap)


def test_bootstrap_ci_bit_identical_with_a_pool(clayton_fit):
    assert (clayton_fit.bootstrap_ci(B=B_CI, seed=5, n_jobs=3)
            == clayton_fit.bootstrap_ci(B=B_CI, seed=5))


@pytest.mark.parametrize("test_fn, extra", [
    (exchangeability_test, ()),
    (radial_symmetry_test, ()),
    (rosenblatt_gof_test, (CopulaGaussian,)),
], ids=["exchangeability", "radial_symmetry", "rosenblatt"])
def test_surrogate_screens_bit_identical_with_a_pool(pairs, test_fn, extra):
    x, y = pairs
    assert (test_fn(x, y, *extra, B=B_SCREEN, seed=2, n_jobs=2)
            == test_fn(x, y, *extra, B=B_SCREEN, seed=2))


# ---------------------------------------------------------------------------
# n_jobs validation
# ---------------------------------------------------------------------------

def test_resolve_n_jobs():
    cpus = _parallel._cpu_count()
    assert _parallel.resolve_n_jobs(None) is None
    assert _parallel.resolve_n_jobs(1) == 1
    assert _parallel.resolve_n_jobs(np.int64(3)) == 3
    assert _parallel.resolve_n_jobs(-1) == cpus >= 1
    for bad in (0, -2, -8):
        with pytest.raises(ValueError, match="n_jobs"):
            _parallel.resolve_n_jobs(bad)
    for bad in (True, 2.0, "2"):
        with pytest.raises(TypeError, match="n_jobs"):
            _parallel.resolve_n_jobs(bad)


@pytest.mark.parametrize("bad, exc", [(0, ValueError), (-3, ValueError), (1.5, TypeError)])
def test_the_six_functions_validate_n_jobs(clayton_fit, pairs, series, bad, exc):
    x, y = pairs
    calls = [
        lambda: clayton_fit.gof_test(B=2, n_jobs=bad),
        lambda: clayton_fit.bootstrap_ci(B=2, n_jobs=bad),
        lambda: exchangeability_test(x, y, B=2, n_jobs=bad),
        lambda: radial_symmetry_test(x, y, B=2, n_jobs=bad),
        lambda: rosenblatt_gof_test(x, y, CopulaGaussian, B=2, n_jobs=bad),
        lambda: _pb(series, bad),
    ]
    for call in calls:
        with pytest.raises(exc, match="n_jobs"):
            call()


# ---------------------------------------------------------------------------
# The pool (pmcprg._parallel)
# ---------------------------------------------------------------------------

def test_results_in_task_order_one_blas_thread_per_worker():
    env_before = {k: os.environ.get(k) for k in _parallel.THREAD_ENV_VARS}
    out = _parallel.map_replicates(W.square, range(20), n_tasks=20, n_jobs=3, shared=10)
    assert [o[0] for o in out] == [10 * b for b in range(20)]
    assert all(o[1] != os.getpid() and o[2] for o in out)        # in a worker
    assert {o[3] for o in out} == {o[4] for o in out} == {"1"}
    # the parent's own environment is left as it was
    assert {k: os.environ.get(k) for k in _parallel.THREAD_ENV_VARS} == env_before


def test_in_process_paths_start_no_worker():
    for n_jobs, n_tasks in ((1, 5), (4, 1)):
        out = _parallel.map_replicates(W.square, range(n_tasks), n_tasks=n_tasks,
                                       n_jobs=n_jobs, shared=2)
        assert [o[0] for o in out] == [2 * b for b in range(n_tasks)]
        assert {o[1] for o in out} == {os.getpid()}


def test_worker_exception_surfaces_with_the_replicate_named():
    with pytest.raises(ValueError, match="replicate 5 is broken") as info:
        _parallel.map_replicates(W.boom, range(12), n_tasks=12, n_jobs=2)
    notes = getattr(info.value, "__notes__", [])
    assert any("replicate 5" in note and "worker process" in note for note in notes)
    assert "Traceback" in str(info.value.__cause__)               # the worker's traceback


def test_the_earliest_failing_replicate_is_raised_whatever_the_timing():
    # replicate 5 (another chunk) fails first in time; the serial loop would
    # stop at replicate 1, and so must the pool
    with pytest.raises(ValueError, match="replicate 1 fails late"):
        _parallel.map_replicates(W.fail_late_early, range(8), n_tasks=8, n_jobs=2,
                                 chunksize=1)


def test_unpicklable_replicate_function_is_refused_before_any_worker():
    with pytest.raises(TypeError, match="top-level function"):
        _parallel.map_replicates(lambda s, t: t, range(4), n_tasks=4, n_jobs=2)
    with pytest.raises(TypeError, match="top-level function"):
        _pb(np.zeros(20), 2, statistic=lambda m, Y, rng: {"m": 0.0})


def test_worker_warnings_are_reemitted_in_the_parent():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default")
        _parallel.map_replicates(W.warn_some, range(8), n_tasks=8, n_jobs=2, chunksize=1)
    got = sorted((w.category.__name__, str(w.message)) for w in caught
                 if w.filename == W.__file__)
    # "default": once per location and text — the twice-raised one shows once,
    # as it would in-process; file, line and category are the worker's.
    assert got == [("RuntimeWarning", "the same warning, twice"),
                   ("UserWarning", "replicate 1 warns"),
                   ("UserWarning", "replicate 2 warns")]


def test_worker_warnings_follow_the_parent_filters():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        _parallel.map_replicates(W.warn_some, range(8), n_tasks=8, n_jobs=2)
    assert sorted(str(w.message) for w in caught if w.filename == W.__file__) == \
        ["replicate 1 warns", "replicate 2 warns"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(UserWarning, match="replicate 1 warns"):
            _parallel.map_replicates(W.warn_some, range(8), n_tasks=8, n_jobs=2)


def test_parametric_bootstrap_warnings_reach_the_caller(series):
    with pytest.warns(UserWarning, match="statistic warning") as rec:
        _pb(series, 2, statistic=W.stat_warns)
    # the observed pass and every replicate: pytest.warns shows them all
    assert sum("statistic warning" in str(w.message) for w in rec) == B_PB + 1


def test_worker_log_records_are_forwarded(caplog):
    with caplog.at_level(logging.WARNING, logger="pmcprg.tests.parallel_workers"):
        _parallel.map_replicates(W.log_some, range(6), n_tasks=6, n_jobs=2)
    msgs = [r.getMessage() for r in caplog.records if r.name == "pmcprg.tests.parallel_workers"]
    assert msgs == ["replicate 2 logs"]                           # DEBUG stays below the level


def test_no_pool_inside_a_worker():
    out = _parallel.map_replicates(W.nested, range(4), n_tasks=4, n_jobs=2)
    for outer_pid, inner_pids in out:
        assert inner_pids == {outer_pid}


def test_multistart_is_serial_inside_a_worker(monkeypatch):
    """``run_multistart`` ignores ``multistart_workers`` in a resampling worker."""
    from pmcprg.pmc import PMCModel
    from pmcprg.pmc import _estim_common as ec

    class _Trace:
        def __init__(self, ll):
            self.log_liks, self.multistart_runs, self.run_tag = [ll], [], ""

    seen = []

    def single_run(init, tag, s):
        seen.append(s)
        return init, _Trace(float(s))

    cfg = dict(ec.shared_estim_defaults(), n_starts=3, multistart_workers=3)
    model = PMCModel("pmcprg/pmc/models/pmc_gauss_k2.toml")
    monkeypatch.setattr(ec, "in_worker", lambda: True)
    # worker=(None, None): the pool path would fail on it
    ec.run_multistart(single_run, model, cfg, label="ICE", worker=(None, None))
    assert seen == [0, 1, 2]
