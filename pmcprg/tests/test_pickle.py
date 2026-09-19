"""Pickling of copulas, their results and PMC models.

Everything here already round-tripped *within one version* before this file
existed — what it pins down is what made pickles a poor storage format:

* **Weight.** Every ``CopulaVirt`` carried three 150 × 150 float64 plotting
  meshes (180 kB each), so a pickled copula was 0.5 MB before any parameter,
  1.1–1.6 MB for a rotated or survival one, ``BivariateLaw`` 1.2 MB and a
  two-state PMC model 2.2 MB — copied into every multistart worker. The meshes
  are deterministic in ``N``: they are now left out and rebuilt on load.
* **Fragility.** ``Enum`` pickles by *value*, i.e. the whole registry entry
  (τ-range, parameter names, MODULE path…). Editing any field of one entry
  made every pickle of that family unloadable. ``CopulaEnum`` now pickles by
  name.
"""

from __future__ import annotations

import copy
import pickle
import subprocess
import sys
import textwrap
import zlib
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from pmcprg.copulas import (
    BivariateLaw,
    CopulaClayton,
    CopulaEnum,
    CopulaGaussian,
    EmpiricalBetaCopula,
    submodel_lr_test,
)

#: A pickled copula is a few hundred bytes of parameters; 0.5 MB was the
#: weight of the meshes alone. 4 kB leaves room for a base copula and future
#: scalar attributes while still failing loudly if a mesh comes back.
MAX_COPULA_PICKLE_BYTES = 4_000


def _seed(*parts) -> int:
    return zlib.crc32(repr(parts).encode())


def _build(entry: CopulaEnum, tau: float = 0.3):
    """An instance of ``entry`` at the admissible point nearest ``tau``."""
    lo, hi = entry.constructible_tau_range()
    tau = min(max(tau, lo), hi)
    if lo < 0.0 < hi and abs(tau) < 0.05:
        tau = 0.05
    return entry.klass(**entry.klass.constructible_params({"tau_k": tau}))


_POINTS = np.random.default_rng(_seed("points")).uniform(0.02, 0.98, size=(120, 2))


# ---------------------------------------------------------------------------
# Every registered family
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("entry", list(CopulaEnum), ids=lambda e: e.name)
def test_every_family_round_trips_bit_for_bit(entry):
    cop = _build(entry)
    payload = pickle.dumps(cop)
    back = pickle.loads(payload)

    assert type(back) is type(cop)
    assert back.copula_enum is cop.copula_enum
    assert back.params == cop.params
    np.testing.assert_array_equal(back.logpdf_array(_POINTS), cop.logpdf_array(_POINTS))
    # the rebuilt meshes are the originals, not merely close to them
    for name in cop._GRID_ATTRS:
        np.testing.assert_array_equal(getattr(back, name), getattr(cop, name))
    assert len(payload) < MAX_COPULA_PICKLE_BYTES, (
        f"{entry.name}: {len(payload):,} bytes — the plotting meshes are "
        "being pickled again (see CopulaVirt.__getstate__)")


@pytest.mark.parametrize("entry", list(CopulaEnum), ids=lambda e: e.name)
def test_every_family_deep_copies(entry):
    cop = _build(entry)
    dup = copy.deepcopy(cop)
    assert dup is not cop and dup.params == cop.params
    np.testing.assert_array_equal(dup.logpdf_array(_POINTS), cop.logpdf_array(_POINTS))


# ---------------------------------------------------------------------------
# The enumeration is pickled by name
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("entry", list(CopulaEnum), ids=lambda e: e.name)
def test_enum_member_is_pickled_by_name(entry):
    payload = pickle.dumps(entry)
    assert pickle.loads(payload) is entry
    assert entry.name.encode() in payload
    # nothing of the registry entry itself — no module path, no τ-range
    assert entry.MODULE.encode() not in payload
    assert entry.SHORT_NAME.encode() not in payload or entry.SHORT_NAME == entry.name


def test_enum_pickle_survives_an_edit_of_the_registry_entry(monkeypatch):
    """Simulate what happened to this registry more than once — a field of an
    entry changes between the version that wrote the pickle and the one that
    reads it. By value that made the member unloadable; by name it does not
    matter."""
    member = CopulaEnum.BB1
    payload = pickle.dumps(member)
    edited = copy.copy(member.value)
    edited.TAU_MIN_MAX = [0.1, 0.9]
    edited.MODULE = "pmcprg.copulas.somewhere.else"
    monkeypatch.setattr(member, "_value_", edited)
    assert pickle.loads(payload) is member


def test_a_pickle_written_by_value_still_loads():
    """Files written before the change store ``CopulaEnum(value)``."""
    class _Legacy:
        def __reduce__(self):
            return CopulaEnum, (CopulaEnum.TAWN3.value,)

    assert pickle.loads(pickle.dumps(_Legacy())) is CopulaEnum.TAWN3


def test_the_state_of_a_pickle_that_still_holds_the_meshes_loads():
    """A copula pickled before the meshes were dropped carries them in its
    state; applying that state must leave a working, identical copula."""
    cop = _build(CopulaEnum.BB1)
    old_state = dict(cop.__dict__)                      # meshes included
    assert all(name in old_state for name in cop._GRID_ATTRS)
    back = type(cop).__new__(type(cop))
    back.__setstate__(old_state)
    np.testing.assert_array_equal(back.logpdf_array(_POINTS), cop.logpdf_array(_POINTS))
    for name in cop._GRID_ATTRS:
        np.testing.assert_array_equal(getattr(back, name), getattr(cop, name))


# ---------------------------------------------------------------------------
# A fresh interpreter, which is what a saved file or a worker really is
# ---------------------------------------------------------------------------

def test_a_pickle_loads_in_a_fresh_interpreter(tmp_path: Path):
    cops = [_build(e) for e in (CopulaEnum.GAUSSIAN, CopulaEnum.BB1, CopulaEnum.TAWN3,
                                CopulaEnum.BB8, CopulaEnum.SURVIVAL_BB190)]
    path = tmp_path / "copulas.pkl"
    path.write_bytes(pickle.dumps(cops))
    pts = tmp_path / "pts.npy"
    np.save(pts, _POINTS)
    script = textwrap.dedent(f"""
        import pickle, sys
        import numpy as np
        cops = pickle.load(open({str(path)!r}, "rb"))
        pts = np.load({str(pts)!r})
        print(repr([float(np.sum(c.logpdf_array(pts))) for c in cops]))
    """)
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                         check=True).stdout
    theirs = eval(out.strip().splitlines()[-1])
    ours = [float(np.sum(c.logpdf_array(_POINTS))) for c in cops]
    assert theirs == ours          # exact, not approx: same code, same numbers


# ---------------------------------------------------------------------------
# Results and objects built on copulas
# ---------------------------------------------------------------------------

def _sample(n=300, seed=1):
    return CopulaClayton(tau_k=0.5).sample(n=n, seed=seed)


def test_fit_result_round_trips_and_is_light():
    res = CopulaClayton.fit(_sample(), method="mle")
    payload = pickle.dumps(res)
    back = pickle.loads(payload)
    assert back.copula.params == res.copula.params
    assert back.log_likelihood == res.log_likelihood
    assert back.aic == res.aic
    # was ~550 kB, all of it the copula's meshes
    assert len(payload) < 30_000


def test_bivariate_law_round_trips_and_is_light():
    law = BivariateLaw(copula=CopulaGaussian(tau_k=0.4),
                       left_margin=(stats.norm, 0.0, 1.0),        # (dist, *params)
                       right_margin=(stats.expon, 0.0, 1.0))
    payload = pickle.dumps(law)
    back = pickle.loads(payload)
    pts = [[0.3, 0.7], [1.1, 0.2], [-0.5, 1.4]]
    np.testing.assert_array_equal([back.pdf(p) for p in pts], [law.pdf(p) for p in pts])
    for name in law._GRID_ATTRS:
        np.testing.assert_array_equal(getattr(back, name), getattr(law, name))
    assert len(payload) < 50_000    # was 1.2 MB


def test_empirical_beta_copula_round_trips():
    emp = EmpiricalBetaCopula(_sample())
    back = pickle.loads(pickle.dumps(emp))
    np.testing.assert_array_equal(back.cdf_array(_POINTS), emp.cdf_array(_POINTS))
    np.testing.assert_array_equal(back.logpdf_array(_POINTS), emp.logpdf_array(_POINTS))


def test_test_result_dataclasses_round_trip():
    from pmcprg.copulas import CopulaBB1
    res = submodel_lr_test(CopulaBB1, CopulaClayton, _sample(200))
    back = pickle.loads(pickle.dumps(res))
    assert back == res


# ---------------------------------------------------------------------------
# PMC models
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pmc_model():
    import pmcprg.pmc as pmc
    return pmc.PMCModel(str(Path(pmc.__file__).parent / "models" / "pmc_gauss_k2.toml"))


def test_pmc_model_round_trips_and_is_light(pmc_model):
    from pmcprg.pmc import classify, simulate
    payload = pickle.dumps(pmc_model)
    back = pickle.loads(payload)
    _, Y = simulate(pmc_model, N=200, seed=0)
    for got, want in zip(classify(back, Y), classify(pmc_model, Y)):
        np.testing.assert_array_equal(got, want)
    # was 2.2 MB: four copula blocks × the meshes
    assert len(payload) < 100_000


def test_pmc_model_deep_copies(pmc_model):
    dup = copy.deepcopy(pmc_model)
    assert dup is not pmc_model
    assert dup.raw == pmc_model.raw


def test_ice_result_round_trips(pmc_model):
    from pmcprg.pmc import ice, simulate
    _, Y = simulate(pmc_model, N=300, seed=2)
    fitted, trace = ice(pmc_model, Y, ice_cfg={"max_iter": 3})
    back_model, back_trace = pickle.loads(pickle.dumps((fitted, trace)))
    assert back_model.raw == fitted.raw
    np.testing.assert_array_equal(back_trace.log_liks, trace.log_liks)


# ---------------------------------------------------------------------------
# PMC models with a missingness mechanism (P6)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("table", [
    {"mechanism": "state", "rates": [0.05, 0.4]},
    {"mechanism": "state-markov", "onset": [0.01, 0.2], "persistence": [0.5, 0.9]},
], ids=["state", "state-markov"])
def test_pmc_model_with_missingness_round_trips(pmc_model, table):
    from pmcprg.pmc import classify, simulate
    m = pmc_model.with_missingness(table)
    back = pickle.loads(pickle.dumps(m))
    assert back.missingness == m.missingness and back.raw == m.raw
    assert pickle.loads(pickle.dumps(m.missingness)) == m.missingness
    _, Y = simulate(pmc_model, N=120, seed=_seed("P6", table["mechanism"]) % 2**31)
    Y[[0, 30, 31, 119]] = np.nan
    for got, want in zip(classify(back, Y), classify(m, Y)):
        np.testing.assert_array_equal(got, want)


def test_pmc_model_pickled_before_missingness_loads_as_ignorable(pmc_model):
    """A model pickled before P6 has no ``_missingness`` in its state: the
    class-level default makes it load as an ignorable model."""
    from pmcprg.pmc import classify, simulate
    legacy = copy.deepcopy(pmc_model)
    del legacy.__dict__["_missingness"]            # the state an older version wrote
    payload = pickle.dumps(legacy)
    assert b"_missingness" not in payload
    back = pickle.loads(payload)
    assert back.missingness is None
    _, Y = simulate(pmc_model, N=120, seed=_seed("P6", "legacy") % 2**31)
    Y[[5, 6, 60]] = np.nan
    for got, want in zip(classify(back, Y), classify(pmc_model, Y)):
        np.testing.assert_array_equal(got, want)
    # and it can still be given a mechanism
    mech = back.with_missingness({"mechanism": "state", "rates": [0.1, 0.2]}).missingness
    assert mech.rates == (0.1, 0.2)
