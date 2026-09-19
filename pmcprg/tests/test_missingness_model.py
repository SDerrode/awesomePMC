"""Non-ignorable missingness (P6) — the ``[missingness]`` table of a model.

TOML round trip (file, ``from_dict``, ``raw``, ``save``), validation,
``with_missingness``, the evidence factors of both mechanisms, and the
boundary rates 0 and 1 (states made impossible at missing or observed rows,
an all-impossible step raising ``IncompatibleObservationError``).
Pickling is in ``test_pickle.py``.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from pmcprg.exceptions import IncompatibleObservationError
from pmcprg.pmc import PMCModel, StateMarkovMissingness, StateMissingness, gaps
from pmcprg.pmc import inference as inf
from pmcprg.pmc.missingness import MECHANISMS, as_missingness, parse_missingness

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "pmcprg" / "pmc" / "models" / "hmc_in_gauss_k2.toml"
GRID_FIXTURE = REPO / "pmcprg" / "pmc" / "models" / "hmc_dn_gauss_k2.toml"

TOML_STATE = """
[missingness]
mechanism = "state"
rates = [0.01, 0.3]
"""
TOML_MARKOV = """
[missingness]
mechanism = "state-markov"
onset = [0.002, 0.02]
persistence = [0.8, 0.95]
"""


def _write(tmp_path, extra, base=FIXTURE):
    p = tmp_path / "model.toml"
    p.write_text(base.read_text() + extra)
    return p


# ---------------------------------------------------------------------------
# TOML round trip
# ---------------------------------------------------------------------------

def test_default_is_ignorable():
    m = PMCModel(FIXTURE)
    assert m.missingness is None and "missingness" not in m.raw
    raw = m.raw
    raw["missingness"] = {"mechanism": "ignorable"}
    assert PMCModel.from_dict(raw).missingness is None
    assert set(MECHANISMS) == {"ignorable", "state", "state-markov"}


@pytest.mark.parametrize("text,cls", [(TOML_STATE, StateMissingness),
                                      (TOML_MARKOV, StateMarkovMissingness)])
def test_toml_round_trip(tmp_path, text, cls):
    m = PMCModel(_write(tmp_path, text))
    mech = m.missingness
    assert isinstance(mech, cls) and mech.K == 2
    if cls is StateMissingness:
        assert mech.mechanism == "state" and mech.rates == (0.01, 0.3)
    else:
        assert mech.mechanism == "state-markov"
        assert mech.onset == (0.002, 0.02) and mech.persistence == (0.8, 0.95)
        np.testing.assert_allclose(mech.stationary, [0.002 / 0.202, 0.02 / 0.07], rtol=1e-15)
    # raw → from_dict, save → load: the same mechanism
    assert PMCModel.from_dict(m.raw).missingness == mech
    out = m.save(tmp_path / "saved.toml")
    back = PMCModel(out)
    assert back.missingness == mech and back.raw == m.raw
    # the table is kept as written
    assert back.raw["missingness"] == m.raw["missingness"]


def test_mechanism_is_read_only_and_frozen():
    m = PMCModel.from_dict({**PMCModel(FIXTURE).raw,
                            "missingness": {"mechanism": "state", "rates": [0.1, 0.2]}})
    with pytest.raises(AttributeError):
        m.missingness = None
    with pytest.raises(AttributeError):
        m.missingness.rates = (0.5, 0.5)
    assert hash(m.missingness) == hash(StateMissingness(rates=(0.1, 0.2)))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("table,match", [
    ({"mechanism": "mnar"}, "Unknown"),
    ({"rates": [0.1, 0.2]}, "'mechanism' key"),
    ({"mechanism": "state"}, "missing key"),
    ({"mechanism": "state", "rates": [0.1, 0.2], "onset": [0.1, 0.1]}, "unexpected key"),
    ({"mechanism": "ignorable", "rates": [0.1, 0.2]}, "unexpected key"),
    ({"mechanism": "state-markov", "onset": [0.1, 0.2]}, "missing key"),
    ({"mechanism": "state", "rates": [0.1]}, "one entry per state"),
    ({"mechanism": "state", "rates": [0.1, 0.2, 0.3]}, "one entry per state"),
    ({"mechanism": "state", "rates": [0.1, 1.2]}, r"\[0, 1\]"),
    ({"mechanism": "state", "rates": [-0.1, 0.2]}, r"\[0, 1\]"),
    ({"mechanism": "state", "rates": [float("nan"), 0.2]}, r"\[0, 1\]"),
    ({"mechanism": "state", "rates": [float("inf"), 0.2]}, r"\[0, 1\]"),
    ({"mechanism": "state", "rates": [True, 0.2]}, "numbers"),
    ({"mechanism": "state", "rates": ["0.1", 0.2]}, "numbers"),
    ({"mechanism": "state", "rates": 0.1}, "list"),
    ({"mechanism": "state", "rates": [[0.1, 0.2]]}, "list"),
    ({"mechanism": "state-markov", "onset": [0.0, 0.1], "persistence": [1.0, 0.5]}, "0/0"),
    ({"mechanism": "state-markov", "onset": [0.1, 0.1], "persistence": [0.5]}, "one entry per state"),
    ("state", "table"),
])
def test_invalid_tables_are_refused(table, match):
    raw = PMCModel(FIXTURE).raw
    raw["missingness"] = table
    with pytest.raises(ValueError, match=match):
        PMCModel.from_dict(raw)


def test_boundary_probabilities_are_allowed():
    for table in ({"mechanism": "state", "rates": [0, 1]},
                  {"mechanism": "state-markov", "onset": [0.0, 1.0], "persistence": [0.0, 1.0]},
                  {"mechanism": "state-markov", "onset": [1.0, 0.3], "persistence": [1.0, 0.0]}):
        assert parse_missingness(table, 2) is not None


# ---------------------------------------------------------------------------
# with_missingness
# ---------------------------------------------------------------------------

def test_with_missingness_returns_a_modified_copy():
    m = PMCModel(FIXTURE)
    ms = m.with_missingness(StateMissingness(rates=[0.1, 0.4]))
    assert m.missingness is None and "missingness" not in m.raw          # untouched
    assert ms.missingness == StateMissingness(rates=(0.1, 0.4))
    assert ms.raw["missingness"] == {"mechanism": "state", "rates": [0.1, 0.4]}
    assert {k: v for k, v in ms.raw.items() if k != "missingness"} == m.raw
    assert ms.path == m.path
    mm = ms.with_missingness({"mechanism": "state-markov", "onset": [0.1, 0.2],
                              "persistence": [0.3, 0.4]})
    assert mm.missingness == StateMarkovMissingness(onset=(0.1, 0.2), persistence=(0.3, 0.4))
    back = mm.with_missingness(None)
    assert back.missingness is None and back.raw == m.raw
    np.testing.assert_array_equal(back.prior_p, m.prior_p)


def test_with_missingness_validates_against_K():
    m = PMCModel(REPO / "pmcprg" / "pmc" / "models" / "hmc_in_gauss_k3.toml")
    with pytest.raises(ValueError, match="one entry per state"):
        m.with_missingness(StateMissingness(rates=[0.1, 0.2]))
    with pytest.raises(TypeError, match="missingness must be"):
        m.with_missingness([0.1, 0.2, 0.3])
    assert as_missingness(None, 3) is None


# ---------------------------------------------------------------------------
# Evidence factors
# ---------------------------------------------------------------------------

def test_evidence_factors_of_both_mechanisms():
    miss = np.array([True, True, False, False, True, False])
    st = StateMissingness(rates=[0.1, 0.4])
    np.testing.assert_array_equal(st.evidence(miss),
                                  [[0.1, 0.4], [0.1, 0.4], [0.9, 0.6], [0.9, 0.6],
                                   [0.1, 0.4], [0.9, 0.6]])
    a, b = np.array([0.1, 0.3]), np.array([0.6, 0.9])
    mk = StateMarkovMissingness(onset=a, persistence=b)
    s = a / (1 - b + a)
    ref = np.array([s, b, 1 - b, 1 - a, a, 1 - b])
    np.testing.assert_allclose(mk.evidence(miss), ref, rtol=1e-15)
    for mech in (st, mk):
        with np.errstate(divide="ignore"):
            np.testing.assert_allclose(mech.log_evidence(miss), np.log(mech.evidence(miss)),
                                       rtol=1e-14)   # log1p(−a) vs log(fl(1 − a)): ≤ 1 ulp of 1
    # a mask sums to 1 over its 2^N values, whatever the state path
    x = np.array([0, 1, 1, 0])
    tot = 0.0
    for bits in range(16):
        mm = np.array([(bits >> k) & 1 for k in range(4)], bool)
        tot += np.prod(mk.evidence(mm)[np.arange(4), x])
    assert tot == pytest.approx(1.0, abs=1e-15)
    assert mk.evidence(np.zeros(0, bool)).shape == (0, 2)
    with pytest.raises(ValueError, match="1-D"):
        st.evidence(np.zeros((3, 2), bool))


# ---------------------------------------------------------------------------
# Rates 0 and 1: impossible states, impossible steps
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", [FIXTURE, GRID_FIXTURE], ids=["exact", "grid"])
def test_rate_zero_or_one_makes_a_state_impossible(path):
    m0 = PMCModel(path)
    Y = np.array([-0.9, np.nan, 1.1, 0.4, np.nan, np.nan, -0.2, 0.7])
    miss = gaps.missing_mask(Y)
    # π_0 = 0: state 0 is impossible at a missing row
    m = m0.with_missingness(StateMissingness(rates=[0.0, 0.5]))
    _, g, ll = inf.classify(m, Y)
    np.testing.assert_array_equal(g[miss, 0], 0.0)
    np.testing.assert_allclose(g.sum(axis=1), 1.0, atol=1e-15)
    # π_0 = 1: state 0 is impossible at an observed row
    m = m0.with_missingness(StateMissingness(rates=[1.0, 0.5]))
    _, g, ll = inf.classify(m, Y)
    np.testing.assert_array_equal(g[~miss, 0], 0.0)
    assert np.isfinite(ll)
    # the log-space passes agree (log 0 = −∞)
    if not gaps.needs_grid(m):
        a, ll_log = inf._forward_log_space(m, Y)
        assert ll_log == pytest.approx(ll, abs=1e-12)
        np.testing.assert_array_equal(inf.smooth(a, inf._backward_log_space(m, Y))[~miss, 0], 0.0)
    else:
        chain = gaps._Chain(m, Y, miss, gaps.reference_grid(m), log=True)
        _, ll_log = gaps._forward_chain(chain)
        assert ll_log == pytest.approx(ll, abs=1e-12)


@pytest.mark.parametrize("path", [FIXTURE, GRID_FIXTURE], ids=["exact", "grid"])
def test_a_step_impossible_under_every_state_raises(path):
    m0 = PMCModel(path)
    Y = np.array([-0.9, 0.3, 1.1, np.nan, 0.4])
    m = m0.with_missingness(StateMissingness(rates=[0.0, 0.0]))       # nothing can be missing
    with pytest.raises(IncompatibleObservationError):
        inf.classify(m, Y)
    with pytest.raises(IncompatibleObservationError):
        inf.forward(m, Y)
    m = m0.with_missingness(StateMissingness(rates=[1.0, 1.0]))       # everything is missing
    with pytest.raises(IncompatibleObservationError):
        inf.classify(m, Y)
    # a leading row impossible under every state
    m = m0.with_missingness(StateMarkovMissingness(onset=[0.0, 0.0], persistence=[0.5, 0.5]))
    with pytest.raises(IncompatibleObservationError):
        inf.forward(m, np.array([np.nan, 0.3, 1.1]))


def test_all_missing_sequence_under_a_mechanism():
    # No observation: p(m) = Σ_x p(x) Π π_{x_n}, and γ_n ∝ the prior reweighted
    # by the mask — exact for a Markov X (HMC-IN) against the K-state product.
    m0 = PMCModel(FIXTURE)
    m = m0.with_missingness(StateMissingness(rates=[0.2, 0.6]))
    Y = np.full(6, np.nan)
    _, g, ll = inf.classify(m, Y)
    pi, A = m.prior_p.sum(axis=0), m.transition_A          # the initial law of forward
    alpha = pi * np.array([0.2, 0.6])
    lik = [alpha.sum()]
    alpha = alpha / alpha.sum()
    for _ in range(5):
        alpha = (alpha @ A) * np.array([0.2, 0.6])
        lik.append(alpha.sum())
        alpha = alpha / alpha.sum()
    assert ll == pytest.approx(sum(math.log(c) for c in lik), abs=1e-13)
    np.testing.assert_allclose(g[-1], alpha, atol=1e-15)
