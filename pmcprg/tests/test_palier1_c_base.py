"""Base-class contracts of AUDIT_COPULES palier 1 (RB-6, RB-9, FR-2, K-9).

* a singular registered endpoint (|τ| = 1) is refused with
  ``CopulaParameterError`` **before** the family's parameter map runs, while
  admissible endpoints stay constructible;
* ``update_tau_k`` follows the same rule and refreshes θ (audit K-9);
* the default ``pdf_array`` / ``logpdf_array`` / ``cdf_array`` do not floor.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from pmcprg.copulas import CopulaEnum
from pmcprg.copulas._base import (
    CopulaVirt,
    constructible_tau_range,
    is_singular_tau,
    padded_tau_range,
)
from pmcprg.exceptions import CopulaParameterError
from pmcprg.numerics import EPS

_AVAILABLE = [e for e in CopulaEnum if e.value.AVAILABLE]


def _kwargs(entry, tau):
    kw = {"tau_k": tau}
    if "df" in entry.value.PARAMETERS_SET_NAME:
        kw["df"] = 4.0
    if "delta" in entry.value.PARAMETERS_SET_NAME:
        kw["delta"] = 1.5
    return kw


_SINGULAR = [pytest.param(e, t, id=f"{e.value.SHORT_NAME}{t:+g}")
             for e in _AVAILABLE for t in set(e.value.TAU_MIN_MAX) if is_singular_tau(t)]


@pytest.mark.parametrize("entry,tau", _SINGULAR)
def test_singular_registered_endpoint_is_refused_before_the_parameter_map(entry, tau, monkeypatch):
    """τ = 1 used to reach the family map: ZeroDivisionError (GH, Clayton, A12,
    A14, survivals), LinAlgError (Gauss, Student), θ = 1e9 (Joe)."""
    calls = []
    monkeypatch.setattr(entry.klass, "_update_params",
                        lambda self: calls.append(self.params["tau_k"]))
    with pytest.raises(CopulaParameterError, match="Fréchet"):
        entry.klass(**_kwargs(entry, tau))
    assert calls == [], "the family's _update_params ran before the refusal"


def test_every_family_registered_at_one_is_covered():
    shorts = {p.values[0].value.SHORT_NAME for p in _SINGULAR}
    assert {"Gauss", "Student", "GH", "Clayton", "A12", "A14", "Joe",
            "SClayton", "SGH", "SJoe", "BB1"} <= shorts


# Admissible endpoints — constructible, with a density. Joe/SJoe at ε need the
# RB-2 kernel fix and BB1 at ε needs δ ≈ 1; they are covered elsewhere.
_ADMISSIBLE = [
    ("Prod", 0.0), ("GH", EPS), ("SGH", EPS), ("Clayton", EPS), ("SClayton", EPS),
    ("A12", 1.0 / 3.0), ("A14", 1.0 / 3.0), ("FGM", -2.0 / 9.0), ("FGM", 2.0 / 9.0),
    ("CubSec", 0.0), ("CubSec", 33.0 / 200.0),
    ("AMH", CopulaEnum.AMH.value.TAU_MIN_MAX[0]), ("AMH", CopulaEnum.AMH.value.TAU_MIN_MAX[1]),
    ("Frank", CopulaEnum.FRANK.value.TAU_MIN_MAX[0]), ("Frank", CopulaEnum.FRANK.value.TAU_MIN_MAX[1]),
    ("Plackett", CopulaEnum.PLACKETT.value.TAU_MIN_MAX[0]),
    ("Plackett", CopulaEnum.PLACKETT.value.TAU_MIN_MAX[1]),
]


@pytest.mark.parametrize("short,tau", _ADMISSIBLE, ids=[f"{s}{t:+.3g}" for s, t in _ADMISSIBLE])
def test_admissible_registered_endpoint_stays_constructible(short, tau):
    entry = CopulaEnum.from_short_name(short)
    lo, hi = entry.constructible_tau_range()
    # Frank's registered ends lie beyond the τ it reaches (θ ≤ 700): they still
    # build, but the constructible range stops at the reachable τ (F1).
    assert lo <= entry.reachable_tau(tau) <= hi
    cop = entry.klass(**_kwargs(entry, tau))
    vals = cop.logpdf_array(np.array([[0.3, 0.6], [0.5, 0.5]]))
    assert not np.any(np.isnan(vals))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "0.3", None])
def test_non_numeric_or_non_finite_tau_is_refused(bad):
    from pmcprg.copulas import CopulaGaussian
    with pytest.raises(CopulaParameterError):
        CopulaGaussian(tau_k=bad)


def test_tau_range_helpers():
    assert constructible_tau_range(-1.0, 1.0) == pytest.approx((-1.0 + 2e-4, 1.0 - 2e-4))
    assert constructible_tau_range(EPS, 1.0) == (EPS, pytest.approx(1.0 - 1e-4))
    assert constructible_tau_range(1 / 3, 1.0)[0] == 1 / 3
    assert constructible_tau_range(0.0, 0.0) == (0.0, 0.0)
    assert padded_tau_range(0.0, 0.0) == (0.0, 0.0)
    lo, hi = padded_tau_range(EPS, 1.0)
    assert lo == pytest.approx(1e-4) and hi == pytest.approx(1.0 - 1e-4)
    # correct_tau keeps its registry contract (the target may be singular).
    assert CopulaEnum.GH.correct_tau(2.0) == 1.0


# --------------------------------------------------------------------------
# update_tau_k — K-9
# --------------------------------------------------------------------------

def test_update_tau_k_refreshes_theta():
    from pmcprg.copulas import CopulaClayton
    cop = CopulaClayton(tau_k=0.2)
    cop.update_tau_k(0.5)
    assert cop.params["tau_k"] == 0.5
    assert cop.theta == pytest.approx(2.0)         # θ = 2τ/(1 − τ)
    assert cop.pdf_array(np.array([[0.3, 0.4]]))[0] == pytest.approx(
        CopulaClayton(tau_k=0.5).pdf_array(np.array([[0.3, 0.4]]))[0], rel=1e-14)


@pytest.mark.parametrize("bad", [5.0, 1.0, float("nan"), -0.1])
def test_update_tau_k_refuses_like_the_constructor_and_leaves_the_copula_unchanged(bad):
    """Formerly: out of range → params τ set to the mid-range, θ left stale."""
    from pmcprg.copulas import CopulaClayton
    cop = CopulaClayton(tau_k=0.5)
    with pytest.raises(CopulaParameterError):
        cop.update_tau_k(bad)
    assert cop.params["tau_k"] == 0.5 and cop.theta == pytest.approx(2.0)


def test_update_tau_k_restores_state_when_the_family_refuses():
    from pmcprg.copulas import CopulaBB1
    cop = CopulaBB1(tau_k=0.6, delta=1.5)
    theta = cop.theta
    with pytest.raises(CopulaParameterError):
        cop.update_tau_k(0.2)                       # θ < 0 at δ = 1.5
    assert cop.params["tau_k"] == 0.6 and cop.theta == theta


def test_update_tau_k_refuses_the_singular_endpoint_of_gauss():
    from pmcprg.copulas import CopulaGaussian
    cop = CopulaGaussian(tau_k=0.3)
    with pytest.raises(CopulaParameterError, match="Fréchet"):
        cop.update_tau_k(-1.0)
    assert cop.params["tau_k"] == 0.3


# --------------------------------------------------------------------------
# Default densities — FR-2 / RB-9
# --------------------------------------------------------------------------

class _TinyBackend:
    @staticmethod
    def pdf(uv):
        return np.array([1e-30, 0.0, 2.0])[: len(uv)]


def _backend_only(backend):
    """An object with only the base-class defaults and a statsmodels-like backend.

    Since palier 1 no shipped family relies on the base defaults any more
    (Gauss, Student and Clayton dropped statsmodels; every family defines its
    own density arrays), so the defaults are exercised on a stand-in.
    """
    from pmcprg.copulas._base import CopulaVirt

    class _BackendOnly:
        _model = backend
        _overrides = CopulaVirt._overrides
        _backend_pdf_array = CopulaVirt._backend_pdf_array
        pdf_array = CopulaVirt.pdf_array
        logpdf_array = CopulaVirt.logpdf_array
        cdf_array = CopulaVirt.cdf_array

    return _BackendOnly()


def test_default_logpdf_is_not_floored_at_log_eps():
    """The default used to return log(max(pdf, EPS)) = −36.04 wherever the
    density was below EPS (Student at τ = 0.95, ν = 30 went to −3086 nat)."""
    cop = _backend_only(_TinyBackend())
    assert not cop._overrides("logpdf_array") and not cop._overrides("pdf_array")
    uv = np.array([[1e-6, 1 - 1e-6], [0.01, 0.99], [0.5, 0.5]])
    with np.errstate(divide="ignore"):
        logc = cop.logpdf_array(uv)
    assert logc[0] == pytest.approx(math.log(1e-30)) and logc[0] < math.log(EPS) - 1.0
    assert logc[1] == -math.inf and logc[2] == pytest.approx(math.log(2.0))
    np.testing.assert_array_equal(cop.pdf_array(uv), [1e-30, 0.0, 2.0])


def test_native_logpdf_defines_the_default_pdf_array():
    """A family with only a native log-density gets pdf = exp(log c).

    Since the palier-1 kernels every shipped family defines both methods, so
    the base-class default is called explicitly on a Clayton instance (the
    registry refuses ad-hoc subclasses).
    """
    from pmcprg.copulas import CopulaClayton
    cop = CopulaClayton(tau_k=0.9)
    assert cop._overrides("logpdf_array")
    uv = np.array([[0.01, 0.99], [0.3, 0.31], [1e-6, 0.5]])
    np.testing.assert_array_equal(CopulaVirt.pdf_array(cop, uv), np.exp(cop.logpdf_array(uv)))


def test_product_defaults():
    from pmcprg.copulas import CopulaProduct
    cop = CopulaProduct(tau_k=0.0)
    uv = np.array([[0.1, 0.9], [0.5, 0.5]])
    np.testing.assert_array_equal(cop.pdf_array(uv), [1.0, 1.0])
    np.testing.assert_array_equal(cop.logpdf_array(uv), [0.0, 0.0])


class _NaNBackend:
    @staticmethod
    def cdf(uv):
        out = np.full(len(uv), 0.25)
        out[0] = np.nan
        return out

    @staticmethod
    def pdf(uv):
        out = np.array([np.nan, 0.0, -1e-18])
        return out[: len(uv)]


def test_default_cdf_and_pdf_arrays_report_backend_failures():
    """cdf_array replaced a non-finite backend value by EPS — plausible in the
    lower corner, wrong wherever C ≈ 1. It is NaN now; pdf keeps NaN and an
    exact 0.0, and only a negative rounding residue becomes 0.0."""
    cop = _backend_only(_NaNBackend())
    uv = np.array([[0.9, 0.9], [0.5, 0.5], [0.2, 0.2]])
    cdf = cop.cdf_array(uv)
    assert math.isnan(cdf[0]) and cdf[1] == 0.25
    pdf = cop.pdf_array(uv)
    assert math.isnan(pdf[0]) and pdf[1] == 0.0 and pdf[2] == 0.0
    with np.errstate(invalid="ignore"):
        logc = cop.logpdf_array(uv)
    assert math.isnan(logc[0]) and logc[1] == -math.inf


def test_perturbed_initial_model_never_lands_on_a_singular_tau():
    """Multistart jitter clipped τ onto the registered bound 1 (RB-6)."""
    from pmcprg.pmc.ice import _perturb_initial_model
    from pmcprg.pmc.model import PMCModel
    raw = {
        "model": {"name": "t", "variant": "PMC", "K": 2, "N_default": 10},
        "margins": [{"i": 0, "dist": "norm", "params": {"loc": -1.0, "scale": 1.0}},
                    {"i": 1, "dist": "norm", "params": {"loc": 1.0, "scale": 1.0}}],
        "copulas": [{"i": i, "j": j, "name": name, "tau": 0.999}
                    for (i, j), name in zip([(0, 0), (0, 1), (1, 0), (1, 1)],
                                            ["GH", "Gauss", "Gauss", "Clayton"])],
        "prior": {"p": [[0.45, 0.05], [0.05, 0.45]]},
    }
    mdl = PMCModel.from_dict(raw)
    rng = np.random.default_rng(0)
    for _ in range(20):
        new = _perturb_initial_model(mdl, rng, jitter=1.0)      # must not raise
        for blk in new.raw["copulas"]:
            assert abs(blk["tau"]) < 1.0
