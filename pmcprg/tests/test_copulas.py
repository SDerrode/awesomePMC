"""Tests for the copula layer — instantiation, PDF, majorant, exceptions."""
import logging
import math

import numpy as np
import pytest

from pmcprg.copulas import (
    CopulaEnum,
    CopulaGaussian, CopulaStudent,
    CopulaGH, CopulaClayton, CopulaA12, CopulaA14,
    CopulaProduct, CopulaFGM, CopulaCubSec,
)
from pmcprg.exceptions import CopulaParameterError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ALL_AVAILABLE = [
    CopulaGaussian(tau_k=0.5),
    CopulaStudent(tau_k=0.5),
    CopulaGH(tau_k=0.2),
    CopulaClayton(tau_k=0.2),
    CopulaA12(tau_k=0.5),
    CopulaA14(tau_k=0.5),
    CopulaProduct(tau_k=0.0),
    CopulaFGM(tau_k=0.1),
    CopulaCubSec(tau_k=0.1),
]

UV = [0.5, 0.7]


# ---------------------------------------------------------------------------
# Basic tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('cop', ALL_AVAILABLE)
def test_pdf_positive(cop):
    assert cop.pdf(UV) > 0.0


@pytest.mark.parametrize('cop', ALL_AVAILABLE)
def test_majorant_geq_pdf(cop):
    u_left = UV[0]
    assert cop.majorant(u_left) >= cop.pdf(UV)


@pytest.mark.parametrize('cop', ALL_AVAILABLE)
def test_update_tau_k(cop):
    old_tau = cop.params['tau_k']
    new_tau = (cop.tau_min + cop.tau_max) / 2.0
    cop.update_tau_k(new_tau)
    assert cop.params['tau_k'] == pytest.approx(new_tau)
    cop.update_tau_k(old_tau)


# ---------------------------------------------------------------------------
# Exception tests
# ---------------------------------------------------------------------------

def test_bad_tau_k_raises_parameter_error():
    with pytest.raises(CopulaParameterError):
        CopulaGaussian(tau_k=5.0)


def test_bad_param_name_raises_parameter_error():
    with pytest.raises(CopulaParameterError):
        CopulaGaussian(rho=0.5)


def test_missing_tau_k_raises_parameter_error():
    with pytest.raises(CopulaParameterError):
        CopulaGaussian()


def test_parameter_error_is_value_error():
    with pytest.raises(ValueError):
        CopulaGaussian(tau_k=5.0)


def test_student_cdf_raises_not_implemented():
    cop = CopulaStudent(tau_k=0.5)
    with pytest.raises(NotImplementedError):
        cop.cdf(UV)


# ---------------------------------------------------------------------------
# CopulaEnum tests
# ---------------------------------------------------------------------------

def test_available_count():
    # 10 originals (FRANK now enabled) + JOE + 3 Survival + BB1 + AMH + Plackett
    # + Galambos (FR-9) = 18
    assert len(CopulaEnum.available()) == 18


def test_cubsec_long_name_english():
    assert CopulaEnum.CUBSEC.LONG_NAME == 'Cubic Section'


def test_from_short_name():
    assert CopulaEnum.from_short_name('Gauss') is CopulaEnum.GAUSSIAN


def test_favorite():
    assert CopulaEnum.favorite() is CopulaEnum.GAUSSIAN


# ---------------------------------------------------------------------------
# Boundary tests — τ_K → τ_max (Archimedean families)
# ---------------------------------------------------------------------------
#
# These tests exercise the upper edge of each Archimedean copula's valid τ
# range, which is precisely where ``θ → ∞`` (perfect concordance) and naive
# implementations either overflow, return non-finite log-pdfs, or saturate
# the inv-h solver. They also confirm that ``CopulaEnum.correct_tau`` clips
# silently-but-loudly at the boundaries (cf. CHANGELOG: clip warnings).

# All Archimedean families with a finite, non-trivial upper τ bound.
ARCHI_FAMILIES = [
    CopulaEnum.GH,
    CopulaEnum.CLAYTON,
    CopulaEnum.JOE,
    CopulaEnum.SURVIVAL_CLAYTON,
    CopulaEnum.SURVIVAL_GH,
    CopulaEnum.SURVIVAL_JOE,
    CopulaEnum.A12,
    CopulaEnum.A14,
    CopulaEnum.AMH,
    CopulaEnum.FRANK,
    CopulaEnum.BB1,
]

# How close to the boundary the "near τ_max / τ_min" tests probe. We use
# a 5 % margin (so τ = τ_min + 0.95·span) instead of a 0.1 % margin: at
# τ → τ_max the underlying θ blows up (Clayton θ → ∞, Joe θ → ∞, GH θ → ∞)
# and the resulting copula concentrates on the diagonal — evaluating any
# off-diagonal pdf is mathematically meaningful but numerically saturates.
# 95 % of the range is still firmly "near the boundary" (well inside the
# region where ``correct_tau`` clips) while leaving room for finite-precision
# arithmetic to remain meaningful.
_BOUNDARY_FRACTION = 0.95


def _instantiate_at_tau(entry: CopulaEnum, tau: float):
    """Build a copula at ``tau`` filling other required params with defaults."""
    cls = entry.klass
    kwargs: dict = {"tau_k": float(tau)}
    # BB1 needs a δ ≥ 1 in addition to τ_K. With δ = 1.5 the *effective*
    # lower τ-bound rises to ~1/3, so the lower-bound test is skipped for
    # BB1 in :func:`test_tau_near_lower_bound` below.
    if "delta" in entry.value.PARAMETERS_SET_NAME:
        kwargs["delta"] = 1.5
    if "df" in entry.value.PARAMETERS_SET_NAME:
        kwargs["df"] = 4.0
    return cls(**kwargs)


def _tau_near_upper(entry: CopulaEnum) -> float:
    tau_min, tau_max = entry.value.TAU_MIN_MAX
    span = tau_max - tau_min
    return tau_min + _BOUNDARY_FRACTION * span


def _tau_near_lower(entry: CopulaEnum) -> float:
    tau_min, tau_max = entry.value.TAU_MIN_MAX
    span = tau_max - tau_min
    return tau_min + (1.0 - _BOUNDARY_FRACTION) * span


@pytest.mark.parametrize("entry", ARCHI_FAMILIES,
                         ids=[e.value.SHORT_NAME for e in ARCHI_FAMILIES])
def test_tau_near_upper_bound(entry):
    """At τ → τ_max the copula must remain finite and produce a positive PDF."""
    cop = _instantiate_at_tau(entry, _tau_near_upper(entry))

    # Off-diagonal eval — avoids the corner singularity at u = v = 1.
    pdf  = cop.pdf([0.4, 0.6])
    assert math.isfinite(pdf), f"{entry.value.SHORT_NAME}: pdf non-finite near τ_max"
    assert pdf >= 0.0, f"{entry.value.SHORT_NAME}: negative pdf near τ_max"

    cdf = cop.cdf([0.4, 0.6])
    assert math.isfinite(cdf)
    assert 0.0 <= cdf <= 1.0

    h = cop.conditional_cdf(0.6, 0.4)
    assert math.isfinite(h)
    assert 0.0 <= h <= 1.0


# BB1 has a τ_min that depends on δ (with δ = 1.5 the effective floor is
# ~1/3, *not* the generic Archimedean ε declared in TAU_MIN_MAX); testing
# near that nominal floor is meaningless, so we exclude BB1 here.
_LOWER_BOUND_FAMILIES = [e for e in ARCHI_FAMILIES if e is not CopulaEnum.BB1]


@pytest.mark.parametrize("entry", _LOWER_BOUND_FAMILIES,
                         ids=[e.value.SHORT_NAME for e in _LOWER_BOUND_FAMILIES])
def test_tau_near_lower_bound(entry):
    """At τ → τ_min the copula must remain numerically well-behaved."""
    cop = _instantiate_at_tau(entry, _tau_near_lower(entry))
    pdf = cop.pdf([0.4, 0.6])
    assert math.isfinite(pdf) and pdf >= 0.0


@pytest.mark.parametrize("entry", ARCHI_FAMILIES,
                         ids=[e.value.SHORT_NAME for e in ARCHI_FAMILIES])
def test_correct_tau_clips_with_warning(entry, caplog):
    """``correct_tau`` must clip out-of-range τ and emit a WARNING.

    Regression guard against silent truncation: callers (notably ICE
    candidate selection) must be able to detect that their τ estimate was
    censored.
    """
    _, tau_max = entry.value.TAU_MIN_MAX
    too_high = tau_max + max(1e-2, 0.1 * (tau_max - entry.value.TAU_MIN_MAX[0]))

    with caplog.at_level(logging.WARNING, logger="pmcprg.copulas._base"):
        clipped = entry.correct_tau(too_high)

    assert clipped == pytest.approx(tau_max), \
        f"{entry.value.SHORT_NAME}: clip target should be τ_max"
    assert any(
        "outside valid range" in rec.message
        for rec in caplog.records
        if rec.levelno >= logging.WARNING
    ), f"{entry.value.SHORT_NAME}: no WARNING emitted on out-of-range τ"


@pytest.mark.parametrize("entry", ARCHI_FAMILIES,
                         ids=[e.value.SHORT_NAME for e in ARCHI_FAMILIES])
def test_correct_tau_handles_nan(entry, caplog):
    """Non-finite τ must be replaced with the midpoint and warned about."""
    tau_min, tau_max = entry.value.TAU_MIN_MAX
    mid = 0.5 * (tau_min + tau_max)

    with caplog.at_level(logging.WARNING, logger="pmcprg.copulas._base"):
        out = entry.correct_tau(float("nan"))

    assert out == pytest.approx(mid)
    assert any("non-finite" in rec.message for rec in caplog.records)


@pytest.mark.parametrize("entry", ARCHI_FAMILIES,
                         ids=[e.value.SHORT_NAME for e in ARCHI_FAMILIES])
def test_logpdf_array_finite_near_upper_bound(entry):
    """Vectorised log-PDF must stay finite at τ → τ_max (no -inf saturation)."""
    cop = _instantiate_at_tau(entry, _tau_near_upper(entry))

    # Random points well inside (0, 1) and concentrated near the principal
    # diagonal — at high τ the copula puts almost all its mass there, and
    # off-diagonal evaluation is legitimately near-zero (which would be a
    # false positive for the saturation guard below).
    rng = np.random.default_rng(0)
    centre = rng.uniform(0.2, 0.8, size=64)
    spread = 0.05 * rng.standard_normal(size=64)          # very tight band around v ≈ u
    uv = np.column_stack((centre, np.clip(centre + spread, 1e-3, 1.0 - 1e-3)))
    log_pdfs = cop.logpdf_array(uv)

    assert log_pdfs.shape == (64,)
    assert np.all(np.isfinite(log_pdfs)), \
        f"{entry.value.SHORT_NAME}: non-finite log-pdf near τ_max"
    # Diagonal samples should give clearly positive log-pdf; the EPS-saturation
    # tell-tale was around -36, so anything well above that means the native
    # log-pdf path is engaging (and not silently falling back to log(EPS)).
    assert log_pdfs.max() > -30.0, \
        f"{entry.value.SHORT_NAME}: log-pdf appears saturated near τ_max"


@pytest.mark.parametrize("entry", ARCHI_FAMILIES,
                         ids=[e.value.SHORT_NAME for e in ARCHI_FAMILIES])
def test_inv_h_array_within_unit_near_upper_bound(entry):
    """``inv_h_array`` must return values strictly inside (0, 1) near τ_max."""
    cop = _instantiate_at_tau(entry, _tau_near_upper(entry))

    rng = np.random.default_rng(1)
    w   = rng.uniform(0.1, 0.9, size=200)
    u   = rng.uniform(0.1, 0.9, size=200)
    v   = cop.inv_h_array(w, u)

    assert v.shape == (200,)
    assert np.all(np.isfinite(v))
    assert np.all(v > 0.0) and np.all(v < 1.0), \
        f"{entry.value.SHORT_NAME}: inv_h escaped (0, 1) near τ_max"


# ---------------------------------------------------------------------------
# Regression: Frank θ-inversion for strong dependence (audit finding N-2)
# ---------------------------------------------------------------------------
#
# The Brent bracket used to be hard-capped at θ = 500 (|τ| ≈ 0.992), so any
# tau_k in the gap (0.992, 1) — reachable from a TOML [[copulas]] block or an
# ICE fit in strong dependence — raised
#   ValueError: f(a) and f(b) must have different signs
# The fix widens the bracket to θ = ±700 and clamps (with a warning) above the
# largest numerically-reachable |τ| instead of crashing.

from pmcprg.copulas import CopulaFrank                               # noqa: E402
from pmcprg.copulas.archimedean.frank import (                      # noqa: E402
    _FRANK_TAU_MAX,
    find_theta_frank,
    kendall_tau_frank,
)


@pytest.mark.parametrize("tau", [0.993, 0.994, -0.993, -0.994])
def test_frank_strong_dependence_does_not_crash(tau):
    """CopulaFrank must build (no ValueError) for |τ| in the former crash gap."""
    cop = CopulaFrank(tau_k=tau)                       # used to raise ValueError
    # θ recovers the requested τ to high accuracy in the reachable range.
    assert math.isfinite(cop.theta)
    assert abs(kendall_tau_frank(cop.theta) - tau) < 1e-3
    # Densities/CDF stay finite and valid off-diagonal.
    assert math.isfinite(cop.pdf([0.4, 0.6]))
    cdf = cop.cdf([0.4, 0.6])
    assert math.isfinite(cdf) and 0.0 <= cdf <= 1.0


@pytest.mark.parametrize("tau", [0.9999, -0.9999])
def test_frank_above_reachable_tau_clamps_with_warning(tau, caplog):
    """|τ| beyond the largest numerically-reachable value clamps, not crashes."""
    with caplog.at_level(logging.WARNING, logger="pmcprg.copulas.archimedean.frank"):
        theta = find_theta_frank(tau)
    assert math.isfinite(theta)
    assert abs(theta) == pytest.approx(700.0)         # clamped to θ_max
    assert (theta > 0) == (tau > 0)                   # sign preserved
    assert any("clamping" in r.message for r in caplog.records)


@pytest.mark.parametrize("tau", [0.9999, -0.9999])
def test_frank_clamped_tau_reported_consistently(tau):
    """When θ is clamped, ``params['tau_k']`` is clipped to the τ the clamped
    θ actually models — traces / saved TOMLs / the GUI must not overstate the
    dependence (audit N-9)."""
    cop = CopulaFrank(tau_k=tau)
    assert abs(cop.params["tau_k"]) == pytest.approx(_FRANK_TAU_MAX)
    assert (cop.params["tau_k"] > 0) == (tau > 0)     # sign preserved
    # The reported τ matches the τ implied by the actual θ.
    assert abs(kendall_tau_frank(cop.theta) - cop.params["tau_k"]) < 1e-9


def test_frank_inversion_roundtrip_safe_range():
    """θ↔τ round-trip stays accurate across the safe dependence range."""
    for tau in (-0.9, -0.5, -0.1, 0.1, 0.5, 0.9):
        theta = find_theta_frank(tau)
        assert abs(kendall_tau_frank(theta) - tau) < 1e-4


# ---------------------------------------------------------------------------
# Regression: A12 scalar CDF must not use np.pow (audit finding N-3)
# ---------------------------------------------------------------------------
#
# A12.cdf used np.pow, an alias added only in NumPy 2.0, while the project
# floor is numpy>=1.24 — so the scalar CDF raised AttributeError on numpy 1.x.
# This guards the scalar path (cdf) agrees with the vectorised one (cdf_array).

def test_a12_scalar_cdf_finite_and_matches_array():
    cop = CopulaA12(tau_k=0.5)
    c = cop.cdf([0.3, 0.7])
    assert math.isfinite(c) and 0.0 <= c <= 1.0
    # Scalar and vectorised closed forms must agree.
    c_arr = float(cop.cdf_array(np.array([[0.3, 0.7]]))[0])
    assert c == pytest.approx(c_arr, abs=1e-9)
