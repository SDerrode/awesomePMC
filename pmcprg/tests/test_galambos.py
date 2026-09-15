"""Galambos copula (FR-9 pilot) — construction, round-trips, τ(θ), fit.

Complements ``test_copula_limits.py`` (which added Galambos to its own
grid and high-precision reference file) with the checks that are specific
to this family: boundary rejection, the h/h⁻¹ round-trip, τ(θ) against a
simulated Kendall's τ, sampling's marginal uniformity, ``fit`` recovery,
and that the reference file addition covers exactly the registered cases.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import kendalltau, kstest

from pmcprg.copulas import CopulaEnum, CopulaGalambos
from pmcprg.copulas.extreme_value.galambos import (
    _galambos_tau_from_theta,
    _galambos_theta_from_tau,
    _THETA_HI,
)
from pmcprg.exceptions import CopulaParameterError
from pmcprg.numerics import EPS

GALAMBOS = CopulaEnum.GALAMBOS


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------

def test_registered_in_copula_enum():
    assert GALAMBOS.value.SHORT_NAME == "Galambos"
    assert GALAMBOS.value.CLASS_NAME == "CopulaGalambos"
    assert GALAMBOS.klass is CopulaGalambos
    assert GALAMBOS.value.AVAILABLE
    assert GALAMBOS in CopulaEnum.available()


def test_tau_range_is_one_sided():
    """Extreme-value copulas have no negative dependence (module docstring)."""
    lo, hi = GALAMBOS.value.TAU_MIN_MAX
    assert lo == pytest.approx(EPS)
    assert hi == 1.0


# --------------------------------------------------------------------------
# Construction / boundary rejection
# --------------------------------------------------------------------------

@pytest.mark.parametrize("tau", [0.0, -0.1, -1.0])
def test_non_positive_tau_is_refused(tau):
    with pytest.raises(CopulaParameterError):
        CopulaGalambos(tau_k=tau)


def test_tau_at_or_beyond_one_is_refused():
    with pytest.raises(CopulaParameterError):
        CopulaGalambos(tau_k=1.0)
    with pytest.raises(CopulaParameterError):
        CopulaGalambos(tau_k=1.5)


def test_a_tau_beyond_the_reachable_bound_is_clamped_and_stored():
    """Same RB-10/F1/G2 convention as Frank and Plackett: θ is capped at
    ``_THETA_HI``, so a τ beyond ``reachable_tau_bounds()[1]`` (a registered
    bound the base class's ``_check_tau_k`` would otherwise accept, since
    < 1.0) builds at the cap and stores the τ it realises, not the one
    asked for — see ``test_other_families_are_unchanged`` in
    ``test_frank_reachable_tau.py``, which excludes Galambos for this
    reason, same as Plackett."""
    lo, hi = GALAMBOS.klass.reachable_tau_bounds()
    assert lo == pytest.approx(EPS)
    assert 0.999 < hi < 1.0
    assert GALAMBOS.reachable_tau(0.999999) == pytest.approx(hi)
    assert GALAMBOS.reachable_tau(0.5) == 0.5
    assert GALAMBOS.constructible_tau_range()[1] <= hi

    cop = CopulaGalambos(tau_k=0.999999)
    assert cop.theta == _THETA_HI
    assert cop.params["tau_k"] == pytest.approx(hi, rel=1e-9)
    assert cop.params["tau_k"] < 0.999999


# --------------------------------------------------------------------------
# τ(θ) — closed-form claims re-derived and cross-checked (module docstring)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("theta,tau_expected", [
    # 30-digit mpmath, Genest & MacKay (1986) integral of A''; see the
    # module docstring for the Gumbel cross-check of the integral itself.
    (1.0, 0.4183991523123016),
    (2.0, 0.6311588944292703),
    (5.0, 0.8248473436972861),
    (20.0, 0.9517135059779276),
])
def test_tau_from_theta_matches_high_precision_quadrature(theta, tau_expected):
    assert _galambos_tau_from_theta(theta) == pytest.approx(tau_expected, rel=1e-6)


def test_tau_theta_round_trip():
    for tau in (1e-8, 1e-4, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95):
        theta = _galambos_theta_from_tau(tau)
        assert _galambos_tau_from_theta(theta) == pytest.approx(tau, rel=1e-6, abs=1e-9)


def test_upper_tail_dependence_matches_the_pickands_identity():
    """λ_U = 2(1 − A(1/2)) for any EV copula (Nelsen 2006, Thm 5.7.3);
    A(1/2) = 1 − 2^{−1−1/θ} for Galambos ⇒ λ_U = 2^{−1/θ} (module docstring)."""
    for theta in (0.2, 1.0, 3.0, 10.0):
        cop = CopulaGalambos(tau_k=_galambos_tau_from_theta(theta))
        lam_l, lam_u = cop.tail_dependence()
        assert lam_l == 0.0
        assert lam_u == pytest.approx(2.0 ** (-1.0 / theta))


def test_tau_increases_with_theta():
    """Independence (θ → 0) to comonotone (θ → ∞); never negative."""
    thetas = [1e-3, 1e-2, 0.1, 1.0, 10.0, 1e3, 1e6]
    taus = [_galambos_tau_from_theta(t) for t in thetas]
    assert all(t >= 0.0 for t in taus)
    assert all(a <= b for a, b in zip(taus, taus[1:]))
    assert taus[-1] > 0.999


# --------------------------------------------------------------------------
# pdf / cdf / h / h⁻¹
# --------------------------------------------------------------------------

@pytest.mark.parametrize("tau", [1e-4, 0.3, 0.7, 0.95])
def test_h_inv_h_round_trip(tau):
    """``inv_h(h(v|u), u) == v`` — checked on the *w = h(v|u)* scale, not on
    v directly: at strong dependence (τ = 0.95) h(·|u) is nearly flat over
    part of its range, so a tiny residual in w (Brent's ``xtol``) can widen
    into a visibly different v without either value being wrong — the
    invariant Brent actually guarantees is on w."""
    cop = CopulaGalambos(tau_k=tau)
    rng = np.random.default_rng(0)
    us = rng.uniform(0.01, 0.99, 40)
    vs = rng.uniform(0.01, 0.99, 40)
    for u, v in zip(us, vs):
        w = cop.conditional_cdf(v, u)
        v_back = cop.inv_h(w, u)
        w_back = cop.conditional_cdf(v_back, u)
        assert w_back == pytest.approx(w, abs=1e-6)


@pytest.mark.parametrize("tau", [1e-4, 0.3, 0.7, 0.95])
def test_pdf_array_is_exp_of_logpdf_array(tau):
    cop = CopulaGalambos(tau_k=tau)
    rng = np.random.default_rng(1)
    uv = rng.uniform(0.001, 0.999, (200, 2))
    logc = cop.logpdf_array(uv)
    finite = np.isfinite(logc) & (logc > -700) & (logc < 700)
    np.testing.assert_allclose(cop.pdf_array(uv)[finite], np.exp(logc[finite]), rtol=1e-12)


@pytest.mark.parametrize("tau", [1e-4, 0.3, 0.7, 0.95])
def test_cdf_is_a_valid_copula_cdf(tau):
    """C(u,0)=C(0,v)=0-ish, C(u,1)=u, C(1,v)=v, and C nondecreasing in each arg."""
    cop = CopulaGalambos(tau_k=tau)
    for u in (0.2, 0.5, 0.8):
        assert cop.cdf([u, 1.0 - EPS]) == pytest.approx(u, abs=1e-6)
        assert cop.cdf([1.0 - EPS, u]) == pytest.approx(u, abs=1e-6)
    grid = np.linspace(0.05, 0.95, 10)
    for u in grid:
        vals = [cop.cdf([u, v]) for v in grid]
        assert all(a <= b + 1e-12 for a, b in zip(vals, vals[1:]))


# --------------------------------------------------------------------------
# Sampling: marginal uniformity and realised τ (Monte-Carlo, module docstring)
# --------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("tau", [0.2, 0.5, 0.8])
def test_sample_has_uniform_margins_and_the_right_tau(tau):
    cop = CopulaGalambos(tau_k=tau)
    uv = cop.sample(n=20000, seed=42)
    assert kstest(uv[:, 0], "uniform").pvalue > 0.001
    assert kstest(uv[:, 1], "uniform").pvalue > 0.001
    tau_hat = kendalltau(uv[:, 0], uv[:, 1]).statistic
    # Monte-Carlo s.e. of Kendall's tau at n=20000 is ~0.006; allow 5 s.e.
    assert abs(tau_hat - tau) < 0.03


# --------------------------------------------------------------------------
# fit
# --------------------------------------------------------------------------

@pytest.mark.parametrize("tau_true", [0.2, 0.5, 0.8])
@pytest.mark.parametrize("method", ["tau", "mle"])
def test_fit_recovers_tau_from_simulated_data(tau_true, method):
    cop = CopulaGalambos(tau_k=tau_true)
    uv = cop.sample(n=3000, seed=7)
    r = CopulaGalambos.fit(uv, method=method)
    assert r.tau_k == pytest.approx(tau_true, abs=0.06)
    assert r.copula.params["tau_k"] == r.tau_k


def test_fit_best_includes_galambos():
    from pmcprg.copulas._base import CopulaVirt
    cop = CopulaGalambos(tau_k=0.6)
    uv = cop.sample(n=800, seed=3)
    results = CopulaVirt.fit_best(uv, families=[CopulaGalambos], method="tau")
    assert len(results) == 1
    assert results[0].copula.__class__ is CopulaGalambos


# --------------------------------------------------------------------------
# High-precision reference file coverage (see test_copula_limits.py)
# --------------------------------------------------------------------------

def test_galambos_cases_are_covered_by_the_reference_file():
    from test_copula_limits import (
        _TAIL_TAUS, _INDEP_TAUS, _file_table, _required_param_keys, _match_file_key,
    )
    assert "Galambos" in _TAIL_TAUS
    assert "Galambos" in _INDEP_TAUS
    table = _file_table()
    need = {k for k in _required_param_keys() if k[0] == "Galambos"}
    assert need, "no Galambos cases registered in test_copula_limits.py"
    for key in need:
        assert _match_file_key(key, table) is not None, f"{key} missing from the reference file"
