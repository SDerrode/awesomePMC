"""Frank, Plackett and Student kernels beyond the acceptance grid (AUDIT_COPULES FR-1, RB-4, RB-9, RB-10).

``test_copula_limits.py`` checks C, log c and h on a 6 × 6 grid. These tests pin
what it does not reach: Frank's inverse h-function and τ(θ) series near
independence and at the θ cap, Plackett's kernels against the textbook
formulas in ``decimal`` near θ = 1 and at the table ends, the realised τ on
both sides of a clamp, and Student's inverse h-function, absence of density
floor and ν bounds.
"""
from __future__ import annotations

import math
from decimal import Decimal as D, localcontext

import numpy as np
import pytest

import scipy

from pmcprg.copulas import CopulaFrank, CopulaPlackett, CopulaStudent
from pmcprg.copulas import _base
from pmcprg.copulas.archimedean import frank as F
from pmcprg.copulas.explicit import plackett as P

_G = (1e-12, 1e-6, 0.01, 0.3, 0.5, 0.77, 1 - 1e-6, 1 - 1e-12)
_UV = np.array([(u, v) for u in _G for v in _G])


# --------------------------------------------------------------------------
# Frank
# --------------------------------------------------------------------------

def _frank_prec(th):
    """Digits: e^{−|θ|} must survive next to 1, plus 40 for the 10⁻¹² tails and the result."""
    return 60 + int(abs(th) / math.log(10.0))


def _frank_ref(u, v, th):
    """C, log c, h(v|u) from the textbook formulas (Nelsen 2006, (4.2.5))."""
    with localcontext() as ctx:
        ctx.prec = _frank_prec(th)
        U, V, T = D(u), D(v), D(th)
        e = lambda z: z.exp()                                          # noqa: E731
        C = -(1 + (e(-T * U) - 1) * (e(-T * V) - 1) / (e(-T) - 1)).ln() / T
        den = e(-T * U) + e(-T * V) - e(-T * (U + V)) - e(-T)
        logc = (T * (1 - e(-T)) * e(-T * (U + V)) / (den * den)).ln()
        h = e(-T * U) * (e(-T * V) - 1) / -den
        return float(C), float(logc), float(h)


def _frank_inv_ref(w, u, th):
    with localcontext() as ctx:
        ctx.prec = _frank_prec(th)
        W, U, T = D(w), D(u), D(th)
        y = ((-T).exp() - 1) / (1 + (1 - W) / W * (-T * U).exp())
        return float(-(1 + y).ln() / T)


@pytest.mark.parametrize("theta", [1e-11, -1e-11, 1e-6, -5e-3, 0.5, -38.0, 700.0, -700.0])
def test_frank_kernels_match_decimal_references(theta):
    ref = np.array([_frank_ref(u, v, theta) for u, v in _UV])
    u, v = _UV[:, 0], _UV[:, 1]
    C = F._frank_cdf(theta, u, v)
    ok = ref[:, 0] > 1e-290
    np.testing.assert_allclose(C[ok], ref[ok, 0], rtol=1e-12)
    np.testing.assert_allclose(F._frank_logpdf(theta, u, v), ref[:, 1], rtol=1e-13, atol=1e-12)
    np.testing.assert_allclose(F._frank_h(theta, v, u), ref[:, 2], rtol=1e-12, atol=1e-300)
    # the inverse: relative precision on v (on 1 − v only up to one ulp)
    W = np.array([(w, uu) for w in _G for uu in _G])
    inv = F._frank_inv_h(theta, W[:, 0], W[:, 1])
    inv_ref = np.array([_frank_inv_ref(w, uu, theta) for w, uu in W])
    np.testing.assert_allclose(inv, inv_ref, rtol=1e-12, atol=4e-16)


@pytest.mark.parametrize("tau", [1e-15, 1e-12, 1e-8, 1e-4, 0.0111, 0.0112, 0.5, -0.9])
def test_frank_tau_theta_round_trip_down_to_independence(tau):
    """RB-9: |τ| < 10⁻¹⁰ used to be rounded to θ = 0; τ = θ/9 + O(θ³)."""
    theta = F.find_theta_frank(tau)
    assert theta != 0.0
    assert F.kendall_tau_frank(theta) == pytest.approx(tau, rel=1e-12)
    if abs(tau) <= 1e-8:
        assert theta == pytest.approx(9.0 * tau, rel=1e-12)


def test_frank_first_order_branch_is_continuous():
    """At |θ| = 10⁻¹² the FGM expansion hands over to the exact kernels."""
    lo, hi = F._FRANK_THETA_SERIES, F._FRANK_THETA_SERIES * (1 + 1e-9)
    u, v = _UV[:, 0], _UV[:, 1]
    np.testing.assert_allclose(F._frank_cdf(lo, u, v), F._frank_cdf(hi, u, v), rtol=1e-14)
    np.testing.assert_allclose(F._frank_h(lo, v, u), F._frank_h(hi, v, u), rtol=1e-14)
    np.testing.assert_allclose(F._frank_logpdf(lo, u, v), F._frank_logpdf(hi, u, v), rtol=0, atol=1e-20)
    np.testing.assert_allclose(F._frank_inv_h(lo, v, u), F._frank_inv_h(hi, v, u), rtol=1e-14)


def test_frank_scalar_paths_have_no_sentinel():
    """RB-9: no EPS density, no w for a failed inverse, no 0/1 step for h."""
    cop = CopulaFrank(tau_k=1e-12)
    assert cop.pdf([0.5, 0.5]) == pytest.approx(1.0, abs=1e-10)
    assert np.isnan(cop.inv_h_array(np.array([np.nan]), np.array([0.5]))[0])
    strong = CopulaFrank(tau_k=0.99)
    assert 0.0 < strong.pdf([1e-12, 1 - 1e-12]) < 1e-100          # exp(log c), not EPS


# --------------------------------------------------------------------------
# Plackett
# --------------------------------------------------------------------------

def _plackett_ref(u, v, th):
    with localcontext() as ctx:
        ctx.prec = 80
        U, V, T = D(u), D(v), D(th)
        if T == 1:
            return float(U * V), 0.0, float(V)
        S = 1 + (T - 1) * (U + V)
        Dl = S * S - 4 * T * (T - 1) * U * V
        C = (S - Dl.sqrt()) / (2 * (T - 1))
        logc = (T * (1 + (T - 1) * (U + V - 2 * U * V)) / (Dl * Dl.sqrt())).ln()
        h = (1 - (S - 2 * T * V) / Dl.sqrt()) / 2
        return float(C), float(logc), float(h)


@pytest.mark.parametrize("theta", [1e-6, 0.02, 0.5, 1 - 4.5e-8, 1.0, 1 + 1e-12, 1 + 4.5e-4, 50.0, 1e6])
def test_plackett_kernels_match_decimal_references(theta):
    ref = np.array([_plackett_ref(u, v, theta) for u, v in _UV])
    u, v = _UV[:, 0], _UV[:, 1]
    np.testing.assert_allclose(P._plackett_cdf(theta, u, v), ref[:, 0], rtol=1e-12)
    np.testing.assert_allclose(P._plackett_logpdf(theta, u, v), ref[:, 1], rtol=0, atol=1e-12)
    np.testing.assert_allclose(P._plackett_h(v, u, theta), ref[:, 2], rtol=1e-12)


@pytest.mark.parametrize("tau", [0.999, -0.999, 0.9999999, -0.9999999])
def test_plackett_clamped_copula_stores_the_realised_tau(tau):
    """RB-10 on both sides of the table."""
    cop = CopulaPlackett(tau_k=tau)
    assert cop.params["tau_k"] == pytest.approx(P._plackett_tau_from_theta(cop.theta), abs=1e-12)
    assert abs(cop.params["tau_k"]) < abs(tau)


def test_plackett_tau_table_unchanged_by_the_new_h():
    """The τ table (quadrature of Hoeffding's identity) moved by < 10⁻¹² with the rationalised h."""
    taus, _ = P._plackett_tau_table()
    assert np.all(np.diff(taus) > 0)
    assert taus[-1] == pytest.approx(0.9935245713, abs=1e-9)
    assert taus[0] == pytest.approx(-taus[-1], abs=1e-9)


# --------------------------------------------------------------------------
# Student
# --------------------------------------------------------------------------

def test_student_density_has_no_floor_and_arrays_agree():
    """RB-4: at ν = 30, τ = 0.95 the old density was floored at EPS (log c = −36.04)."""
    cop = CopulaStudent(tau_k=0.95, df=30.0)
    uv = np.array([[1e-12, 1 - 1e-12], [1e-6, 0.3], [0.5, 0.5]])
    logc = cop.logpdf_array(uv)
    assert logc[0] < -70.0 and logc[1] < -36.1
    np.testing.assert_allclose(cop.pdf_array(uv), np.exp(logc), rtol=1e-15)
    assert cop.pdf(list(uv[0])) == pytest.approx(math.exp(logc[0]), rel=1e-14)


def _scipy_at_least(major: int, minor: int) -> bool:
    got = tuple(int(x) for x in scipy.__version__.split(".")[:2])
    return got >= (major, minor)


@pytest.mark.parametrize("tau,df", [(-0.9, 2.5), (0.3, 4.0), (0.95, 30.0)])
def test_student_inverse_h_is_closed_form_and_accurate_in_the_tails(tau, df):
    cop = CopulaStudent(tau_k=tau, df=df)
    G = np.array([1e-12, 1e-6, 0.01, 0.3, 0.5])
    U, V = (a.ravel() for a in np.meshgrid(np.concatenate([G, 1 - G[:-1]]), G))
    W = cop._h_array(V, U)
    ok = (W > 1e-300) & (W < 0.5)
    back = cop.inv_h_array(W[ok], U[ok])
    # The round trip goes through scipy's Student-t quantiles: 1e-9 from scipy 1.17,
    # 5.6e-9 on 1.13–1.16 and 5.5e-6 on 1.10–1.12 (measured, oldest supported 1.10).
    rtol = 1e-9 if _scipy_at_least(1, 17) else 1e-5
    np.testing.assert_allclose(back, V[ok], rtol=rtol)
    assert cop.inv_h(float(W[ok][0]), float(U[ok][0])) == pytest.approx(back[0], rel=1e-15)


def test_student_fit_reads_the_registry_bounds_for_df(monkeypatch):
    """RB-4: ν is bounded (and started) by EXTRA_PARAM_BOUNDS_BY_PARAM['df'], read at call time."""
    data = CopulaStudent(tau_k=0.4, df=60.0).sample(400, seed=2)
    monkeypatch.setitem(_base.EXTRA_PARAM_BOUNDS_BY_PARAM, "df", (2.5, 6.0, 3.0))
    r = CopulaStudent.fit(data)
    assert 2.5 <= r.copula.df <= 6.0
    assert math.isfinite(r.log_likelihood)
