"""``CopulaVirt.transposed()`` for every registered family, and ``BivariateLaw``'s right margin (audit FR-11).

The h-functions of the package condition on the first argument; the parity
tests with pyvinecopulib, VineCopula and R ``copula``
(``test_parity_interior.py``) showed that nothing in the package computed
∂C/∂v, and that ``BivariateLaw`` conditioned on its right margin by swapping
the copula's arguments — right for an exchangeable copula only. For the
90°/270° rotations and the Tawn models it returned the conditional law of
the other side. Conditioning on the second argument is now conditioning on
the first of the transpose Cᵀ(u, v) = C(v, u), which this module checks
for all 41 families: its density and CDF are the copula's at the swapped
point, its h-function is ∂C/∂v, the ``exchangeable`` flag is honest, and
``BivariateLaw`` uses it.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from pmcprg.copulas import BivariateLaw, CopulaEnum
from pmcprg.numerics import EPS, ONE_MINUS_EPS

POINTS = np.array([[0.2, 0.7], [0.7, 0.2], [0.35, 0.9], [0.9, 0.35], [0.5, 0.1], [0.1, 0.5],
                   [0.6, 0.6], [0.05, 0.8], [0.8, 0.05], [0.97, 0.4], [0.4, 0.97]])
SWAPPED = POINTS[:, ::-1].copy()

# Asymmetric weights for the Tawn models, at a τ they reach without repair.
EXTRA = {"TAWN1": {"psi": 0.5}, "TAWN2": {"psi": 0.5}, "TAWN3": {"psi_u": 0.4, "psi_v": 0.8}}
TAU = {"TAWN1": 0.3, "TAWN2": 0.3, "TAWN3": 0.3}
NOT_EXCHANGEABLE = {e.name for e in CopulaEnum.available()
                    if e.name.endswith(("90", "270")) or e.name.startswith("TAWN")}
ENTRIES = list(CopulaEnum.available())


def _build(entry):
    """The family at mid-range τ — positive dependence when the range allows it."""
    lo, hi = entry.constructible_tau_range()
    tau = 0.5 * hi if lo < 0.0 < hi else 0.5 * (lo + hi)
    tau = TAU.get(entry.name, tau)
    params = {"tau_k": tau, **EXTRA.get(entry.name, {})}
    built = entry.klass.constructible_params(params)
    assert built is params, entry          # no repair: the parameters above are the ones tested
    return entry.klass(**params)


def _has_cdf(cop) -> bool:
    try:
        cop.cdf([0.5, 0.5])
    except NotImplementedError:          # Student
        return False
    return True


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda e: e.name)
def test_transpose_is_the_copula_of_the_swapped_pair(entry):
    cop = _build(entry)
    tr = cop.transposed()
    # Measured: 1.8e-15 (Frank, whose kernel is not evaluated symmetrically
    # in its two coordinates), bit for bit for most families.
    np.testing.assert_allclose(tr.pdf_array(POINTS), cop.pdf_array(SWAPPED), rtol=1e-14, atol=0.0)
    if _has_cdf(cop):
        # Measured: 1.4e-16 (Frank).
        np.testing.assert_allclose(tr.cdf_array(POINTS), cop.cdf_array(SWAPPED), rtol=0.0, atol=1e-15)
    back = tr.transposed()
    assert type(back) is type(cop) and back.params == cop.params


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda e: e.name)
def test_exchangeable_flag_is_honest(entry):
    cop = _build(entry)
    pdf = cop.pdf_array(POINTS)
    asym = np.max(np.abs(pdf - cop.pdf_array(SWAPPED)) / pdf)
    if entry.name in NOT_EXCHANGEABLE:
        # Measured: at least 0.85 (the BB1 rotations) at these parameters.
        assert not cop.exchangeable and asym > 0.5
        assert cop.transposed() is not cop
    else:
        # Measured: 1.8e-15 (Frank); 0 for most families.
        assert cop.exchangeable and asym <= 1e-14
        assert cop.transposed() is cop


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda e: e.name)
def test_transposed_h_is_dC_dv_and_its_inverse_inverts_it(entry):
    cop = _build(entry)
    tr = cop.transposed()
    h2 = np.array([tr.conditional_cdf(a, b) for a, b in POINTS])
    if _has_cdf(cop):
        d = 1e-6
        fd = np.array([(cop.cdf([a, b + d]) - cop.cdf([a, b - d])) / (2 * d) for a, b in POINTS])
        # Central differences: O(δ²) + ε/δ; measured 1.0e-10 (BB7).
        np.testing.assert_allclose(h2, fd, rtol=0.0, atol=1e-9)
    x = tr.inv_h_array(POINTS[:, 0].copy(), POINTS[:, 1].copy())
    back = np.array([tr.conditional_cdf(xx, b) for xx, b in zip(x, POINTS[:, 1])])
    # Measured: up to 8.3e-9 (A12) for the families inverted by the base
    # class's Brent (xtol 1e-8), ≤ 2.7e-14 for the closed forms and kernel
    # inverses (rotated BB1).
    np.testing.assert_allclose(back, POINTS[:, 0], rtol=0.0, atol=2e-8)


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda e: e.name)
def test_bivariate_law_right_margin_goes_through_the_transpose(entry):
    """P(Y_left ≤ a | Y_right = b) is ∂C/∂v, and its sampler inverts it.

    Uniform margins, so the law's values are the copula's. Before the fix the
    right margin read h1(b, a) = ``cop.conditional_cdf(a, b)``: off by up to
    0.46 for the rotations on the parity grid.
    """
    cop = _build(entry)
    tr = cop.transposed()
    law = BivariateLaw(cop, (stats.uniform,), (stats.uniform,))
    for a, b in POINTS:
        assert law.conditional_cdf(a, b, which="right") == tr.conditional_cdf(a, b)
        assert law.conditional_cdf(b, a, which="left") == cop.conditional_cdf(b, a)
        # both conditional densities are the joint density c(u_left, u_right)
        assert law.conditional_pdf(a, b, which="right") == pytest.approx(cop.pdf([a, b]), rel=1e-14)
        assert law.conditional_pdf(b, a, which="left") == pytest.approx(cop.pdf([a, b]), rel=1e-14)
    law.set_seed(7)
    drawn = law.sample_conditional(0.35, which="right", n=16)
    ws = np.random.default_rng(7).uniform(EPS, ONE_MINUS_EPS, 16)
    np.testing.assert_array_equal(drawn, tr.inv_h_array(ws, np.full(16, 0.35)))
    law.set_seed(7)
    one = law._sample_left_given_right(0.35)
    w = np.random.default_rng(7).uniform(EPS, ONE_MINUS_EPS)
    assert one == tr.inv_h(float(w), 0.35)
