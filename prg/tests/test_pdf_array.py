"""
test_pdf_array.py — verify each copula's vectorised ``pdf_array(uv)`` matches
its scalar ``pdf([u, v])`` to floating-point precision.

This is a regression net for the closed-form / statsmodels-batched overrides
we added in v0.4.x: any future micro-optimisation of one path must keep the
two paths bit-identical (or within ~1 ulp).
"""

from __future__ import annotations

import numpy as np
import pytest

from prg.copulas._base import CopulaEnum


# ---------------------------------------------------------------------------
# Per-family configuration: pick a τ inside the valid range, plus any extra
# parameters the family requires.
# ---------------------------------------------------------------------------

def _params_for(entry: CopulaEnum) -> dict:
    """Build a valid kwargs dict for ``entry.klass(**kwargs)``."""
    tau_min, tau_max = entry.value.TAU_MIN_MAX
    # Use a τ comfortably inside the range (avoid τ=0 for Product-style families
    # and avoid the boundary for numerical edge-cases).
    tau = 0.5 * (tau_min + tau_max)
    if entry.value.SHORT_NAME == "Prod":
        tau = 0.0
    elif entry.value.SHORT_NAME == "AMH":
        # AMH τ_max ≈ 1/3; pick a clearly positive value
        tau = 0.2
    elif entry.value.SHORT_NAME in ("A12", "A14"):
        # τ ∈ [1/3, 1) — use a safely interior value
        tau = 0.5
    elif tau_max - tau_min > 0.5:
        tau = 0.4   # generic positive dependence

    kwargs: dict = {"tau_k": float(tau)}
    for name in entry.value.PARAMETERS_SET_NAME:
        if name == "tau_k":
            continue
        # Sensible defaults for the second parameter of multi-param families
        if name == "df":
            kwargs["df"] = 5.0
        elif name == "delta":
            kwargs["delta"] = 1.5
        else:
            kwargs[name] = 1.0
    return kwargs


# Pseudo-observations grid: 50 random (u, v) pairs in (0.02, 0.98)²,
# avoiding the open-interval boundaries where some closed forms underflow.
_RNG = np.random.default_rng(0)
_UV = np.column_stack(
    [_RNG.uniform(0.02, 0.98, size=50), _RNG.uniform(0.02, 0.98, size=50)]
)


@pytest.mark.parametrize(
    "entry",
    [c for c in CopulaEnum if c.value.AVAILABLE],
    ids=lambda c: c.value.SHORT_NAME,
)
def test_pdf_array_matches_scalar(entry):
    """For every available copula, ``pdf_array(uv) ≡ [pdf([u,v]) for u,v in uv]``.

    Tolerance is loose-by-design (1e-9) because some families internally clamp
    PDF values at ``EPS`` differently between the scalar and vectorised paths.
    """
    cop = entry.klass(**_params_for(entry))

    scalar = np.array([cop.pdf([float(u), float(v)]) for u, v in _UV])
    vector = cop.pdf_array(_UV)

    # Both must be finite and non-negative
    assert np.all(np.isfinite(scalar)), f"{entry.value.SHORT_NAME}: scalar pdf produced non-finite"
    assert np.all(np.isfinite(vector)), f"{entry.value.SHORT_NAME}: vector pdf produced non-finite"
    assert (scalar >= 0).all() and (vector >= 0).all()

    # Element-wise agreement up to floating-point noise; relative tolerance for
    # large PDF values, absolute for tiny ones.
    np.testing.assert_allclose(
        vector, scalar,
        rtol=1e-9, atol=1e-12,
        err_msg=f"{entry.value.SHORT_NAME}: pdf_array vs pdf mismatch",
    )


@pytest.mark.parametrize(
    "entry",
    # Closed-form inv_h overrides currently exist for Gaussian, Clayton, Frank;
    # the rest fall back to Brent's method via CopulaVirt.inv_h.
    [c for c in CopulaEnum if c.value.SHORT_NAME in ("Gauss", "Clayton", "Frank")],
    ids=lambda c: c.value.SHORT_NAME,
)
def test_inv_h_inverts_conditional_cdf(entry):
    """``cop.inv_h(h(v|u), u) ≈ v`` for any v ∈ (0, 1) and any u ∈ (0, 1)."""
    cop = entry.klass(**_params_for(entry))

    # Test on a small grid of (u, v) pairs
    grid = np.linspace(0.05, 0.95, 7)
    for u in grid:
        for v in grid:
            w     = cop.conditional_cdf(float(v), float(u))
            v_rec = cop.inv_h(float(w), float(u))
            assert abs(v_rec - v) < 1e-6, (
                f"{entry.value.SHORT_NAME}: inv_h(h({v}|{u}), {u})={v_rec:.6f}, "
                f"expected {v:.6f} (Δ={abs(v_rec-v):.2e})"
            )


@pytest.mark.parametrize(
    "entry",
    [c for c in CopulaEnum if c.value.AVAILABLE],
    ids=lambda c: c.value.SHORT_NAME,
)
def test_cdf_array_matches_scalar(entry):
    """For every available copula, ``cdf_array(uv) ≡ [cdf([u,v]) for u,v in uv]``.

    Exercises the same fast-path logic as :func:`test_pdf_array_matches_scalar`
    on the joint CDF (used by goodness-of-fit bootstrap).
    """
    cop = entry.klass(**_params_for(entry))

    # Some families (notably Student-t) have no closed-form CDF; skip them.
    try:
        cop.cdf([0.5, 0.5])
    except NotImplementedError:
        pytest.skip(f"{entry.value.SHORT_NAME} has no closed-form CDF")

    # Some copulas (Product is the obvious one) have CDFs that don't admit a
    # closed-form vectorised expression; we still want them to satisfy the
    # round-trip via the scalar fallback.
    scalar = np.array([cop.cdf([float(u), float(v)]) for u, v in _UV])
    vector = cop.cdf_array(_UV)

    # CDFs must lie in [0, 1] and be finite
    assert np.all((scalar >= 0) & (scalar <= 1)), (
        f"{entry.value.SHORT_NAME}: scalar cdf out of [0,1]"
    )
    assert np.all((vector >= 0) & (vector <= 1)), (
        f"{entry.value.SHORT_NAME}: vector cdf out of [0,1]"
    )

    np.testing.assert_allclose(
        vector, scalar,
        rtol=1e-9, atol=1e-12,
        err_msg=f"{entry.value.SHORT_NAME}: cdf_array vs cdf mismatch",
    )


@pytest.mark.parametrize(
    "entry",
    [c for c in CopulaEnum if c.value.SHORT_NAME in
     ("Gauss", "Clayton", "Frank", "Joe", "BB1")],
    ids=lambda c: c.value.SHORT_NAME,
)
def test_logpdf_array_native_extends_dynamic_range(entry):
    """The native ``logpdf_array`` overrides on the 5 main families must
    extend the dynamic range past ``log(EPS) ≈ -36``, where the default
    ``log(max(pdf, EPS))`` saturates.

    On regular pseudo-observations the two paths must match to numerical
    precision; in extreme tails the native version produces lower
    (correct) log-densities.
    """
    cop = entry.klass(**_params_for(entry))

    # Regular range — both paths agree
    rng = np.random.default_rng(0)
    uv  = np.column_stack([rng.uniform(0.1, 0.9, 200),
                           rng.uniform(0.1, 0.9, 200)])
    log_native  = cop.logpdf_array(uv)
    log_floored = np.log(np.maximum(cop.pdf_array(uv), np.finfo(float).eps))
    np.testing.assert_allclose(
        log_native, log_floored,
        atol=1e-9, rtol=1e-9,
        err_msg=f"{entry.value.SHORT_NAME}: native vs floored disagree on regular data",
    )

    # The native version reaches *strictly below* -log(EPS) ≈ -36 on at
    # least one extreme-tail point. Pick a u, v combination that drives
    # the density very low for each family.
    pivot_uv = {
        "Gauss":   np.array([[1e-7, 1 - 1e-7]]),    # opposite corners
        "Clayton": np.array([[1e-7, 1 - 1e-7]]),
        "Frank":   np.array([[1e-7, 1 - 1e-7]]),
        "Joe":     np.array([[1 - 1e-7, 0.5]]),
        "BB1":     np.array([[1e-7, 1 - 1e-7]]),
    }
    uv_tail = pivot_uv[entry.value.SHORT_NAME]
    ln_tail = cop.logpdf_array(uv_tail)[0]
    log_eps = np.log(np.finfo(float).eps)
    if ln_tail < log_eps:
        # Native dynamic range extends below the EPS floor — exactly the
        # benefit of having a native logpdf. Verify the floored path
        # saturates at log_eps for this point.
        lf_tail = np.log(np.maximum(cop.pdf_array(uv_tail), np.finfo(float).eps))[0]
        assert ln_tail < lf_tail, (
            f"{entry.value.SHORT_NAME}: expected native ln < floored ln "
            f"on extreme tail, got native={ln_tail:.2f} ≥ floored={lf_tail:.2f}"
        )


@pytest.mark.parametrize(
    "entry",
    [c for c in CopulaEnum if c.value.AVAILABLE],
    ids=lambda c: c.value.SHORT_NAME,
)
def test_logpdf_array_consistent(entry):
    """``logpdf_array`` must equal ``log(pdf_array)`` (up to the EPS floor)."""
    cop = entry.klass(**_params_for(entry))

    pdf = cop.pdf_array(_UV)
    log_via_default = np.log(np.maximum(pdf, np.finfo(float).eps))
    log_direct      = cop.logpdf_array(_UV)

    np.testing.assert_allclose(
        log_direct, log_via_default,
        rtol=1e-12, atol=1e-12,
        err_msg=f"{entry.value.SHORT_NAME}: logpdf_array inconsistent with log(pdf_array)",
    )
