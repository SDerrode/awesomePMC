"""Unit tests of the dependent multiplier sequence (audit FR-5).

Everything the module docstring of :mod:`pmcprg.diagnostics.dependent_multiplier`
claims about the *construction* is checked here on generated sequences rather
than argued on paper:

* mean 0 and **variance 1**, which is what the ``Σ_k w_k² = 1`` normalisation
  buys and the only thing that makes the smoothed variates usable as
  multipliers at all;
* the **autocorrelation** ``ρ(h) = Σ_k w_k w_{k+h}``, and that for
  ``kernel="bartlett"`` it is exactly the Bartlett kernel ``1 − |h|/ℓ``;
* the identity that lets ``ℓ`` reuse the package's HAC bandwidth:
  ``Var(Σ_i a_i ξ_i) = hac_variance(a, ℓ − 1)``;
* ``ℓ = 1`` reproduces the pre-FR-5 i.i.d. draw **bit for bit**.

Seeds are integer literals (never ``hash()``, which is salted per process —
see commit 5e56fda).
"""
from __future__ import annotations

import numpy as np
import pytest

from pmcprg.diagnostics.dependent_multiplier import (
    MULTIPLIER_KERNELS,
    auto_block_length,
    draw_multipliers,
    multiplier_autocorrelation,
    multiplier_weights,
    resolve_block_length,
)
from pmcprg.diagnostics.model_selection import hac_variance

ELLS = (1, 2, 5, 10, 20)


# ---------------------------------------------------------------------------
# Weights and their normalisation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kernel", MULTIPLIER_KERNELS)
@pytest.mark.parametrize("ell", ELLS)
def test_weights_are_normalised_to_unit_squared_sum(ell, kernel):
    w = multiplier_weights(ell, kernel)
    assert np.isclose(float(np.dot(w, w)), 1.0), (ell, kernel, w)
    assert np.all(w >= 0.0)
    assert w.size == (ell if kernel == "bartlett" else 2 * ell - 1)


@pytest.mark.parametrize("kernel", MULTIPLIER_KERNELS)
def test_block_length_one_is_the_identity_weight(kernel):
    assert multiplier_weights(1, kernel).tolist() == [1.0]


def test_bartlett_autocorrelation_is_the_bartlett_kernel():
    """``ρ(h) = 1 − h/ℓ`` exactly — the identity the HAC reuse rests on."""
    for ell in ELLS:
        rho = multiplier_autocorrelation(ell, "bartlett")
        expected = 1.0 - np.arange(ell) / ell
        assert np.allclose(rho, expected), (ell, rho, expected)


def test_parzen_autocorrelation_is_smooth_and_longer_ranged():
    rho = multiplier_autocorrelation(10, "parzen")
    assert rho.size == 19                       # 2*ell - 1, vs. ell for Bartlett
    assert np.isclose(rho[0], 1.0)
    assert np.all(np.diff(rho) <= 1e-12)        # monotone decreasing
    assert rho[-1] < 1e-6


@pytest.mark.parametrize("bad", [0, -3, 1.5, "auto"])
def test_weights_reject_a_bad_block_length(bad):
    with pytest.raises(ValueError):
        multiplier_weights(bad, "bartlett")


def test_weights_reject_an_unknown_kernel():
    with pytest.raises(ValueError):
        multiplier_weights(4, "epanechnikov")


# ---------------------------------------------------------------------------
# The generated sequence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kernel", MULTIPLIER_KERNELS)
@pytest.mark.parametrize("ell", ELLS)
@pytest.mark.parametrize("law", ["normal", "rademacher"])
def test_generated_sequence_has_mean_zero_and_unit_variance(ell, kernel, law):
    """The whole point of the ``Σ w² = 1`` normalisation, measured.

    40 000 independent columns of length 120: the s.e. of the variance
    estimate is ~``sqrt(2/(n B))`` ≈ 6e-4 for the normal law, so 0.01 is a
    generous band and a ``Σ|w|``-normalised sequence (variance ``ℓ`` at
    ``ℓ = 20``) would miss it by a factor of 20.
    """
    xi = draw_multipliers(120, 40_000, np.random.default_rng(20250917), law,
                          block_length=ell, kernel=kernel)
    assert xi.shape == (120, 40_000)
    assert abs(float(xi.mean())) < 0.01
    assert abs(float(xi.var()) - 1.0) < 0.01


@pytest.mark.parametrize("kernel", MULTIPLIER_KERNELS)
@pytest.mark.parametrize("ell", [2, 5, 10])
def test_generated_autocorrelation_matches_the_weights(ell, kernel):
    """``Corr(ξ_i, ξ_{i+h}) = Σ_k w_k w_{k+h}``, measured across replicates."""
    n, B = 120, 40_000
    xi = draw_multipliers(n, B, np.random.default_rng(4242), "normal",
                          block_length=ell, kernel=kernel)
    rho = multiplier_autocorrelation(ell, kernel)
    emp = np.array([float(np.mean(xi[: n - h] * xi[h:])) for h in range(rho.size)])
    assert np.allclose(emp, rho, atol=0.01), (ell, kernel, emp, rho)
    # ... and nothing beyond the kernel's reach: the sequence is finitely dependent.
    beyond = float(np.mean(xi[: n - rho.size] * xi[rho.size:]))
    assert abs(beyond) < 0.01, beyond


def test_conditional_variance_is_exactly_the_package_hac_variance():
    """``Var(Σ_i a_i ξ_i) = hac_variance(a, ℓ − 1)``.

    This is the identity that makes reusing
    :func:`pmcprg.diagnostics.model_selection.newey_west_bandwidth` for ``ℓ``
    legitimate rather than a loose analogy: with the Bartlett-autocorrelation
    multipliers, the dependent multiplier bootstrap reproduces *exactly* the
    Newey–West long-run variance the package already uses elsewhere.
    """
    n = 150
    a = np.random.default_rng(999).standard_normal(n)
    for ell in (1, 2, 5, 10):
        xi = draw_multipliers(n, 200_000, np.random.default_rng(31337), "normal",
                              block_length=ell)
        measured = float((a @ xi).var())
        exact = hac_variance(a, ell - 1)
        assert abs(measured - exact) < 0.03 * exact, (ell, measured, exact)


@pytest.mark.parametrize("law", ["normal", "rademacher"])
@pytest.mark.parametrize("kernel", MULTIPLIER_KERNELS)
def test_block_length_one_is_bit_identical_to_the_iid_draw(law, kernel):
    """Not "statistically equivalent": the very same array.

    ``bootstrap="multiplier"`` must stay untouched by FR-5, so
    ``block_length=1`` has to consume the RNG in exactly the pre-FR-5 way and
    return exactly the pre-FR-5 values.
    """
    got = draw_multipliers(64, 9, np.random.default_rng(7), law,
                           block_length=1, kernel=kernel)
    rng = np.random.default_rng(7)
    want = (rng.standard_normal(size=(64, 9)) if law == "normal"
            else rng.choice(np.array([-1.0, 1.0]), size=(64, 9)))
    assert np.array_equal(got, want)


def test_unknown_law_raises():
    with pytest.raises(ValueError):
        draw_multipliers(10, 2, np.random.default_rng(1), "cauchy")


# ---------------------------------------------------------------------------
# Block-length selection
# ---------------------------------------------------------------------------

def _gaussian_copula_chain(n, tau, seed):
    """A Markov chain whose transition copula is Gaussian; consecutive pairs.

    The minimal version of FR-5's setting: ``(u_t, u_{t+1})`` and
    ``(u_{t+1}, u_{t+2})`` share ``u_{t+1}``.
    """
    from pmcprg.copulas import CopulaGaussian
    cop = CopulaGaussian(tau_k=tau)
    rng = np.random.default_rng(seed)
    u = np.empty(n + 1)
    u[0] = rng.random()
    w = rng.random(n)
    for t in range(n):
        u[t + 1] = float(cop.inv_h_array(w[t:t + 1], u[t:t + 1])[0])
    return u[:-1], u[1:]


def test_auto_block_length_is_larger_on_a_chain_than_on_an_iid_sample():
    """The rule has to *see* the serial dependence, not just be well typed.

    Both samples have the same size and the same Kendall's τ; only the serial
    dependence differs. See the module docstring for why this is a ratio test
    and not an "``ℓ = 1`` on i.i.d. data" test: the Newey–West plug-in is
    known to be biased up when the true autocovariances vanish.
    """
    from pmcprg.copulas import CopulaGaussian
    n = 400
    chain = [auto_block_length(*_gaussian_copula_chain(n, 0.5, 300 + r))
             for r in range(12)]
    iid = [auto_block_length(*CopulaGaussian(tau_k=0.5).sample(n, seed=400 + r).T)
           for r in range(12)]
    assert np.median(chain) >= 2 * np.median(iid), (np.median(chain), np.median(iid))
    assert all(1 <= e <= n // 4 for e in chain + iid)


def test_auto_block_length_is_one_on_a_sample_too_short_to_select_on():
    assert auto_block_length(np.array([0.2, 0.7]), np.array([0.4, 0.9])) == 1


def test_auto_block_length_respects_the_cap():
    x, y = _gaussian_copula_chain(400, 0.7, 11)
    assert auto_block_length(x, y, max_block=3) <= 3


def test_auto_block_length_rejects_mismatched_inputs():
    with pytest.raises(ValueError):
        auto_block_length(np.zeros(5), np.zeros(6))
    with pytest.raises(ValueError):
        auto_block_length(np.zeros(5), np.zeros(5), kernel="triangular")


def test_parzen_auto_block_length_targets_the_same_dependence_length():
    """``ℓ_parzen`` is about half ``ℓ_bartlett``, because its autocorrelation
    reaches ``2ℓ − 2`` instead of ``ℓ``."""
    x, y = _gaussian_copula_chain(400, 0.6, 77)
    lb = auto_block_length(x, y, kernel="bartlett")
    lp = auto_block_length(x, y, kernel="parzen")
    assert lb >= 4                                   # the design is dependent
    assert lp == int(np.ceil((lb + 1) / 2.0)), (lb, lp)


@pytest.mark.parametrize("bad", ["0", "AUTO", 0, -1, 2.5, True])
def test_resolve_block_length_rejects_bad_values(bad):
    with pytest.raises(ValueError):
        resolve_block_length(bad, np.zeros(10), np.zeros(10))


def test_resolve_block_length_passes_an_int_through():
    assert resolve_block_length(7, np.zeros(10), np.zeros(10)) == 7
