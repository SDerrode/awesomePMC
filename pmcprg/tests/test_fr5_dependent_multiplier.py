"""Size and power of the dependent multiplier bootstrap on genuinely serially
dependent pseudo-observations (audit FR-5).

The point of FR-5 is a claim about *data that are not i.i.d.*, so the study
below does not use i.i.d. samples: every replicate simulates a whole series
from a two-state PMC with :func:`pmcprg.pmc.simulate` and extracts the
pseudo-observations of **one state pair**, exactly the sample a per-pair
copula diagnostic is run on. Consecutive kept pairs share an observation,
which is the dependence the finding is about.

The design
----------
* Two states, both margins ``N(0, 1)``, transition matrix with persistence
  0.98 (so state-0 runs are long and the extracted pair sample is nearly
  contiguous in time — the dependence survives the extraction).
* Null: all four transition copulas Gaussian at ``τ = 0.7``. Gaussian is
  radially symmetric, so ``H0`` of :func:`radial_symmetry_test` is exactly
  true, and the serial dependence is strong.
* Alternative: the ``(0, 0)`` copula is Clayton at ``τ = 0.7`` (not radially
  symmetric), the other three Gaussian.
* Pairs are labelled with the **true** hidden states, not an E-step: this
  isolates the serial dependence from the assignment noise that
  ``test_model_selection_mc.py``'s genuine-ICE study measures separately.
* Matched i.i.d. controls (a Gaussian / Clayton copula sample at the same
  ``n`` and ``τ``) separate "the bootstrap is wrong under dependence" from
  "the statistic is off anyway".

What was measured (R = 500 replicates, B = 200, α = 5 %, ±3.3 binomial s.d.
≈ ±3.2 points; the bands asserted below are those)
----------------------------------------------------------------------------
Size, ``radial_symmetry_test``, n ≈ 392 kept pairs out of 799:

    multipliers          PMC state pair      i.i.d. control
    i.i.d. (ℓ = 1)          19.8 %               3.2 %
    dependent, ℓ = 2        12.8 %               4.0 %
    dependent, ℓ = 5         6.6 %               4.8 %
    dependent, ℓ = 10        3.4 %               4.6 %
    dependent, ℓ = 20        3.0 %               4.2 %
    dependent, ℓ auto        3.2 %  (ℓ̄ = 14.1)   4.6 %  (ℓ̄ = 6.1)
    dependent, auto/Parzen   4.0 %  (ℓ̄ =  7.8)   4.6 %  (ℓ̄ = 3.8)

That is FR-5's claim, quantified: on serially dependent pseudo-observations
the i.i.d. multiplier bootstrap rejects a **true** null 19.8 % of the time at
a nominal 5 % — four times the level — while on the matched i.i.d. control
the same code is at 3.2 %. The dependent version brings the chain back to
nominal and leaves the i.i.d. control alone, which is the other half of the
claim: it is a strict generalisation, not a different test.

Power, Clayton alternative, n ≈ 127 (a much shorter series, chosen so power
is informative rather than saturated at 1):

    multipliers          PMC state pair      i.i.d. control
    i.i.d. (ℓ = 1)          49.6 %              86.6 %
    dependent, ℓ = 2        39.0 %              86.6 %
    dependent, ℓ = 5        26.4 %              86.2 %
    dependent, ℓ = 10       19.0 %              86.0 %
    dependent, ℓ = 20       15.2 %              86.4 %
    dependent, ℓ auto       18.4 %              85.8 %

Raw power falls with ``ℓ``, but the i.i.d. row is not a fair comparison: a
test rejecting 19.8 % under the null is not "more powerful", it is
mis-levelled. Two readings settle it. First, the i.i.d. control column is
flat at ~86 % for every ``ℓ``: the dependent multipliers cost essentially
nothing when the data really are independent. Second, judging each method
against its **own** empirical 5 % null quantile at the power design's own
``n`` (R = 500, size and power measured on the same chain, ``n ≈ 120``/127):

    multipliers    size    raw power    size-adjusted power
    i.i.d.        14.5 %     49.6 %          33.0 %
    ℓ = 2          8.2 %     39.0 %          33.6 %
    ℓ = 5          3.6 %     26.4 %          33.6 %
    ℓ = 10         1.8 %     19.0 %          30.4 %
    ℓ = 20         1.2 %     15.2 %          24.0 %
    ℓ auto         2.8 %     18.4 %          26.6 %

Size-adjusted power is flat to ``ℓ = 5`` and only mildly below it after: the
i.i.d. bootstrap's apparent advantage was the level distortion, not power.
Over-shooting ``ℓ`` does eventually cost something real (24.0 % at ``ℓ = 20``
on a sample of 120), which is the sensitivity FR-5 asks to be measured.

``exchangeability_test`` on the same null is already far below nominal
(0.4 % with i.i.d. multipliers, 0.0 % dependent): FR-5's over-rejection does
not show up for that statistic on this design, so there is nothing for the
dependent version to repair there — it only makes an already conservative
test slightly more so. Reported, not hidden.

Left for later
--------------
``CopulaVirt.bootstrap_ci`` / ``.gof_test``
(``pmcprg/copulas/_bivariate_fit.py``, ``_fit.py``) are the *other* i.i.d.
bootstrap FR-5 names. They resample pairs rather than reweight an empirical
process, so the dependent multiplier sequence does not drop into them; a
block bootstrap would be the analogue. Their coverage under serial dependence
is **not** measured here.

Seeds are integer literals (never ``hash()``, salted per process — see commit
5e56fda).
"""
from __future__ import annotations

import numpy as np
import pytest

from pmcprg.copulas import CopulaClayton, CopulaGaussian
from pmcprg.diagnostics import exchangeability_test, radial_symmetry_test
from pmcprg.pmc.model import PMCModel
from pmcprg.pmc.simulate import simulate

ALPHA = 0.05
TAU = 0.7
STAY = 0.98
B = 200
R = 500
#: 3.3 binomial s.d. at p = 0.05, R = 500.
MC_BAND = 0.032

ELLS = (2, 5, 10, 20)


def _raw(name00: str, N: int) -> dict:
    """Two-state PMC; only the ``(0, 0)`` transition copula ever changes."""
    off = (1.0 - STAY) / 2.0
    return {
        "model": {"name": "fr5", "variant": "PMC", "K": 2, "N_default": N},
        "prior": {"p": [[STAY / 2.0, off], [off, STAY / 2.0]]},
        "margins": [{"i": 0, "dist": "norm", "params": {"loc": 0.0, "scale": 1.0}},
                    {"i": 1, "dist": "norm", "params": {"loc": 0.0, "scale": 1.0}}],
        "copulas": [{"i": i, "j": j, "tau": TAU,
                     "name": name00 if (i, j) == (0, 0) else "Gauss"}
                    for i in range(2) for j in range(2)],
    }


def _pair_sample(model, N: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """The ``(0, 0)`` state pair's observations, in time order.

    True hidden states, not an E-step — see the module docstring.
    """
    X, Y = simulate(model, N=N, seed=seed)
    keep = (X[:-1] == 0) & (X[1:] == 0)
    return Y[:-1][keep], Y[1:][keep]


def _methods():
    yield "iid", {"bootstrap": "multiplier"}
    for ell in ELLS:
        yield f"l={ell}", {"bootstrap": "dependent-multiplier", "block_length": ell}
    yield "auto", {"bootstrap": "dependent-multiplier", "block_length": "auto"}


def _rates(test, samples) -> dict[str, float]:
    out = {}
    for name, opts in _methods():
        out[name] = float(np.mean([
            bool(test(x, y, B=B, seed=k, alpha=ALPHA, **opts).reject)
            for k, (x, y) in enumerate(samples)
        ]))
    return out


# ---------------------------------------------------------------------------
# Fast checks: the option is wired in, and ell = 1 changes nothing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("test,kw", [
    (radial_symmetry_test, {}),
    (exchangeability_test, {}),
])
def test_block_length_one_reproduces_the_iid_multiplier_bootstrap_exactly(test, kw):
    """``bootstrap='dependent-multiplier', block_length=1`` *is* the old path.

    Not "agrees to Monte-Carlo error": the same statistic and the same
    p-value, because the multiplier draw is bit-identical (see
    ``test_dependent_multiplier.py``). This is what makes the FR-5 option an
    addition rather than a change.
    """
    s = CopulaGaussian(tau_k=0.5).sample(180, seed=515)
    x, y = s[:, 0], s[:, 1]
    a = test(x, y, bootstrap="multiplier", B=150, seed=3, **kw)
    b = test(x, y, bootstrap="dependent-multiplier", block_length=1, B=150, seed=3, **kw)
    assert a.statistic == b.statistic
    assert a.p_value == b.p_value
    assert a.block_length == 1 and b.block_length == 1


def test_rosenblatt_block_length_one_reproduces_the_iid_multiplier_bootstrap():
    from pmcprg.diagnostics import rosenblatt_gof_test
    s = CopulaGaussian(tau_k=0.5).sample(180, seed=516)
    x, y = s[:, 0], s[:, 1]
    a = rosenblatt_gof_test(x, y, CopulaGaussian, bootstrap="multiplier", B=150, seed=3)
    b = rosenblatt_gof_test(x, y, CopulaGaussian, bootstrap="dependent-multiplier",
                            block_length=1, B=150, seed=3)
    assert a.statistic == b.statistic
    assert a.p_value == b.p_value


@pytest.mark.parametrize("test,kw", [
    (radial_symmetry_test, {}),
    (exchangeability_test, {}),
])
def test_the_parametric_default_is_untouched_by_the_new_keywords(test, kw):
    """Passing ``block_length`` must not perturb ``bootstrap='parametric'``."""
    s = CopulaGaussian(tau_k=0.5).sample(120, seed=517)
    x, y = s[:, 0], s[:, 1]
    a = test(x, y, B=60, seed=1, **kw)
    b = test(x, y, B=60, seed=1, block_length=13, block_kernel="parzen", **kw)
    assert a == b
    assert a.bootstrap == "parametric" and a.block_length == 1


@pytest.mark.parametrize("test,kw", [
    (radial_symmetry_test, {}),
    (exchangeability_test, {}),
])
def test_bad_dependent_multiplier_arguments_raise(test, kw):
    s = CopulaGaussian(tau_k=0.4).sample(60, seed=518)
    x, y = s[:, 0], s[:, 1]
    with pytest.raises(ValueError):
        test(x, y, bootstrap="dependent-multiplier", block_kernel="boxcar", **kw)
    with pytest.raises(ValueError):
        test(x, y, bootstrap="dependent-multiplier", block_length=0, B=10, **kw)
    with pytest.raises(ValueError):
        test(x, y, bootstrap="block", **kw)


def test_a_dependent_bootstrap_widens_the_null_on_a_dependent_sample_fast():
    """One replicate, cheap: the mechanism, before the 500-replicate study.

    On a strongly serially dependent state pair the dependent multipliers
    must produce a *stochastically larger* bootstrap null than the i.i.d.
    ones — that is the whole repair. Compared through the p-value of the same
    observed statistic, so no separate quantile machinery is needed.
    """
    model = PMCModel.from_dict(_raw("Gauss", 800))
    x, y = _pair_sample(model, 800, 50_000)
    p_iid = radial_symmetry_test(x, y, bootstrap="multiplier", B=400, seed=1).p_value
    res = radial_symmetry_test(x, y, bootstrap="dependent-multiplier",
                               block_length=15, B=400, seed=1)
    assert res.p_value > p_iid
    assert res.block_length == 15


# ---------------------------------------------------------------------------
# The study itself
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_iid_multipliers_are_anticonservative_on_a_pmc_state_pair():
    """FR-5's core claim, measured: 19.8 % at a nominal 5 %.

    The same statistic, the same code, on a matched **i.i.d.** sample is at
    3.2 % — so this is the serial dependence, not the statistic.
    """
    N = 800
    null = [_pair_sample(PMCModel.from_dict(_raw("Gauss", N)), N, 50_000 + r)
            for r in range(R)]
    n_mean = float(np.mean([x.size for x, _ in null]))
    assert 350 <= n_mean <= 430, n_mean          # ~392 kept pairs out of 799

    chain = _rates(radial_symmetry_test, null)
    ctl = _rates(radial_symmetry_test,
                 [tuple(CopulaGaussian(tau_k=TAU).sample(int(round(n_mean)),
                                                         seed=70_000 + r).T)
                  for r in range(R)])

    # i.i.d. multipliers: broken on the chain, fine on the control.
    assert chain["iid"] >= 0.15, chain
    assert abs(ctl["iid"] - 0.032) <= MC_BAND, ctl

    # Dependent multipliers: back to nominal on the chain, control untouched.
    assert abs(chain["l=10"] - 0.034) <= MC_BAND, chain
    assert abs(chain["auto"] - 0.032) <= MC_BAND, chain
    assert abs(ctl["auto"] - 0.046) <= MC_BAND, ctl

    # Monotone in ell over the measured range: 19.8 / 12.8 / 6.6 / 3.4 / 3.0.
    ladder = [chain["iid"], chain["l=2"], chain["l=5"], chain["l=10"], chain["l=20"]]
    assert all(a >= b - MC_BAND for a, b in zip(ladder, ladder[1:])), ladder
    assert chain["l=5"] < chain["iid"] - 0.05, chain


@pytest.mark.slow
def test_power_cost_of_the_block_width_is_paid_only_under_dependence():
    """Power at ℓ ∈ {1, …, 20}, chain vs. matched i.i.d. control.

    The control is the honest measurement of what ``ℓ`` costs in itself: ~86 %
    at every ``ℓ``, i.e. nothing. On the chain raw power does fall with ``ℓ``
    (49.6 % → 15.2 %), but the ``ℓ = 1`` end of that range is the test whose
    level is 19.8 %, so the fall is mostly the level being repaired — see
    ``test_size_adjusted_power_removes_most_of_the_apparent_iid_advantage``.
    """
    N = 250
    alt = [_pair_sample(PMCModel.from_dict(_raw("Clayton", N)), N, 60_000 + r)
           for r in range(R)]
    n_mean = float(np.mean([x.size for x, _ in alt]))
    assert 100 <= n_mean <= 160, n_mean

    chain = _rates(radial_symmetry_test, alt)
    ctl = _rates(radial_symmetry_test,
                 [tuple(CopulaClayton(tau_k=TAU).sample(int(round(n_mean)),
                                                        seed=80_000 + r).T)
                  for r in range(R)])

    # Flat in ell on i.i.d. data: the dependent multipliers cost no power there.
    assert abs(ctl["iid"] - 0.866) <= 0.06, ctl
    for name in ("l=2", "l=5", "l=10", "l=20", "auto"):
        assert abs(ctl[name] - ctl["iid"]) <= 0.05, (name, ctl)

    # On the chain, raw power decreases with ell, and stays usable.
    assert chain["l=5"] >= 0.15, chain
    assert chain["auto"] >= 0.10, chain
    assert chain["l=20"] < chain["iid"], chain


@pytest.mark.slow
def test_size_adjusted_power_removes_most_of_the_apparent_iid_advantage():
    """Compare power at a level-matched critical value, not at p < 0.05.

    Measured at ``n ≈ 120``/127 (R = 500), size and power on the same chain:
    the i.i.d. bootstrap's raw 49.6 % rests on a 5 %-critical p-value of
    0.014, far below the nominal 0.05, i.e. it is testing at a much larger
    level than it reports. Once each method is judged against its *own*
    empirical 5 % null quantile the apparent advantage disappears — 33.0 %
    (i.i.d.), 33.6 % (ℓ = 2), 33.6 % (ℓ = 5), 30.4 % (ℓ = 10), 24.0 %
    (ℓ = 20), 26.6 % (auto).

    A handful of replicates produce fewer than ``MIN_N`` kept pairs and a NaN
    p-value (2 of 500 in the null); they are dropped rather than counted as
    non-rejections, which is the only place this test differs from the
    ``.reject``-based studies above.
    """
    N = 250
    null = [_pair_sample(PMCModel.from_dict(_raw("Gauss", N)), N, 90_000 + r)
            for r in range(R)]
    alt = [_pair_sample(PMCModel.from_dict(_raw("Clayton", N)), N, 60_000 + r)
           for r in range(R)]

    adjusted, size = {}, {}
    for name, opts in _methods():
        p0 = np.array([radial_symmetry_test(x, y, B=B, seed=k, **opts).p_value
                       for k, (x, y) in enumerate(null)])
        p1 = np.array([radial_symmetry_test(x, y, B=B, seed=k, **opts).p_value
                       for k, (x, y) in enumerate(alt)])
        p0, p1 = p0[np.isfinite(p0)], p1[np.isfinite(p1)]
        size[name] = float(np.mean(p0 < ALPHA))
        adjusted[name] = float(np.mean(p1 <= float(np.quantile(p0, ALPHA))))

    # The level distortion is there at this n too (14.5 % vs. 5 %) and the
    # dependent versions remove it.
    assert size["iid"] >= 0.10, size
    assert size["l=5"] <= 0.07 and size["auto"] <= 0.07, size

    # Size-adjusted, nothing is lost up to ell = 5 and little beyond.
    assert adjusted["iid"] >= 0.25, adjusted
    for name in ("l=2", "l=5", "l=10"):
        assert adjusted[name] >= adjusted["iid"] - MC_BAND, (name, adjusted)
    assert adjusted["auto"] >= 0.7 * adjusted["iid"], adjusted


@pytest.mark.slow
def test_exchangeability_is_already_conservative_on_this_design():
    """Reported because it was measured, not because it is flattering.

    ``exchangeability_test`` rejects the true null 0.4 % of the time with
    i.i.d. multipliers here and 0.0 % with dependent ones: FR-5's
    over-rejection simply does not appear for this statistic on this design,
    so the dependent version has nothing to repair and only adds
    conservativeness.
    """
    N = 800
    null = [_pair_sample(PMCModel.from_dict(_raw("Gauss", N)), N, 50_000 + r)
            for r in range(R)]
    rates = _rates(exchangeability_test, null)
    assert rates["iid"] <= 0.05, rates
    assert rates["auto"] <= rates["iid"] + 0.01, rates
