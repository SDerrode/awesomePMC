"""
missingness_lr.py — likelihood-ratio test of state-dependent missingness (P6).

Does the mask of the missing rows carry information on the hidden states?
:func:`missingness_lr_test` tests a state-independent mechanism (H0) against
a state-dependent one (H1) of :mod:`pmcprg.pmc.missingness`, both fitted by
ICE (:func:`pmcprg.pmc.ice.ice`, config key ``missingness``).

Nulls and alternatives
----------------------
N rows, M of them missing (mask m), K states.

* ``alternative="state"``, ``null="common"`` — H1: P(m_n = 1 | x_n = i) = π_i;
  H0: π_i = π for every i. Under H0 the mask factor does not depend on x, so
  p(y_obs, m) = p(y_obs) p(m): the parameters θ of the chain are those of the
  ignorable fit, π̂ = M / N is the MLE of π, and

      LL0 = LL_ignorable(θ̂) + M log(M/N) + (N − M) log(1 − M/N)

  exactly. df = K − 1.
* ``alternative="state-markov"``, ``null="common"`` — H1: onset a_i and
  persistence b_i per state; H0: a common (a, b). Under H0 the mask is a
  two-state Markov chain with a stationary start, independent of x; its MLE
  (â, b̂) is :func:`pmcprg.pmc.missingness.common_mechanism` (closed form up
  to the initial term, then maximised exactly) and LL0 = LL_ignorable(θ̂) +
  log p(m | â, b̂). df = 2 (K − 1).
* ``alternative="state-markov"``, ``null="state"`` — the nesting a_i = b_i =
  π_i (``"state"`` with π is ``"state-markov"`` with a = b = π, the initial
  law included: s = π / (1 − π + π) = π). LL0 is the ``"state"`` fit. df = K.

The fits
--------
Both hypotheses are fitted by ICE. The alternative is fitted from the null
fit: θ̂0 with the null mechanism is its starting point. Both runs go to
ICE's fixed point (the test's defaults, :data:`_LR_DEFAULTS`: ``tol =
1e-8``, and ``patience`` large enough never to stop them). ICE's default
``patience = 3`` would stop both runs a few iterations in, as soon as the
log-likelihood drops three times (measured on HMC-DN with gaps, N = 500: the
alternative returned its start, LR = 2e-12). The estimated mechanisms carry
the boundary guard of :mod:`pmcprg.pmc.missingness` (one pseudo-observation
at a pooled rate): the fits are posterior modes.

The statistic
-------------
LR = 2 (sup_H1 LL − sup_H0 LL) needs the two suprema of the observed-data
log-likelihood log p(y_obs, m). Where ICE is EM — HMC-IN, HMC-IN2, PMC-IN
with state margins: every M-step exact — its fixed points are maxima.
Elsewhere (copulas, pair margins: the grid variants) they are not, and the
difference of the two fits, LL1(θ̂1, φ̂1) − LL0(θ̂0), mixes how well each
fit optimised θ with the effect of the mechanism. Measured on HMC-DN,
N = 500, Markov null (``report/missing_state/lr_diagnosis.py``, README
"Diagnosis"): against a direct maximisation over all the parameters (10
for the null, 14 for the alternative; 36 replications), the null fit ends a
median 0.27 nat below the maximum, the alternative 0.51
(both up to 4.6); the alternative's run, started at the null fit, can climb
and then drift to a fixed point at another θ below its start — 9 of 200 null
replications gave a negative difference (min −1.42, whose exact LR is 4.73),
and the part due to θ̂1 ≠ θ̂0 alone, 2 (LL0(θ̂1) − LL0(θ̂0)), has an sd of
0.9 nat (−5.9 to +4.7).

The test therefore approximates each supremum by the best its hypothesis
reaches at the θ of either fit. With ℓ_h(θ) the log-likelihood maximised
over hypothesis h's mechanism at θ fixed,

    sup_H0 ≈ max(ℓ0(θ̂0), ℓ0(θ̂1)),
    sup_H1 ≈ max(ℓ1(θ̂0), ℓ1(θ̂1), sup_H0),      LR = 2 (sup_H1 − sup_H0) ≥ 0.

The null's points are the alternative's (H0 ⊂ H1), so LR ≥ 0 without
clipping. When the same θ gives both maxima, LR is the mechanism's LR at
that θ and the θ-difference cancels. ℓ_h(θ) is computed by EM over the
mechanism alone, θ held fixed (:func:`_profile`): exact posteriors, the exact
M-step of :func:`~pmcprg.pmc.missingness.estimate_mechanism` without the
guard (a profile is a maximum of the likelihood, not a posterior mode; the
guard is for starts at a rate of 0 or 1, where EM stays, and these starts
are the fits' mechanisms, inside (0, 1)) and ICE's stopping rule. The best
point of its path is kept, and the path starts at the fit's mechanism, so
ℓ1(θ̂1) ≥ LL1 and ℓ1(θ̂0) ≥ ℓ0(θ̂0). For a common null the mask's MLE does
not depend on θ: ℓ0(θ) = LL_ignorable(θ) + log p(m | common MLE) exactly.
Measured: the EM stops within 1.3e-4 nat of an exact maximiser (L-BFGS-B on
the logits, exact gradient; HMC-DN, N = 500, 16 profiles, 6 to 214 EM
steps) and within 2.6e-6 nat of a Nelder–Mead maximum on HMC-IN; against
the direct maximisation above (36 null replications) |LR − LR_exact| has a
mean 0.61 and a max 3.9, against 1.02 and 6.2 for the difference of the
fits. Where ICE is EM the two statistics differ by the guard only (HMC-IN,
N = 500, null replications: median 0.008, up to 2.1 for the Markov test —
a state with one or two bursts, whose b̂ the guard shrinks; ≤ 0.017 for
"state"). Cost: two EM runs over the mechanism (four for the nested test),
a forward-backward pass per step — 17 % of the test's CPU time on HMC-DN,
N = 500, 38–45 % on HMC-IN, where the whole test takes 0.2 s.
:attr:`MissingnessLRTest.statistic_fits` keeps the difference of the fits,
a diagnostic.

p-values
--------
* **Asymptotic**: χ²(df). The null values are interior (a common rate in
  (0, 1), a = b inside the square), so Wilks' theorem applies to the exact
  LR as N grows. At a finite N it may not have taken hold: on HMC-DN,
  N = 500 (~9 bursts), the exact LR of the Markov test (direct maximisation,
  30 null replications) has a mean 3.40 against 2, and 5 of 30 exceed the
  5 % point.
* **Parametric bootstrap** (``n_bootstrap = B > 0``): B series simulated
  from the fitted null model θ̂0 (:func:`pmcprg.pmc.simulate`, one seed
  stream), their masks drawn under the fitted null mechanism by
  :func:`pmcprg.missing.patterns.state_dependent` or
  :func:`~pmcprg.missing.patterns.state_markov` (another stream: a shared
  seed would correlate the path and its mask), both models refitted by the
  same ICE procedure, started from θ̂0, the same statistic, and
  p = (1 + #{LR_b ≥ LR}) / (B_valid + 1) (Davison & Hinkley 1997, ch. 4). A
  replicate whose mask is empty or full, or whose fit fails, is dropped
  (``n_bootstrap_valid``). It does not repair the case above: series
  simulated from the fitted null have a χ²-like statistic (B = 99 on six
  HMC-DN null series: 95 % quantiles 5.2–6.9), and the four largest null
  values of the study (10.6–14.3) get p = 0.01.

Measured (``report/missing_state/lr_study.py``, K = 2, nominal 5 %;
bootstrap by the warp-speed method). With this statistic (HMC-IN at N = 500,
HMC-DN Markov cells; paired with the study of the fits' statistic, which
covers N = 500–2000): on HMC-IN the size is 0.058 (χ²) / 0.050 (bootstrap)
for "state" vs common, 0.052 / 0.035 for "state-markov" vs common, 0.020 /
0.020 nested; on HMC-DN the Markov test is anticonservative at N = 500,
0.125 / 0.150, and holds at N = 1000, 0.045 / 0.010 — use it on grid
variants from ~20 bursts on. The fits' statistic gave on HMC-IN 0.045–0.060
and 0.045–0.065 for "state" at N = 500–2000, 0.028–0.050 and 0.028–0.058
for "state-markov", 0.020–0.048 and 0.022–0.072 nested, on HMC-DN
0.060 / 0.040–0.060 for "state". Power of "state" against π = 0.1 ∓ 0.025
(HMC-IN): 0.37, 0.62, 0.94 at N = 500, 1000, 2000; against 0.1 ∓ 0.05:
0.885, 0.995, 1 (the same at N = 500 with this statistic). The tables, the
diagnosis and the recovery of π̂, â, b̂ are in the study's README
(``pmcprg/tests/test_missingness_lr.py`` holds a reduced version).

References
----------
* Wilks, S. S. (1938). The large-sample distribution of the likelihood ratio
  for testing composite hypotheses. *Ann. Math. Statist.* 9(1), 60–62.
* Davison, A. C. & Hinkley, D. V. (1997). *Bootstrap Methods and Their
  Application*. Cambridge University Press, ch. 4.
* Little, R. J. A. & Rubin, D. B. (2019). *Statistical Analysis with Missing
  Data*, 3rd ed., Wiley — selection models for data missing not at random.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from pmcprg.pmc.missingness import common_mechanism, mask_log_likelihood

logger = logging.getLogger(__name__)

__all__ = ["MissingnessLRTest", "bootstrap_replicate", "missingness_lr_test"]

#: The implemented (alternative, null) pairs and their degrees of freedom
#: as a function of K.
_TESTS = {
    ("state", "common"):        lambda K: K - 1,
    ("state-markov", "common"): lambda K: 2 * (K - 1),
    ("state-markov", "state"):  lambda K: K,
}

#: ICE settings of the test's fits (and stopping rule of its profiles,
#: :func:`_profile`) unless the model's ``[ice]`` table or ``ice_cfg`` sets
#: them. The statistic compares log-likelihoods of converged fits:
#: ICE's default ``tol = 1e-4`` is relative to |LL| (~3000 nat at N = 2000),
#: i.e. a stop when an iteration gains less than 0.3 nat — the order of the
#: χ²(1) critical value's decimals; 1e-8 stops below 3e-5 nat per iteration.
#: ``patience = 500`` (= ``max_iter``): ICE's log-likelihood regresses near its
#: fixed point (by ~1e-4 nat on HMC-IN, the SR symmetrisation of the prior;
#: by 0.1 nat per iteration for several iterations on HMC-DN with gaps), which
#: the default ``patience = 3`` takes for a stop short of the fixed point.
_LR_DEFAULTS = {"tol": 1e-8, "max_iter": 500, "patience": 500}

# Seed streams of the bootstrap: paths and masks never share a seed.
_SIM_STREAM = 0x53494D    # "SIM"
_MASK_STREAM = 0x4D534B   # "MSK"


@dataclass(frozen=True)
class MissingnessLRTest:
    """Result of :func:`missingness_lr_test`.

    Fields
    ------
    alternative        : ``"state"`` or ``"state-markov"`` (H1).
    null               : ``"common"`` (state-independent) or ``"state"`` (H0).
    statistic          : LR = 2 (sup_log_lik_alt − sup_log_lik_null) ≥ 0
                         (module docstring, "The statistic").
    df                 : degrees of freedom of the asymptotic χ².
    p_value            : asymptotic, χ²(df) survival function at ``statistic``.
    p_value_bootstrap  : parametric-bootstrap p-value, NaN without bootstrap.
    n_bootstrap        : replicates requested (B).
    n_bootstrap_valid  : replicates that produced a statistic.
    bootstrap_statistics : the replicate statistics LR_b (valid ones).
    log_lik_null, log_lik_alt : log p(y_obs, m) of the two ICE fits.
    null_model, alt_model : the fitted models (:class:`~pmcprg.pmc.model.PMCModel`),
                         each carrying its mechanism (``missingness``).
    null_params, alt_params : the ``[missingness]`` tables of the two fitted
                         mechanisms.
    n_obs, n_missing   : N and M.
    sup_log_lik_null, sup_log_lik_alt : the approximations of the two
                         suprema, max over the θ of both fits of the profile
                         log-likelihoods ``profile_log_liks``.
    profile_log_liks   : ``{"null_at_null_theta": ℓ0(θ̂0), "null_at_alt_theta":
                         ℓ0(θ̂1), "alt_at_null_theta": ℓ1(θ̂0),
                         "alt_at_alt_theta": ℓ1(θ̂1)}`` — ℓ_h(θ) the
                         log-likelihood maximised over hypothesis h's
                         mechanism at θ.
    statistic_fits     : 2 (log_lik_alt − log_lik_null), the difference of the
                         two ICE fits alone — a diagnostic: it can be negative
                         where ICE is not EM (module docstring).
    """

    alternative: str
    null: str
    statistic: float
    df: int
    p_value: float
    p_value_bootstrap: float
    n_bootstrap: int
    n_bootstrap_valid: int
    bootstrap_statistics: tuple
    log_lik_null: float
    log_lik_alt: float
    null_model: object
    alt_model: object
    null_params: dict
    alt_params: dict
    n_obs: int
    n_missing: int
    sup_log_lik_null: float
    sup_log_lik_alt: float
    profile_log_liks: dict
    statistic_fits: float

    def summary(self) -> str:
        """A few lines for a log or a console."""
        h0 = {"common": "state-independent (common) mechanism",
              "state": "'state' (π_i per state)"}[self.null]
        h1 = {"state": "'state' (π_i per state)",
              "state-markov": "'state-markov' (a_i, b_i per state)"}[self.alternative]
        boot = ("no bootstrap" if self.n_bootstrap == 0 else
                f"p (bootstrap, {self.n_bootstrap_valid}/{self.n_bootstrap} "
                f"replicates) = {self.p_value_bootstrap:.4g}")

        def params(tbl):
            return ", ".join(f"{k} = [{', '.join(f'{v:.4g}' for v in tbl[k])}]"
                             for k in ("rates", "onset", "persistence") if k in tbl)

        return "\n".join([
            "Likelihood-ratio test of state-dependent missingness",
            f"  H0: {h0}",
            f"  H1: {h1}",
            f"  N = {self.n_obs}, missing = {self.n_missing} "
            f"({100.0 * self.n_missing / max(self.n_obs, 1):.1f} %)",
            f"  log-lik  H0 = {self.sup_log_lik_null:.4f}   H1 = {self.sup_log_lik_alt:.4f}"
            f"   (ICE fits: {self.log_lik_null:.4f}, {self.log_lik_alt:.4f})",
            f"  LR = {self.statistic:.4f}   df = {self.df}   "
            f"p (chi2) = {self.p_value:.4g}   {boot}",
            f"  H0 estimate: {params(self.null_params)}",
            f"  H1 estimate: {params(self.alt_params)}",
        ])

    def __repr__(self) -> str:
        boot = ("" if self.n_bootstrap == 0 else
                f", p_boot={self.p_value_bootstrap:.3g} (B={self.n_bootstrap_valid})")
        return (f"MissingnessLRTest({self.alternative} vs {self.null}, "
                f"LR={self.statistic:.4g}, df={self.df}, p={self.p_value:.3g}{boot})")


def _check(alternative: str, null: str) -> None:
    if (alternative, null) not in _TESTS:
        raise ValueError(
            f"missingness_lr_test: no test of alternative={alternative!r} against "
            f"null={null!r}. Implemented (alternative, null): {sorted(_TESTS)}."
        )


def _fit_cfg(model, ice_cfg: dict | None) -> dict:
    """ICE cfg of the test's fits: :data:`_LR_DEFAULTS` under the model's
    ``[ice]`` table and ``ice_cfg``."""
    own = model.ice_config()
    cfg = {k: v for k, v in _LR_DEFAULTS.items() if k not in own}
    cfg.update(ice_cfg or {})
    return cfg


def _log_lik(model, Y, miss, cfg) -> float:
    """log p(y_obs[, m]) of a fitted model — the forward pass of ICE's E-step,
    with the ``gap_nodes`` the fit used."""
    from pmcprg.pmc._estim_common import evaluate_log_lik
    from pmcprg.pmc.ice import _parse_ice_cfg
    return float(evaluate_log_lik(model, Y, miss, _parse_ice_cfg(model, cfg).get("gap_nodes")))


def _profile(model, Y, miss, mechanism: str, start, cfg) -> tuple[float, object]:
    """max over the ``mechanism`` of log p(y_obs, m | θ, mechanism), θ of
    ``model`` held fixed: EM from ``start``, the exact M-step of
    :func:`~pmcprg.pmc.missingness.estimate_mechanism` without the boundary
    guard (``pseudo_count = 0``) on the exact posteriors given (y_obs, m).
    Stops as ICE does (a step gaining less than ``tol`` |LL|, or
    ``max_iter`` steps). Returns the best (log-likelihood, mechanism) of the
    path — its first point is ``start``, so the result is never below it.
    """
    from pmcprg.pmc.gaps import gap_posterior
    from pmcprg.pmc.ice import _parse_ice_cfg
    from pmcprg.pmc.missingness import estimate_mechanism

    parsed = _parse_ice_cfg(model, cfg)
    gap_nodes, tol, max_iter = parsed.get("gap_nodes"), float(parsed["tol"]), int(parsed["max_iter"])
    mech, prev = start, None
    best_ll, best_mech = -np.inf, start
    for _ in range(max(max_iter, 1)):
        post = gap_posterior(model.with_missingness(mech), Y, gap_nodes=gap_nodes, xi=False)
        ll = float(post.log_lik)
        if ll > best_ll:
            best_ll, best_mech = ll, mech
        if prev is not None and abs(ll - prev) < tol * (abs(prev) + 1e-10):
            break
        prev = ll
        mech = estimate_mechanism(mechanism, post.gamma, miss, pseudo_count=0.0)
    return best_ll, best_mech


@dataclass(frozen=True)
class _Fits:
    """The two ICE fits and the profile log-likelihoods of the statistic."""

    null_model: object
    alt_model: object
    ll_null: float                # the null fit (θ̂0, its mechanism)
    ll_alt: float                 # the alternative fit (θ̂1, φ̂1)
    profiles: dict                # ℓ0(θ̂0), ℓ0(θ̂1), ℓ1(θ̂0), ℓ1(θ̂1)
    sup_null: float
    sup_alt: float

    @property
    def statistic(self) -> float:
        return 2.0 * (self.sup_alt - self.sup_null)


#: Keys of :attr:`MissingnessLRTest.profile_log_liks`: ℓ_h(θ̂_k), the
#: hypothesis h's mechanism maximised at the θ of fit k.
_PROFILE_KEYS = ("null_at_null_theta", "null_at_alt_theta",
                 "alt_at_null_theta", "alt_at_alt_theta")


def _fits(start, Y, miss, alternative: str, null: str, cfg: dict) -> _Fits:
    """Both fits by ICE from ``start``, then the profiles of the statistic
    (module docstring, "The statistic")."""
    from pmcprg.pmc.ice import _missingness_start, ice

    K = start.K
    if null == "common":
        fit0, _ = ice(start, Y, {**cfg, "missingness": "ignorable"})
        mech0 = common_mechanism(alternative, miss, K)
        mask_ll = mask_log_likelihood(mech0, miss)
        ll0 = _log_lik(fit0, Y, miss, cfg) + mask_ll
        null_model = fit0.with_missingness(mech0)
    else:                                   # null == "state"
        null_model, _ = ice(start, Y, {**cfg, "missingness": "state"})
        ll0 = _log_lik(null_model, Y, miss, cfg)
    # H1 from the null fit: its first log-likelihood is LL0 (the null
    # mechanism is its start — "state" π becomes a = b = π).
    alt_model, _ = ice(null_model, Y, {**cfg, "missingness": alternative, "init": "model"})
    ll1 = _log_lik(alt_model, Y, miss, cfg)

    # ℓ0(θ) = max over the null's mechanism at θ. Common: the mask MLE does
    # not depend on θ, so ℓ0(θ) = LL_ignorable(θ) + log p(m | â, b̂) exactly.
    if null == "common":
        p0_null, mech_null0 = float(ll0), null_model.missingness
        p0_alt = _log_lik(alt_model.with_missingness(None), Y, miss, cfg) + mask_ll
    else:
        p0_null, mech_null0 = _profile(null_model, Y, miss, "state",
                                       null_model.missingness, cfg)
        p0_null = max(p0_null, float(ll0))
        p0_alt, _ = _profile(alt_model, Y, miss, "state", null_model.missingness, cfg)
    # ℓ1(θ) = max over the alternative's mechanism at θ: from the null's
    # maximiser at θ̂0 (so ℓ1(θ̂0) ≥ ℓ0(θ̂0)), from φ̂1 at θ̂1.
    p1_null, _ = _profile(null_model, Y, miss, alternative,
                          _missingness_start(null_model.with_missingness(mech_null0),
                                             alternative, miss), cfg)
    # (the start's value, recomputed by the profile's pass, may differ from
    # the fit's in the last bits: the max keeps ℓ1 ≥ ℓ0 and ℓ1(θ̂1) ≥ LL1)
    p1_null = max(p1_null, float(p0_null))
    p1_alt, _ = _profile(alt_model, Y, miss, alternative, alt_model.missingness, cfg)
    p1_alt = max(p1_alt, float(ll1))
    profiles = dict(zip(_PROFILE_KEYS, (float(p0_null), float(p0_alt),
                                        float(p1_null), float(p1_alt))))
    sup_null = max(profiles["null_at_null_theta"], profiles["null_at_alt_theta"])
    # the null's points are the alternative's too (H0 ⊂ H1)
    sup_alt = max(profiles["alt_at_null_theta"], profiles["alt_at_alt_theta"], sup_null)
    return _Fits(null_model=null_model, alt_model=alt_model, ll_null=float(ll0),
                 ll_alt=float(ll1), profiles=profiles, sup_null=sup_null, sup_alt=sup_alt)


def _draw_mask(model, Y, X, seed: int):
    """Y with rows removed by ``model.missingness`` (a mechanism), given X."""
    from pmcprg.missing.patterns import state_dependent, state_markov
    mech = model.missingness
    if mech.mechanism == "state":
        return state_dependent(Y, X, mech.rates, seed=seed)[0]
    return state_markov(Y, X, mech.onset, mech.persistence, seed=seed)[0]


def bootstrap_replicate(null_model, N: int, seed_sim: int, seed_mask: int, *,
                        alternative: str = "state", null: str = "common",
                        ice_cfg: dict | None = None) -> float | None:
    """One replicate LR_b of the parametric bootstrap (module docstring).

    Simulates (X, Y) of length N from ``null_model`` — the ``null_model`` of
    a :class:`MissingnessLRTest` — with seed ``seed_sim``, removes rows by its
    mechanism given X (seed ``seed_mask``; :func:`missingness_lr_test` draws
    the two from separate streams), refits both models from ``null_model``
    with the test's ICE settings (``ice_cfg`` as in
    :func:`missingness_lr_test`) and returns 2 (LL1 − LL0); ``None`` when the
    mask is empty or full or a fit fails. Public for Monte-Carlo studies of
    the test (``report/missing_state/``).
    """
    from pmcprg.pmc.gaps import missing_mask
    from pmcprg.pmc.simulate import simulate
    _check(alternative, null)
    cfg = _fit_cfg(null_model, ice_cfg)
    try:
        X_b, Y_b = simulate(null_model, N=int(N), seed=int(seed_sim))
        Y_b = _draw_mask(null_model, Y_b, X_b, int(seed_mask))
        miss_b = missing_mask(Y_b)
        if not 0 < int(miss_b.sum()) < int(N):
            return None
        return _fits(null_model, Y_b, miss_b, alternative, null, cfg).statistic
    except Exception as exc:                                   # pragma: no cover
        logger.debug("missingness LR bootstrap replicate failed: %s", exc)
        return None


def missingness_lr_test(
    model,
    Y,
    *,
    alternative: str = "state",
    null: str = "common",
    ice_cfg: dict | None = None,
    n_bootstrap: int = 0,
    seed: int = 0,
    progress_cb=None,
) -> MissingnessLRTest:
    """Likelihood-ratio test of state-independent against state-dependent missingness.

    Parameters
    ----------
    model : PMCModel
        Starting point of the null fit (its own mechanism, if any, is only a
        start: the null fit replaces it). Its ``[ice]`` table and ``ice_cfg``
        configure both ICE fits (``fit_margins``, ``missing_strategy``, …).
    Y : array (N,) or (N, d)
        Observations; the rows with a non-finite value are the mask m. Y must
        have at least one missing and one observed row.
    alternative : ``"state"`` or ``"state-markov"`` — H1.
    null : ``"common"`` (a state-independent mechanism of the same kind,
        default) or ``"state"`` (only against ``"state-markov"``) — H0.
    ice_cfg : dict, optional
        ICE settings of the fits, over the model's ``[ice]`` table and the
        test's defaults (``tol = 1e-8``, ``max_iter = 500``, ``patience =
        500``: ICE's fixed points). ``missingness`` is set by the test, and
        the alternative starts from the null fit (``init = "model"``).
    n_bootstrap : int
        Parametric-bootstrap replicates B (0: asymptotic p-value only). Each
        costs two ICE fits.
    seed : int
        Seed of the bootstrap (separate streams for the paths and the masks).
    progress_cb : callable, optional
        ``progress_cb(b, B, 0.0, "bootstrap")`` before each replicate.

    Returns
    -------
    :class:`MissingnessLRTest` (see the module docstring for the statistic,
    the null log-likelihood and the two p-values).
    """
    from scipy.stats import chi2

    from pmcprg.pmc.gaps import missing_mask

    _check(alternative, null)
    B = int(n_bootstrap)
    if B < 0 or B != n_bootstrap:
        raise ValueError(f"n_bootstrap must be an integer ≥ 0, got {n_bootstrap!r}.")
    Y = np.asarray(Y, dtype=float)
    miss = missing_mask(Y)
    N, M = int(miss.size), int(miss.sum())
    if M == 0 or M == N:
        raise ValueError(
            f"missingness_lr_test: Y has {M} missing rows of {N} — the mask must "
            f"have both missing and observed rows for its law to be tested."
        )
    cfg = _fit_cfg(model, ice_cfg)
    df = _TESTS[(alternative, null)](model.K)

    fits = _fits(model, Y, miss, alternative, null, cfg)
    null_model, alt_model = fits.null_model, fits.alt_model
    lr = fits.statistic
    p_asym = float(chi2.sf(lr, df))
    logger.info("missingness LR test (%s vs %s): LR = %.4f, df = %d, p = %.4g",
                alternative, null, lr, df, p_asym)

    boot: list[float] = []
    if B:
        rng_sim = np.random.default_rng(np.random.SeedSequence([int(seed), _SIM_STREAM]))
        rng_mask = np.random.default_rng(np.random.SeedSequence([int(seed), _MASK_STREAM]))
        for b in range(B):
            if progress_cb is not None:
                try:
                    progress_cb(b, B, 0.0, "bootstrap")
                except Exception:                              # pragma: no cover
                    pass
            s_sim = int(rng_sim.integers(0, 2**31 - 1))
            s_mask = int(rng_mask.integers(0, 2**31 - 1))
            lr_b = bootstrap_replicate(null_model, N, s_sim, s_mask,
                                       alternative=alternative, null=null, ice_cfg=cfg)
            if lr_b is not None:
                boot.append(lr_b)
    boot_arr = np.asarray(boot, dtype=float)
    p_boot = (float((1 + np.sum(boot_arr >= lr)) / (boot_arr.size + 1))
              if B and boot_arr.size else float("nan"))

    return MissingnessLRTest(
        alternative=alternative, null=null, statistic=float(lr), df=int(df),
        p_value=p_asym, p_value_bootstrap=p_boot, n_bootstrap=B,
        n_bootstrap_valid=int(boot_arr.size), bootstrap_statistics=tuple(boot),
        log_lik_null=fits.ll_null, log_lik_alt=fits.ll_alt, null_model=null_model,
        alt_model=alt_model, null_params=null_model.missingness.to_table(),
        alt_params=alt_model.missingness.to_table(), n_obs=N, n_missing=M,
        sup_log_lik_null=fits.sup_null, sup_log_lik_alt=fits.sup_alt,
        profile_log_liks=dict(fits.profiles),
        statistic_fits=2.0 * (fits.ll_alt - fits.ll_null),
    )
