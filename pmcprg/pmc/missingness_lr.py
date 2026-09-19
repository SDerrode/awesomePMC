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

The statistic is LR = 2 (LL1 − LL0), each log-likelihood that of the model
the ICE run returns (recomputed by a forward pass). The alternative is fitted
from the null fit: θ̂0 with the null mechanism is its starting point, whose
log-likelihood is LL0. Both runs go to ICE's fixed point (the test's
defaults, :data:`_LR_DEFAULTS`: ``tol = 1e-8``, and ``patience`` large
enough never to stop them). Where ICE is EM — HMC-IN, HMC-IN2, PMC-IN with
state margins: every M-step exact, the mechanism's included — that fixed
point is a maximum, the run from θ̂0 does not lose likelihood, and LR ≥ 0 up
to the ~1e-4 nat of the SR symmetrisation of the prior. Elsewhere (copulas,
pair margins: grid variants) ICE is not EM, its log-likelihood is not
monotone, and LR compares the two ICE estimates: it may be slightly negative
(χ² p-value 1). It is not clipped, so that the bootstrap compares like with
like. The estimated mechanisms carry the boundary guard of
:mod:`pmcprg.pmc.missingness` (one pseudo-observation at a pooled rate):
the fits are posterior modes, not exact maxima, which is invisible against
a common rate (smallest LR over 1 200 HMC-IN null replications: 5.7e-6) but
lets the nested test end up to 0.1 nat below its start at N = 500 (the two
mechanisms pull towards different pooled rates; without the guard those
replications give LR ≥ 0.007). The fixed point, not the best iterate, because it is ICE's estimator
— the best iterate depends on the path (measured on HMC-DN, N = 1000, three
data sets: the two LR differ by 0.004–0.35). ICE's default ``patience = 3``
would stop both runs a few iterations in, as soon as the log-likelihood
drops three times (measured on HMC-DN with gaps, N = 500: the alternative
returned its start, LR = 2e-12).

p-values
--------
* **Asymptotic**: χ²(df). The null values are interior (a common rate in
  (0, 1), a = b inside the square), so Wilks' theorem applies to the exact
  MLE — for ICE only when its M-step is EM's (HMC-IN: margins, prior and the
  mechanism all have exact M-steps). Elsewhere ICE is an approximation of the
  MLE and the χ² reference with it.
* **Parametric bootstrap** (``n_bootstrap = B > 0``) — the reference: B
  series simulated from the fitted null model θ̂0 (:func:`pmcprg.pmc.simulate`,
  one seed stream), their masks drawn under the fitted null mechanism by
  :func:`pmcprg.missing.patterns.state_dependent` or
  :func:`~pmcprg.missing.patterns.state_markov` (another stream: a shared
  seed would correlate the path and its mask), both models refitted by the
  same ICE procedure, started from θ̂0, and
  p = (1 + #{LR_b ≥ LR}) / (B_valid + 1) (Davison & Hinkley 1997, ch. 4). A
  replicate whose mask is empty or full, or whose fit fails, is dropped
  (``n_bootstrap_valid``).

Measured (``report/missing_state/lr_study.py``, K = 2, nominal 5 %; the
bootstrap by the warp-speed method): on HMC-IN the size is 0.045–0.060 (χ²)
and 0.045–0.065 (bootstrap) for "state" vs common at N = 500–2000, 0.028–0.050
and 0.028–0.058 for "state-markov" vs common, 0.020–0.048 and 0.022–0.072 for
the nested test; on HMC-DN 0.060 / 0.040–0.060 for "state", but 0.135 (χ²)
and 0.175 (bootstrap) for "state-markov" at N = 500 (~9 bursts; 0.030 /
0.015 at N = 1000). Power of "state" against π = 0.1 ∓ 0.025 (HMC-IN): 0.37,
0.62, 0.94 at N = 500, 1000, 2000; against 0.1 ∓ 0.05: 0.885, 0.995, 1.
The tables, and the recovery of π̂, â, b̂, are in the study's README
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

#: ICE settings of the test's fits unless the model's ``[ice]`` table or
#: ``ice_cfg`` sets them. The statistic is a difference of two fixed points:
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
    statistic          : LR = 2 (log_lik_alt − log_lik_null); ≥ 0 where ICE
                         is EM, possibly slightly negative elsewhere (module
                         docstring).
    df                 : degrees of freedom of the asymptotic χ².
    p_value            : asymptotic, χ²(df) survival function at ``statistic``
                         (1 for a negative statistic).
    p_value_bootstrap  : parametric-bootstrap p-value, NaN without bootstrap.
    n_bootstrap        : replicates requested (B).
    n_bootstrap_valid  : replicates that produced a statistic.
    bootstrap_statistics : the replicate statistics LR_b (valid ones).
    log_lik_null, log_lik_alt : log p(y_obs, m) of the two fits.
    null_model, alt_model : the fitted models (:class:`~pmcprg.pmc.model.PMCModel`),
                         each carrying its mechanism (``missingness``).
    null_params, alt_params : the ``[missingness]`` tables of the two fitted
                         mechanisms.
    n_obs, n_missing   : N and M.
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
            f"  log-lik  H0 = {self.log_lik_null:.4f}   H1 = {self.log_lik_alt:.4f}",
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


def _fits(start, Y, miss, alternative: str, null: str, cfg: dict):
    """(null_model, LL0, alt_model, LL1) — both fits by ICE from ``start``."""
    from pmcprg.pmc.ice import ice

    K = start.K
    if null == "common":
        fit0, _ = ice(start, Y, {**cfg, "missingness": "ignorable"})
        mech0 = common_mechanism(alternative, miss, K)
        ll0 = _log_lik(fit0, Y, miss, cfg) + mask_log_likelihood(mech0, miss)
        null_model = fit0.with_missingness(mech0)
    else:                                   # null == "state"
        null_model, _ = ice(start, Y, {**cfg, "missingness": "state"})
        ll0 = _log_lik(null_model, Y, miss, cfg)
    # H1 from the null fit: its first log-likelihood is LL0 (the null
    # mechanism is its start — "state" π becomes a = b = π).
    alt_model, _ = ice(null_model, Y, {**cfg, "missingness": alternative, "init": "model"})
    return null_model, float(ll0), alt_model, _log_lik(alt_model, Y, miss, cfg)


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
        _, l0, _, l1 = _fits(null_model, Y_b, miss_b, alternative, null, cfg)
        return 2.0 * (l1 - l0)
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

    null_model, ll0, alt_model, ll1 = _fits(model, Y, miss, alternative, null, cfg)
    lr = 2.0 * (ll1 - ll0)
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
        log_lik_null=ll0, log_lik_alt=ll1, null_model=null_model, alt_model=alt_model,
        null_params=null_model.missingness.to_table(),
        alt_params=alt_model.missingness.to_table(), n_obs=N, n_missing=M,
    )
