"""
gaps.py — supervised inference with missing observations (NaN) for PMC/HMC models.

Public API
----------
gap_posterior(model, Y, *, gap_nodes=64) -> GapPosterior
    Filtering/smoothing quantities of a NaN-bearing sequence: log-likelihood of
    the observed data, α̂, β̂, γ, ξ and, for the grid variants, the posterior of
    (x_n, y_n) on the quadrature nodes at every missing position.
impute(model, Y, *, gap_nodes=64, quantiles=(0.05, 0.5, 0.95), n_samples=0, rng=None)
    -> Imputation
    Posterior mean, standard deviation and quantiles of every missing y_n given
    all the observed data; optionally joint FFBS draws of the missing values.
forecast(model, Y, h, *, gap_nodes=64, quantiles=(0.05, 0.5, 0.95)) -> Forecast
    h-step predictive law of (x_{N+k}, y_{N+k}), k = 1..h, given the observed
    part of Y (Y may itself contain NaN).
missing_mask(Y), needs_grid(model), reference_grid(model, gap_nodes)
    Helpers (mask of missing rows, whether a model needs the quadrature grid,
    the quadrature grid itself).

The NaN-aware entry points of :mod:`pmcprg.pmc.inference` (``classify``,
``forward``, ``backward``, ``precompute_weights``, ``sample_posterior``)
dispatch here; a Y without missing values never reaches this module.

Missing observations
--------------------
A row of Y is *missing* when it holds a non-finite value (NaN, ±inf):
``miss = ~np.isfinite(Y)`` for d = 1; for d > 1 a row with any non-finite
component is missing as a whole (partially observed rows are not supported in
this version — their finite components are ignored). Missingness is assumed
ignorable (MCAR/MAR): the quantities below are those of the observed-data
likelihood p(y_obs) = ∫ p(y) dy_miss.

Z = (X, Y) is a Markov chain with transition q(j, y' | i, y) = W[n, i, j] for
y = y_n, y' = y_{n+1} (see :mod:`pmcprg.pmc.inference`) and initial density
μ(i, y) = α_1(i) evaluated at y = y_1.

1. **Exact shortcut** (HMC-IN, HMC-IN2, PMC-IN with state margins). The
   transition q(j, y' | i, y) = A_ij f_j(y') does not depend on y, and
   ∫ f_j(y') dy' = 1: a missing y_{n+1} contributes the factor 1 in place of
   f_j(y_{n+1}), and a missing y_1 gives α_1(j) = Σ_i p_ij. This is exact and
   stays a K-state recursion, so :func:`pmcprg.pmc.inference.precompute_weights`
   returns the marginalised weights (f_pdf row = 1 at a missing n) and every
   K-state function of :mod:`pmcprg.pmc.inference` is exact on them. Given the
   states, the missing y_n are independent with law f_{x_n}: imputation and
   forecasting use the exact mixtures Σ_j P(x_n = j | y_obs) f_j.

2. **Quadrature grid** (HMC-DN, PMC, PMC-IN with pair margins). The
   transition depends on the value of y_n through the copula and/or the pair
   margins, so a missing y_n is integrated out on an augmented state (i, g)
   over quadrature nodes y_g.

   * Reference law g_ref(y) = Σ_i μ(i, y) = Σ_{i,j} p_ij f_ij(y) (pair margins)
     or Σ_j (Σ_i p_ij) f_j(y) (state margins; π_j f_j for the HMC variants) —
     the law of y_1, the stationary law of y_n for a stationary chain.
   * Nodes y_g = G_ref^{-1}(t_g), t_g = ψ(s_g), with (s_g, ws_g) the G-point
     Gauss–Legendre rule on (0, 1) and ψ(s) = s² / (s² + (1 − s)²) an
     endpoint transform; the weights w_g = ws_g ψ'(s_g) integrate over t, and
     ω_g = w_g / g_ref(y_g), so ∫ h(y) dy ≈ Σ_g ω_g h(y_g). Without ψ (plain
     Gauss–Legendre in t = G_ref(y)) the integrands, a conditional density
     over g_ref, behave like t^β at t → 0, 1 (e.g. β = 2ρ²/(1 − ρ²) for a
     Gaussian copula ρ between N(0, 1) margins): an endpoint singularity that
     makes the rule converge only algebraically — 1e-4 errors at G = 64 for
     ρ = 0.5 — whereas ψ restores fast convergence (1e-7 at G = 64, see the
     accuracy tests). G_ref^{-1} is a vectorised bisection on the mixture CDF
     (on the survival function in the upper half), bracketed by the component
     quantiles. Default G = 64 (``gap_nodes``).
   * Transitions of the augmented chain (augmented index u = i·G + g):

       observed n → missing n+1  E[i, (j, g)]        = q(j, y_g | i, y_n) ω_g
       missing n → missing n+1   Q[(i, g), (j, g')]  = q(j, y_g' | i, y_g) ω_g'
       missing n → observed n+1  X[(i, g), j]        = q(j, y_{n+1} | i, y_g)
       observed → observed       W[n, i, j]          (unchanged)
       missing y_1               α̃_1(i, g)           = μ(i, y_g) ω_g

     Q does not depend on the data and is built once per call.
   * Renormalisation (Tauchen & Hussey 1991). The quadrature does not
     integrate the transition density exactly, but two of its integrals are
     known: Σ_g of the block (i, y) → (j, ·) must be P(x_{n+1} = j | x_n = i,
     y_n = y) — A_ij, or p_ij f_ij(y) / Σ_k p_ik f_ik(y) with pair margins —
     and the block (·, g) of α̃_1 must be P(x_1 = i). Every block of E, Q and
     α̃_1 is rescaled to that mass (then every row to 1, a no-op unless a
     block vanished). A trailing gap then contributes exactly log C = 0, a
     sequence with no observed value has log-likelihood 0, and for state
     margins (X Markov) the state marginals across a gap are exact.
     Measured on the references of the test-suite, this also lowers the
     likelihood and moment errors 2–5× against plain row renormalisation.
   * Forward and backward run the Devijver/Rabiner-scaled recursions of
     :mod:`pmcprg.pmc.inference` on this chain of variable width (K at an
     observed position, K·G at a missing one); log p(y_obs) = Σ_n log C_n.
     γ_n(i) = Σ_g α̃_n(i, g) β̃_n(i, g) at a missing n, ξ_n by summing the
     augmented pair posterior over the nodes. Copula arguments are clipped to
     [EPS, 1 − EPS] as in ``precompute_weights``.
   * Log space. When a linear step underflows (a normaliser not finite or
     below MIN_POSITIVE) or a kernel value overflows (e.g. a copula density
     beyond 1.8e308), the whole pass is redone in log space from the exact
     log-densities (``logpdf_array`` of the copulas).
   * Laws of the missing values. Moments of y_n are the quadratures
     Σ_g mass_g y_g^k of the node posterior. Quantiles need the CDF between
     the nodes: the density of y_n at any y is interpolated from the messages
     of the neighbouring positions through the exact kernels (Nyström
     interpolation, :func:`_nystrom_masses`) on a grid four times finer, and
     the CDF of its Legendre interpolant is inverted. (Interpolating the
     G node masses directly loses half of the polynomial degree: quantile
     errors 1e-4 where the moments are at 1e-6.) ``forecast`` uses the
     normalised forward messages of a trailing gap of length h.

Returned α̂ and β̂ (K-state view of the augmented chain)
-------------------------------------------------------
``alpha_hat[n, i] = P(x_n = i | y_obs ∩ 1:n)`` at every n (missing or not).
``beta_hat[n]`` is the normalised backward message at an observed n; at a
missing n there is no K-state backward message independent of the forward
(y_n couples x_{n-1} and x_{n+1}), so it is defined as
``beta_hat[n, i] ∝ Σ_g α̃_n(i, g) β̃_n(i, g) / Σ_g α̃_n(i, g)`` — the
likelihood of the future observations given x_n = i and the past ones. With
this definition ``smooth(alpha_hat, beta_hat)`` is exact at every n. No K×K
tensor W makes ``joint_posteriors(alpha_hat, W, beta_hat)`` exact for the grid
variants (the posterior of X given y_obs is not Markov across a gap): use
``gap_posterior(model, Y).xi``.

References
----------
* Tauchen, G. & Hussey, R. (1991). Quadrature-based methods for obtaining
  approximate solutions to nonlinear asset pricing models. *Econometrica*
  59(2), 371–396 — row-renormalised quadrature discretisation of a Markov
  kernel.
* Kitagawa, G. (1987). Non-Gaussian state-space modeling of nonstationary
  time series. *J. Amer. Statist. Assoc.* 82(400), 1032–1041 — numerical
  integration of filtering densities on a grid.
* Nyström, E. J. (1930). Über die praktische Auflösung von Integralgleichungen
  mit Anwendungen auf Randwertaufgaben. *Acta Math.* 54, 185–204 — the
  interpolation of a quadrature solution through its kernel.
* Little, R. J. A. & Rubin, D. B. (2019). *Statistical Analysis with Missing
  Data*, 3rd ed., Wiley — ignorable missingness, observed-data likelihood.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.polynomial import legendre as _leg
from scipy import stats as _ss

from pmcprg.exceptions import IncompatibleObservationError
from pmcprg.numerics import EPS, ONE_MINUS_EPS, MIN_POSITIVE
from pmcprg.pmc import inference as _inf
from pmcprg.pmc.model import PMCModel

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_GAP_NODES",
    "DEFAULT_QUANTILES",
    "QuadratureGrid",
    "GapPosterior",
    "Imputation",
    "Forecast",
    "missing_mask",
    "needs_grid",
    "reference_grid",
    "gap_posterior",
    "impute",
    "forecast",
]

#: Default number of Gauss–Legendre nodes of the quadrature grid.
DEFAULT_GAP_NODES = 64
#: Default quantile levels reported by :func:`impute` and :func:`forecast`.
DEFAULT_QUANTILES = (0.05, 0.5, 0.95)


# ---------------------------------------------------------------------------
# Masks and dispatch helpers
# ---------------------------------------------------------------------------

def missing_mask(Y) -> np.ndarray:
    """Boolean mask (N,) of the missing rows of Y (any non-finite component)."""
    fin = np.isfinite(np.asarray(Y, dtype=float))
    if fin.ndim > 1:
        fin = fin.reshape(fin.shape[0], -1).all(axis=1)
    return ~fin


def needs_grid(model: PMCModel) -> bool:
    """True when a missing y must be integrated on the quadrature grid.

    The transition depends on the value of y_n for HMC-DN and PMC (copula) and
    for PMC-IN with pair margins (A_ij(y_n) ∝ p_ij f_ij(y_n)); HMC-IN, HMC-IN2
    and PMC-IN with state margins admit the exact shortcut.
    """
    return bool(model.variant.uses_copula or model.margin_structure == "pair")


def _resolve_nodes(gap_nodes) -> int:
    G = DEFAULT_GAP_NODES if gap_nodes is None else int(gap_nodes)
    if G < 2:
        raise ValueError(f"gap_nodes must be an integer ≥ 2, got {gap_nodes!r}.")
    return G


def _check_quantiles(quantiles) -> tuple[float, ...]:
    qs = tuple(float(q) for q in quantiles)
    if any(not (0.0 < q < 1.0) for q in qs):
        raise ValueError(f"quantiles must lie in (0, 1); got {qs}.")
    return qs


def _check_grid_supported(model: PMCModel) -> None:
    if getattr(model, "d", 1) != 1:
        raise NotImplementedError(
            f"Missing observations for variant {model.variant.value} with "
            f"{model.margin_structure} margins need the quadrature grid, which "
            f"is implemented for scalar observations only (model.d = {model.d})."
        )


# ---------------------------------------------------------------------------
# Mixtures of scalar laws: CDF and quantile by vectorised bisection
# ---------------------------------------------------------------------------

def _frozen(margin):
    fr = getattr(margin, "_frozen", None)
    if fr is None:
        raise TypeError(f"margin {margin!r} exposes no frozen scipy.stats law.")
    return fr


def _mixture_ppf(weights: np.ndarray, dists: list, q: np.ndarray,
                 qc: np.ndarray | None = None) -> np.ndarray:
    """Quantiles of the mixtures Σ_c weights[m, c] · dists[c].

    ``weights`` (M, C) rows summing to 1, ``q`` (M, Q) levels in (0, 1) and
    optionally ``qc`` = 1 − q computed without cancellation (upper tail).
    Bracket [min_c F_c^{-1}(q), max_c F_c^{-1}(q)] over the components of
    positive weight (the mixture CDF is ≤ q below every component quantile and
    ≥ q above them), then bisection to relative width 1e-13 — on the CDF for
    q ≤ 1/2 and on the survival function against ``qc`` above, so that levels
    within 1e-16 of 1 keep their precision.
    """
    weights = np.asarray(weights, dtype=float)
    q = np.asarray(q, dtype=float)
    qc = (1.0 - q) if qc is None else np.asarray(qc, dtype=float)
    upper = q > 0.5
    q = np.clip(q, MIN_POSITIVE, 1.0)
    qc = np.clip(qc, MIN_POSITIVE, 1.0)
    pos = weights > 0.0
    lo = np.full(q.shape, np.inf)
    hi = np.full(q.shape, -np.inf)
    for c, dist in enumerate(dists):
        with np.errstate(all="ignore"):
            yc = np.where(upper, dist.isf(qc), dist.ppf(q))
        active = pos[:, c][:, None]
        lo = np.where(active & (yc < lo), yc, lo)
        hi = np.where(active & (yc > hi), yc, hi)
    target = np.where(upper, qc, q)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        F = np.zeros(q.shape)
        for c, dist in enumerate(dists):
            F += weights[:, c][:, None] * np.where(upper, dist.sf(mid), dist.cdf(mid))
        left = np.where(upper, F > target, F < target)      # the root lies above mid
        lo = np.where(left, mid, lo)
        hi = np.where(left, hi, mid)
        if np.all(hi - lo <= 1e-13 * np.maximum(1.0, np.abs(mid))):
            break
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Reference law and quadrature grid
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QuadratureGrid:
    """Quadrature grid of the reference law g_ref (module docstring).

    Attributes
    ----------
    s       : (G,) Gauss–Legendre nodes on (0, 1).
    ws      : (G,) Gauss–Legendre weights on (0, 1), summing to 1.
    power   : exponent p of the endpoint transform t = ψ_p(s).
    t       : (G,) ψ_p(s_g), the nodes in the probability scale of g_ref.
    w       : (G,) ws_g ψ_p'(s_g) — weights of ∫_0^1 · dt, summing to 1.
    nodes   : (G,) y_g = G_ref^{-1}(t_g), increasing.
    ref_pdf : (G,) g_ref(y_g).
    omega   : (G,) ω_g = w_g / g_ref(y_g) — ∫ h(y) dy ≈ Σ_g ω_g h(y_g).
    weights : (C,) mixture weights of the reference law.
    dists   : list of the C frozen component laws.
    """

    s: np.ndarray
    ws: np.ndarray
    power: int
    t: np.ndarray
    w: np.ndarray
    nodes: np.ndarray
    ref_pdf: np.ndarray
    omega: np.ndarray
    weights: np.ndarray
    dists: list

    @property
    def G(self) -> int:
        return int(self.s.size)

    def y_of_s(self, s: np.ndarray) -> np.ndarray:
        """G_ref^{-1}(ψ_p(s)), elementwise (upper tail without cancellation)."""
        s = np.asarray(s, dtype=float)
        t, tc, _ = _psi(s.reshape(1, -1), self.power)
        out = _mixture_ppf(self.weights[None, :], self.dists, t, tc)
        return out.reshape(s.shape)


def _psi(s: np.ndarray, p: int):
    """Endpoint transform t = s^p / (s^p + (1-s)^p): (t, 1 − t, dt/ds).

    p = 1 is the identity. For p ≥ 2, ψ' vanishes to order p − 1 at both ends,
    which turns an integrand behaving like t^β at t → 0 or 1 (an algebraic
    endpoint singularity, typical of a conditional density divided by g_ref)
    into one behaving like s^{p(β+1)−1}, restoring fast Gauss–Legendre
    convergence.
    """
    s = np.asarray(s, dtype=float)
    if p == 1:
        return s, 1.0 - s, np.ones_like(s)
    a, b = s ** p, (1.0 - s) ** p
    den = a + b
    dt = p * (s * (1.0 - s)) ** (p - 1) / (den * den)
    return a / den, b / den, dt


def _reference_mixture(model: PMCModel) -> tuple[np.ndarray, list]:
    """Weights and frozen components of g_ref = Σ_i μ(i, ·)."""
    p = model.prior_p
    K = model.K
    if model.margin_structure == "pair":
        pairs = [(p[i, j], model.margin(i, j)) for i in range(K) for j in range(K)]
    else:
        col = p.sum(axis=0)
        pairs = [(col[j], model.margin(j)) for j in range(K)]
    w = np.array([max(float(c[0]), 0.0) for c in pairs])
    keep = w > 0.0
    if not keep.any():
        raise ValueError("The prior gives zero probability to every state.")
    dists = [_frozen(mg) for (_, mg), k in zip(pairs, keep) if k]
    return w[keep] / w[keep].sum(), dists


#: Exponent p of the endpoint transform ψ_p of the Gauss–Legendre nodes (see
#: :func:`_psi`). p = 2 measured best overall on the Gaussian AR(1) reference
#: (pmcprg/tests/test_gaps.py): p = 1 converges only algebraically for weak or
#: moderate dependence (|ρ| ≲ 0.7), p = 3 under-resolves the interior for
#: ρ ≈ 0.99. Private switch kept for accuracy studies.
_T_POWER = 2


def reference_grid(model: PMCModel, gap_nodes: int | None = None) -> QuadratureGrid:
    """The G-node quadrature grid of the reference law g_ref of ``model``."""
    _check_grid_supported(model)
    G = _resolve_nodes(gap_nodes)
    x, W = _leg.leggauss(G)
    s = 0.5 * (x + 1.0)
    ws = 0.5 * W
    t, tc, dt = _psi(s, _T_POWER)
    w = ws * dt
    weights, dists = _reference_mixture(model)
    nodes = _mixture_ppf(weights[None, :], dists, t[None, :], tc[None, :])[0]
    ref_pdf = np.zeros(G)
    for c, dist in enumerate(dists):
        ref_pdf += weights[c] * dist.pdf(nodes)
    if not np.all(np.isfinite(ref_pdf) & (ref_pdf > 0.0)):
        raise ValueError(
            "reference_grid: the reference density g_ref vanishes at a "
            "quadrature node; the margins' supports leave a gap the grid "
            "cannot represent."
        )
    return QuadratureGrid(s=s, ws=ws, power=_T_POWER, t=t, w=w, nodes=nodes,
                          ref_pdf=ref_pdf, omega=w / ref_pdf, weights=weights,
                          dists=dists)


# ---------------------------------------------------------------------------
# Transition kernel q(j, y' | i, y) on arbitrary pairs — linear and log
# ---------------------------------------------------------------------------

def _margin_eval(model: PMCModel, y: np.ndarray, *, log: bool):
    """f[:, i, j] = f_ij(y) (log f if ``log``), F[:, i, j] = F_ij(y) clipped.

    Shapes (M, K, K); for state margins both are broadcast from the K
    per-state evaluations, as in ``precompute_weights``.
    """
    K = model.K
    M = len(y)
    uses_cop = model.variant.uses_copula
    f = np.empty((M, K, K))
    F = np.empty((M, K, K)) if uses_cop else None
    if model.margin_structure == "pair":
        for ii in range(K):
            for jj in range(K):
                mg = model.margin(ii, jj)
                f[:, ii, jj] = (_inf._margin_logpdf_vec(mg, y) if log else mg.pdf_vec(y))
                if uses_cop:
                    F[:, ii, jj] = mg.cdf_vec(y)
    else:
        for kk in range(K):
            mg = model.margin(kk)
            f[:, kk, :] = (_inf._margin_logpdf_vec(mg, y) if log else mg.pdf_vec(y))[:, None]
            if uses_cop:
                F[:, kk, :] = np.asarray(mg.cdf_vec(y))[:, None]
    if F is not None:
        np.clip(F, EPS, ONE_MINUS_EPS, out=F)
    return f, F


def _kernel(model: PMCModel, fL, FL, fR, FR) -> tuple[np.ndarray, bool]:
    """q(j, y_R | i, y_L) on (M,) pairs — the formulas of ``precompute_weights``.

    Returns ``(Wk, overflow)`` with Wk (M, K, K) cleaned like
    ``precompute_weights`` (non-finite → 0, negatives → 0) and ``overflow``
    True when a raw value was +inf or NaN (the caller then switches to log
    space).
    """
    var = model.variant
    K = model.K
    f_next_T = fR.transpose(0, 2, 1)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        if var.has_markov_prior:
            Wk = model.transition_A[None, :, :] * f_next_T
        else:
            p = model.prior_p
            joint = p[None, :, :] * fL * f_next_T
            D = np.einsum("ij,nij->ni", p, fL)
            Wk = joint / np.maximum(D, MIN_POSITIVE)[:, :, None]
        if var.uses_copula:
            for ii in range(K):
                for jj in range(K):
                    uv = np.column_stack((FL[:, ii, jj], FR[:, jj, ii]))
                    Wk[:, ii, jj] *= model.copula(ii, jj).pdf_array(uv)
    bad = ~np.isfinite(Wk)
    overflow = bool(bad.any())
    if overflow:
        Wk[bad] = 0.0
    np.clip(Wk, 0.0, None, out=Wk)
    return Wk, overflow


def _log_kernel(model: PMCModel, lfL, FL, lfR, FR) -> np.ndarray:
    """log q(j, y_R | i, y_L) on (M,) pairs — the formulas of ``_log_transition_weights``."""
    var = model.variant
    K = model.K
    lf_next_T = lfR.transpose(0, 2, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = model.prior_p
        log_p = np.log(p)
        if var.has_markov_prior:
            L = np.log(model.transition_A)[None, :, :] + lf_next_T
        elif model.margin_structure == "pair":
            left = log_p[None, :, :] + lfL
            log_d = _inf._lse(left, axis=2)
            L = left - log_d[:, :, None] + lf_next_T
        else:
            log_kernel = log_p - np.log(p.sum(axis=1, keepdims=True))
            L = log_kernel[None, :, :] + lf_next_T
        if var.uses_copula:
            for ii in range(K):
                for jj in range(K):
                    uv = np.column_stack((FL[:, ii, jj], FR[:, jj, ii]))
                    L[:, ii, jj] += model.copula(ii, jj).logpdf_array(uv)
    L[np.isnan(L) | (L == np.inf)] = -np.inf
    return L


def _initial(model: PMCModel, f: np.ndarray, *, log: bool) -> np.ndarray:
    """μ(i, y) = α_1(i) at the rows of f (M, K, K) — the ``forward`` initialisation."""
    p = model.prior_p
    if not log:
        if model.margin_structure == "pair":
            return np.einsum("ij,nij->ni", p, f)
        return np.einsum("ij,nji->nj", p, f)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.log(p)
        if model.margin_structure == "pair":
            out = _inf._lse(log_p[None, :, :] + f, axis=2)
        else:
            out = _inf._lse(log_p, axis=0)[None, :] + f[:, :, 0]
    out[np.isnan(out) | (out == np.inf)] = -np.inf
    return out


#: Renormalisation of the quadrature rows of E, Q and α̃_1 (module docstring):
#: ``"block"`` (production) rescales every block (i, g) → (j, ·) to the exact
#: transition probability P(x_{n+1} = j | x_n = i, y_n), then every row to 1;
#: ``"row"`` only rescales rows; ``None`` keeps the raw quadrature. Private
#: switch kept for accuracy studies.
_RENORMALISE = "block"


def _x_transition(model: PMCModel, f: np.ndarray, *, log: bool) -> np.ndarray:
    """P(x_{n+1} = j | x_n = i, y_n) at the rows of f (M, K, K) — log if ``log``.

    A_ij for the HMC variants and for state margins (p_ij / Σ_k p_ik); for
    pair margins p_ij f_ij(y) / Σ_k p_ik f_ik(y) (DerrodePieczynski_CSDA2013
    Eq. 13), with the linear-space guard of ``precompute_weights``.
    """
    M, K = f.shape[0], model.K
    p = model.prior_p
    with np.errstate(divide="ignore", invalid="ignore", under="ignore"):
        if model.variant.has_markov_prior or model.margin_structure != "pair":
            A = model.transition_A if model.variant.has_markov_prior else p / p.sum(axis=1, keepdims=True)
            T = np.broadcast_to(np.log(A) if log else A, (M, K, K))
        elif log:
            left = np.log(p)[None, :, :] + f
            T = left - _inf._lse(left, axis=2)[:, :, None]
        else:
            joint = p[None, :, :] * f
            T = joint / np.maximum(joint.sum(axis=2), MIN_POSITIVE)[:, :, None]
    return T


def _normalise_blocks(B: np.ndarray, target: np.ndarray | None, *, log: bool):
    """Rescale the quadrature tensor B (..., K, G) to its exact block masses.

    ``B[..., j, g]`` holds the weights of the destination (j, y_g) and
    ``target[..., j]`` the exact mass Σ_g of each block (a transition
    probability). Blocks are rescaled to ``target`` (``"block"`` mode), then the
    flattened rows (..., K·G) to sum to 1 (``"block"`` and ``"row"``); a block
    whose quadrature mass vanished keeps 0 and its mass goes to the other
    blocks through the row step.

    Returns ``(rows, factors)``: the (..., K·G) rows and the (..., K) factors
    by which each block was multiplied, in linear scale (the Nyström
    interpolation of :func:`_nystrom_masses` applies them to kernel values off
    the nodes).
    """
    shape = B.shape[:-2] + (B.shape[-2] * B.shape[-1],)
    if _RENORMALISE is None:
        return B.reshape(shape), np.ones(B.shape[:-1])
    with np.errstate(divide="ignore", invalid="ignore", under="ignore", over="ignore"):
        if log:
            fac = np.zeros(B.shape[:-1])
            if _RENORMALISE == "block" and target is not None:
                sb = _inf._lse(B, axis=-1)
                ok = np.isfinite(sb) & np.isfinite(target)
                fac = np.where(ok, np.where(ok, target, 0.0) - np.where(ok, sb, 0.0), -np.inf)
                B = np.where(ok[..., None], B + np.where(ok, fac, 0.0)[..., None], -np.inf)
            R = B.reshape(shape)
            sr = _inf._lse(R, axis=-1)
            fin = np.isfinite(sr)
            R = np.where(fin[..., None], R - np.where(fin, sr, 0.0)[..., None], R)
            fac = np.exp(fac - np.where(fin, sr, 0.0)[..., None])
            return R, np.where(np.isfinite(fac), fac, 0.0)
        fac = np.ones(B.shape[:-1])
        if _RENORMALISE == "block" and target is not None:
            sb = B.sum(axis=-1)
            ok = np.isfinite(sb) & (sb > 0.0)
            fac = np.where(ok, target / np.where(ok, sb, 1.0), 0.0)
            B = B * fac[..., None]
        R = B.reshape(shape)
        sr = R.sum(axis=-1, keepdims=True)
        ok = np.isfinite(sr) & (sr > 0.0)
        R = np.divide(R, np.where(ok, sr, 1.0), out=np.zeros_like(R), where=ok)
        fac = fac / np.where(ok, sr, 1.0)
        return R, np.broadcast_to(fac, B.shape[:-1]).copy()


# ---------------------------------------------------------------------------
# The augmented chain
# ---------------------------------------------------------------------------

class _Chain:
    """Transitions of the chain on the augmented state (module docstring).

    ``trans[n]`` maps position n to n+1 and has shape S_n × S_{n+1}, with
    S_n = K at an observed n and K·G at a missing one (index i·G + g); in log
    space when ``log``. ``init`` has shape (S_0,).
    """

    def __init__(self, model: PMCModel, Y: np.ndarray, miss: np.ndarray,
                 grid: QuadratureGrid, *, log: bool):
        self.model = model
        self.K = K = model.K
        self.G = G = grid.G
        self.N = N = len(Y)
        self.miss = miss
        self.log = log
        self.grid = grid
        self.overflow = False
        self.fac_Q = self.fac_init = None
        self.fac_E = {}
        Y = np.asarray(Y, dtype=float)
        self.Y = Y
        Yf = np.where(miss, grid.nodes[G // 2], Y)
        if log:
            Wobs, _ = _inf._log_transition_weights(model, Yf)
        else:
            Wobs, _ = _inf.precompute_weights(model, Yf)
        self.Wobs = Wobs
        lw = np.log(grid.omega)

        fN, FN = _margin_eval(model, grid.nodes, log=log)

        def kernel(fl, Fl, fr, Fr):
            if log:
                return _log_kernel(model, fl, Fl, fr, Fr)
            Wk, over = _kernel(model, fl, Fl, fr, Fr)
            self.overflow |= over
            return Wk

        def take(F, idx):
            return None if F is None else F[idx]

        m0, m1 = miss[:-1], miss[1:]
        inner = m0 & m1
        entries = np.nonzero(~m0 & m1)[0]
        exits = np.nonzero(m0 & ~m1)[0]

        Q = None
        if inner.any():
            rep, til = np.repeat(np.arange(G), G), np.tile(np.arange(G), G)
            ker = kernel(fN[rep], take(FN, rep), fN[til], take(FN, til)).reshape(G, G, K, K)
            Q = ker.transpose(2, 0, 3, 1)                      # (K, G, K, G)
            Q = (Q + lw[None, None, None, :]) if log else Q * grid.omega[None, None, None, :]
            T = _x_transition(model, fN, log=log).transpose(1, 0, 2)          # (K, G, K)
            Q, self.fac_Q = _normalise_blocks(Q, T, log=log)                   # fac (K, G, K)
            Q = Q.reshape(K * G, K * G)

        E = {}
        if entries.size:
            fo, Fo = _margin_eval(model, Y[entries], log=log)
            e_rep = np.repeat(np.arange(entries.size), G)
            g_til = np.tile(np.arange(G), entries.size)
            ker = kernel(fo[e_rep], take(Fo, e_rep), fN[g_til], take(FN, g_til))
            ker = ker.reshape(entries.size, G, K, K).transpose(0, 2, 3, 1)   # (n_e, K, K, G)
            ker = (ker + lw) if log else ker * grid.omega
            ker, fac = _normalise_blocks(ker, _x_transition(model, fo, log=log), log=log)
            E = {int(n): ker[e] for e, n in enumerate(entries)}
            self.fac_E = {int(n): fac[e] for e, n in enumerate(entries)}           # (K, K)

        X = {}
        if exits.size:
            fo, Fo = _margin_eval(model, Y[exits + 1], log=log)
            x_rep = np.repeat(np.arange(exits.size), G)
            g_til = np.tile(np.arange(G), exits.size)
            ker = kernel(fN[g_til], take(FN, g_til), fo[x_rep], take(Fo, x_rep))
            ker = ker.reshape(exits.size, G, K, K).transpose(0, 2, 1, 3)     # (n_x, K, G, K)
            ker = ker.reshape(exits.size, K * G, K)
            X = {int(n): ker[x] for x, n in enumerate(exits)}

        if miss[0]:
            mu = _initial(model, fN, log=log).T                  # (K, G)
            if not log and not np.all(np.isfinite(mu)):
                self.overflow = True
                mu = np.where(np.isfinite(mu), mu, 0.0)
            mu = (mu + lw[None, :]) if log else mu * grid.omega[None, :]
            prior = _initial(model, np.zeros((1, K, K)) if log else np.ones((1, K, K)), log=log)
            init, fac = _normalise_blocks(mu[None], prior, log=log)
            self.init, self.fac_init = init[0], fac[0]                          # fac (K,)
        else:
            f0, _ = _margin_eval(model, Y[:1], log=log)
            self.init = _initial(model, f0, log=log)[0]

        trans = []
        for n in range(N - 1):
            if not m0[n] and not m1[n]:
                trans.append(Wobs[n])
            elif not m0[n]:
                trans.append(E[n])
            elif m1[n]:
                trans.append(Q)
            else:
                trans.append(X[n])
        self.trans = trans

    def size(self, n: int) -> int:
        return self.K * self.G if self.miss[n] else self.K


def _forward_chain(chain: _Chain):
    """Scaled forward pass; ``None`` when a linear step underflows.

    Returns ``(alphas, log_lik)`` with ``alphas[n]`` the normalised message in
    *linear* scale (S_n,).
    """
    N = chain.N
    alphas = [None] * N
    if chain.log:
        la = chain.init
        lc = float(_inf._lse(la))
        if not np.isfinite(lc):
            raise IncompatibleObservationError(
                f"Forward pass (missing data): Y[0] has zero density under every "
                f"state even in log space (log C_1 = {lc!r})."
            )
        ll = lc
        la = la - lc
        alphas[0] = np.exp(la)
        for n in range(N - 1):
            r = _inf._lse(la[:, None] + chain.trans[n], axis=0)
            lc = float(_inf._lse(r))
            if not np.isfinite(lc):
                raise IncompatibleObservationError(
                    f"Forward pass (missing data): Y[{n + 1}] has zero density "
                    f"under every state reachable from step {n} (log C = {lc!r}), "
                    f"even in log space. The observation sequence is incompatible "
                    f"with the model."
                )
            ll += lc
            la = r - lc
            alphas[n + 1] = np.exp(la)
        return alphas, float(ll)

    a = chain.init
    C = float(a.sum())
    if not (np.isfinite(C) and C >= MIN_POSITIVE):
        return None
    ll = np.log(C)
    alphas[0] = a / C
    for n in range(N - 1):
        raw = alphas[n] @ chain.trans[n]
        C = float(raw.sum())
        if not (np.isfinite(C) and C >= MIN_POSITIVE):
            return None
        ll += np.log(C)
        alphas[n + 1] = raw / C
    return alphas, float(ll)


def _backward_chain(chain: _Chain):
    """Scaled backward pass (own normaliser per step); ``None`` on linear underflow."""
    N = chain.N
    betas = [None] * N
    S = chain.size(N - 1)
    if chain.log:
        lb = np.full(S, -np.log(S))
        betas[N - 1] = np.exp(lb)
        n_void = 0
        for n in range(N - 2, -1, -1):
            r = _inf._lse(chain.trans[n] + lb[None, :], axis=1)
            ld = float(_inf._lse(r))
            if np.isfinite(ld):
                lb = r - ld
            else:
                n_void += 1
                s = chain.size(n)
                lb = np.full(s, -np.log(s))
            betas[n] = np.exp(lb)
        if n_void:
            logger.warning(
                "Backward (missing data, log space): %d step(s) with zero weight "
                "under every state — β̂ reset to uniform there.", n_void,
            )
        return betas
    b = np.full(S, 1.0 / S)
    betas[N - 1] = b
    for n in range(N - 2, -1, -1):
        raw = chain.trans[n] @ betas[n + 1]
        D = float(raw.sum())
        if not (np.isfinite(D) and D >= MIN_POSITIVE):
            return None
        betas[n] = raw / D
    return betas


def _run_chain(model, Y, miss, grid, *, backward: bool):
    """Build the chain and run forward (and backward), linear first, log on failure."""
    chain = _Chain(model, Y, miss, grid, log=False)
    fw = bw = None
    if not chain.overflow:
        fw = _forward_chain(chain)
        if fw is not None and backward:
            bw = _backward_chain(chain)
    if chain.overflow or fw is None or (backward and bw is None):
        logger.warning(
            "Missing-data pass: %s in linear space; recomputing in log space.",
            "a kernel value overflows" if chain.overflow else "a step underflows",
        )
        chain = _Chain(model, Y, miss, grid, log=True)
        fw = _forward_chain(chain)
        bw = _backward_chain(chain) if backward else None
    return chain, fw[0], fw[1], bw


# ---------------------------------------------------------------------------
# Posterior summaries of the augmented chain
# ---------------------------------------------------------------------------

@dataclass
class GapPosterior:
    """Posterior quantities of a sequence with missing observations.

    Attributes
    ----------
    miss       : (N,) bool — missing rows.
    log_lik    : float — log p(y_obs), the observed-data log-likelihood.
    alpha_hat  : (N, K) — P(x_n = i | observations up to n).
    beta_hat   : (N, K) — normalised backward messages (module docstring for
                 their definition at a missing n).
    gamma      : (N, K) — P(x_n = i | y_obs).
    xi         : (N-1, K, K) or None — P(x_n = i, x_{n+1} = j | y_obs).
    method     : ``"exact"`` (K-state shortcut) or ``"grid"``.
    grid       : QuadratureGrid or None (exact shortcut).
    node_post  : (M, K, G) or None — P(x_n = i, y_n ∈ node g | y_obs) at the
                 M missing positions ``np.nonzero(miss)[0]`` (grid variants).
    """

    miss: np.ndarray
    log_lik: float
    alpha_hat: np.ndarray
    beta_hat: np.ndarray
    gamma: np.ndarray
    xi: np.ndarray | None
    method: str
    grid: QuadratureGrid | None = None
    node_post: np.ndarray | None = None

    @property
    def index(self) -> np.ndarray:
        """Positions of the missing rows."""
        return np.nonzero(self.miss)[0]


def _marginal_alpha(chain: _Chain, alphas) -> np.ndarray:
    K, G = chain.K, chain.G
    out = np.empty((chain.N, K))
    for n in range(chain.N):
        a = alphas[n]
        out[n] = a.reshape(K, G).sum(axis=1) if chain.miss[n] else a
    return out


def _chain_posterior(chain: _Chain, alphas, betas, *, want_xi: bool) -> dict:
    K, G, N, miss = chain.K, chain.G, chain.N, chain.miss
    alpha_hat = _marginal_alpha(chain, alphas)
    beta_hat = np.empty((N, K))
    gamma = np.empty((N, K))
    idx = np.nonzero(miss)[0]
    node_post = np.empty((idx.size, K, G))
    obs = np.nonzero(~miss)[0]
    if obs.size:
        A = np.array([alphas[n] for n in obs])
        B = np.array([betas[n] for n in obs])
        beta_hat[obs] = B
        gamma[obs] = _inf.smooth(A, B)
    for m, n in enumerate(idx):
        a = alphas[n].reshape(K, G)
        b = betas[n].reshape(K, G)
        joint = a * b
        tot = joint.sum()
        if np.isfinite(tot) and tot > 0.0:
            node_post[m] = joint / tot
        else:
            node_post[m] = a / max(a.sum(), MIN_POSITIVE)
        gamma[n] = node_post[m].sum(axis=1)
        am = a.sum(axis=1)
        bm = np.where(am > 0.0, joint.sum(axis=1) / np.where(am > 0.0, am, 1.0), b.mean(axis=1))
        s = bm.sum()
        beta_hat[n] = bm / s if (np.isfinite(s) and s > 0.0) else np.full(K, 1.0 / K)

    xi = None
    if want_xi and N > 1:
        xi = np.empty((N - 1, K, K))
        with np.errstate(divide="ignore", under="ignore", over="ignore", invalid="ignore"):
            for n in range(N - 1):
                T = chain.trans[n]
                if chain.log:
                    la = np.log(alphas[n])[:, None]
                    lb = np.log(betas[n + 1])[None, :]
                    L = la + T + lb
                    mx = np.max(L)
                    J = np.exp(L - mx) if np.isfinite(mx) else np.zeros_like(L)
                else:
                    J = alphas[n][:, None] * T * betas[n + 1][None, :]
                if miss[n]:
                    J = J.reshape(K, G, -1).sum(axis=1)
                if miss[n + 1]:
                    J = J.reshape(K, K, G).sum(axis=2)
                s = J.sum()
                if np.isfinite(s) and s > 0.0:
                    xi[n] = J / s
                else:
                    xi[n] = gamma[n][:, None] * gamma[n + 1][None, :]
    return dict(alpha_hat=alpha_hat, beta_hat=beta_hat, gamma=gamma, xi=xi,
                node_post=node_post)


def _check_length(Y) -> None:
    if len(Y) < 2:
        raise ValueError(
            f"Inference with missing observations requires N ≥ 2 rows, got N={len(Y)}."
        )


def gap_posterior(
    model: PMCModel,
    Y: np.ndarray,
    *,
    gap_nodes: int | None = None,
    xi: bool = True,
) -> GapPosterior:
    """Exact (up to quadrature) posterior quantities of a NaN-bearing Y.

    Works for every variant and for a Y without missing values too (then
    identical to the functions of :mod:`pmcprg.pmc.inference`). The exact
    shortcut variants use ``precompute_weights`` / ``forward`` / ``backward``
    / ``smooth`` / ``joint_posteriors`` on the marginalised weights; the grid
    variants run the augmented chain (module docstring).

    Parameters
    ----------
    model     : PMCModel — known parameters.
    Y         : (N,) or (N, d) observations, NaN (any non-finite value) where missing.
    gap_nodes : number G of quadrature nodes (default 64; grid variants only).
    xi        : also compute the pairwise posteriors ξ (default True).
    """
    return _posterior(model, Y, gap_nodes, xi)[0]


def _posterior(model, Y, gap_nodes, xi):
    """:func:`gap_posterior` plus the run it came from.

    Returns ``(GapPosterior, run)`` with ``run = (W, None, None)`` for the
    exact shortcut (the marginalised weights) and ``(chain, alphas, betas)``
    for the grid.
    """
    Y = np.asarray(Y, dtype=float)
    _check_length(Y)
    miss = missing_mask(Y)
    if not needs_grid(model) or not miss.any():
        W, f_pdf = _inf.precompute_weights(model, Y)
        a, ll = _inf.forward(model, Y, W=W, f_pdf=f_pdf)
        b = _inf.backward(model, Y, W=W)
        g = _inf.smooth(a, b)
        x = _inf.joint_posteriors(a, W, b) if xi else None
        return (GapPosterior(miss=miss, log_lik=float(ll), alpha_hat=a, beta_hat=b,
                             gamma=g, xi=x, method="exact"), (W, None, None))
    _check_grid_supported(model)
    grid = reference_grid(model, gap_nodes)
    chain, alphas, ll, betas = _run_chain(model, Y, miss, grid, backward=True)
    post = _chain_posterior(chain, alphas, betas, want_xi=xi)
    return (GapPosterior(miss=miss, log_lik=float(ll), method="grid", grid=grid, **post),
            (chain, alphas, betas))


# ---- entry points used by pmcprg.pmc.inference --------------------------------

def _grid_forward(model, Y, miss, gap_nodes):
    _check_length(Y)
    _check_grid_supported(model)
    grid = reference_grid(model, gap_nodes)
    chain, alphas, ll, _ = _run_chain(model, np.asarray(Y, dtype=float), miss, grid,
                                      backward=False)
    return _marginal_alpha(chain, alphas), ll


def _grid_backward(model, Y, miss, gap_nodes):
    return gap_posterior(model, Y, gap_nodes=gap_nodes, xi=False).beta_hat


def _grid_sample(model, Y, miss, rng, gap_nodes, return_y):
    Y = np.asarray(Y, dtype=float)
    _check_length(Y)
    _check_grid_supported(model)
    grid = reference_grid(model, gap_nodes)
    chain, alphas, _, _ = _run_chain(model, Y, miss, grid, backward=False)
    U = _ffbs(chain.trans, alphas, chain.log, rng, 1)[0]
    X, Yc = _decode_path(U, miss, chain.G, grid.nodes, Y)
    return (X, Yc) if return_y else X


# ---------------------------------------------------------------------------
# Forward-filter backward-sample on a chain of variable width (vectorised)
# ---------------------------------------------------------------------------

def _draw_rows(w: np.ndarray, rng: np.random.Generator, fallback: np.ndarray) -> np.ndarray:
    """One index per row of w (S, M), with probability ∝ the row."""
    S, M = w.shape
    tot = w.sum(axis=1)
    bad = ~(np.isfinite(tot) & (tot > 0.0))
    if bad.any():
        fb = np.asarray(fallback, dtype=float)
        s = fb.sum()
        fb = fb / s if (np.isfinite(s) and s > 0.0) else np.full(M, 1.0 / M)
        w = w.copy()
        w[bad] = fb
    cum = np.cumsum(w, axis=1)
    u = rng.random(S) * cum[:, -1]
    idx = np.sum(cum <= u[:, None], axis=1)
    over = idx >= M
    if over.any():
        last_pos = M - 1 - np.argmax(w[over, ::-1] > 0.0, axis=1)
        idx[over] = last_pos
    return idx


def _ffbs(trans, alphas, log: bool, rng: np.random.Generator, S: int) -> np.ndarray:
    """S joint draws of the (augmented) path; (S, N) indices."""
    N = len(alphas)
    U = np.empty((S, N), dtype=int)
    last = np.broadcast_to(alphas[N - 1], (S, alphas[N - 1].size))
    U[:, N - 1] = _draw_rows(np.array(last), rng, alphas[N - 1])
    with np.errstate(divide="ignore", under="ignore", over="ignore", invalid="ignore"):
        for n in range(N - 2, -1, -1):
            T = trans[n][:, U[:, n + 1]].T                 # (S, S_n)
            if log:
                L = np.log(alphas[n])[None, :] + T
                mx = np.max(L, axis=1, keepdims=True)
                w = np.where(np.isfinite(mx), np.exp(L - np.where(np.isfinite(mx), mx, 0.0)), 0.0)
            else:
                w = alphas[n][None, :] * T
            U[:, n] = _draw_rows(w, rng, alphas[n])
    return U


def _decode_path(U: np.ndarray, miss: np.ndarray, G: int, nodes: np.ndarray, Y: np.ndarray):
    """Augmented indices → (states, Y with the missing rows set to the drawn nodes)."""
    X = np.where(miss, U // G, U).astype(int)
    Yc = np.array(Y, dtype=float, copy=True)
    Yc[miss] = nodes[U[miss] % G]
    return X, Yc


# ---------------------------------------------------------------------------
# Nyström interpolation of the posterior density of a missing observation
# ---------------------------------------------------------------------------

#: Size of the fine grid of the Nyström interpolation, in multiples of G.
_NYSTROM_FACTOR = 4


def _nystrom_masses(chain: _Chain, alphas, betas, positions: np.ndarray,
                    fine: QuadratureGrid) -> np.ndarray | None:
    """Posterior masses of y_n on the fine grid, for the missing ``positions``.

    The density of y_n = y given the observations follows from the messages
    of the neighbouring positions through the kernels, at any y:

        π_n(y) ∝ Σ_i F_n(i, y) B_n(i, y),
        F_n(i, y) = Σ_u α̃_{n-1}(u) φ_u(i) q(i, y | u),
        B_n(i, y) = Σ_v T_{(i, y) → v} β̃_{n+1}(v),

    with u, v the (augmented) states at n − 1 and n + 1, φ_u(i) the block
    factors applied by the chain to the row u (:func:`_normalise_blocks`) and
    T_{(i, y) → v} the chain's transition out of (i, y), built like a row of Q
    (or the exit density q(j, y_{n+1} | i, y) when y_{n+1} is observed).
    ``F_n(i, y_g) ω_g`` and ``B_n(i, y_g)`` are the chain's own messages at the
    nodes, so π_n interpolates the node posterior with the exact kernels
    (Nyström 1930) — accurate to the quadrature error of the messages, where
    a polynomial interpolation of the node masses loses half of the degree.
    ``betas`` None means B ≡ 1 (a trailing gap, for :func:`forecast`).

    Returns (P, M) masses π_n(y_f) ω_f normalised per row, or ``None`` when a
    density is not representable in linear scale (the caller then uses the
    coarse node masses).
    """
    model, K, G = chain.model, chain.K, chain.G
    miss, N, Y = chain.miss, chain.N, chain.Y
    M = fine.G
    positions = np.asarray(positions, dtype=int)
    P = positions.size
    fN, FN = _margin_eval(model, chain.grid.nodes, log=False)
    fF, FF = _margin_eval(model, fine.nodes, log=False)

    def take(F, idx):
        return None if F is None else F[idx]

    def kern(fl, Fl, fr, Fr):
        return _kernel(model, fl, Fl, fr, Fr)[0]

    with np.errstate(divide="ignore", invalid="ignore", under="ignore", over="ignore"):
        F = np.zeros((P, K, M))
        B = np.ones((P, K, M))
        prev_first = positions == 0
        prev_miss = np.zeros(P, bool)
        prev_miss[~prev_first] = miss[positions[~prev_first] - 1]
        next_last = positions == N - 1
        next_miss = np.zeros(P, bool)
        next_miss[~next_last] = miss[positions[~next_last] + 1]

        # ---- incoming side ------------------------------------------------
        if prev_first.any():
            mu = _initial(model, fF, log=False).T * chain.fac_init[:, None]      # (K, M)
            F[prev_first] = mu[None]
        sel = np.nonzero(~prev_first & ~prev_miss)[0]
        if sel.size:
            n_prev = positions[sel] - 1
            fo, Fo = _margin_eval(model, Y[n_prev], log=False)
            rep_, til = np.repeat(np.arange(sel.size), M), np.tile(np.arange(M), sel.size)
            ker = kern(fo[rep_], take(Fo, rep_), fF[til], take(FF, til)).reshape(sel.size, M, K, K)
            fac = np.array([chain.fac_E[int(n)] for n in n_prev])                # (P', K, K)
            ker = ker * fac[:, None, :, :]                                         # [p, f, h, i]
            a = np.array([alphas[int(n)] for n in n_prev])                        # (P', K)
            F[sel] = np.einsum("ph,pfhi->pif", a, ker)
        sel = np.nonzero(~prev_first & prev_miss)[0]
        if sel.size:
            rep_, til = np.repeat(np.arange(G), M), np.tile(np.arange(M), G)
            ker = kern(fN[rep_], take(FN, rep_), fF[til], take(FF, til)).reshape(G, M, K, K)
            C = ker.transpose(2, 0, 3, 1) * chain.fac_Q[:, :, :, None]           # (K, G, K, M)
            C = C.reshape(K * G, K * M)
            a = np.array([alphas[int(n) - 1] for n in positions[sel]])            # (P', KG)
            F[sel] = (a @ C).reshape(sel.size, K, M)

        # ---- outgoing side ------------------------------------------------
        if betas is not None:
            sel = np.nonzero(~next_last & ~next_miss)[0]
            if sel.size:
                n_next = positions[sel] + 1
                fo, Fo = _margin_eval(model, Y[n_next], log=False)
                rep_, til = np.repeat(np.arange(sel.size), M), np.tile(np.arange(M), sel.size)
                ker = kern(fF[til], take(FF, til), fo[rep_], take(Fo, rep_)).reshape(sel.size, M, K, K)
                b = np.array([betas[int(n)] for n in n_next])                     # (P', K)
                B[sel] = np.einsum("pfij,pj->pif", ker, b)
            sel = np.nonzero(~next_last & next_miss)[0]
            if sel.size:
                rep_, til = np.repeat(np.arange(M), G), np.tile(np.arange(G), M)
                ker = kern(fF[rep_], take(FF, rep_), fN[til], take(FN, til)).reshape(M, G, K, K)
                T = ker.transpose(2, 0, 3, 1) * chain.grid.omega                    # (K, M, K, G)
                T, _ = _normalise_blocks(T, _x_transition(model, fF, log=False).transpose(1, 0, 2),
                                         log=False)
                T = T.reshape(K * M, K * G)
                b = np.array([betas[int(n) + 1] for n in positions[sel]])          # (P', KG)
                B[sel] = (b @ T.T).reshape(sel.size, K, M)
            # the chain rescales β̃ by its own sum at every step: any positive
            # constant per position cancels in the normalisation below.

        mass = (F * B).sum(axis=1) * fine.omega[None, :]
        tot = mass.sum(axis=1, keepdims=True)
    if not (np.all(np.isfinite(mass)) and np.all(tot > 0.0)):
        return None
    return mass / tot


# ---------------------------------------------------------------------------
# Laws on the grid: moments and quantiles
# ---------------------------------------------------------------------------

def _grid_summary(mass: np.ndarray, grid: QuadratureGrid, qs: tuple[float, ...],
                  fine_mass: np.ndarray | None = None, fine: QuadratureGrid | None = None):
    """Mean, sd and quantiles of the laws with node masses ``mass`` (M, G).

    Moments are the quadratures Σ_g mass_g y_g^k. Quantiles invert the CDF of
    the Legendre interpolant, in the Gauss–Legendre variable s (y =
    G_ref^{-1}(ψ_p(s))), of the density h(s) = mass_g / ws_g — spectrally
    accurate for a smooth h — by bisection on s, then map s to y. With
    ``fine_mass`` (the Nyström masses on the grid ``fine``) the quantiles use
    that finer representation instead.
    """
    mass = np.asarray(mass, dtype=float)
    tot = mass.sum(axis=1, keepdims=True)
    mass = mass / np.where(tot > 0.0, tot, 1.0)
    y = grid.nodes
    mean = mass @ y
    var = mass @ (y * y) - mean * mean
    sd = np.sqrt(np.maximum(var, 0.0))
    M = mass.shape[0]
    if not qs:
        return mean, sd, np.empty((M, 0))
    if fine_mass is not None:
        mass, grid = np.asarray(fine_mass, dtype=float), fine
    G = grid.G
    x = 2.0 * grid.s - 1.0
    V = _leg.legvander(x, G - 1)                                   # (G, G)
    c = (mass @ V) * (2.0 * np.arange(G) + 1.0)                    # (M, G)
    cint = _leg.legint(c, lbnd=-1, axis=1).T                       # (G+1, M)
    target = np.array(qs)[:, None] * np.ones((1, M))              # (Q, M)
    lo = np.zeros_like(target)
    hi = np.ones_like(target)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        H = 0.5 * _leg.legval(2.0 * mid - 1.0, cint, tensor=False)
        below = H < target
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return mean, sd, grid.y_of_s(0.5 * (lo + hi)).T                # (M, Q)


def _state_dists(model: PMCModel) -> list:
    return [_frozen(model.margin(k)) for k in range(model.K)]


def _mixture_summary(model: PMCModel, weights: np.ndarray, qs: tuple[float, ...]):
    """Mean, sd, quantiles of Σ_k weights[m, k] f_k (state margins, exact).

    d = 1: shapes (M,), (M,), (M, Q). d > 1 (multivariate normal margins):
    (M, d), (M, d) and per-component quantiles (M, Q, d).
    """
    K = model.K
    d = getattr(model, "d", 1)
    weights = np.asarray(weights, dtype=float)
    M = weights.shape[0]
    if d == 1:
        dists = _state_dists(model)
        mu = np.array([float(dd.mean()) for dd in dists])
        v = np.array([float(dd.var()) for dd in dists])
        mean = weights @ mu
        sd = np.sqrt(np.maximum(weights @ (v + mu * mu) - mean * mean, 0.0))
        qv = (_mixture_ppf(weights, dists, np.broadcast_to(np.array(qs), (M, len(qs))))
              if qs else np.empty((M, 0)))
        return mean, sd, qv
    mus = np.array([model.margin(k).params["mean"] for k in range(K)], dtype=float)       # (K, d)
    vs = np.array([np.diag(np.asarray(model.margin(k).params["cov"], dtype=float))
                   for k in range(K)])                                                     # (K, d)
    mean = weights @ mus
    sd = np.sqrt(np.maximum(weights @ (vs + mus * mus) - mean * mean, 0.0))
    qv = np.empty((M, len(qs), d))
    for c in range(d):
        dists = [_ss.norm(loc=mus[k, c], scale=np.sqrt(vs[k, c])) for k in range(K)]
        if qs:
            qv[:, :, c] = _mixture_ppf(weights, dists, np.broadcast_to(np.array(qs), (M, len(qs))))
    return mean, sd, qv


def _mixture_density(model: PMCModel, weights: np.ndarray, y: np.ndarray) -> np.ndarray:
    dists = _state_dists(model)
    dens = np.stack([dd.pdf(y) for dd in dists], axis=0)          # (K, G)
    return np.asarray(weights, dtype=float) @ dens


# ---------------------------------------------------------------------------
# Imputation
# ---------------------------------------------------------------------------

@dataclass
class Imputation:
    """Posterior law of the missing observations given all the observed ones.

    Attributes
    ----------
    index           : (M,) positions of the missing rows.
    mean, sd        : (M,) — or (M, d) — posterior mean and standard deviation
                      of y_n given y_obs.
    quantiles       : the requested levels.
    quantile_values : (M, Q) — or (M, Q, d), per component — posterior quantiles.
    gamma           : (M, K) — P(x_n = i | y_obs) at the missing positions.
    log_lik         : float — log p(y_obs).
    Y_mean          : Y with every missing row replaced by its posterior mean.
    method          : ``"exact"`` or ``"grid"``.
    nodes           : (G,) or None — grid y_g (d = 1).
    density         : (M, G) or None — posterior density of y_n at the nodes
                      (the exact mixture density for the shortcut variants,
                      mass / ω on the grid otherwise).
    x_samples       : (S, N) int or None — FFBS draws of the whole state path.
    y_samples       : (S, M) — or (S, M, d) — or None: the matching draws of the
                      missing observations (exact continuous draws for the
                      shortcut variants; drawn on the quadrature nodes — the
                      discrete law whose moments are ``mean`` and ``sd`` — for
                      the grid variants).
    """

    index: np.ndarray
    mean: np.ndarray
    sd: np.ndarray
    quantiles: tuple
    quantile_values: np.ndarray
    gamma: np.ndarray
    log_lik: float
    Y_mean: np.ndarray
    method: str
    nodes: np.ndarray | None = None
    density: np.ndarray | None = None
    x_samples: np.ndarray | None = None
    y_samples: np.ndarray | None = None


def impute(
    model: PMCModel,
    Y: np.ndarray,
    *,
    gap_nodes: int | None = DEFAULT_GAP_NODES,
    quantiles=DEFAULT_QUANTILES,
    n_samples: int = 0,
    rng=None,
) -> Imputation:
    """Posterior law of every missing y_n given all the observed data.

    Parameters
    ----------
    model     : PMCModel — known parameters.
    Y         : (N,) or (N, d) with NaN (any non-finite value) at missing rows.
    gap_nodes : quadrature nodes G for the grid variants (default 64).
    quantiles : levels in (0, 1) of the reported posterior quantiles.
    n_samples : number S of joint FFBS draws of (x_{1:N}, y_miss) (default 0).
    rng       : seed or :class:`numpy.random.Generator` for the draws.

    Returns
    -------
    Imputation — see its docstring. A Y without missing rows gives empty
    per-position arrays.
    """
    Y = np.asarray(Y, dtype=float)
    qs = _check_quantiles(quantiles)
    n_samples = int(n_samples)
    if n_samples < 0:
        raise ValueError(f"n_samples must be ≥ 0, got {n_samples}.")
    post, run = _posterior(model, Y, gap_nodes, False)
    idx = post.index
    d = getattr(model, "d", 1)
    nodes = density = None
    if post.method == "grid":
        grid = post.grid
        mass = post.node_post.sum(axis=1)                              # (M, G)
        fine_mass = fine = None
        if qs and idx.size:
            chain, alphas, betas = run
            fine = reference_grid(model, _NYSTROM_FACTOR * grid.G)
            fine_mass = _nystrom_masses(chain, alphas, betas, idx, fine)
        mean, sd, qv = _grid_summary(mass, grid, qs, fine_mass, fine)
        nodes, density = grid.nodes, mass / grid.omega[None, :]
    elif idx.size == 0:
        # Complete data (any variant): nothing to impute.
        tail = (d,) if d > 1 else ()
        mean, sd = np.empty((0,) + tail), np.empty((0,) + tail)
        qv = np.empty((0, len(qs)) + tail)
    else:
        w = post.gamma[idx]
        mean, sd, qv = _mixture_summary(model, w, qs)
        if d == 1:
            nodes = reference_grid(model, gap_nodes).nodes
            density = _mixture_density(model, w, nodes)
    Y_mean = np.array(Y, copy=True)
    Y_mean[idx] = mean

    x_s = y_s = None
    if n_samples > 0:
        gen = np.random.default_rng(rng)
        miss = post.miss
        if post.method == "grid":
            chain, alphas, _ = run
            U = _ffbs(chain.trans, alphas, chain.log, gen, n_samples)
            x_s = np.where(miss[None, :], U // chain.G, U).astype(int)
            y_s = post.grid.nodes[U[:, idx] % chain.G]
        else:
            W = run[0]
            trans = [W[n] for n in range(len(Y) - 1)]
            alphas = list(post.alpha_hat)
            x_s = _ffbs(trans, alphas, False, gen, n_samples)
            y_s = _draw_state_margins(model, x_s[:, idx], gen)
    return Imputation(index=idx, mean=mean, sd=sd, quantiles=qs, quantile_values=qv,
                      gamma=post.gamma[idx], log_lik=post.log_lik, Y_mean=Y_mean,
                      method=post.method, nodes=nodes, density=density,
                      x_samples=x_s, y_samples=y_s)


def _draw_state_margins(model: PMCModel, states: np.ndarray, rng) -> np.ndarray:
    """y ~ f_{state} elementwise for integer ``states`` of any shape."""
    d = getattr(model, "d", 1)
    out = np.empty(states.shape + ((d,) if d > 1 else ()))
    for k in range(model.K):
        sel = states == k
        cnt = int(sel.sum())
        if cnt:
            draws = np.asarray(model.margin(k).rvs(cnt, rng), dtype=float)
            out[sel] = draws.reshape((cnt, d) if d > 1 else (cnt,))
    return out


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------

@dataclass
class Forecast:
    """h-step predictive law of (x_{N+k}, y_{N+k}) given the observed part of Y.

    Attributes
    ----------
    h               : horizon.
    state_probs     : (h, K) — P(x_{N+k} = j | y_obs), k = 1..h.
    mean, sd        : (h,) — or (h, d) — predictive mean and standard deviation.
    quantiles       : the requested levels.
    quantile_values : (h, Q) — or (h, Q, d) — predictive quantiles.
    log_lik         : float — log p(y_obs) of the conditioning sequence.
    method          : ``"exact"`` or ``"grid"``.
    nodes           : (G,) or None — quadrature nodes y_g (d = 1).
    density         : (h, G) or None — predictive density at the nodes.
    """

    h: int
    state_probs: np.ndarray
    mean: np.ndarray
    sd: np.ndarray
    quantiles: tuple
    quantile_values: np.ndarray
    log_lik: float
    method: str
    nodes: np.ndarray | None = None
    density: np.ndarray | None = None


def forecast(
    model: PMCModel,
    Y: np.ndarray,
    h: int,
    *,
    gap_nodes: int | None = DEFAULT_GAP_NODES,
    quantiles=DEFAULT_QUANTILES,
) -> Forecast:
    """Predictive laws of the next h observations, a trailing gap of length h.

    Runs the forward filter on Y followed by h missing rows: the normalised
    filter at N + k is P(x_{N+k}, y_{N+k} | y_obs) (no backward message is
    needed for a trailing gap). Y may contain missing rows.

    Parameters
    ----------
    model     : PMCModel — known parameters.
    Y         : (N,) or (N, d) conditioning observations (N ≥ 1).
    h         : horizon ≥ 1.
    gap_nodes : quadrature nodes G for the grid variants (default 64).
    quantiles : levels in (0, 1) of the predictive quantiles.
    """
    Y = np.asarray(Y, dtype=float)
    qs = _check_quantiles(quantiles)
    h = int(h)
    if h < 1:
        raise ValueError(f"forecast horizon h must be ≥ 1, got {h}.")
    if len(Y) < 1:
        raise ValueError("forecast needs at least one row of Y (it may be NaN).")
    tail = np.full((h,) + Y.shape[1:], np.nan)
    Yx = np.concatenate([Y, tail], axis=0)
    N = len(Y)
    miss = missing_mask(Yx)
    d = getattr(model, "d", 1)
    nodes = density = None
    if needs_grid(model):
        _check_grid_supported(model)
        grid = reference_grid(model, gap_nodes)
        chain, alphas, ll, _ = _run_chain(model, Yx, miss, grid, backward=False)
        K, G = chain.K, chain.G
        joint = np.array([alphas[N + k].reshape(K, G) for k in range(h)])   # (h, K, G)
        joint /= joint.sum(axis=(1, 2), keepdims=True)
        state_probs = joint.sum(axis=2)
        mass = joint.sum(axis=1)
        fine = reference_grid(model, _NYSTROM_FACTOR * grid.G) if qs else None
        fine_mass = (_nystrom_masses(chain, alphas, None, np.arange(N, N + h), fine)
                     if qs else None)
        mean, sd, qv = _grid_summary(mass, grid, qs, fine_mass, fine)
        method = "grid"
        nodes, density = grid.nodes, mass / grid.omega[None, :]
    else:
        W, f_pdf = _inf.precompute_weights(model, Yx)
        a, ll = _inf.forward(model, Yx, W=W, f_pdf=f_pdf)
        state_probs = a[N:]
        state_probs = state_probs / state_probs.sum(axis=1, keepdims=True)
        mean, sd, qv = _mixture_summary(model, state_probs, qs)
        method = "exact"
        if d == 1:
            nodes = reference_grid(model, gap_nodes).nodes
            density = _mixture_density(model, state_probs, nodes)
    return Forecast(h=h, state_probs=state_probs, mean=mean, sd=sd, quantiles=qs,
                    quantile_values=qv, log_lik=float(ll), method=method,
                    nodes=nodes, density=density)
