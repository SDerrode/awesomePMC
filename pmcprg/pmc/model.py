"""
PMCModel — load, validate, and access PMC/HMC model parameters from TOML.

Supported variants (A16 §2.1, Eqs. 5–9)
---------------------------------------
HMC-IN   : hidden Markov chain, independent noise, p(y_n | x_n).
HMC-IN2  : hidden Markov chain, independent noise of order 2, p(y_n | x_{n-1}, x_n).
HMC-DN   : hidden Markov chain, dependent noise (copula between y_{n-1} and y_n).
PMC-IN   : pairwise Markov chain, independent noise.
PMC      : general pairwise Markov chain, copula-based transitions.

Margin structure
----------------
The package works with stationary reversible PMCs (SR-PMC, A16 §2.1–2.2),
characterised by the joint prior p_ij = p(x_1 = i, x_2 = j) and by K²
bivariate densities (A16 Eq. 12)

    f_ij(y_1, y_2) = f_ij(y_1) · f_ji(y_2) · c_ij(F_ij(y_1), F_ji(y_2)),

whose margins f_ij(y) = p(y_1 | x_1 = i, x_2 = j) are indexed by the *pair*
of states. Reversibility only makes the right margin of the pair (i, j) the
left margin of (j, i) — the index inversion f_ji of Eq. 12; it does not make
f_ij independent of j. The Proposition of A16 §2.1 states that for an SR-PMC

    X is a Markov chain  ⇔  p(y_2 | x_1, x_2) = p(y_2 | x_2)
                         ⇔  p(y_n | x_{1:N}) = p(y_n | x_n) for all n.

A :class:`PMCModel` therefore has one of two margin structures
(:attr:`PMCModel.margin_structure`):

* ``"state"`` — K densities f_i (f_ij = f_i). By the Proposition this is
  exactly the case where X is Markov: the transition
  p(x_{n+1} = j | x_n = i, y_n) ∝ p_ij f_i(y_n) (A16 Eq. 13) no longer
  depends on y_n, so a PMC with state margins is an SR HMC-DN, and a PMC-IN
  with state margins an HMC-IN. Every HMC-* variant has state margins.
* ``"pair"`` — K² densities f_ij: the general PMC of A16 Eqs. 12–14, where
  X is not Markov. Allowed for the PMC and PMC-IN variants only.

Versions 0.5.0–0.8.x collapsed K² margins to f_ij = f_i for every variant,
attributing the collapse to reversibility; that made the package's "PMC" an
SR HMC-DN. The collapse is still available on request
(``[model].margin_structure = "state"``).

TOML format (see pmcprg/pmc/models/*.toml for examples)
------------------------------------------------------
[model]
name      = "My model"
variant   = "PMC"        # one of: HMC-IN | HMC-IN2 | HMC-DN | PMC-IN | PMC
K         = 2
N_default = 5000
# margin_structure = "state"  # optional, "state" | "pair"; default: inferred
#                             # from [[margins]] (K blocks → state, K² → pair)

[prior]
# HMC-* variants → key "A" (K×K row-stochastic transition matrix)
# PMC-* variants → key "p" (K×K joint distribution, sums to 1)
p = [[0.45, 0.05], [0.05, 0.45]]

# State margins: K blocks keyed by i only.
[[margins]]
i      = 0
dist   = "norm"
params = {loc = 0.0, scale = 1.0}
# … one block per state

# — or — pair margins: K² blocks keyed by (i, j), f_ij = p(y_n | x_n=i, x_{n+1}=j).
# [[margins]]
# i = 0
# j = 1
# dist   = "norm"
# params = {loc = 0.3, scale = 1.6}
# … one block per pair; an optional ``candidates`` list (GICE) per block.

[[copulas]]               # only for HMC-DN and PMC; K² blocks
i   = 0
j   = 0
name = "Gauss"            # SHORT_NAME from CopulaEnum
tau  = 0.7
# optional extra params: df = 4.0 (Student), delta = 1.5 (BB1), psi = 1.0 (Tawn1/Tawn2)

Reference
---------
A16 — Derrode, S. & Pieczynski, W. (2013). Unsupervised data classification
using pairwise Markov chains with automatic copulas selection. *Computational
Statistics & Data Analysis* 63, 81–98. doi:10.1016/j.csda.2013.01.027.
"""

import copy
import logging
import tomllib
from enum import Enum, unique
from pathlib import Path

import numpy as np
import scipy.stats as _ss
import tomli_w

from pmcprg.copulas._base import CopulaEnum, CopulaVirt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Variant enum
# ---------------------------------------------------------------------------

@unique
class Variant(Enum):
    """The five PMC/HMC model variants.

    Properties
    ----------
    uses_copula         : True if Y_n depends on Y_{n-1} via a copula (HMC-DN, PMC).
    has_markov_prior    : True if the prior is a transition matrix A (HMC-*).
                          False if the prior is a joint p[i,j] (PMC-*).
    allows_pair_margins : True if the variant admits pair-indexed margins
                          f_ij (PMC, PMC-IN).
    per_class_margin    : True if the variant forces state-indexed margins
                          f_i (HMC-*); the negation of allows_pair_margins.
    """

    HMC_IN  = "HMC-IN"
    HMC_IN2 = "HMC-IN2"
    HMC_DN  = "HMC-DN"
    PMC_IN  = "PMC-IN"
    PMC     = "PMC"

    @property
    def uses_copula(self) -> bool:
        return self in (Variant.HMC_DN, Variant.PMC)

    @property
    def has_markov_prior(self) -> bool:
        return self in (Variant.HMC_IN, Variant.HMC_IN2, Variant.HMC_DN)

    @property
    def allows_pair_margins(self) -> bool:
        """True for PMC and PMC-IN, whose margins may be pair-indexed (f_ij).

        For an HMC-* variant X is a Markov chain, and by the Proposition of
        A16 §2.1 an SR-PMC has a Markov X iff p(y_2 | x_1, x_2) = p(y_2 | x_2):
        its margins are state-indexed. Whether a given model *uses* pair
        margins is :attr:`PMCModel.margin_structure`.
        """
        return self in (Variant.PMC_IN, Variant.PMC)

    @property
    def per_class_margin(self) -> bool:
        """True if the variant forces state-indexed margins f_i (HMC-*).

        For PMC and PMC-IN the margins are state-indexed or pair-indexed
        depending on the model (:attr:`PMCModel.margin_structure`), so this
        is False. (Versions 0.5.0–0.8.x returned True for every variant,
        wrongly attributing f_ij = f_i to reversibility; the Proposition of
        A16 §2.1 ties it to X being Markov.)
        """
        return not self.allows_pair_margins


# Backward-compatible aliases — kept so external code that imports these sets
# still works. New code should use the Variant.* properties above.
USES_COPULA      = {v for v in Variant if v.uses_copula}
MARKOV_PRIOR     = {v for v in Variant if v.has_markov_prior}
PER_CLASS_MARGIN = {v for v in Variant if v.per_class_margin}

#: Values of :attr:`PMCModel.margin_structure` (and of ``[model].margin_structure``).
MARGIN_STRUCTURES = ("state", "pair")

# The INFO notice that K²-format margins are pair-indexed (no longer collapsed)
# is logged once per process: ICE rebuilds the model at every iteration.
_pair_notice_logged = False


def _log_pair_notice() -> None:
    """INFO, once per process: K²-format margins are now kept pair-indexed."""
    global _pair_notice_logged
    if _pair_notice_logged:
        return
    _pair_notice_logged = True
    logger.info(
        "[[margins]] in K²-format (keys i, j) are pair-indexed margins f_ij "
        "(general PMC, A16 Eqs. 12–14); versions 0.5.0–0.8 collapsed them to "
        "f_i. Set [model].margin_structure = \"state\" to restore the collapse "
        "(f_i = f_i0, X Markov — A16 §2.1 Proposition)."
    )


# ---------------------------------------------------------------------------
# Thin wrapper around a frozen scipy.stats distribution
# ---------------------------------------------------------------------------

# scipy.stats family names treated as multivariate margins. Public so other
# modules (e.g. pmcprg.pmc.ice) don't reach for an underscore-private name
# (audit A-8); ``_MULTIVARIATE_DISTS`` kept as a backward-compatible alias.
MULTIVARIATE_DISTS = frozenset({"multivariate_normal"})
_MULTIVARIATE_DISTS = MULTIVARIATE_DISTS


def _build_margin(dist_name: str, params: dict):
    """Factory: scalar ``_MarginDist`` or ``_VectorMargin`` based on ``dist_name``.

    Multivariate distributions (currently: ``multivariate_normal``) get the
    vector wrapper; everything else is treated as a scalar scipy.stats family.
    """
    if dist_name in _MULTIVARIATE_DISTS:
        return _VectorMargin(dist_name, params)
    return _MarginDist(dist_name, params)


class _MarginDist:
    """Thin wrapper around a frozen scalar ``scipy.stats`` distribution."""

    is_multivariate = False
    d = 1

    def __init__(self, dist_name: str, params: dict):
        self.dist_name = dist_name
        self.params = dict(params)
        try:
            dist_cls = getattr(_ss, dist_name)
        except AttributeError:
            raise ValueError(
                f"Unknown scipy.stats distribution: {dist_name!r}. "
                "Check the 'dist' key in your TOML [[margins]] block."
            )
        self._frozen = dist_cls(**params)

    # scalar wrappers — return Python float for consistency with copula API
    def pdf(self, y: float) -> float:
        return float(self._frozen.pdf(y))

    def logpdf(self, y: float) -> float:
        return float(self._frozen.logpdf(y))

    def cdf(self, y: float) -> float:
        return float(self._frozen.cdf(y))

    def ppf(self, q: float) -> float:
        return float(self._frozen.ppf(q))

    # vectorised wrappers — return np.ndarray for forward-backward
    def pdf_vec(self, y: np.ndarray) -> np.ndarray:
        return self._frozen.pdf(np.asarray(y, dtype=float))

    def cdf_vec(self, y: np.ndarray) -> np.ndarray:
        return self._frozen.cdf(np.asarray(y, dtype=float))

    def rvs(self, size: int, rng) -> np.ndarray:
        return self._frozen.rvs(size=size, random_state=rng)

    def __repr__(self) -> str:
        return f"_MarginDist({self.dist_name!r}, {self.params})"


class _VectorMargin:
    """Wrapper for a multivariate ``scipy.stats`` distribution.

    Only ``multivariate_normal`` is supported in this version (the only
    multivariate margin GICE/ICE knows how to update). Copula variants
    (HMC-DN, PMC) are forbidden when any margin is multivariate — see
    :meth:`PMCModel._parse` validation — so :meth:`cdf` / :meth:`ppf` are
    not part of the contract.

    Parameters
    ----------
    dist_name : str
        Currently must be ``"multivariate_normal"``.
    params : dict with keys
        ``mean`` : list/array of length d.
        ``cov``  : list-of-lists / array of shape (d, d).
    """

    is_multivariate = True

    def __init__(self, dist_name: str, params: dict):
        if dist_name not in _MULTIVARIATE_DISTS:
            raise ValueError(
                f"_VectorMargin: unsupported multivariate distribution "
                f"{dist_name!r}. Supported: {sorted(_MULTIVARIATE_DISTS)}"
            )
        self.dist_name = dist_name
        self.params    = {
            "mean": [float(x) for x in params["mean"]],
            "cov":  [[float(x) for x in row] for row in params["cov"]],
        }
        mean = np.asarray(self.params["mean"], dtype=float)
        cov  = np.asarray(self.params["cov"],  dtype=float)
        if mean.ndim != 1:
            raise ValueError(
                f"_VectorMargin {dist_name}: mean must be 1D, got shape {mean.shape}."
            )
        d = mean.size
        if cov.shape != (d, d):
            raise ValueError(
                f"_VectorMargin {dist_name}: cov must be ({d}, {d}), got {cov.shape}."
            )
        self.d = d
        # ``allow_singular=True`` gives a graceful pinv-based density when ICE
        # produces a near-singular covariance for a starved state; without it
        # the M-step can crash mid-iteration on real data.
        self._frozen = _ss.multivariate_normal(mean=mean, cov=cov, allow_singular=True)

    # ── scalar-input methods (single observation y ∈ ℝ^d) ────────────────
    def pdf(self, y) -> float:
        return float(self._frozen.pdf(np.asarray(y, dtype=float)))

    def logpdf(self, y) -> float:
        return float(self._frozen.logpdf(np.asarray(y, dtype=float)))

    # ── vectorised pdf — accepts Y of shape (N, d) ───────────────────────
    def pdf_vec(self, Y: np.ndarray) -> np.ndarray:
        Y = np.asarray(Y, dtype=float)
        if Y.ndim != 2 or Y.shape[1] != self.d:
            raise ValueError(
                f"_VectorMargin.pdf_vec expects Y of shape (N, {self.d}); "
                f"got {Y.shape}."
            )
        return self._frozen.pdf(Y)

    # ── CDF / PPF are undefined here: variants requiring them are
    #    forbidden when d > 1 (validation in PMCModel). Keep stub raisers
    #    in case a future code path hits them — clearer than a cryptic
    #    AttributeError deep in forward-backward.
    def cdf(self, y) -> float:
        raise NotImplementedError(
            "Multivariate margins have no scalar CDF; copula variants are "
            "not allowed when d > 1."
        )

    def cdf_vec(self, Y: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "Multivariate margins have no scalar CDF; copula variants are "
            "not allowed when d > 1."
        )

    def ppf(self, q) -> float:
        raise NotImplementedError(
            "Multivariate margins have no scalar quantile function."
        )

    def rvs(self, size: int, rng) -> np.ndarray:
        return self._frozen.rvs(size=size, random_state=rng)

    def __repr__(self) -> str:
        return (
            f"_VectorMargin({self.dist_name!r}, "
            f"d={self.d}, mean={self.params['mean']}, cov=[…])"
        )


# ---------------------------------------------------------------------------
# Helper: build a CopulaVirt from a TOML [[copulas]] block
# ---------------------------------------------------------------------------

def _build_copula(block: dict) -> CopulaVirt:
    """Instantiate a ``CopulaVirt`` from a TOML copula block dict."""
    short = block.get("name", "")
    tau   = float(block["tau"])

    enum_entry = CopulaEnum.from_short_name(short)
    if enum_entry is None:
        available = [c.value.SHORT_NAME for c in CopulaEnum.available()]
        raise ValueError(
            f"Unknown copula name {short!r} in [[copulas]] block. "
            f"Available SHORT_NAMEs: {available}"
        )

    # Collect all declared parameters from TOML block
    params: dict = {"tau_k": tau}
    for k in enum_entry.value.PARAMETERS_SET_NAME:
        if k != "tau_k" and k in block:
            params[k] = float(block[k])

    # Cached lazy import via CopulaEnum.klass
    return enum_entry.klass(**params)


# ---------------------------------------------------------------------------
# Stationary distribution helper
# ---------------------------------------------------------------------------

def _stationary_distribution(
    A: np.ndarray,
    *,
    max_iter: int = 1000,
    tol: float = 1e-12,
) -> np.ndarray:
    """Compute the stationary distribution π of a row-stochastic matrix A.

    Solves  π A = π  with  Σ π_i = 1, π_i ≥ 0.

    Uses power iteration starting from the uniform distribution. This is
    much more robust than eigen-decomposition for nearly-reducible chains,
    where multiple eigenvalues sit very close to 1 and ``np.linalg.eig``
    can return any vector in the degenerate eigenspace.

    For a reducible chain (multiple closed classes), this converges to a
    stationary distribution that depends on the starting point — uniform
    init is the natural neutral choice and biases toward weight-balanced
    classes.

    Parameters
    ----------
    A        : (K, K) row-stochastic matrix.
    max_iter : maximum power-iteration steps (default 1000).
    tol      : ‖π_{t+1} − π_t‖_∞ convergence threshold (default 1e-12).

    Returns
    -------
    pi : (K,) probability vector summing to 1.
    """
    K  = A.shape[0]
    pi = np.full(K, 1.0 / K, dtype=float)

    delta = np.inf
    for it in range(max_iter):
        pi_new = pi @ A
        # Numerical safety: clamp to non-negative and renormalise
        pi_new = np.maximum(pi_new, 0.0)
        s = pi_new.sum()
        if s <= 0.0:
            # Degenerate — fall back to uniform
            logger.warning(
                "Stationary iter %d: row sum %.3e ≤ 0 — falling back to uniform π.",
                it, s,
            )
            return np.full(K, 1.0 / K, dtype=float)
        pi_new /= s
        delta = float(np.max(np.abs(pi_new - pi)))
        if delta < tol:
            return pi_new
        pi = pi_new

    # Power iteration stalls on periodic or very slowly mixing chains. Solve
    # π (A − I) = 0, Σ π = 1 directly instead (least squares on the stacked
    # system), keeping the last iterate only if the solve is not a valid
    # probability vector.
    M = np.vstack([(A - np.eye(K)).T, np.ones((1, K))])
    rhs = np.concatenate([np.zeros(K), [1.0]])
    pi_ls = np.linalg.lstsq(M, rhs, rcond=None)[0]
    if np.all(np.isfinite(pi_ls)) and np.all(pi_ls > -1e-10):
        pi_ls = np.maximum(pi_ls, 0.0)
        pi_ls /= pi_ls.sum()
        logger.info(
            "Stationary distribution: power iteration stopped after %d steps "
            "(last ‖Δ‖_∞=%.2e); solved π (A − I) = 0 directly.",
            max_iter, delta,
        )
        return pi_ls
    logger.warning(
        "Stationary distribution did not converge within %d iterations "
        "(last ‖Δ‖_∞=%.2e) and the direct solve failed. Using last iterate.",
        max_iter, delta,
    )
    return pi


# ---------------------------------------------------------------------------
# PMCModel
# ---------------------------------------------------------------------------

class PMCModel:
    """
    Container for a PMC/HMC model loaded from a TOML configuration file.

    Parameters
    ----------
    path : str or Path
        Path to the ``.toml`` configuration file.

    Attributes (read-only)
    ----------------------
    variant          : Variant
    K                : int   Number of hidden states.
    N_default        : int   Default sequence length for simulation.
    name             : str   Human-readable label from [model].
    margin_structure : str   ``"state"`` (K margins f_i) or ``"pair"``
                             (K² margins f_ij) — see the module docstring.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, path: str | Path):
        self._path = Path(path)
        with open(self._path, "rb") as fh:
            raw = tomllib.load(fh)
        self._raw = raw
        self._parse(raw)

    @classmethod
    def from_dict(cls, raw: dict, path: str | Path = "memory") -> "PMCModel":
        """Build a ``PMCModel`` directly from a dict (useful for unit tests).

        The dict is deep-copied so subsequent mutations of either ``raw`` or
        ``model._raw`` do not affect the other.
        """
        obj = object.__new__(cls)
        obj._path = Path(path)
        obj._raw  = copy.deepcopy(raw)
        obj._parse(obj._raw)
        return obj

    # ------------------------------------------------------------------
    # Parsing & validation
    # ------------------------------------------------------------------

    def _parse(self, raw: dict) -> None:  # noqa: C901  (intentionally long)
        # ── [model] ───────────────────────────────────────────────────
        m = raw.get("model", {})
        self.name = m.get("name", "unnamed")

        try:
            self.variant = Variant(m["variant"])
        except KeyError:
            raise ValueError("[model] section must include a 'variant' key.")
        except ValueError:
            raise ValueError(
                f"Unknown variant {m['variant']!r}. "
                f"Valid values: {[v.value for v in Variant]}"
            )

        self.K = int(m.get("K", 2))
        if self.K < 2:
            raise ValueError(f"K must be >= 2, got {self.K}.")
        self.N_default = int(m.get("N_default", 5000))

        structure_req = m.get("margin_structure")
        if structure_req is not None and structure_req not in MARGIN_STRUCTURES:
            raise ValueError(
                f"[model].margin_structure must be one of {list(MARGIN_STRUCTURES)}; "
                f"got {structure_req!r}."
            )

        # ── Observation dimension d (default 1 for backward compat) ──────
        # When d > 1, observations are vectors in ℝ^d (e.g. RGB images), all
        # margins must be ``multivariate_normal``, and copula-using variants
        # (HMC-DN, PMC) are forbidden — copulas operate on scalar CDFs which
        # are undefined for multivariate margins.
        self.d = int(m.get("d", 1))
        if self.d < 1:
            raise ValueError(f"[model].d must be ≥ 1, got {self.d}.")
        if self.d > 1 and self.variant.uses_copula:
            raise ValueError(
                f"Variant {self.variant.value} uses temporal copulas, which "
                f"require scalar margins; got [model].d = {self.d}. "
                f"For multivariate observations, use HMC-IN, HMC-IN2 or PMC-IN."
            )

        # ── [prior] ───────────────────────────────────────────────────
        prior_raw = raw.get("prior", {})

        if self.variant.has_markov_prior:
            # Transition matrix A[i,j] = P(X_{n+1}=j | X_n=i)
            if "A" not in prior_raw:
                raise ValueError(
                    "[prior] must contain key 'A' for HMC-* variants."
                )
            A = np.array(prior_raw["A"], dtype=float)
            if A.shape != (self.K, self.K):
                raise ValueError(
                    f"[prior].A must be {self.K}×{self.K}, got {A.shape}."
                )
            row_sums = A.sum(axis=1)
            if not np.allclose(row_sums, 1.0, atol=1e-6):
                raise ValueError(
                    f"[prior].A rows must sum to 1; got {row_sums}."
                )
            self._A = A
            # Stationary distribution π — power iteration is numerically more
            # robust than eigen-decomposition for nearly-reducible chains
            # (where several eigenvalues sit very close to 1). Eigen-based
            # extraction can pick the wrong eigenvector in that regime.
            self._pi = _stationary_distribution(A)
            # Joint distribution p[i,j] = π[i] * A[i,j]
            self._p = self._pi[:, None] * A

        else:
            # Joint distribution p[i,j] = P(X_n=i, X_{n+1}=j)
            if "p" not in prior_raw:
                raise ValueError(
                    "[prior] must contain key 'p' for PMC-* variants."
                )
            p = np.array(prior_raw["p"], dtype=float)
            if p.shape != (self.K, self.K):
                raise ValueError(
                    f"[prior].p must be {self.K}×{self.K}, got {p.shape}."
                )
            if not np.isclose(p.sum(), 1.0, atol=1e-6):
                raise ValueError(
                    f"[prior].p must sum to 1; got {p.sum():.8f}."
                )
            self._p  = p
            self._pi = p.sum(axis=1)           # marginal π[i]
            # Avoid divide-by-zero for zero-probability states
            with np.errstate(invalid="ignore", divide="ignore"):
                self._A = np.where(
                    self._pi[:, None] > 0,
                    p / self._pi[:, None],
                    0.0,
                )

            # SR-PMC stationary-reversibility diagnostics. The package
            # commits to the SR-PMC factorisation, which requires:
            #   (a) p[i,j] = p[j,i]                  (detailed balance);
            #   (b) Σ_j p[i,j] = Σ_j p[j,i]          (row sum = column sum,
            #                                         which (a) implies).
            # These properties guarantee that the implied chain X_n is
            # stationary and reversible. Violations don't crash inference
            # but the model is no longer SR-PMC, and forward log-lik /
            # ICE updates may produce statistically dubious results.
            sym_dev = float(np.max(np.abs(p - p.T)))
            row_col_dev = float(np.max(np.abs(p.sum(axis=0) - p.sum(axis=1))))
            if sym_dev > 1e-6:
                logger.warning(
                    "[prior].p is not symmetric (max |p[i,j] - p[j,i]| = %.3e). "
                    "SR-PMC requires detailed balance; the implied chain is "
                    "not reversible.",
                    sym_dev,
                )
            elif row_col_dev > 1e-6:
                # Symmetric but row sums differ from column sums (impossible
                # for a symmetric matrix; left here for safety on numerical
                # asymmetries).
                logger.warning(
                    "[prior].p row sums and column sums differ by %.3e. "
                    "The chain X_n is not stationary.",
                    row_col_dev,
                )

        # ── [[margins]] ───────────────────────────────────────────────
        # Two structures (module docstring): K state margins f_i, listed in
        # the K-format (key ``i``), or K² pair margins f_ij, listed in the
        # K²-format (keys ``i``, ``j``). ``[model].margin_structure = "state"``
        # collapses a K²-format list to f_i = f_i0 (the v0.5.0–0.8 behaviour).
        margins_raw = raw.get("margins", [])
        self._margin_structure = self._resolve_margin_structure(margins_raw, structure_req)
        if self._margin_structure == "pair":
            # ``_state_margins`` is None: there is no density per state.
            self._state_margins = None
            self._margins, self._raw_margin_blocks = self._parse_margins_k2_pair(margins_raw)
            labelled = {f"(i={i}, j={j})": mg for (i, j), mg in self._margins.items()}
            if structure_req is None:
                _log_pair_notice()
        else:
            self._state_margins, self._raw_margin_blocks = self._parse_margins(margins_raw)
            labelled = {f"state i={i}": mg for i, mg in self._state_margins.items()}

        # ── Post-margin dimensionality check ─────────────────────────────
        # Every margin must agree with [model].d. Univariate margins have
        # ``d == 1``; multivariate margins expose their own ``d`` matching
        # the length of ``mean``.
        for label, mg in labelled.items():
            mg_d = getattr(mg, "d", 1)
            if mg_d != self.d:
                raise ValueError(
                    f"Margin for {label} has dimension {mg_d}, but "
                    f"[model].d = {self.d}. All margins must match."
                )
            if self.d > 1 and not getattr(mg, "is_multivariate", False):
                raise ValueError(
                    f"Margin for {label} is scalar ({mg.dist_name!r}), "
                    f"but [model].d = {self.d}. Use 'multivariate_normal' "
                    f"for multivariate observations."
                )
        if self._margin_structure == "state":
            # ``self._margins[(i, j)]``: for state margins every entry with
            # the same ``i`` is the same density object f_i.
            self._margins: dict[tuple[int, int], _MarginDist] = {
                (i, j): self._state_margins[i]
                for i in range(self.K) for j in range(self.K)
            }

        # ── [[copulas]] ───────────────────────────────────────────────
        copulas_raw = raw.get("copulas", [])
        self._copulas: dict[tuple[int, int], CopulaVirt] = {}

        if self.variant.uses_copula:
            expected_cop = self.K * self.K
            if len(copulas_raw) != expected_cop:
                raise ValueError(
                    f"{self.variant.value} requires {expected_cop} [[copulas]] blocks; "
                    f"got {len(copulas_raw)}."
                )
            seen_cop: set[tuple[int, int]] = set()
            for n_blk, block in enumerate(copulas_raw):
                if "i" not in block or "j" not in block:
                    raise ValueError(
                        f"[[copulas]] block #{n_blk} is missing required keys 'i' and 'j'."
                    )
                i = int(block["i"])
                j = int(block["j"])
                if not (0 <= i < self.K and 0 <= j < self.K):
                    raise ValueError(
                        f"Copula indices (i={i}, j={j}) out of range [0, {self.K-1}] "
                        f"in block #{n_blk}."
                    )
                if (i, j) in seen_cop:
                    raise ValueError(
                        f"Duplicate [[copulas]] entry for (i={i}, j={j}) in block #{n_blk}."
                    )
                seen_cop.add((i, j))
                self._copulas[(i, j)] = _build_copula(block)
            missing_cop = sorted(
                {(i, j) for i in range(self.K) for j in range(self.K)} - seen_cop
            )
            if missing_cop:
                raise ValueError(
                    f"{self.variant.value}: missing [[copulas]] blocks for (i,j)={missing_cop}."
                )
        elif copulas_raw:
            logger.warning(
                "[[copulas]] blocks present in TOML but variant %s does not "
                "use copulas — blocks ignored.",
                self.variant.value,
            )

    # ------------------------------------------------------------------
    # Margin parsing — state margins (K-format, or K²-format collapsed)
    # and pair margins (K²-format)
    # ------------------------------------------------------------------

    def _resolve_margin_structure(
        self,
        margins_raw: list[dict],
        requested: str | None,
    ) -> str:
        """Margin structure from the ``[[margins]]`` format and ``[model].margin_structure``.

        ============  ====================  ===================================
        format        margin_structure      result
        ============  ====================  ===================================
        K (key i)     absent / "state"      ``"state"``
        K (key i)     "pair"                ValueError
        K² (i, j)     absent / "pair"       ``"pair"`` (HMC-* → ValueError)
        K² (i, j)     "state"               ``"state"`` (collapse to f_i0)
        ============  ====================  ===================================

        The format is read from the first block, as the parsers do.
        """
        if not margins_raw:
            raise ValueError("[[margins]] section is required (got empty list).")
        k2_format = "j" in margins_raw[0]
        if not k2_format:
            if requested == "pair":
                raise ValueError(
                    "[model].margin_structure = \"pair\" needs pair-indexed margins: "
                    f"K²={self.K * self.K} [[margins]] blocks keyed by 'i' and 'j'; "
                    "got the K-format (blocks keyed by 'i' only)."
                )
            return "state"
        if requested == "state":
            return "state"
        if not self.variant.allows_pair_margins:
            raise ValueError(
                f"Variant {self.variant.value} has a Markov hidden chain X, so its "
                f"margins cannot be pair-indexed: for a stationary reversible PMC, "
                f"X is Markov iff p(y_2 | x_1, x_2) = p(y_2 | x_2) (Proposition of "
                f"Derrode & Pieczynski 2013, §2.1). Pair margins f_ij exist only for "
                f"PMC and PMC-IN. Give K [[margins]] blocks keyed by 'i', or set "
                f"[model].margin_structure = \"state\" to collapse the K² blocks "
                f"to f_i = f_i0."
            )
        return "pair"

    def _parse_margins(
        self,
        margins_raw: list[dict],
    ) -> tuple[dict[int, _MarginDist], list[dict]]:
        """Parse [[margins]] into K state-indexed densities (``margin_structure == "state"``).

        Two TOML schemas are accepted:

        * **K-format** — one block per state, keyed by ``i`` only::

              [[margins]]
              i = 0
              dist = "norm"
              params = {loc = -1.0, scale = 1.0}

        * **K²-format with** ``[model].margin_structure = "state"`` — K²
          blocks keyed by ``(i, j)``, collapsed to K state densities: the
          entry kept for state i is the one at (i, 0). A WARNING lists the
          blocks that differ from their anchor (the collapse then discards
          information); without ``margin_structure = "state"`` the K²-format
          gives pair margins instead (:meth:`_parse_margins_k2_pair`).

        Returns
        -------
        state_margins   : ``dict[int, _MarginDist]`` — one density per
                          state in ``range(K)``.
        raw_for_blocks  : ``list[dict]`` — the K-format blocks
                          (for ``margin_blocks()``).
        """
        K = self.K
        if not margins_raw:
            raise ValueError("[[margins]] section is required (got empty list).")

        # Detect format from the first block.
        format_is_k2 = "j" in margins_raw[0]

        if not format_is_k2:
            return self._parse_margins_k_format(margins_raw)
        return self._parse_margins_k2_legacy(margins_raw, K)

    def _parse_margins_k_format(
        self,
        margins_raw: list[dict],
    ) -> tuple[dict[int, _MarginDist], list[dict]]:
        """Parse the K-format margins (one block per state)."""
        K = self.K
        if len(margins_raw) != K:
            raise ValueError(
                f"K-format [[margins]] requires exactly K={K} blocks "
                f"(one per state); got {len(margins_raw)}."
            )
        seen: set[int] = set()
        state_margins: dict[int, _MarginDist] = {}
        raw_blocks: list[dict] = []
        for n_blk, block in enumerate(margins_raw):
            if "i" not in block:
                raise ValueError(
                    f"[[margins]] block #{n_blk} is missing the 'i' key."
                )
            if "j" in block:                                 # defensive
                raise ValueError(
                    f"[[margins]] block #{n_blk} mixes K-format with a 'j' "
                    f"key; choose one schema."
                )
            i = int(block["i"])
            if not (0 <= i < K):
                raise ValueError(
                    f"Margin index i={i} out of range [0, {K - 1}] in block #{n_blk}."
                )
            if i in seen:
                raise ValueError(
                    f"Duplicate [[margins]] entry for i={i} (block #{n_blk})."
                )
            seen.add(i)
            if "dist" not in block:
                raise ValueError(
                    f"[[margins]] block #{n_blk} (i={i}) missing the 'dist' key."
                )
            state_margins[i] = _build_margin(block["dist"], block.get("params", {}))
            kept = {
                "i": i,
                "dist": block["dist"],
                "params": dict(block.get("params", {})),
            }
            # GICE — preserve the optional ``candidates`` list so the M-step
            # can select among them at each iteration (SP-2016 §3).
            if "candidates" in block:
                cands = list(block["candidates"])
                if not cands:
                    raise ValueError(
                        f"[[margins]] block #{n_blk} (i={i}): 'candidates' "
                        f"list cannot be empty."
                    )
                kept["candidates"] = cands
            raw_blocks.append(kept)
        missing = sorted(set(range(K)) - seen)
        if missing:
            raise ValueError(f"K-format [[margins]]: missing entries for i={missing}.")
        return state_margins, raw_blocks

    @staticmethod
    def _index_k2_blocks(margins_raw: list[dict], K: int) -> dict[tuple[int, int], dict]:
        """Validate a K²-format ``[[margins]]`` list and index its blocks by (i, j)."""
        expected = K * K
        if len(margins_raw) != expected:
            raise ValueError(
                f"K²-format [[margins]] requires K²={expected} blocks (one per "
                f"pair (i, j)); got {len(margins_raw)}. State margins use the "
                f"K-format (one block per state, indexed by 'i' only)."
            )
        by_ij: dict[tuple[int, int], dict] = {}
        for n_blk, block in enumerate(margins_raw):
            if "i" not in block or "j" not in block:
                raise ValueError(
                    f"K²-format [[margins]] block #{n_blk} missing "
                    f"required keys 'i' and 'j'."
                )
            i, j = int(block["i"]), int(block["j"])
            if not (0 <= i < K and 0 <= j < K):
                raise ValueError(
                    f"Margin indices (i={i}, j={j}) out of range "
                    f"[0, {K - 1}] in block #{n_blk}."
                )
            if (i, j) in by_ij:
                raise ValueError(
                    f"Duplicate [[margins]] entry for (i={i}, j={j}) in "
                    f"block #{n_blk}."
                )
            if "dist" not in block:
                raise ValueError(
                    f"[[margins]] block #{n_blk} (i={i}, j={j}) missing "
                    f"the 'dist' key."
                )
            by_ij[(i, j)] = block
        missing_ij = sorted({(i, j) for i in range(K) for j in range(K)} - set(by_ij))
        if missing_ij:
            raise ValueError(
                f"K²-format [[margins]]: missing entries for (i, j)={missing_ij}."
            )
        return by_ij

    def _parse_margins_k2_legacy(
        self,
        margins_raw: list[dict],
        K: int,
    ) -> tuple[dict[int, _MarginDist], list[dict]]:
        """Collapse K²-format margins to K state margins (``margin_structure = "state"``).

        The v0.5.0–0.8 behaviour, now only on request. WARNs (without
        raising) when blocks within the same ``i`` declare different
        densities: state margins require f_ij = f_i — by the Proposition of
        A16 §2.1, a hidden chain X that is Markov — so the collapse discards
        the j-dependence. The entry kept is the one at ``(i, 0)``; the
        conflicting (i, j>0) entries are listed in the warning.
        """
        by_ij = self._index_k2_blocks(margins_raw, K)

        # Collapse to K state densities, reporting untied blocks.
        state_margins: dict[int, _MarginDist] = {}
        raw_blocks: list[dict] = []
        conflicts: list[str] = []
        for i in range(K):
            anchor = by_ij[(i, 0)]
            anchor_dist   = anchor["dist"]
            anchor_params = dict(anchor.get("params", {}))
            for j in range(K):
                blk = by_ij[(i, j)]
                if blk["dist"] != anchor_dist or dict(blk.get("params", {})) != anchor_params:
                    conflicts.append(
                        f"(i={i}, j={j}): {blk['dist']}({blk.get('params', {})}) "
                        f"≠ anchor (i={i}, j=0): {anchor_dist}({anchor_params})"
                    )
            state_margins[i] = _build_margin(anchor_dist, anchor_params)
            raw_blocks.append({"i": i, "dist": anchor_dist, "params": anchor_params})

        if conflicts:
            logger.warning(
                "[[margins]] K²-format collapsed to state margins "
                "([model].margin_structure = \"state\"), but f_ij depends on j: "
                "the anchor (j=0) entry is kept for each state and the pair "
                "margins below are discarded (remove margin_structure to keep "
                "them as a general PMC):\n  - %s",
                "\n  - ".join(conflicts),
            )
        else:
            logger.info(
                "[[margins]] K²-format collapsed to state margins "
                "([model].margin_structure = \"state\", the legacy v0.5.0–0.8 "
                "behaviour): all K² blocks are tied, f_ij = f_i. The K-format "
                "(one block per state, indexed by 'i') states the same model."
            )
        return state_margins, raw_blocks

    def _parse_margins_k2_pair(
        self,
        margins_raw: list[dict],
    ) -> tuple[dict[tuple[int, int], _MarginDist], list[dict]]:
        """Parse K²-format margins as pair margins f_ij (``margin_structure == "pair"``).

        Returns
        -------
        pair_margins : ``dict[(i, j), _MarginDist]`` — f_ij, the density of
                       y_n given (x_n, x_{n+1}) = (i, j) (A16 §2.2).
        raw_blocks   : ``list[dict]`` — K² blocks ``{"i", "j", "dist",
                       "params"[, "candidates"]}`` sorted by (i, j).
        """
        K = self.K
        by_ij = self._index_k2_blocks(margins_raw, K)
        pair_margins: dict[tuple[int, int], _MarginDist] = {}
        raw_blocks: list[dict] = []
        for (i, j) in sorted(by_ij):
            block = by_ij[(i, j)]
            pair_margins[(i, j)] = _build_margin(block["dist"], block.get("params", {}))
            kept = {
                "i": i,
                "j": j,
                "dist": block["dist"],
                "params": dict(block.get("params", {})),
            }
            if "candidates" in block:                        # GICE (SP-2016 §3)
                cands = list(block["candidates"])
                if not cands:
                    raise ValueError(
                        f"[[margins]] block (i={i}, j={j}): 'candidates' "
                        f"list cannot be empty."
                    )
                kept["candidates"] = cands
            raw_blocks.append(kept)
        return pair_margins, raw_blocks

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def prior_p(self) -> np.ndarray:
        """K×K joint distribution p[i,j] = P(X_n=i, X_{n+1}=j)."""
        return self._p.copy()

    @property
    def transition_A(self) -> np.ndarray:
        """K×K row-stochastic matrix A[i,j] = P(X_{n+1}=j | X_n=i)."""
        return self._A.copy()

    @property
    def stationary_pi(self) -> np.ndarray:
        """Marginal distribution π[i] = P(X_n=i)."""
        return self._pi.copy()

    @property
    def path(self) -> Path:
        """Path the model was loaded from (or 'memory' if built via from_dict)."""
        return self._path

    @property
    def raw(self) -> dict:
        """Deep copy of the raw TOML dict — safe to mutate."""
        return copy.deepcopy(self._raw)

    def ice_config(self) -> dict:
        """Return the ``[ice]`` TOML section as a dict (empty if absent)."""
        return copy.deepcopy(self._raw.get("ice", {}))

    def sem_config(self) -> dict:
        """Return the ``[sem]`` TOML section as a dict (empty if absent).

        SEM (:func:`pmcprg.pmc.sem.sem`) merges this on top of ``[ice]`` so a
        model file can override shared estimator keys (or set SEM-specific ones
        like ``sem_seed`` / ``max_iter``) in a dedicated ``[sem]`` block while
        still inheriting the common ``[ice]`` settings.
        """
        return copy.deepcopy(self._raw.get("sem", {}))

    def margin_blocks(self) -> list[dict]:
        """Return a deep copy of the ``[[margins]]`` blocks, in a stable order.

        * ``margin_structure == "state"`` — K blocks keyed by ``i``
          (``{"i", "dist", "params"[, "candidates"]}``), in K-format even
          when the source listed K² blocks collapsed by
          ``[model].margin_structure = "state"``;
        * ``margin_structure == "pair"`` — K² blocks keyed by ``i`` and ``j``
          (``{"i", "j", "dist", "params"[, "candidates"]}``), sorted by (i, j).
        """
        return copy.deepcopy(self._raw_margin_blocks)

    def copula_blocks(self) -> list[dict]:
        """Return a deep copy of the ``[[copulas]]`` blocks."""
        return copy.deepcopy(self._raw.get("copulas", []))

    @property
    def margin_structure(self) -> str:
        """``"state"`` (K margins f_i) or ``"pair"`` (K² margins f_ij).

        State margins are exactly the SR-PMCs whose hidden chain X is Markov
        (Proposition of A16 §2.1); pair margins give the general PMC of A16
        Eqs. 12–14 (PMC and PMC-IN variants only). See the module docstring.
        """
        return self._margin_structure

    # ------------------------------------------------------------------
    # Margin access
    # ------------------------------------------------------------------

    def margin(self, i: int, j: int | None = None) -> _MarginDist:
        """Return the margin f_ij of the observation y_n given (x_n, x_{n+1}) = (i, j).

        * State margins (``margin_structure == "state"``): the density is f_i
          whatever the other state, so ``j`` is ignored and may be omitted —
          ``margin(i, j) is margin(i)``.
        * Pair margins (``margin_structure == "pair"``): ``j`` is required.
          ``margin(i, j)`` is f_ij, the left margin of the pair (i, j) in
          A16 Eq. 12; the right margin of that pair (the density of
          y_{n+1}) is ``margin(j, i)`` = f_ji. ``margin(i)`` raises
          ``ValueError``: no density is attached to a state alone.
        """
        if self._margin_structure == "pair":
            if j is None:
                raise ValueError(
                    f"margin({i}) without j is ambiguous for pair-indexed margins "
                    f"(margin_structure = 'pair'): the density of y_n depends on "
                    f"both states of the pair, f_ij(y) = p(y_n | x_n = i, "
                    f"x_{{n+1}} = j). Call margin(i, j) — margin(j, i) for the "
                    f"right observation of the pair (i, j)."
                )
            mg = self._margins.get((i, j))
            if mg is None:
                raise KeyError(
                    f"No margin defined for the pair (i={i}, j={j}). "
                    f"Available pairs: {sorted(self._margins)}"
                )
            return mg
        if i not in self._state_margins:
            raise KeyError(
                f"No margin defined for state i={i}. "
                f"Available states: {sorted(self._state_margins)}"
            )
        return self._state_margins[i]

    def pdf(self, i: int, j: int | None, y: float) -> float:
        """f_ij(y) — PDF of ``margin(i, j)`` at ``y`` (f_i(y) for state margins)."""
        return self.margin(i, j).pdf(y)

    def cdf(self, i: int, j: int | None, y: float) -> float:
        """F_ij(y) — CDF of ``margin(i, j)`` at ``y`` (F_i(y) for state margins)."""
        return self.margin(i, j).cdf(y)

    def ppf(self, i: int, j: int | None, q: float) -> float:
        """F_ij^{-1}(q) — quantile of ``margin(i, j)`` (F_i^{-1} for state margins)."""
        return self.margin(i, j).ppf(q)

    # ------------------------------------------------------------------
    # Copula access
    # ------------------------------------------------------------------

    def copula(self, i: int, j: int) -> CopulaVirt:
        """Return the copula C_{ij} for the pair (i, j)."""
        if not self.variant.uses_copula:
            raise RuntimeError(
                f"Variant {self.variant.value} does not use copulas."
            )
        key = (i, j)
        if key not in self._copulas:
            raise KeyError(
                f"No copula defined for (i={i}, j={j}). "
                f"Available keys: {sorted(self._copulas)}"
            )
        return self._copulas[key]

    # ------------------------------------------------------------------
    # Unified transition weight  w(i→j, y_n, y_{n+1})
    # ------------------------------------------------------------------

    def weight(self, i: int, j: int, y_n: float, y_n1: float) -> float:
        """
        Un-normalised weight of the pair (x_n, x_{n+1}) = (i, j) given (y_n, y_{n+1}).

        The margin of the left observation y_n is ``margin(i, j)`` = f_ij and
        that of the right observation y_{n+1} is ``margin(j, i)`` = f_ji —
        the index inversion of A16 Eq. 12. With state margins
        (``margin_structure == "state"``) both reduce to the per-state
        densities f_i and f_j.

        Formulas by variant
        -------------------
        HMC-IN  : A[i,j] · f_j(y_{n+1})
        HMC-IN2 : A[i,j] · f_j(y_{n+1})
        HMC-DN  : A[i,j] · f_j(y_{n+1}) · c_ij(F_i(y_n), F_j(y_{n+1}))
        PMC-IN  : p[i,j] · f_ij(y_n) · f_ji(y_{n+1})
        PMC     : p[i,j] · f_ij(y_n) · f_ji(y_{n+1}) · c_ij(F_ij(y_n), F_ji(y_{n+1}))

        (HMC-* variants have state margins, so f_ji = f_j there.)

        Relation to forward–backward
        ----------------------------
        For HMC-* the weight is the transition p(z_{n+1} | z_n) itself:
        ``precompute_weights(model, Y)[0][n, i, j] = weight(i, j, Y[n], Y[n+1])``.
        For PMC-* it is the summand of the pair density of A16 Eq. 12,
        p(x_n = i, x_{n+1} = j, y_n, y_{n+1}); the transition used by
        forward–backward divides it by p(x_n = i, y_n) = Σ_k p[i,k] f_ik(y_n)
        (A16 Eqs. 13–14):

            W[n, i, j] = weight(i, j, y_n, y_{n+1}) / Σ_k p[i,k] · f_ik(y_n).

        (``precompute_weights`` clips the CDF values to [EPS, 1 − EPS]
        before the copula density; ``weight`` does not.)

        Why the margins may depend on the pair
        --------------------------------------
        Reversibility (A16 §2.2) only gives p(y_1 | x_1, x_2) = p(y_2 | x_2, x_1),
        whence f_ji for the right observation. It does not make f_ij
        independent of j: by the Proposition of A16 §2.1, f_ij = f_i holds
        exactly when the hidden chain X is Markov, in which case the PMC is an
        SR HMC-DN (the transition p(x_{n+1} = j | x_n = i, y_n) ∝ p_ij f_i(y_n)
        no longer depends on y_n).
        """
        v = self.variant

        # Observation density at y_{n+1}: right margin of the pair, f_ji
        # (f_j for state margins).
        f_next = self.margin(j, i).pdf(y_n1)

        if v.has_markov_prior:
            prior = float(self._A[i, j])
        else:
            prior = float(self._p[i, j])

        if v == Variant.HMC_IN:
            return prior * f_next

        if v == Variant.HMC_IN2:
            return prior * f_next

        if v == Variant.HMC_DN:
            F_curr = self.margin(i, j).cdf(y_n)
            F_next = self.margin(j, i).cdf(y_n1)
            c_val  = self.copula(i, j).pdf([F_curr, F_next])
            return prior * f_next * c_val

        # PMC-IN and PMC also multiply by the left margin f_ij(y_n)
        f_curr = self.margin(i, j).pdf(y_n)

        if v == Variant.PMC_IN:
            return prior * f_curr * f_next

        # PMC (full)
        F_curr = self.margin(i, j).cdf(y_n)
        F_next = self.margin(j, i).cdf(y_n1)
        c_val  = self.copula(i, j).pdf([F_curr, F_next])
        return prior * f_curr * f_next * c_val

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path | None = None) -> Path:
        """Write the model back to a TOML file.  Returns the path written."""
        out = Path(path) if path is not None else self._path
        out.write_bytes(tomli_w.dumps(self._raw).encode("utf-8"))
        logger.info("PMCModel saved to %s", out)
        return out

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"PMCModel(name={self.name!r}, variant={self.variant.value!r}, "
            f"K={self.K}, N_default={self.N_default})"
        )


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

    models_dir = pathlib.Path(__file__).parent / "models"
    for toml_file in sorted(models_dir.glob("*.toml")):
        print(f"\n── {toml_file.name} ──")
        mdl = PMCModel(toml_file)
        print(mdl)
        print(f"  π  = {mdl.stationary_pi}")
        print(f"  A  =\n{mdl.transition_A}")
        print(f"  p  =\n{mdl.prior_p}")
        print(f"  pdf(0,0,0) = {mdl.pdf(0, 0, 0.0):.6f}")
        if mdl.variant.uses_copula:
            cop = mdl.copula(0, 0)
            print(f"  copula(0,0) = {cop}")
        w = mdl.weight(0, 1, 0.5, -0.5)
        print(f"  weight(0→1, 0.5, -0.5) = {w:.6e}")
