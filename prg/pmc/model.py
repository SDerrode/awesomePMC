"""
PMCModel — load, validate, and access PMC/HMC model parameters from TOML.

Supported variants
------------------
HMC-IN   : Hidden Markov Chain, Independent observations (same margin per class).
HMC-IN2  : Hidden Markov Chain, Independent observations (margin depends on pair (i,j)).
HMC-DN   : Hidden Markov Chain, Dependent observations via copula.
PMC-IN   : Pairwise Markov Chain, Independent observations.
PMC      : Full Pairwise Markov Chain with copula-based transitions.

TOML format (see prg/pmc/models/*.toml for examples)
------------------------------------------------------
[model]
name     = "My model"
variant  = "PMC"        # one of: HMC-IN | HMC-IN2 | HMC-DN | PMC-IN | PMC
K        = 2
N_default = 5000

[prior]
# HMC-* variants → key "A" (K×K row-stochastic transition matrix)
# PMC-* variants → key "p" (K×K joint distribution, sums to 1)
p = [[0.45, 0.05], [0.05, 0.45]]

[[margins]]               # K blocks for HMC-IN; K² blocks for all others
i = 0; j = 0             # j is absent/ignored for HMC-IN
dist   = "norm"
params = {loc = 0.0, scale = 1.0}
# … repeat for every (i,j)

[[copulas]]               # only for HMC-DN and PMC; K² blocks
i   = 0; j = 0
name = "Gaussian"         # SHORT_NAME from CopulaEnum
tau  = 0.7
# optional extra params: df = 4.0 (Student), delta = 1.5 (BB1)
"""

import copy
import logging
import tomllib
from enum import Enum, unique
from pathlib import Path

import numpy as np
import scipy.stats as _ss
import tomli_w

from prg.copulas._base import CopulaEnum, CopulaVirt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Variant enum
# ---------------------------------------------------------------------------

@unique
class Variant(Enum):
    """The five PMC/HMC model variants.

    Properties
    ----------
    uses_copula      : True if Y_n depends on Y_{n-1} via a copula (HMC-DN, PMC).
    has_markov_prior : True if the prior is a transition matrix A (HMC-*).
                       False if the prior is a joint p[i,j] (PMC-*).
    per_class_margin : True if margins are indexed only by i (HMC-IN);
                       False if indexed by the pair (i, j).
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
    def per_class_margin(self) -> bool:
        """Always ``True``: under SR-PMC every variant uses K state-indexed
        margins. The legacy K² (i, j) margin format is still accepted at the
        TOML level for back-compat but collapses to K margins internally —
        see :meth:`PMCModel._parse_margins`. Property kept for API stability;
        it now describes a contract, not a per-variant difference.
        """
        return True


# Backward-compatible aliases — kept so external code that imports these sets
# still works. New code should use the Variant.* properties above.
USES_COPULA      = {v for v in Variant if v.uses_copula}
MARKOV_PRIOR     = {v for v in Variant if v.has_markov_prior}
PER_CLASS_MARGIN = {v for v in Variant if v.per_class_margin}


# ---------------------------------------------------------------------------
# Thin wrapper around a frozen scipy.stats distribution
# ---------------------------------------------------------------------------

_MULTIVARIATE_DISTS = frozenset({"multivariate_normal"})


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
        if np.max(np.abs(pi_new - pi)) < tol:
            return pi_new
        pi = pi_new

    logger.warning(
        "Stationary distribution did not converge within %d iterations "
        "(final ‖Δ‖_∞=%.2e). Using last iterate.",
        max_iter, np.max(np.abs(pi_new - pi)),
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
    variant   : Variant
    K         : int          Number of hidden states.
    N_default : int          Default sequence length for simulation.
    name      : str          Human-readable label from [model].
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
        # SR-PMC contract: there are exactly K marginal densities, one per
        # state. The K-format TOML lists them with a single ``i`` key
        # (canonical, since SR-PMC was adopted). The legacy K²-format TOML
        # uses ``(i, j)`` keys with K² blocks; that format is still
        # accepted for back-compat: the parser validates that every f_{i,*}
        # is identical (raising on conflict, warning on near-tied numerical
        # noise) and collapses them to K state-indexed entries.
        margins_raw = raw.get("margins", [])
        self._state_margins, self._raw_margin_blocks = self._parse_margins(margins_raw)

        # ── Post-margin dimensionality check ─────────────────────────────
        # Every margin must agree with [model].d. Univariate margins have
        # ``d == 1``; multivariate margins expose their own ``d`` matching
        # the length of ``mean``.
        for i, mg in self._state_margins.items():
            mg_d = getattr(mg, "d", 1)
            if mg_d != self.d:
                raise ValueError(
                    f"Margin for state i={i} has dimension {mg_d}, but "
                    f"[model].d = {self.d}. All margins must match."
                )
            if self.d > 1 and not getattr(mg, "is_multivariate", False):
                raise ValueError(
                    f"Margin for state i={i} is scalar ({mg.dist_name!r}), "
                    f"but [model].d = {self.d}. Use 'multivariate_normal' "
                    f"for multivariate observations."
                )
        # ``self._margins[(i, j)]`` retained for back-compat; under SR-PMC
        # all entries with the same ``i`` point to the same _MarginDist.
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
    # Margin parsing — supports both the K-format and legacy K² format
    # ------------------------------------------------------------------

    def _parse_margins(
        self,
        margins_raw: list[dict],
    ) -> tuple[dict[int, _MarginDist], list[dict]]:
        """Parse [[margins]] into K state-indexed densities.

        Two TOML schemas are accepted:

        * **K-format (canonical, SR-PMC)** — one block per state, keyed by
          ``i`` only::

              [[margins]]
              i = 0
              dist = "norm"
              params = {loc = -1.0, scale = 1.0}

        * **K²-format (legacy)** — K² blocks keyed by ``(i, j)``. Under
          SR-PMC all blocks with the same ``i`` *must* declare the same
          density. We validate this and collapse to K state-indexed entries.
          A WARNING is logged when blocks differ within a state (the
          canonical entry kept is the one with j = 0; conflicts across j
          are reported one-by-one).

        Returns
        -------
        state_margins   : ``dict[int, _MarginDist]`` — one density per
                          state in ``range(K)``.
        raw_for_blocks  : ``list[dict]`` — the canonical K-format blocks
                          (for ``margin_blocks()`` and TOML round-trip).
        """
        K = self.K
        if not margins_raw:
            raise ValueError("[[margins]] section is required (got empty list).")

        # Detect format from the first block.
        format_is_legacy = "j" in margins_raw[0]

        if not format_is_legacy:
            return self._parse_margins_k_format(margins_raw)
        return self._parse_margins_k2_legacy(margins_raw, K)

    def _parse_margins_k_format(
        self,
        margins_raw: list[dict],
    ) -> tuple[dict[int, _MarginDist], list[dict]]:
        """Parse the canonical K-format margins."""
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

    def _parse_margins_k2_legacy(
        self,
        margins_raw: list[dict],
        K: int,
    ) -> tuple[dict[int, _MarginDist], list[dict]]:
        """Parse the legacy K²-format and collapse to K state-indexed margins.

        WARNs (without raising) when blocks within the same ``i`` declare
        different densities — that violates SR-PMC and would yield K different
        estimates of the same theoretical quantity. The canonical entry kept
        is the one at ``(i, 0)``; conflicting (i, j>0) entries are listed
        in the warning so the user can fix the TOML.
        """
        expected = K * K
        if len(margins_raw) != expected:
            raise ValueError(
                f"Legacy K²-format [[margins]] requires K²={expected} blocks; "
                f"got {len(margins_raw)}. Use the K-format (one block per "
                f"state, indexed by 'i' only) for new files."
            )
        # First pass: index blocks by (i, j).
        by_ij: dict[tuple[int, int], dict] = {}
        for n_blk, block in enumerate(margins_raw):
            if "i" not in block or "j" not in block:
                raise ValueError(
                    f"Legacy K²-format [[margins]] block #{n_blk} missing "
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
                f"Legacy K²-format [[margins]]: missing entries for (i, j)={missing_ij}."
            )

        # Second pass: collapse to K state densities, validating ties.
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
                "[[margins]] legacy K²-format violates SR-PMC tying — "
                "f_{i,j} should depend on i only. The anchor (j=0) entry "
                "is kept for each state; conflicting entries:\n  - %s",
                "\n  - ".join(conflicts),
            )
        else:
            logger.info(
                "[[margins]] legacy K²-format detected: all K² blocks "
                "respect SR-PMC tying. Consider migrating the TOML to "
                "the canonical K-format (one block per state, indexed by 'i')."
            )
        return state_margins, raw_blocks

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

    def margin_blocks(self) -> list[dict]:
        """Return a deep copy of the K canonical ``[[margins]]`` blocks.

        The format is **always** K-format (one block per state indexed by
        ``i``), regardless of how the source TOML was written. This is the
        SR-PMC canonical schema; legacy K²-format TOMLs are collapsed at
        load time.
        """
        return copy.deepcopy(self._raw_margin_blocks)

    def copula_blocks(self) -> list[dict]:
        """Return a deep copy of the ``[[copulas]]`` blocks."""
        return copy.deepcopy(self._raw.get("copulas", []))

    # ------------------------------------------------------------------
    # Margin access
    # ------------------------------------------------------------------

    def margin(self, i: int, j: int | None = None) -> _MarginDist:
        """Return the marginal density ``f_i`` of the observation at state ``i``.

        Under SR-PMC the density depends **only on the state** of the
        observation, not on the other state of the pair — so the second
        argument is unused. It is kept for back-compat with callers that
        still pass ``(i, j)`` (e.g. the simulator's ``model.margin(j, i)``
        idiom for the right-margin of pair ``(i, j)``); when ``j`` is
        omitted, the call is purely state-indexed.
        """
        if i not in self._state_margins:
            raise KeyError(
                f"No margin defined for state i={i}. "
                f"Available states: {sorted(self._state_margins)}"
            )
        return self._state_margins[i]

    def pdf(self, i: int, j: int | None, y: float) -> float:
        """f_i(y) — PDF of margin at state ``i`` evaluated at ``y``."""
        return self.margin(i).pdf(y)

    def cdf(self, i: int, j: int | None, y: float) -> float:
        """F_i(y) — CDF of margin at state ``i`` evaluated at ``y``."""
        return self.margin(i).cdf(y)

    def ppf(self, i: int, j: int | None, q: float) -> float:
        """F_i^{-1}(q) — quantile function of margin at state ``i``."""
        return self.margin(i).ppf(q)

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
        Un-normalised transition weight for the pair (i→j) given observations.

        Used in forward-backward:
            α_{n+1}(j) ∝  Σ_i  α_n(i) · weight(i, j, y_n, y_{n+1})

        Formulas by variant (under the SR-PMC contract — only K margins,
        one per state, so f_{ij}(y) = f_i(y))
        ---------------------------------------------------------------
        HMC-IN  : A[i,j] · f_j(y_{n+1})
        HMC-IN2 : A[i,j] · f_j(y_{n+1})
        HMC-DN  : A[i,j] · f_j(y_{n+1}) · c_{ij}(F_i(y_n), F_j(y_{n+1}))
        PMC-IN  : p[i,j] · f_i(y_n) · f_j(y_{n+1})
        PMC     : p[i,j] · f_i(y_n) · f_j(y_{n+1}) · c_{ij}(F_i(y_n), F_j(y_{n+1}))

        Note
        ----
        Historic notation in the code base used "f_{ji}" for the right
        margin and "f_{ij}" for the left margin of the bivariate joint
        f_{i,j}. Under SR-PMC reversibility (CSDA 2013, Eq. 14) those
        reduce to the per-state densities: f_{ji} = f_j (right margin
        depends only on its own state j) and f_{ij} = f_i (left margin
        depends only on its own state i).
        """
        v = self.variant

        # Observation density at y_{n+1}: per-state margin f_j (the state
        # to which Y_{n+1} is conditioned).
        f_next = self.margin(j).pdf(y_n1)

        if v.has_markov_prior:
            prior = float(self._A[i, j])
        else:
            prior = float(self._p[i, j])

        if v == Variant.HMC_IN:
            return prior * f_next

        if v == Variant.HMC_IN2:
            return prior * f_next

        if v == Variant.HMC_DN:
            F_curr = self.margin(i).cdf(y_n)
            F_next = self.margin(j).cdf(y_n1)
            c_val  = self.copula(i, j).pdf([F_curr, F_next])
            return prior * f_next * c_val

        # PMC-IN and PMC also multiply by f_i(y_n)
        f_curr = self.margin(i).pdf(y_n)

        if v == Variant.PMC_IN:
            return prior * f_curr * f_next

        # PMC (full)
        F_curr = self.margin(i).cdf(y_n)
        F_next = self.margin(j).cdf(y_n1)
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
