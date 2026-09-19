"""
missingness.py — missingness mechanisms of a PMC/HMC model (``[missingness]``).

The mask of an observation sequence is m_n = 1 iff row n of Y is missing
(:func:`pmcprg.pmc.gaps.missing_mask`). A :class:`~pmcprg.pmc.model.PMCModel`
carries the law of that mask in :attr:`~pmcprg.pmc.model.PMCModel.missingness`:

* ``None`` — **ignorable** missingness (MCAR/MAR, the default): the mask
  carries no information on the states and inference uses the observed-data
  likelihood p(y_obs) = ∫ p(y) dy_miss (Little & Rubin 2019).
* a mechanism object below — **non-ignorable** (missing not at random), a
  selection model in the sense of Little & Rubin (2019): the mask depends
  on the hidden states only, m ⟂ y | x, with p(m | x) = Π_n e_n(x_n).
  Inference then uses

      p(y_obs, m) = Σ_x ∫ p(x, y) Π_n e_n(x_n) dy_miss,

  and every posterior (γ, ξ, MPM, FFBS draws, imputations) is given
  (y_obs, m). Given x, y_miss does not depend on m, so the laws of the
  missing values given the states are unchanged; only the state posteriors
  change.

The factor e_n(i) is a per-position, per-state likelihood. Every mechanism
here makes it a function of x_n alone once the mask is observed, so the
inference of :mod:`pmcprg.pmc.inference` and :mod:`pmcprg.pmc.gaps` applies it
as an evidence vector: the initial message is multiplied by e_0 and the
columns of every transition into position n by e_n (see those modules).

Mechanisms
----------
``"state"`` — :class:`StateMissingness` (``rates`` π, length K)
    Given X the m_n are independent, P(m_n = 1 | x_n = i) = π_i:
    e_n(i) = π_i if m_n = 1, 1 − π_i otherwise.

``"state-markov"`` — :class:`StateMarkovMissingness` (``onset`` a,
``persistence`` b, length K each)
    Given X the mask is a two-state Markov chain whose transition depends on
    the current state:

        P(m_n = 1 | m_{n-1} = 0, x_n = i) = a_i     (a gap starts)
        P(m_n = 1 | m_{n-1} = 1, x_n = i) = b_i     (a gap goes on)
        P(m_0 = 1 | x_0 = i) = s_i = a_i / (1 − b_i + a_i),

    s_i being the stationary missing probability of the mask chain under a
    constant state i. e_n(i) = p(m_n | m_{n-1}, x_n = i), e_0(i) = s_i or
    1 − s_i = (1 − b_i) / (1 − b_i + a_i). A burst of L missing rows counts
    as one onset and L − 1 continuations, where ``"state"`` counts L
    independent pieces of evidence π_i^L. ``"state"`` with rates π is
    ``"state-markov"`` with a = b = π. a_i = 0 with b_i = 1 is refused: the
    mask chain never moves and s_i is 0/0.

Every probability may be 0 or 1: a state i with e_n(i) = 0 is then
impossible at n (log e = −∞ in the log-space passes); a step at which every
state is impossible raises :class:`~pmcprg.exceptions.IncompatibleObservationError`.

TOML
----
::

    [missingness]
    mechanism = "state"          # "ignorable" (same as no table) | "state" | "state-markov"
    rates = [0.01, 0.3]          # "state": π_i, one per state

    # or
    [missingness]
    mechanism = "state-markov"
    onset = [0.002, 0.02]        # a_i
    persistence = [0.8, 0.95]    # b_i

The table round-trips through ``PMCModel.from_dict`` / ``raw`` / ``save``;
:meth:`pmcprg.pmc.model.PMCModel.with_missingness` returns a copy of a model
with another mechanism (or ``None``).

Estimation
----------
ICE and SEM (:mod:`pmcprg.pmc.ice`, :mod:`pmcprg.pmc.sem`) read the config
key ``missingness``: ``"model"`` (default) carries the mechanism of the
initial model with its parameters fixed, ``"ignorable"`` drops it, and
``"state"`` / ``"state-markov"`` estimate it. Every E-step gives posteriors
given (y_obs, m); the mask factors e_n(x_n) involve no other parameter, so
the other M-step formulas are unchanged, and the mechanism's M-step
maximises Σ_n Σ_i γ_n(i) log e_n(i) (ICE: the exact γ, with either
``missing_strategy``; SEM: the one-hot drawn path) —
:func:`estimate_mechanism`:

* ``"state"`` — π_i = Σ_n γ_n(i) m_n / Σ_n γ_n(i) (:func:`estimate_state`).
* ``"state-markov"`` — for n ≥ 1 the transitions split on m_{n−1}:
  a_i = Σ_{n: m_{n−1}=0} γ_n(i) m_n / Σ_{n: m_{n−1}=0} γ_n(i), b_i the same
  over m_{n−1} = 1. The n = 0 term γ_0(i) log p(m_0 | s_i), s_i = a_i /
  (1 − b_i + a_i), couples a_i and b_i; it is maximised exactly, per state,
  from that closed form (:func:`estimate_state_markov`, two parameters,
  L-BFGS-B on the logits). Measured (``report/missing_state/design_measurements.py``,
  HMC-IN, a = (0.005, 0.03), b = (0.7, 0.9), 100 ICE runs each): dropping
  the term moves the estimates by 4–7 % of their standard deviation at
  N = 500 and ≤ 4 % at N = 2000, with no systematic shift (|mean Δ| ≤ 0.02
  sd); the log-likelihood reached is higher with the exact step by a median
  0.002 nat — but by more than 0.1 nat in 11 runs of 100 at N = 500, up to
  2.5 nat: a series that starts inside a long burst, where log s_i is the
  only term that ties a small a_i to a b_i near 1 (â_0 = 9.7e-4 exact
  against 9.2e-5 without). The exact step costs K small optimisations per
  iteration (8 % of an HMC-IN ICE iteration at N = 2000).

**Start.** A mechanism of the estimated kind carried by the initial model is
the start (a ``"state"`` π becomes a = b = π for ``"state-markov"``);
otherwise the start is the state-independent MLE of the mask
(:func:`common_mechanism`: π = M / N, or the common Markov chain). Its
factors do not depend on the state, so the first E-step gives the ignorable
posteriors and the first M-step estimates the mechanism from them. That
start is not a fixed point on data with a real state dependence: from π =
(0.16, 0.16) the first M-step gives (0.055, 0.27) on average and ICE
converges to the truth (0.02, 0.30) (50 runs, N = 2000). Where the states are
i.i.d. (A with equal rows) π is not identified — only the observed mixture
weights ∝ p_i (1 − π_i) and Σ p_i π_i are — and the estimates wander
(sd 0.06–0.08 around (0.14, 0.19) for the same truth).

**On real data, start from the ignorable fit.** On PAMAP2
(``report/missing_state/pamap2``), up to 17 % of unsupervised HMC-IN fits
that estimated the mechanism from a k-means start ended in a worse basin
than "ignorable fit + common mask" (down to −100 nat, and in one case 5 % →
72 % error at the missing windows); from the ignorable fit
(``ice(fit_ignorable, Y, {"missingness": ..., "init": "model"})``) every fit
gained. The mask also helps only when the dropout rate is homogeneous within
each state: there, a model state that mixed activities with different
dropout rates (walking in "locomotion") flipped whole bouts at K = 3.

**Boundary guard.** A rate at 0 or 1 makes its state impossible at every
missing (or observed) row, so the posterior weight that would move it is 0:
an absorbing state for EM. Each M-step therefore adds c = :data:`PSEUDO_COUNT`
= 1 pseudo-observation at the pooled rate of the mask (π̄ = (M + ½)/(N + 1);
ā, b̄ from the transition counts the same way) — the posterior mode under a
Beta prior worth one observation, which also gives a state without weight
the pooled rate instead of 0/0. Measured (same script, 100 runs per cell,
c ∈ {0, 0.1, 1, 10}): from a start π_0 = 0 (truth 0.05) the estimate stays
exactly 0 without the guard in all 50 runs, 58 nat below the fit from the
common start at N = 2000; with c = 1 it recovers in 6–9 iterations to the
same log-likelihood. On π = (0.02, 0.3) and (0, 0.2) c = 1 changes bias and
RMSE by at most 0.001 (c = 10 biases the small rate by +0.008 at N = 500);
for the persistence of a state with one or two bursts (b = 0.5, a = 0.002)
it trades bias (+0.21 at N = 500, +0.09 at N = 2000) for a lower RMSE (0.29
against 0.37, 0.19 against 0.29). c = 0.1 already leaves rates of 1e-6.
Not a config key: nothing measured calls for another value.

**Complete Y.** m = 0 everywhere: the MLE is π̂ = 0 (â = 0, b̂ unidentified),
whose factors are all 1 — the ignorable model. ICE and SEM then log a
WARNING and fit with ``missingness = "ignorable"``.

The likelihood-ratio test of a state-independent against a state-dependent
mechanism is :func:`pmcprg.pmc.missingness_lr.missingness_lr_test`.

References
----------
* Little, R. J. A. & Rubin, D. B. (2019). *Statistical Analysis with Missing
  Data*, 3rd ed., Wiley — ignorability of the missingness mechanism, and
  selection models p(y) p(m | y) for data missing not at random.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

__all__ = [
    "MECHANISMS",
    "PSEUDO_COUNT",
    "StateMissingness",
    "StateMarkovMissingness",
    "parse_missingness",
    "as_missingness",
    "estimate_state",
    "estimate_state_markov",
    "estimate_mechanism",
    "common_mechanism",
    "mask_log_likelihood",
    "describe_mechanism",
]

#: Values of ``[missingness].mechanism``; ``"ignorable"`` is the absence of a
#: mechanism (``PMCModel.missingness is None``).
MECHANISMS: tuple[str, ...] = ("ignorable", "state", "state-markov")


def _probabilities(name: str, values, K: int | None = None) -> tuple[float, ...]:
    """Validate a vector of probabilities in [0, 1] (length K when given)."""
    if isinstance(values, (str, bytes)) or np.ndim(values) != 1:
        raise ValueError(f"[missingness].{name} must be a list of probabilities, got {values!r}.")
    out = []
    for v in values:
        if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)):
            raise ValueError(f"[missingness].{name} must hold numbers, got {v!r} in {values!r}.")
        f = float(v)
        if not (math.isfinite(f) and 0.0 <= f <= 1.0):
            raise ValueError(
                f"[missingness].{name} must hold probabilities in [0, 1], got {v!r}."
            )
        out.append(f)
    if not out:
        raise ValueError(f"[missingness].{name} is empty.")
    if K is not None and len(out) != K:
        raise ValueError(
            f"[missingness].{name} must have one entry per state (K = {K}), got {len(out)}."
        )
    return tuple(out)


def _mask(miss) -> np.ndarray:
    miss = np.asarray(miss, dtype=bool)
    if miss.ndim != 1:
        raise ValueError(f"the missingness mask must be 1-D (N,), got shape {miss.shape}.")
    return miss


@dataclass(frozen=True)
class StateMissingness:
    """Mechanism ``"state"``: P(m_n = 1 | x_n = i) = π_i, independently given X.

    Attributes
    ----------
    rates : tuple of K floats in [0, 1] — π_i.
    """

    mechanism: ClassVar[str] = "state"
    rates: tuple[float, ...]

    def __post_init__(self):
        object.__setattr__(self, "rates", _probabilities("rates", self.rates))

    @property
    def K(self) -> int:
        return len(self.rates)

    def evidence(self, miss) -> np.ndarray:
        """(N, K) factors e_n(i) = π_i if ``miss[n]`` else 1 − π_i."""
        miss = _mask(miss)
        r = np.asarray(self.rates, dtype=float)
        return np.where(miss[:, None], r[None, :], (1.0 - r)[None, :])

    def log_evidence(self, miss) -> np.ndarray:
        """(N, K) log e_n(i) (−∞ where e_n(i) = 0), log(1 − π) by ``log1p``."""
        miss = _mask(miss)
        r = np.asarray(self.rates, dtype=float)
        with np.errstate(divide="ignore"):
            return np.where(miss[:, None], np.log(r)[None, :], np.log1p(-r)[None, :])

    def to_table(self) -> dict:
        """The ``[missingness]`` TOML table of this mechanism."""
        return {"mechanism": self.mechanism, "rates": list(self.rates)}


@dataclass(frozen=True)
class StateMarkovMissingness:
    """Mechanism ``"state-markov"``: a Markov mask whose transition depends on x_n.

    Attributes
    ----------
    onset       : tuple of K floats in [0, 1] — a_i = P(m_n = 1 | m_{n-1} = 0, x_n = i).
    persistence : tuple of K floats in [0, 1] — b_i = P(m_n = 1 | m_{n-1} = 1, x_n = i).

    The initial mask follows the stationary law of the mask chain under a
    constant state: P(m_0 = 1 | x_0 = i) = s_i = a_i / (1 − b_i + a_i)
    (:attr:`stationary`). a_i = 0 with b_i = 1 is refused (s_i = 0/0).
    """

    mechanism: ClassVar[str] = "state-markov"
    onset: tuple[float, ...]
    persistence: tuple[float, ...]

    def __post_init__(self):
        a = _probabilities("onset", self.onset)
        b = _probabilities("persistence", self.persistence, len(a))
        stuck = [i for i in range(len(a)) if a[i] == 0.0 and b[i] == 1.0]
        if stuck:
            raise ValueError(
                f"[missingness] state-markov: onset = 0 and persistence = 1 for "
                f"state(s) {stuck} — the mask chain never moves, so the initial "
                f"missing probability a / (1 − b + a) is 0/0. Give that state a "
                f"positive onset or a persistence below 1."
            )
        object.__setattr__(self, "onset", a)
        object.__setattr__(self, "persistence", b)

    @property
    def K(self) -> int:
        return len(self.onset)

    def _arrays(self):
        a = np.asarray(self.onset, dtype=float)
        b = np.asarray(self.persistence, dtype=float)
        return a, b, (1.0 - b) + a

    @property
    def stationary(self) -> np.ndarray:
        """(K,) s_i = a_i / (1 − b_i + a_i) — the law of m_0 given x_0 = i."""
        a, _, den = self._arrays()
        return a / den

    def evidence(self, miss) -> np.ndarray:
        """(N, K) factors e_n(i) = p(m_n | m_{n-1}, x_n = i); e_0(i) = p(m_0 | x_0 = i)."""
        miss = _mask(miss)
        a, b, den = self._arrays()
        out = np.empty((miss.size, a.size))
        if miss.size == 0:
            return out
        out[0] = np.where(miss[0], a / den, (1.0 - b) / den)
        prev, cur = miss[:-1, None], miss[1:, None]
        out[1:] = np.where(cur, np.where(prev, b, a), np.where(prev, 1.0 - b, 1.0 - a))
        return out

    def log_evidence(self, miss) -> np.ndarray:
        """(N, K) log e_n(i) (−∞ where e_n(i) = 0), log(1 − ·) by ``log1p``."""
        miss = _mask(miss)
        a, b, den = self._arrays()
        out = np.empty((miss.size, a.size))
        if miss.size == 0:
            return out
        with np.errstate(divide="ignore"):
            la, lb, l1a, l1b, lden = np.log(a), np.log(b), np.log1p(-a), np.log1p(-b), np.log(den)
        out[0] = np.where(miss[0], la - lden, l1b - lden)
        prev, cur = miss[:-1, None], miss[1:, None]
        out[1:] = np.where(cur, np.where(prev, lb, la), np.where(prev, l1b, l1a))
        return out

    def to_table(self) -> dict:
        """The ``[missingness]`` TOML table of this mechanism."""
        return {"mechanism": self.mechanism, "onset": list(self.onset),
                "persistence": list(self.persistence)}


_KEYS = {
    "ignorable": set(),
    "state": {"rates"},
    "state-markov": {"onset", "persistence"},
}


def parse_missingness(table, K: int):
    """The mechanism of a ``[missingness]`` table (``None``: ignorable).

    ``table`` None (no table) or ``mechanism = "ignorable"`` give ``None``.
    Unknown mechanisms, missing or unexpected keys, vectors of the wrong
    length and values outside [0, 1] raise ``ValueError``.
    """
    if table is None:
        return None
    if not isinstance(table, dict):
        raise ValueError(f"[missingness] must be a table, got {table!r}.")
    if "mechanism" not in table:
        raise ValueError(
            f"[missingness] needs a 'mechanism' key, one of {list(MECHANISMS)}."
        )
    mech = table["mechanism"]
    if mech not in MECHANISMS:
        raise ValueError(
            f"Unknown [missingness].mechanism {mech!r}. Valid: {list(MECHANISMS)}."
        )
    keys = set(table) - {"mechanism"}
    need = _KEYS[mech]
    if keys != need:
        extra, missing = sorted(keys - need), sorted(need - keys)
        parts = ([f"missing key(s) {missing}"] if missing else []) + (
            [f"unexpected key(s) {extra}"] if extra else [])
        raise ValueError(
            f"[missingness] mechanism = {mech!r} takes the key(s) {sorted(need)}: "
            f"{'; '.join(parts)}."
        )
    if mech == "ignorable":
        return None
    if mech == "state":
        return StateMissingness(rates=_probabilities("rates", table["rates"], K))
    return StateMarkovMissingness(onset=_probabilities("onset", table["onset"], K),
                                  persistence=_probabilities("persistence", table["persistence"], K))


def as_missingness(spec, K: int):
    """Normalise ``spec`` — ``None``, a mechanism object or a ``[missingness]``
    table — to a mechanism object of K states (or ``None``)."""
    if spec is None:
        return None
    if isinstance(spec, (StateMissingness, StateMarkovMissingness)):
        return parse_missingness(spec.to_table(), K)
    if isinstance(spec, dict):
        return parse_missingness(spec, K)
    raise TypeError(
        f"missingness must be None, a StateMissingness / StateMarkovMissingness "
        f"or a [missingness] table (dict), got {type(spec).__name__}."
    )


# ---------------------------------------------------------------------------
# Estimation — M-steps of ICE / SEM and the state-independent (null) MLE
# ---------------------------------------------------------------------------

#: Pseudo-count c of the boundary guard of :func:`estimate_mechanism` (module
#: docstring, "Estimation"): every rate is the posterior mode under a Beta
#: prior worth c observations centred on the pooled rate of the mask.
PSEUDO_COUNT: float = 1.0

#: Bound on the logits of the numerical "state-markov" M-step: a rate is kept
#: in [σ(−36), σ(36)] = [2.3e-16, 1 − 2.3e-16]. Reached only by the unguarded
#: (c = 0) state-independent MLE when a count is zero — e.g. no burst of two
#: missing rows, b̂ = 0 — where the log-likelihood lost is below 1e-13.
_LOGIT_MAX: float = 36.0


def _gamma(gamma, N: int) -> np.ndarray:
    g = np.asarray(gamma, dtype=float)
    if g.ndim != 2 or g.shape[0] != N:
        raise ValueError(
            f"the state posteriors must have shape (N, K) with N = {N} (the mask "
            f"length), got {g.shape}."
        )
    return g


def _pooled(k, n) -> float:
    """(k + ½) / (n + 1): a pooled rate of the mask, never 0 or 1 (the target
    of the guard; Jeffreys' Beta(½, ½) posterior mean)."""
    return (float(k) + 0.5) / (float(n) + 1.0)


def _ratio(num, den, fallback):
    """num / den where den > 0, ``fallback`` elsewhere, clipped to [0, 1]."""
    num, den = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0.0, num / np.where(den > 0.0, den, 1.0), fallback)
    return np.clip(out, 0.0, 1.0)


def estimate_state(gamma, miss, *, pseudo_count: float | None = None) -> StateMissingness:
    """M-step of ``"state"``: π_i = (Σ_n γ_n(i) m_n + c π̄) / (Σ_n γ_n(i) + c).

    ``gamma`` (N, K) are the state posteriors given (y_obs, m) — soft for ICE,
    the one-hot drawn path for SEM — and ``miss`` the mask m. With c = 0 this
    is the exact maximiser of the expected complete-data log-likelihood
    Σ_n Σ_i γ_n(i) [m_n log π_i + (1 − m_n) log(1 − π_i)]; c > 0 adds c
    pseudo-observations at the pooled rate π̄ = (M + ½)/(N + 1), M = Σ m_n,
    which keeps every π_i in (0, 1) (module docstring, "Estimation"). A state
    of zero weight gets π̄.
    """
    miss = _mask(miss)
    g = _gamma(gamma, miss.size)
    c = PSEUDO_COUNT if pseudo_count is None else float(pseudo_count)
    target = _pooled(miss.sum(), miss.size)
    S, S1 = g.sum(axis=0), g[miss].sum(axis=0)
    rates = _ratio(S1 + c * target, S + c, target)
    return StateMissingness(rates=tuple(float(r) for r in rates))


def _markov_counts(g: np.ndarray, miss: np.ndarray):
    """Posterior-weighted transition counts of the mask, n ≥ 1, per state."""
    prev, cur, gn = miss[:-1], miss[1:], g[1:]
    return (gn[~prev & cur].sum(axis=0), gn[~prev & ~cur].sum(axis=0),     # onset: 1, 0
            gn[prev & cur].sum(axis=0), gn[prev & ~cur].sum(axis=0))       # persistence: 1, 0


def _log_sigmoid(u):
    """log σ(u) = −log(1 + e^{−u}), stable for any sign of u."""
    return -np.logaddexp(0.0, -u)


def _markov_maximise(na1: float, na0: float, nb1: float, nb0: float, g0: float,
                     start: tuple[float, float]) -> tuple[float, float]:
    """(a, b) maximising the per-state objective of the ``"state-markov"`` M-step

        F(a, b) = na1 log a + na0 log(1 − a) + nb1 log b + nb0 log(1 − b)
                  − g0 log(1 − b + a),

    in which the initial term γ_0(i) log p(m_0 | s_i), s = a / (1 − b + a),
    has been split: its numerator γ_0 m_0 log a or γ_0 (1 − m_0) log(1 − b)
    is already counted in na1 or nb0, and −γ_0 log(1 − b + a) couples a and
    b. L-BFGS-B on the logits (u, v) = (logit a, logit b), analytic gradient

        ∂F/∂u = na1 (1 − a) − na0 a − g0 a (1 − a) / (1 − b + a),
        ∂F/∂v = nb1 (1 − b) − nb0 b + g0 b (1 − b) / (1 − b + a),

    started from ``start``; logits bounded by ±:data:`_LOGIT_MAX`.
    """
    from scipy.optimize import minimize

    scale = max(na1 + na0 + nb1 + nb0 + g0, 1e-300)

    def neg(z):
        u, v = z
        la, l1a = _log_sigmoid(u), _log_sigmoid(-u)
        lb, l1b = _log_sigmoid(v), _log_sigmoid(-v)
        a, b = math.exp(la), math.exp(lb)
        one_b = math.exp(l1b)
        den = one_b + a
        F = na1 * la + na0 * l1a + nb1 * lb + nb0 * l1b - g0 * math.log(den)
        da = a * (1.0 - a)
        db = b * one_b
        gu = na1 * (1.0 - a) - na0 * a - g0 * da / den
        gv = nb1 * one_b - nb0 * b + g0 * db / den
        return -F / scale, np.array([-gu / scale, -gv / scale])

    def logit(p):
        p = min(max(float(p), 1e-300), 1.0)
        if p >= 1.0:
            return _LOGIT_MAX
        return float(np.clip(math.log(p) - math.log1p(-p), -_LOGIT_MAX, _LOGIT_MAX))

    z0 = np.array([logit(start[0]), logit(start[1])])
    res = minimize(neg, z0, jac=True, method="L-BFGS-B",
                   bounds=[(-_LOGIT_MAX, _LOGIT_MAX)] * 2,
                   options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 500})
    u, v = (res.x if np.all(np.isfinite(res.x)) and res.fun <= neg(z0)[0] else z0)
    return 1.0 / (1.0 + math.exp(-u)), 1.0 / (1.0 + math.exp(-v))


def estimate_state_markov(gamma, miss, *,
                          pseudo_count: float | None = None) -> StateMarkovMissingness:
    """M-step of ``"state-markov"`` — exact per state (module docstring, "Estimation").

    Per state i the expected complete-data log-likelihood of the mask is

        Σ_{n ≥ 1} γ_n(i) log p(m_n | m_{n−1}, a_i, b_i) + γ_0(i) log p(m_0 | s_i),

    plus c pseudo-observations at the pooled onset and persistence rates ā, b̄
    of the mask (the guard; ā = (#{0→1} + ½) / (#{0→·} + 1), b̄ likewise).
    Without the initial term it is maximised in closed form,

        a_i = (Σ_{n ≥ 1: m_{n−1} = 0} γ_n(i) m_n + c ā) / (Σ_{n ≥ 1: m_{n−1} = 0} γ_n(i) + c),
        b_i = the same sum over n ≥ 1 with m_{n−1} = 1;

    the initial term couples a_i and b_i through s_i = a_i / (1 − b_i + a_i),
    so :func:`_markov_maximise` maximises the exact objective from that closed
    form (two parameters per state; skipped when γ_0(i) = 0, where the closed
    form is exact).
    """
    miss = _mask(miss)
    g = _gamma(gamma, miss.size)
    K = g.shape[1]
    c = PSEUDO_COUNT if pseudo_count is None else float(pseudo_count)
    na1, na0, nb1, nb0 = _markov_counts(g, miss)
    prev, cur = miss[:-1], miss[1:]
    a_bar = _pooled((~prev & cur).sum(), (~prev).sum())
    b_bar = _pooled((prev & cur).sum(), prev.sum())
    na1, na0 = na1 + c * a_bar, na0 + c * (1.0 - a_bar)
    nb1, nb0 = nb1 + c * b_bar, nb0 + c * (1.0 - b_bar)
    a_cf = _ratio(na1, na1 + na0, a_bar)
    b_cf = _ratio(nb1, nb1 + nb0, b_bar)
    onset, persistence = [], []
    m0 = bool(miss[0]) if miss.size else False
    for i in range(K):
        g0 = float(g[0, i]) if miss.size else 0.0
        if g0 <= 0.0:
            a, b = float(a_cf[i]), float(b_cf[i])
        else:
            a, b = _markov_maximise(float(na1[i]) + g0 * m0, float(na0[i]),
                                    float(nb1[i]), float(nb0[i]) + g0 * (not m0), g0,
                                    (float(a_cf[i]), float(b_cf[i])))
        onset.append(a)
        persistence.append(b)
    return StateMarkovMissingness(onset=tuple(onset), persistence=tuple(persistence))


def estimate_mechanism(mechanism: str, gamma, miss, *, pseudo_count: float | None = None):
    """The M-step of ``mechanism`` (``"state"`` or ``"state-markov"``) given the
    state posteriors ``gamma`` (N, K) and the mask ``miss`` (module docstring,
    "Estimation"): :func:`estimate_state` or :func:`estimate_state_markov`."""
    if mechanism == "state":
        return estimate_state(gamma, miss, pseudo_count=pseudo_count)
    if mechanism == "state-markov":
        return estimate_state_markov(gamma, miss, pseudo_count=pseudo_count)
    raise ValueError(f"Cannot estimate mechanism {mechanism!r}: 'state' or 'state-markov'.")


def common_mechanism(mechanism: str, miss, K: int):
    """Maximum-likelihood state-independent mechanism of the mask ``miss``.

    The null of :func:`pmcprg.pmc.missingness_lr.missingness_lr_test`: the
    rates do not depend on the state, so the mask is independent of X and its
    likelihood p(m) is maximised on the mask alone.

    * ``"state"`` — π_i = M / N for every i (M missing rows of N).
    * ``"state-markov"`` — the two-state Markov chain with stationary start:
      (a, b) maximises log p(m_0 | s) + Σ_{n ≥ 1} log p(m_n | m_{n−1}), the
      objective of :func:`estimate_state_markov` with γ ≡ 1 and no guard,
      from its closed form without the initial term.

    Returns the K-state mechanism with these common values. Its evidence
    factors do not depend on the state: every posterior is the ignorable one
    and log p(y_obs, m) = log p(y_obs) + log p(m).
    """
    miss = _mask(miss)
    N, M = miss.size, int(miss.sum())
    if mechanism == "state":
        return StateMissingness(rates=(M / N if N else 0.0,) * int(K))
    if mechanism != "state-markov":
        raise ValueError(f"Unknown mechanism {mechanism!r}: 'state' or 'state-markov'.")
    fit = estimate_state_markov(np.ones((N, 1)), miss, pseudo_count=0.0)
    return StateMarkovMissingness(onset=fit.onset * int(K), persistence=fit.persistence * int(K))


def mask_log_likelihood(mechanism, miss) -> float:
    """log p(m) of a **state-independent** mechanism (every state the same
    factors — e.g. :func:`common_mechanism`): Σ_n log e_n(i) for any i."""
    le = mechanism.log_evidence(miss)
    if le.size and not np.allclose(le, le[:, :1], rtol=0.0, atol=0.0, equal_nan=True):
        raise ValueError("mask_log_likelihood needs a state-independent mechanism.")
    return float(le[:, 0].sum()) if le.size else 0.0


def describe_mechanism(mechanism) -> str | None:
    """A one-line summary of a mechanism's ``[missingness]`` table, e.g.
    ``"state  (rates = [0.05, 0.30])"`` — shared by the CLI ('estimate') and
    the GUI (the estimation-result log) to report a carried or estimated
    mechanism next to a fit's other parameters. ``None`` (ignorable) returns
    ``None``; the caller decides whether/how to report that case."""
    if mechanism is None:
        return None
    tbl = mechanism.to_table()
    detail = ", ".join(
        f"{k} = [{', '.join(f'{v:.4g}' for v in tbl[k])}]"
        for k in ("rates", "onset", "persistence") if k in tbl
    )
    return f"{tbl['mechanism']}  ({detail})"
