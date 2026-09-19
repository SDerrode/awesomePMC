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

Estimation. ICE and SEM (:mod:`pmcprg.pmc.ice`, :mod:`pmcprg.pmc.sem`) carry
the mechanism of the initial model unchanged: its parameters are held fixed
and every E-step uses the posteriors given (y_obs, m). They are not
estimated in this version.

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
    "StateMissingness",
    "StateMarkovMissingness",
    "parse_missingness",
    "as_missingness",
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
