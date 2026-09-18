"""Shared M-step / warm-start infrastructure for unsupervised PMC estimators.

This module is the **stable internal API** between the ICE
(:mod:`pmcprg.pmc.ice`) and SEM (:mod:`pmcprg.pmc.sem`) estimators — and is the
single import surface that any future sister estimator (Variational EM,
Gibbs samplers, …) should target. It replaces the previous arrangement,
where ``sem.py`` reached directly into underscore-private symbols of
``ice.py``.

Why a separate module?
----------------------
The M-step machinery (prior update, copula selection, margin selection,
warm-start, multistart perturbation, per-iteration snapshots) is genuinely
shared between ICE and SEM: both algorithms produce posterior weights
(soft for ICE, hard for SEM) and feed them through the *same* parameter
update. Without a shared surface, ``sem.py`` had to import
``_m_step``, ``_warmstart_from_kmeans``, ``_snapshot_*`` etc. from
``ice.py`` — an underscore-private import that the audit flagged.

The actual implementations still live in :mod:`pmcprg.pmc.ice` (canonical
home — that is where the matching documentation, references, and tests
already are). This module simply re-exports them under public names so
the dependency direction is explicit:

    ice.py        ──┐
                    ├── re-exported by → _estim_common
    sem.py        ──┘                       ▲
                                            │
    future estim. ──────── imports ─────────┘

The names exported here mirror the underscore-private names in ``ice.py``
without the leading underscore. The underscore-private names remain in
``ice.py`` as the canonical definitions; this module is a thin facade.
"""

from __future__ import annotations

import itertools
import logging
import multiprocessing
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np

from pmcprg.copulas._base import CopulaEnum
from pmcprg.pmc.ice import (
    COPULA_MARGIN_MODES,
    DEFAULT_COPULA_MARGINS,
    DEFAULT_MARGIN_SELECTION_RULE,
    DEFAULT_MISSING_DRAWS,
    DEFAULT_MISSING_STRATEGY,
    DEFAULT_SELECTION_CRITERION,
    EXTRA_PARAM_BOUNDS,
    INIT_STRATEGIES,
    MARGIN_SELECTION_RULES,
    MISSING_STRATEGIES,
    SELECTION_CRITERIA,
    IceResult,
    IceTrace,
    _BestIterate          as BestIterate,
    _best_iter            as best_iter_of,
    _DEFAULT_CANDIDATES   as DEFAULT_CANDIDATES,
    _check_copula_margins as check_copula_margins,
    _check_init_strategy  as check_init_strategy,
    _check_missing_cfg    as check_missing_cfg,
    _evaluate_log_lik     as evaluate_log_lik,
    _gap_e_step           as gap_e_step,
    _kmeans_label_assignment as kmeans_label_assignment,
    _m_step               as m_step,
    _perturb_initial_model as perturb_initial_model,
    _record_copula_margins as record_copula_margins,
    _snapshot_margins     as snapshot_margins,
    _snapshot_prior_p     as snapshot_prior_p,
    _snapshot_tau_family  as snapshot_tau_family,
    _warmstart_from_kmeans as warmstart_from_kmeans,
)
from pmcprg.pmc.model import PMCModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Copula-family multistart (config key ``multistart_families``)
# ---------------------------------------------------------------------------
#
# ICE (and SEM) can have several fixed points that differ by the copula
# *family* of some pairs: on the CSDA-2013 Exp. 3 design, a start at
# independence ends 20–58 nat (median) below the run started at the truth and
# selects Gumbel instead of the true Gaussian copula on a diagonal pair in
# 94–100 % of runs. Parameter jitter (``perturb_initial_model``) keeps the
# families and cannot leave that basin; these modes change the families of
# the starting points.

#: Values of the ``multistart_families`` config key.
#:
#: * ``"none"``   — parameter jitter only (the historical behaviour).
#: * ``"random"`` — every start after the first also redraws the copula family
#:   of every pair (i, j) uniformly among the candidates, with τ uniform on
#:   the central band of the family's range (see :data:`RANDOM_TAU_BAND`).
#: * ``"sweep"``  — the starts enumerate every combination of candidate
#:   families on the diagonal pairs (i, i) at τ = :data:`SWEEP_TAU`.
MULTISTART_FAMILY_MODES: tuple[str, ...] = ("none", "random", "sweep")

#: τ of the diagonal pairs of a ``"sweep"`` start, clipped to each family's
#: constructible range (FGM → 2/9, CubSec → 33/200, Product → 0).
SWEEP_TAU: float = 0.5

#: A ``"random"`` start draws τ uniformly on ``[a + b·s, a + (1 − b)·s]``, the
#: central ``1 − 2b`` of the family's registered range ``[a, a + s]`` (b = 0.2:
#: Gauss/Frank/Plackett/Student [−0.6, 0.6], Clayton/GH/Joe/BB1/survival
#: families [0.2, 0.8], A12/A14 [0.47, 0.87], FGM [−0.13, 0.13]), then clips it
#: to the constructible range. Away from independence, where every family
#: looks alike, and from the comonotone end, where the densities degenerate.
RANDOM_TAU_BAND: float = 0.2

#: ``"sweep"`` refuses more than this many family combinations (|C|^K).
MAX_SWEEP_COMBINATIONS: int = 256

# Parameter names of multi-parameter families other than τ (``df``, ``delta``).
_EXTRA_COPULA_KEYS: frozenset[str] = frozenset(
    p for c in CopulaEnum for p in c.value.PARAMETERS_SET_NAME if p != "tau_k"
)

# Separate RNG stream for the family draws, so that the parameter jitter of
# start k is the same under ``"random"`` as under ``"none"`` for a given seed.
_FAMILY_STREAM: int = 0x46414D   # "FAM"


def check_multistart_families(value) -> str:
    """Validate the ``multistart_families`` config key."""
    if not isinstance(value, str) or value not in MULTISTART_FAMILY_MODES:
        raise ValueError(
            f"Unknown multistart_families {value!r}. "
            f"Valid: {list(MULTISTART_FAMILY_MODES)}"
        )
    return value


def check_return_best_iterate(value) -> bool:
    """Validate the ``return_best_iterate`` config key (a boolean, or 0 / 1).

    Strict on purpose: ``bool("false")`` is ``True``, so a string from a
    hand-written config would silently switch the option on.
    """
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    raise ValueError(f"return_best_iterate must be a boolean, got {value!r}.")


def returned_log_lik(trace, cfg: dict) -> float:
    """Log-likelihood by which multistart ranks a start.

    ``trace.log_liks[-1]`` (the historical rule); with ``return_best_iterate``
    that of the model the start returns, ``log_liks[trace.returned_iter]``.
    ``-inf`` for an empty trace.
    """
    lls = trace.log_liks
    if not lls:
        return -np.inf
    if cfg.get("return_best_iterate", False) and 0 <= trace.returned_iter < len(lls):
        return lls[trace.returned_iter]
    return lls[-1]


def _family_candidates(cfg: dict, label: str) -> list:
    """Resolvable, de-duplicated ``CopulaEnum`` entries of ``cfg["candidates"]``."""
    out, seen, unknown = [], set(), []
    for short in cfg.get("candidates", []) or []:
        entry = CopulaEnum.from_short_name(str(short))
        if entry is None:
            unknown.append(short)
        elif entry.value.SHORT_NAME not in seen:
            seen.add(entry.value.SHORT_NAME)
            out.append(entry)
    if unknown:
        logger.warning(
            "%s multistart: unknown copula candidate(s) %s ignored for the "
            "family starts.", label, unknown,
        )
    return out


def _resolve_family_mode(model, cfg: dict, label: str) -> tuple[str, list]:
    """``(mode, candidates)`` actually used — ``"none"`` when families do not apply."""
    mode = check_multistart_families(cfg.get("multistart_families", "none"))
    if mode == "none":
        return mode, []
    if not model.variant.uses_copula:
        logger.info(
            "%s multistart: multistart_families=%r ignored — variant %s has "
            "no copula; parameter jitter only.", label, mode, model.variant.value,
        )
        return "none", []
    cands = _family_candidates(cfg, label)
    if not cands:
        logger.info(
            "%s multistart: multistart_families=%r ignored — no copula "
            "candidate; parameter jitter only.", label, mode,
        )
        return "none", []
    if mode == "sweep":
        n_comb = len(cands) ** model.K
        if n_comb > MAX_SWEEP_COMBINATIONS:
            raise ValueError(
                f"multistart_families='sweep' would enumerate |C|^K = "
                f"{len(cands)}^{model.K} = {n_comb} family combinations on the "
                f"diagonal pairs (limit {MAX_SWEEP_COMBINATIONS}). Use "
                f"multistart_families='random' with a chosen n_starts, or fewer "
                f"candidates."
            )
    return mode, cands


def _set_copula_family(blk: dict, entry, tau: float) -> None:
    """Put ``blk`` on family ``entry`` at ``tau`` (clipped), in place.

    A family change resets the extra parameters to the new family's initial
    values (:data:`pmcprg.pmc.ice.EXTRA_PARAM_BOUNDS`) and drops those it does not
    have; an unchanged family keeps them. When the family refuses the new τ
    with those extras — BB1's initial δ = 1.5 needs τ > 1/3, a kept δ needs
    τ > 1 − 1/δ — the extras are moved onto the admissible set
    (:meth:`CopulaVirt.constructible_params`); an accepted pair is kept.
    """
    changed = blk.get("name") != entry.value.SHORT_NAME
    blk["name"] = entry.value.SHORT_NAME
    # Plain clip, not ``correct_tau``: τ = 0.5 outside FGM's or CubSec's range
    # is the documented rule, not an input error worth a WARNING.
    blk["tau"] = float(np.clip(float(tau), *entry.constructible_tau_range()))
    own = EXTRA_PARAM_BOUNDS.get(entry.value.CLASS_NAME, {})
    if changed:
        for key in _EXTRA_COPULA_KEYS:
            if key in own:
                blk[key] = float(own[key][2])
            else:
                blk.pop(key, None)
    # Joint constraint of the family (G1): no RNG draw, only a refused pair moves.
    params = {"tau_k": blk["tau"], **{k: blk[k] for k in own if k in blk}}
    blk.update((k, val) for k, val in entry.klass.constructible_params(params).items()
               if k != "tau_k")


def _random_family_start(model, rng, fam_rng, jitter: float, cands: list):
    """Parameter jitter (``perturb_initial_model``) + a family redraw per pair.

    Per copula block, in declaration order: one ``integers`` draw picks the
    family uniformly among ``cands``, one ``uniform`` draw sets τ on the
    central band of its range (:data:`RANDOM_TAU_BAND`).
    """
    raw = perturb_initial_model(model, rng, jitter=jitter).raw
    for blk in raw.get("copulas", []):
        entry = cands[int(fam_rng.integers(len(cands)))]
        lo, hi = entry.value.TAU_MIN_MAX
        span = hi - lo
        tau = fam_rng.uniform(lo + RANDOM_TAU_BAND * span, hi - RANDOM_TAU_BAND * span)
        _set_copula_family(blk, entry, tau)
    return PMCModel.from_dict(raw)


def _sweep_start(model, combo: tuple):
    """``model`` with the diagonal pair (i, i) on ``combo[i]`` at τ = SWEEP_TAU."""
    raw = model.raw
    for blk in raw.get("copulas", []):
        i, j = int(blk["i"]), int(blk["j"])
        if i == j:
            _set_copula_family(blk, combo[i], SWEEP_TAU)
    return PMCModel.from_dict(raw)


def _sweep_label(combo: tuple) -> str:
    return "family-sweep:" + "/".join(e.value.SHORT_NAME for e in combo)


def build_multistart_inits(model, cfg: dict, *, label: str = "") -> list[tuple]:
    """Starting points ``[(PMCModel, run_tag), …]`` of a multistart run.

    The first start is always ``(model, "unperturbed")``; the list has
    ``max(1, n_starts)`` entries. Deterministic given ``multistart_seed``.

    * ``"none"``   — start s ≥ 1 is ``perturb_initial_model`` draw s
      (``"perturbed-s"``).
    * ``"random"`` — the same jittered model, then every pair (i, j) redrawn
      by :func:`_random_family_start` (``"family-random-s"``). The family
      draws use their own RNG stream, so prior and margins of start s are
      those of ``"none"``.
    * ``"sweep"``  — the ``|C|^K`` combinations of candidate families on the
      diagonal pairs, in ``itertools.product`` order over the candidates as
      listed (pair (0, 0) varies slowest), τ = :data:`SWEEP_TAU`, off-diagonal
      pairs and every other parameter as in ``model``, no jitter
      (``"family-sweep:Gauss/Clayton"``). ``n_starts < 1 + |C|^K`` keeps the
      first combinations and logs the dropped ones at WARNING; a larger
      ``n_starts`` fills the rest with ``"random"`` starts.

    The mode falls back to ``"none"`` (INFO) for a variant without copulas or
    an empty candidate list; ``"sweep"`` raises ``ValueError`` beyond
    :data:`MAX_SWEEP_COMBINATIONS` combinations.
    """
    n_starts = max(1, int(cfg["n_starts"]))
    mode, cands = _resolve_family_mode(model, cfg, label)
    rng    = np.random.default_rng(int(cfg["multistart_seed"]))
    jitter = float(cfg["multistart_jitter"])

    if mode == "none":
        return [(model, "unperturbed")] + [
            (perturb_initial_model(model, rng, jitter=jitter), f"perturbed-{s}")
            for s in range(1, n_starts)
        ]

    fam_rng = np.random.default_rng(
        np.random.SeedSequence([int(cfg["multistart_seed"]), _FAMILY_STREAM])
    )
    inits = [(model, "unperturbed")]
    if mode == "sweep":
        combos = list(itertools.product(cands, repeat=model.K))
        kept, dropped = combos[: n_starts - 1], combos[n_starts - 1:]
        if dropped:
            names = [_sweep_label(c) for c in dropped]
            shown = ", ".join(names[:8]) + (", …" if len(names) > 8 else "")
            logger.warning(
                "%s multistart: multistart_families='sweep' needs n_starts = "
                "1 + %d^%d = %d; n_starts=%d drops %d combination(s): %s",
                label, len(cands), model.K, 1 + len(combos), n_starts,
                len(dropped), shown,
            )
        inits += [(_sweep_start(model, c), _sweep_label(c)) for c in kept]
    for s in range(len(inits), n_starts):
        inits.append((_random_family_start(model, rng, fam_rng, jitter, cands),
                      f"family-random-{s}"))
    return inits


def run_multistart(single_run, model, cfg, *, label: str, worker=None):
    """Generic multistart driver shared by ICE and SEM (audit Q-1).

    Both estimators ran a near-identical ~40-line loop: short-circuit on
    ``n_starts == 1``, otherwise perturb the model for each subsequent start,
    run, keep the highest final log-likelihood, and attach the losing traces
    to the winner. That loop now lives here once.

    The starting points come from :func:`build_multistart_inits`
    (``multistart_families``: parameter jitter only, random copula families,
    or a sweep of the diagonal families). The winning start's label is its
    trace's ``run_tag``. Starts are ranked by :func:`returned_log_lik`: the
    last log-likelihood of the trace, or with ``return_best_iterate`` that of
    the iterate the start returns.

    Parameters
    ----------
    single_run : callable ``(init_model, run_tag, start_index) -> (PMCModel, trace)``
        Runs one fit. ``trace`` must expose ``log_liks`` (list) and a settable
        ``multistart_runs`` attribute. ``start_index`` lets a stochastic
        estimator (SEM) offset its RNG seed per start; deterministic ICE
        ignores it.
    model : PMCModel — the (already warm-started) initial model.
    cfg   : dict — merged estimator config (reads ``n_starts``,
        ``multistart_seed``, ``multistart_jitter``, ``multistart_workers``,
        ``multistart_families``, ``candidates``).
    label : str — ``"ICE"`` / ``"SEM"``, used as the log-message prefix.
    worker : optional ``(fn, Y)`` pair enabling process-based parallelism
        when ``cfg["multistart_workers"] > 1``. ``fn`` must be a *top-level*
        (picklable) callable ``(raw_dict, Y, cfg, run_tag, start_index) ->
        (PMCModel, trace)`` — each subprocess rebuilds the model from its raw
        dict. The starting points are pre-drawn with the same RNG consumption
        as the sequential path, so parallel and sequential runs produce the
        **same** result; only wall-clock and log order differ. ``progress_cb``
        cannot cross process boundaries: parallel mode logs per-run
        completion instead of per-iteration progress.

    Returns
    -------
    (PMCModel, trace) — the best run, its ``multistart_runs`` populated with
    the non-best traces (empty list when ``n_starts == 1``).
    """
    n_starts = max(1, int(cfg["n_starts"]))
    families = check_multistart_families(cfg.get("multistart_families", "none"))
    if n_starts == 1:
        if families != "none":
            # Validates the mode against the model (fallback INFO, sweep size
            # guard) and warns that a sweep is truncated to the model itself.
            build_multistart_inits(model, cfg, label=label)
        return single_run(model, "", 0)

    jitter  = float(cfg["multistart_jitter"])
    workers = min(max(1, int(cfg["multistart_workers"])), n_starts)
    if families == "none":
        logger.info(
            "%s multistart: n_starts=%d  jitter=%.2f  workers=%d",
            label, n_starts, jitter, workers,
        )
    else:
        logger.info(
            "%s multistart: n_starts=%d  jitter=%.2f  workers=%d  families=%s",
            label, n_starts, jitter, workers, families,
        )

    # Pre-draw every starting point up-front. RNG consumption is identical to
    # the former lazy per-iteration draws (one perturbation per start, in
    # order), and identical between the sequential and parallel paths — a
    # given (multistart_seed, jitter, multistart_families) always explores the
    # same starts.
    inits = build_multistart_inits(model, cfg, label=label)

    # results[s] keeps submission order so best-run selection (and its
    # first-wins tie-breaking) is independent of completion order.
    results: list[tuple[float, object, object] | None] = [None] * n_starts

    what = ("returned (best-iterate) log-lik" if cfg.get("return_best_iterate", False)
            else "final log-lik")

    def _record(s: int, tag: str, fitted_s, trace_s) -> None:
        final_ll = returned_log_lik(trace_s, cfg)
        logger.info(
            "%s multistart: run %d (%s) → %s = %.4f  (iters=%d)",
            label, s, tag, what, final_ll, len(trace_s.log_liks),
        )
        results[s] = (final_ll, fitted_s, trace_s)

    if workers > 1 and worker is not None:
        fn, Y = worker
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            futures = {
                pool.submit(fn, mdl.raw, Y, cfg, tag, s): (s, tag)
                for s, (mdl, tag) in enumerate(inits)
            }
            for fut in as_completed(futures):
                s, tag = futures[fut]
                try:
                    fitted_s, trace_s = fut.result()
                except Exception as exc:                      # pragma: no cover
                    logger.warning(
                        "%s multistart: run %d (%s) failed: %s", label, s, tag, exc,
                    )
                    continue
                _record(s, tag, fitted_s, trace_s)
    else:
        for s, (init_mdl, tag) in enumerate(inits):
            try:
                fitted_s, trace_s = single_run(init_mdl, tag, s)
            except Exception as exc:                          # pragma: no cover
                logger.warning(
                    "%s multistart: run %d (%s) failed: %s", label, s, tag, exc,
                )
                continue
            _record(s, tag, fitted_s, trace_s)

    runs = [r for r in results if r is not None]
    if not runs:                                              # pragma: no cover
        return single_run(model, "", 0)

    best_idx, (best_ll, best_mdl, best_trace) = max(
        enumerate(runs), key=lambda kv: kv[1][0],
    )
    logger.info(
        "%s multistart: best run = #%d [%s]  (%s = %.4f)",
        label, best_idx, getattr(best_trace, "run_tag", ""), what, best_ll,
    )
    best_trace.multistart_runs = [
        trace for k, (_, _, trace) in enumerate(runs) if k != best_idx
    ]
    return best_mdl, best_trace


def linearise_image(model, img, *, what: str) -> np.ndarray:
    """Validate a 2D/3D image against ``model.d`` and linearise it (audit Q-3).

    Shared by :func:`pmcprg.pmc.ice.ice_image` and :func:`pmcprg.pmc.sem.sem_image`,
    which had byte-identical validation. ``what`` names the caller for error
    messages (``"ice_image"`` / ``"sem_image"``).
    """
    from pmcprg.pmc.peano import image_to_signal

    if img.ndim not in (2, 3):
        raise ValueError(
            f"{what} expects 2D (H, W) or 3D (H, W, d); got shape {img.shape}."
        )
    img_d = 1 if img.ndim == 2 else img.shape[2]
    if img_d != model.d:
        raise ValueError(
            f"Image channels ({img_d}) do not match model.d = {model.d}."
        )
    return image_to_signal(img)


# ---------------------------------------------------------------------------
# Degenerate fitted states (a diagnostic — nothing in the estimate changes)
# ---------------------------------------------------------------------------
#
# The likelihood of a mixture-like model is unbounded: a margin whose variance
# collapses on one observation, or on an atom of discretised data, gains nats
# without limit, and a copula driven to τ = ±1 concentrates its mass on a
# curve. ICE and SEM can walk into such a point and nothing warned. The
# thresholds below were read off the real-series study (``report/real_series``:
# 266 fits of tsNH4, Beijing PM2.5 and PAMAP2, ``results/fits.csv``, fitted
# models in ``report/out/real_series/full/models``). With them
# :func:`degenerate_states` flags exactly the 15 fits the study called
# degenerate and none of the other 251.

#: A state whose stationary probability π_i is below this is degenerate. In
#: the study the 13 fits with a state below 0.5 % (12 tsNH4, 1 Huairou; all
#: K = 3 except one SEM start at K = 2) have π_min ≤ 0.24 % — five of them
#: exactly 0 — and 9 of them also a collapsed margin; the smallest π_min of the
#: other fits is 0.59 %, the next 0.97 %. On a series of N values a state with
#: π < 0.5 % holds fewer than N/200 of them (≈ 23 on tsNH4) to estimate a
#: margin and K copulas. A genuinely rare regime of a long series falls below
#: it too: the warning is then a false alarm, the fit is unchanged.
DEGENERATE_MIN_WEIGHT: float = 0.005

#: A margin (state f_i or pair f_ij) whose standard deviation is below this
#: fraction of the standard deviation of the observed data is degenerate. In
#: the study the 11 fits concerned sit at the fitting floors — sd 1e-8 (the
#: numerical margin fit), 1e-4 (√ of the 1e-8 variance floor: the raw ln PM2.5
#: atom of Huairou, among others), 1e-3 (the GICE initial-value floor), i.e.
#: 1.4e-8 … 1.4e-3 of the data sd — or below them (sd 4e-5), or at 0.6 % and
#: 0.9 % of the data sd (rare pair margins of tsNH4, K = 3). The smallest ratio of any other
#: fit is 3.5 % (tsNH4, K = 2 pair margins), and a real narrow regime, the
#: 3 µg/m³ detection floor of Huairou after dequantisation, 8 %. A ratio of
#: 1e-3 would miss the fit stuck at the 1e-3 floor (1.4e-3) and the two tsNH4
#: pair fits. Two regimes reach 1 % legitimately only when they are about 200
#: within-regime standard deviations apart.
DEGENERATE_SD_RATIO: float = 0.01

#: A copula whose τ is within this distance of a singular end of its family's
#: range (τ = ±1: comonotone / countermonotone, no density) is degenerate. The
#: bounded τ search stops 1e-4 × span short of it (τ = 0.9999 for Clayton or
#: Gumbel, ±0.9998 for Gauss). In the study 3 fits have such a τ (0.9998 to
#: 0.9999: the Huairou atom's Clayton, printed τ = 1.000, and two tsNH4 K = 3
#: pair fits), all with a collapsed margin too; the largest |τ| of the other
#: fits is 0.9984 (a degenerate SEM fit) and 0.9960 among the healthy ones (a
#: Frank copula, past its reachable 0.9943). A hint rather than a proof — a
#: pair of very small weight can end there — reported with the other findings.
#: The admissible ends of bounded families (FGM ±2/9, Clayton's τ = 0) are
#: not degenerate and are not checked.
DEGENERATE_TAU_EDGE: float = 1e-3

#: Normal-consistent interquartile range: IQR / 1.349 = σ for a Gaussian. The
#: scale of a margin whose standard deviation is undefined (Cauchy, t with
#: df ≤ 2).
_IQR_TO_SD: float = 1.3489795003921634


@dataclass(frozen=True)
class DegenerateFinding:
    """One sign of degeneracy of a fitted model — see :func:`degenerate_states`.

    Fields
    ------
    cause     : ``"state_weight"``, ``"margin_sd"`` or ``"copula_tau"``.
    i, j      : the state i (``j is None``: a state weight or a state margin
                f_i), or the pair (i, j) of a pair margin f_ij or a copula c_ij.
    value     : π_i, the margin's standard deviation, or τ_ij.
    threshold : the bound crossed — :data:`DEGENERATE_MIN_WEIGHT`; the
                absolute sd bound ``ratio × data_sd``; the distance to ±1.
    family    : the margin's or copula's family (``""`` for a weight).
    data_sd   : standard deviation of the observed data (``"margin_sd"``;
                that of the coordinate for a multivariate margin).
    coord     : coordinate of a multivariate margin, else ``None``.
    """

    cause: str
    i: int
    j: int | None
    value: float
    threshold: float
    family: str = ""
    data_sd: float | None = None
    coord: int | None = None

    @property
    def where(self) -> str:
        """``"state 2"``, ``"margin f_1"``, ``"margin f_12"``, ``"copula c_00"``."""
        if self.cause == "state_weight":
            return f"state {self.i}"
        sep = "" if self.j is None or max(self.i, self.j) < 10 else ","
        idx = f"{self.i}" if self.j is None else f"{self.i}{sep}{self.j}"
        return ("copula c_" if self.cause == "copula_tau" else "margin f_") + idx

    def __str__(self) -> str:
        if self.cause == "state_weight":
            return (f"{self.where}: stationary weight π = {self.value:.3g} "
                    f"< {self.threshold:.3g}")
        fam = f" ({self.family})" if self.family else ""
        if self.cause == "margin_sd":
            coord = "" if self.coord is None else f", coordinate {self.coord}"
            ratio = self.threshold / self.data_sd if self.data_sd else float("nan")
            return (f"{self.where}{fam}{coord}: sd = {self.value:.3g} < "
                    f"{self.threshold:.3g} ({ratio:.3g} × data sd {self.data_sd:.3g})")
        edge = 1.0 if self.value > 0 else -1.0
        return (f"{self.where}{fam}: τ = {self.value:.6g} within "
                f"{self.threshold:.3g} of {edge:+.0f}")


def _margin_scales(margin) -> np.ndarray:
    """Per-coordinate scale of a model margin: its standard deviation.

    Scalar margins: ``std()`` of the scipy law, or the normal-consistent IQR
    when the standard deviation is undefined; multivariate normal: √diag(cov).
    """
    if getattr(margin, "is_multivariate", False):
        cov = np.asarray(margin.params["cov"], dtype=float)
        return np.sqrt(np.clip(np.diag(cov), 0.0, None))
    import scipy.stats as ss
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        law = getattr(ss, margin.dist_name)(**margin.params)
        sd = float(law.std())
        if not np.isfinite(sd):
            sd = float((law.ppf(0.75) - law.ppf(0.25)) / _IQR_TO_SD)
    return np.array([sd])


def degenerate_states(
    model,
    Y: np.ndarray,
    *,
    min_weight: float | None = None,
    sd_ratio: float | None = None,
    tau_edge: float | None = None,
) -> list[DegenerateFinding]:
    """Signs that a fitted model sits at a degenerate point of its likelihood.

    Three checks, each on the model as it is (no inference, nothing changed):

    * **state weight** — a state i with stationary probability
      π_i < ``min_weight`` (:data:`DEGENERATE_MIN_WEIGHT`, 0.5 %);
    * **margin sd** — a margin, state f_i or pair f_ij (every block of
      ``model.margin_blocks()``), whose standard deviation is below
      ``sd_ratio`` (:data:`DEGENERATE_SD_RATIO`, 1 %) × the standard
      deviation of the observed rows of ``Y``. The margin's own standard
      deviation when defined, the normal-consistent IQR otherwise; per
      coordinate (√diag(cov) against each column's sd) for a multivariate
      normal margin;
    * **copula τ** — a copula c_ij whose τ is within ``tau_edge``
      (:data:`DEGENERATE_TAU_EDGE`, 1e-3) of a singular end ±1 of its
      family's range; a hint, usually seen together with a collapsed margin.

    The thresholds and the data that set them are documented at the three
    constants. A keyword left to ``None`` reads the module constant at call
    time.

    Parameters
    ----------
    model : PMCModel — typically a fitted model.
    Y     : np.ndarray, shape (N,) or (N, d) — the data it was fitted to;
            rows with a non-finite value are ignored. Only its standard
            deviation is used.

    Returns
    -------
    list[DegenerateFinding] — in the order: state weights, margins (block
    order), copulas (block order); empty for a healthy model.
    """
    from pmcprg.pmc.gaps import missing_mask

    min_weight = DEGENERATE_MIN_WEIGHT if min_weight is None else float(min_weight)
    sd_ratio = DEGENERATE_SD_RATIO if sd_ratio is None else float(sd_ratio)
    tau_edge = DEGENERATE_TAU_EDGE if tau_edge is None else float(tau_edge)
    findings: list[DegenerateFinding] = []

    for i, w in enumerate(np.asarray(model.stationary_pi, dtype=float)):
        if w < min_weight:
            findings.append(DegenerateFinding("state_weight", i, None, float(w), min_weight))

    Y = np.asarray(Y, dtype=float)
    Yc = Y.reshape(len(Y), -1)
    obs = Yc[~missing_mask(Y)]
    data_sd = obs.std(axis=0) if len(obs) > 1 else np.full(Yc.shape[1], np.nan)
    for blk in model.margin_blocks():
        i = int(blk["i"])
        j = int(blk["j"]) if "j" in blk else None
        margin = model.margin(i) if j is None else model.margin(i, j)
        scales = _margin_scales(margin)
        multi = len(scales) > 1
        for c, (sd, ref) in enumerate(zip(scales, data_sd)):
            if not (np.isfinite(ref) and ref > 0.0 and np.isfinite(sd)):
                continue
            bound = sd_ratio * float(ref)
            if sd < bound:
                findings.append(DegenerateFinding(
                    "margin_sd", i, j, float(sd), bound, family=str(blk["dist"]),
                    data_sd=float(ref), coord=c if multi else None,
                ))

    if model.variant.uses_copula:
        for blk in model.copula_blocks():
            tau = float(blk["tau"])
            entry = CopulaEnum.from_short_name(str(blk.get("name", "")))
            if entry is None or not np.isfinite(tau):
                continue
            ends = [e for e in entry.value.TAU_MIN_MAX if abs(float(e)) == 1.0]
            if any(abs(tau - float(e)) < tau_edge for e in ends):
                findings.append(DegenerateFinding(
                    "copula_tau", int(blk["i"]), int(blk["j"]), tau, tau_edge,
                    family=entry.value.SHORT_NAME,
                ))
    return findings


def report_degenerate_states(model, trace, Y: np.ndarray, *, label: str) -> list:
    """Run :func:`degenerate_states` on an estimator's returned model.

    Stores the findings in ``trace.degenerate`` and, when there are any, logs
    them as **one** WARNING line prefixed by ``label`` (``"ICE"`` / ``"SEM"``).
    A diagnostic must not break a fit: an exception is logged at DEBUG and
    leaves ``trace.degenerate`` at ``None`` (not checked).
    """
    try:
        findings = degenerate_states(model, Y)
    except Exception as exc:                                  # pragma: no cover
        logger.debug("%s: degenerate-state check failed: %s", label, exc, exc_info=True)
        return []
    trace.degenerate = findings
    if findings:
        logger.warning(
            "%s: degenerate fitted model — %s. Its likelihood may be unbounded "
            "(a collapsing variance or τ → ±1 gains nats without limit): do not "
            "compare it with other fits (pmcprg.pmc._estim_common.degenerate_states).",
            label, "; ".join(str(f) for f in findings),
        )
    return findings


def shared_estim_defaults() -> dict:
    """Default values for the cfg keys shared by ICE and SEM.

    Returned as a fresh dict on every call, so callers can mutate freely.
    :func:`ice_estim_defaults` / :func:`sem_estim_defaults` add each
    estimator's non-shared keys on top.

    This single source of truth ensures that adding a new shared option
    (a new selection criterion, a new warm-start strategy, …) requires
    only one change — both ICE and SEM pick it up automatically, with no
    silent default-drift between the two estimators.
    """
    return {
        "fit_margins":            False,
        "candidates":             list(DEFAULT_CANDIDATES),
        "selection_criterion":    DEFAULT_SELECTION_CRITERION,
        "margin_selection_rule":  DEFAULT_MARGIN_SELECTION_RULE,
        "init":                   "model",
        "kmeans_seed":            0,
        "n_starts":               1,
        "multistart_seed":        0,
        "multistart_jitter":      0.10,
        # Process-based parallelism across multistart runs. 1 = sequential
        # (default). Results are identical either way (starts are pre-drawn);
        # >1 trades per-iteration progress reporting for wall-clock speed.
        "multistart_workers":     1,
        # Copula families of the starts after the first: "none" (parameter
        # jitter only), "random" or "sweep" — see MULTISTART_FAMILY_MODES.
        "multistart_families":    "none",
        # Return the iterate with the highest log-likelihood instead of the
        # last one (ICE: ``_parse_ice_cfg``; SEM: a heuristic, see ``sem``).
        "return_best_iterate":    False,
    }


def ice_estim_defaults() -> dict:
    """Full ICE default cfg: shared keys + the ICE-only convergence keys.

    The single source read by ``_parse_ice_cfg`` **and** the GUI (spinbox
    initial values, ``load()`` fallbacks) — audit Q-9: these literals used to
    be re-hardcoded at every site.
    """
    return {
        **shared_estim_defaults(),
        # ICE-only convergence-control keys (SEM is stochastic and has no
        # deterministic convergence criterion).
        "max_iter": 50,
        "tol":      1e-4,
        "patience": 3,
    }


def sem_estim_defaults() -> dict:
    """Full SEM default cfg: shared keys + the SEM-only keys."""
    return {
        **shared_estim_defaults(),
        "max_iter": 30,   # smaller default than ICE — SEM does not converge
                          # in the deterministic sense.
        "sem_seed": 0,    # base RNG seed for the FFBS draws.
    }


def ice_missing_defaults() -> dict:
    """ICE defaults of the keys read only when Y has missing observations.

    * ``missing_strategy`` — ``"available"`` or ``"impute"`` (see
      :func:`pmcprg.pmc.ice.ice`, section "Missing observations");
    * ``missing_draws``    — completed series per iteration (``"impute"``);
    * ``missing_seed``     — RNG seed of those draws;
    * ``gap_nodes``        — quadrature nodes of the grid variants.

    Merged by ``_parse_ice_cfg`` on top of :func:`ice_estim_defaults`, but
    kept out of it: that dict is also the ``test_ice_tab_exposes_every_api_config_key``
    contract for keys that *must* get a widget from ``ice_estim_defaults`` /
    ``sem_estim_defaults`` alone. ``missing_strategy``, ``missing_draws`` and
    ``missing_seed`` do have their own widgets in ``_IceTab`` (``gap_nodes``
    does not — it runs with its default here), sourced from this function's
    defaults rather than duplicated as literals.
    """
    from pmcprg.pmc.gaps import DEFAULT_GAP_NODES
    return {
        "missing_strategy": DEFAULT_MISSING_STRATEGY,
        "missing_draws":    DEFAULT_MISSING_DRAWS,
        "missing_seed":     0,
        "gap_nodes":        DEFAULT_GAP_NODES,
    }


def copula_margin_defaults() -> dict:
    """ICE and SEM default of the ``copula_margins`` key (AUDIT_COPULES FR-7 a).

    ``"parametric"`` (:data:`DEFAULT_COPULA_MARGINS`) — the historical copula
    step; ``"empirical"`` is documented at :func:`pmcprg.pmc.ice._parse_ice_cfg`.
    Merged by ``_parse_ice_cfg`` and ``_parse_sem_cfg`` but kept out of
    :func:`ice_estim_defaults` / :func:`sem_estim_defaults` for the reason
    given in :func:`ice_missing_defaults`: those two dicts are the
    ``test_ice_tab_exposes_every_api_config_key`` contract, and this key has
    no widget in ``_IceTab`` yet. It is reachable from ``ice_cfg`` /
    ``sem_cfg`` and from the TOML ``[ice]`` / ``[sem]`` tables, which a GUI
    round trip preserves (``test_ice_section_round_trip_preserves_keys_without_widgets``).
    """
    return {"copula_margins": DEFAULT_COPULA_MARGINS}


def sem_missing_defaults() -> dict:
    """SEM defaults of the keys read only when Y has missing observations.

    SEM always completes the data by one joint posterior draw per iteration
    (its own ``sem_seed`` stream), so only ``gap_nodes`` applies. Kept out of
    :func:`sem_estim_defaults` for the reason given in
    :func:`ice_missing_defaults`.
    """
    from pmcprg.pmc.gaps import DEFAULT_GAP_NODES
    return {"gap_nodes": DEFAULT_GAP_NODES}


__all__ = [
    # constants
    "COPULA_MARGIN_MODES",
    "DEFAULT_COPULA_MARGINS",
    "DEGENERATE_MIN_WEIGHT",
    "DEGENERATE_SD_RATIO",
    "DEGENERATE_TAU_EDGE",
    "MAX_SWEEP_COMBINATIONS",
    "MULTISTART_FAMILY_MODES",
    "RANDOM_TAU_BAND",
    "SWEEP_TAU",
    "DEFAULT_CANDIDATES",
    "DEFAULT_MARGIN_SELECTION_RULE",
    "DEFAULT_MISSING_DRAWS",
    "DEFAULT_MISSING_STRATEGY",
    "DEFAULT_SELECTION_CRITERION",
    "INIT_STRATEGIES",
    "MARGIN_SELECTION_RULES",
    "MISSING_STRATEGIES",
    "SELECTION_CRITERIA",
    # shared result/trace dataclasses (subclassed by SEM)
    "IceResult",
    "IceTrace",
    # config plumbing
    "check_copula_margins",
    "copula_margin_defaults",
    "record_copula_margins",
    "ice_estim_defaults",
    "ice_missing_defaults",
    "sem_estim_defaults",
    "sem_missing_defaults",
    "shared_estim_defaults",
    # best iterate / degenerate states
    "BestIterate",
    "DegenerateFinding",
    "best_iter_of",
    "check_return_best_iterate",
    "degenerate_states",
    "evaluate_log_lik",
    "report_degenerate_states",
    "returned_log_lik",
    # missing observations
    "check_missing_cfg",
    "gap_e_step",
    # init / warm-start
    "check_init_strategy",
    "kmeans_label_assignment",
    "warmstart_from_kmeans",
    # parameter updates / snapshots
    "m_step",
    "snapshot_margins",
    "snapshot_prior_p",
    "snapshot_tau_family",
    # multistart / image
    "build_multistart_inits",
    "check_multistart_families",
    "perturb_initial_model",
    "run_multistart",
    "linearise_image",
]
