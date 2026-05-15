"""Shared M-step / warm-start infrastructure for unsupervised PMC estimators.

This module is the **stable internal API** between the ICE
(:mod:`prg.pmc.ice`) and SEM (:mod:`prg.pmc.sem`) estimators — and is the
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

The actual implementations still live in :mod:`prg.pmc.ice` (canonical
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

from prg.pmc.ice import (
    DEFAULT_MARGIN_SELECTION_RULE,
    DEFAULT_SELECTION_CRITERION,
    INIT_STRATEGIES,
    MARGIN_SELECTION_RULES,
    SELECTION_CRITERIA,
    _DEFAULT_CANDIDATES   as DEFAULT_CANDIDATES,
    _check_init_strategy  as check_init_strategy,
    _kmeans_label_assignment as kmeans_label_assignment,
    _m_step               as m_step,
    _perturb_initial_model as perturb_initial_model,
    _snapshot_margins     as snapshot_margins,
    _snapshot_prior_p     as snapshot_prior_p,
    _snapshot_tau_family  as snapshot_tau_family,
    _warmstart_from_kmeans as warmstart_from_kmeans,
)


def shared_estim_defaults() -> dict:
    """Default values for the cfg keys shared by ICE and SEM.

    Returned as a fresh dict on every call, so callers can mutate freely.
    The estimators add their own non-shared keys on top:

    * **ICE-only**: ``max_iter`` (50), ``tol`` (1e-4), ``patience`` (3).
    * **SEM-only**: ``max_iter`` (30 — SEM never early-stops, so the
      default is smaller), ``sem_seed`` (0).

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
    }


__all__ = [
    # constants
    "DEFAULT_CANDIDATES",
    "DEFAULT_MARGIN_SELECTION_RULE",
    "DEFAULT_SELECTION_CRITERION",
    "INIT_STRATEGIES",
    "MARGIN_SELECTION_RULES",
    "SELECTION_CRITERIA",
    # config plumbing
    "shared_estim_defaults",
    # init / warm-start
    "check_init_strategy",
    "kmeans_label_assignment",
    "warmstart_from_kmeans",
    # parameter updates / snapshots
    "m_step",
    "snapshot_margins",
    "snapshot_prior_p",
    "snapshot_tau_family",
    # multistart
    "perturb_initial_model",
]
