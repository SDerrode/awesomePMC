"""
sem.py — Stochastic EM (SEM) for unsupervised PMC/HMC fitting.

Public API
----------
sem(model, Y, sem_cfg=None) -> (PMCModel, SemTrace)
    Fit a :class:`PMCModel` to the observation sequence ``Y`` using SEM.
    Returns the fitted model and a :class:`SemTrace` capturing per-iteration
    diagnostics.

sem_image(model, img, sem_cfg=None) -> (PMCModel, SemTrace)
    Convenience wrapper that linearises a 2D image along the Generalized
    Hilbert ("gilbert") path and runs SEM on it.

Algorithm
---------
Given an initial model θ⁽⁰⁾ and observations Y = y_{1:N}, SEM alternates:

  E-step  (filter)
    Compute α̂_n(j) = P(X_n=j | Y_{1:n})  via :func:`forward`.

  S-step  (stochastic completion)
    Draw one realisation  X̃ ~ P(X | Y)  via Forward-Filter Backward-Sample
    (see :func:`sample_posterior`). Build hard one-hot posteriors

        γ̃_n(i) = 1[X̃_n = i]
        ξ̃_n(i, j) = 1[X̃_n = i, X̃_{n+1} = j]

  M-step  (supervised-style)
    Update prior, margins (optional), copula τ from the hard posteriors —
    delegated to :func:`prg.pmc._estim_common.m_step` so the implementation
    stays identical to ICE's M-step body.

Where SEM differs from ICE
--------------------------
* **Stochastic E-step.** Each iteration draws a *single* X̃ ~ P(X|Y)
  instead of computing the soft posteriors ξ, γ. Successive iterations
  are random; the log-likelihood trace fluctuates rather than
  monotonically increasing.
* **Convergence detection.** The log-lik does NOT converge in the
  deterministic sense. We therefore stop only at ``max_iter`` (no
  ``patience`` / ``tol`` early-stop). For inference, prefer averaging
  the post burn-in estimates externally (the full trace is exposed).
* **Multistart and K-means warm-start** are supported with the same
  semantics as ICE — see :func:`prg.pmc._estim_common.warmstart_from_kmeans`.

Reference
---------
The SEM idea is borrowed from the companion project ``markovchain_todelete``
(``prg/PMC_Estim.py`` + ``prg/tools/probaPMC.py``, function
``simulRealisationAP``), itself going back to Celeux & Diebolt's classical
Stochastic-EM literature for finite mixture and Markov models. The
copulasformm implementation is a from-scratch rewrite that reuses the
shared M-step / warm-start helpers in :mod:`prg.pmc._estim_common` to
avoid duplicating the update logic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from prg.pmc._estim_common import (
    DEFAULT_CANDIDATES,
    DEFAULT_MARGIN_SELECTION_RULE,
    DEFAULT_SELECTION_CRITERION,
    check_init_strategy,
    m_step,
    perturb_initial_model,
    shared_estim_defaults,
    snapshot_margins,
    snapshot_prior_p,
    snapshot_tau_family,
    warmstart_from_kmeans,
)
from prg.pmc.inference import forward, precompute_weights, sample_posterior
from prg.pmc.model     import PMCModel

logger = logging.getLogger(__name__)


__all__ = [
    "SemResult",
    "SemTrace",
    "sem",
    "sem_image",
]


# ---------------------------------------------------------------------------
# SemTrace — captured per-iteration diagnostics
# ---------------------------------------------------------------------------

@dataclass
class SemTrace:
    """Per-iteration SEM history, captured for diagnostic visualisations.

    Same structure as :class:`prg.pmc.ice.IceTrace` so the two can share
    GUI/plotting code. Snapshots correspond to the *current* model state at
    the time ``log_liks[t]`` was computed (i.e. before the M-step that
    produces the model used for ``log_liks[t+1]``).

    Fields
    ------
    log_liks            : list[float]
    tau_history         : (T, K, K) float, ``NaN`` where no copula.
    family_history      : list-of-lists-of-strings, same (T, K, K) shape.
    p_history           : (T, K, K) joint prior trace.
    margin_history      : list[list[dict]] — margin blocks per iteration.
    sampled_X_history   : (T, N) int — the stochastic completion X̃ used at
                          each M-step. SEM-specific (no counterpart in
                          :class:`IceTrace`).
    multistart_runs     : list[SemTrace] — non-best runs when ``n_starts > 1``.
    run_tag             : str — label of this run.
    candidates          : list[str] — copula SHORT_NAMEs SEM iterated over.
    """
    log_liks:         list[float]              = field(default_factory=list)
    tau_history:      np.ndarray               = field(
        default_factory=lambda: np.empty((0, 0, 0), dtype=float)
    )
    family_history:   list[list[list[str]]]    = field(default_factory=list)
    p_history:        np.ndarray               = field(
        default_factory=lambda: np.empty((0, 0, 0), dtype=float)
    )
    margin_history:   list[list[dict]]         = field(default_factory=list)
    sampled_X_history: np.ndarray              = field(
        default_factory=lambda: np.empty((0, 0), dtype=int)
    )
    multistart_runs:  list["SemTrace"]         = field(default_factory=list)
    run_tag:          str                      = ""
    candidates:       list[str]                = field(default_factory=list)

    @property
    def n_iters(self) -> int:
        return len(self.log_liks)

    def __len__(self) -> int:
        return self.n_iters


@dataclass
class SemResult:
    """Bundle returned by GUI/diagnostics layers around an SEM run.

    Mirrors :class:`prg.pmc.ice.IceResult`.
    """
    initial_model: PMCModel
    fitted_model:  PMCModel
    Y:             np.ndarray
    trace:         SemTrace


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------

def _parse_sem_cfg(model: PMCModel, sem_cfg: dict | None) -> dict:
    """Merge TOML ``[sem]`` (or ``[ice]`` fallback) section with defaults.

    SEM reuses most ICE keys (``fit_margins``, ``candidates``,
    ``selection_criterion``, ``margin_selection_rule``, ``init``,
    ``kmeans_seed``, ``n_starts``, ``multistart_seed``, ``multistart_jitter``)
    and adds:

    * ``max_iter``  (int, default 30) — fixed number of SEM iterations
                                        (no early stop — SEM does not converge
                                        in the deterministic sense).
    * ``sem_seed``  (int, default 0)  — base RNG seed for the per-iteration
                                        :func:`sample_posterior` draws.

    Resolution order:
      1. defaults
      2. TOML ``[sem]`` section (``model.sem_config()`` if available)
      3. TOML ``[ice]`` section (fallback — keeps the common keys shared)
      4. caller-supplied ``sem_cfg`` (highest priority)
    """
    # Shared defaults come from a single source of truth so that ICE and
    # SEM cannot silently drift apart on common keys (see audit: "cfg
    # leaks"). Only SEM-specific additions are declared inline here.
    defaults: dict = {
        **shared_estim_defaults(),
        # SEM-specific keys.
        "max_iter":  30,                  # smaller default than ICE — SEM
                                          # does not converge in the
                                          # deterministic sense.
        "sem_seed":  0,                   # base RNG seed for the FFBS draws.
    }
    # TOML [ice] section is the fallback for keys not duplicated under [sem].
    cfg = {**defaults, **model.ice_config()}
    # Optional [sem] section (only if the model exposes one).
    if hasattr(model, "sem_config"):
        cfg.update(model.sem_config())
    if sem_cfg:
        cfg.update(sem_cfg)
    return cfg


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sem(
    model: PMCModel,
    Y: np.ndarray,
    sem_cfg: dict | None = None,
    progress_cb=None,
) -> tuple[PMCModel, SemTrace]:
    """Stochastic EM (SEM) for unsupervised PMC/HMC fitting.

    With ``n_starts > 1`` (config), runs SEM several times from random
    perturbations of ``model`` and keeps the run with the highest final
    log-likelihood — the same multistart strategy as
    :func:`prg.pmc.ice.ice`.

    Parameters
    ----------
    model        : PMCModel — initial parameter values.
    Y            : np.ndarray, shape (N,) — observation sequence.
    sem_cfg      : dict, optional — override SEM settings. See
                    :func:`_parse_sem_cfg` for recognised keys.
    progress_cb  : callable, optional — invoked as
                    ``progress_cb(it, max_iter, log_lik, run_tag)`` after
                    every iteration.

    Returns
    -------
    fitted_model : PMCModel — the model at the last iteration (or the best
                              run when multistart is enabled).
    trace        : SemTrace — per-iteration history.
    """
    cfg       = _parse_sem_cfg(model, sem_cfg)
    n_starts  = max(1, int(cfg.get("n_starts", 1)))

    # Optional K-means warm-start (shared with ICE).
    init_strategy = check_init_strategy(str(cfg.get("init", "model")))
    if init_strategy == "kmeans":
        logger.info(
            "SEM init='kmeans': clustering Y (N=%d, K=%d, seed=%d) "
            "before estimation.",
            len(Y), model.K, int(cfg.get("kmeans_seed", 0)),
        )
        model = warmstart_from_kmeans(
            model, Y,
            random_state=int(cfg.get("kmeans_seed", 0)),
            fit_margins=bool(cfg.get("fit_margins", False)),
            candidates=list(cfg.get("candidates", DEFAULT_CANDIDATES)),
            selection_criterion=str(cfg.get(
                "selection_criterion", DEFAULT_SELECTION_CRITERION,
            )),
            margin_selection_rule=str(cfg.get(
                "margin_selection_rule", DEFAULT_MARGIN_SELECTION_RULE,
            )),
        )

    if n_starts == 1:
        return _sem_single_run(model, Y, cfg, progress_cb=progress_cb)

    rng_ms = np.random.default_rng(int(cfg.get("multistart_seed", 0)))
    jitter = float(cfg.get("multistart_jitter", 0.10))

    logger.info("SEM multistart: n_starts=%d  jitter=%.2f", n_starts, jitter)

    runs: list[tuple[float, PMCModel, SemTrace]] = []
    for s in range(n_starts):
        if s == 0:
            init_mdl = model
            tag = "unperturbed"
        else:
            init_mdl = perturb_initial_model(model, rng_ms, jitter=jitter)
            tag = f"perturbed-{s}"
        try:
            fitted_s, trace_s = _sem_single_run(
                init_mdl, Y, cfg, run_tag=tag, progress_cb=progress_cb,
            )
            final_ll = trace_s.log_liks[-1] if trace_s.log_liks else -np.inf
        except Exception as exc:                            # pragma: no cover
            logger.warning("SEM multistart: run %d (%s) failed: %s", s, tag, exc)
            continue
        logger.info(
            "SEM multistart: run %d (%s) → final log-lik = %.4f  (iters=%d)",
            s, tag, final_ll, len(trace_s.log_liks),
        )
        runs.append((final_ll, fitted_s, trace_s))

    if not runs:                                            # pragma: no cover
        return _sem_single_run(model, Y, cfg, progress_cb=progress_cb)

    best_idx, (best_ll, best_mdl, best_trace) = max(
        enumerate(runs), key=lambda kv: kv[1][0],
    )
    logger.info(
        "SEM multistart: best run = #%d  (final log-lik = %.4f)",
        best_idx, best_ll,
    )
    best_trace.multistart_runs = [
        trace for k, (_, _, trace) in enumerate(runs) if k != best_idx
    ]
    return best_mdl, best_trace


def sem_image(
    model: PMCModel,
    img: np.ndarray,
    sem_cfg: dict | None = None,
    progress_cb=None,
) -> tuple[PMCModel, SemTrace]:
    """SEM on a 2D image — linearises along the gilbert path then calls :func:`sem`.

    Same API as :func:`prg.pmc.ice.ice_image`.
    """
    from prg.pmc.peano import image_to_signal

    if img.ndim not in (2, 3):
        raise ValueError(
            f"sem_image expects 2D (H, W) or 3D (H, W, d); got shape {img.shape}."
        )
    img_d = 1 if img.ndim == 2 else img.shape[2]
    if img_d != model.d:
        raise ValueError(
            f"Image channels ({img_d}) do not match model.d = {model.d}."
        )
    Y = image_to_signal(img)
    return sem(model, Y, sem_cfg=sem_cfg, progress_cb=progress_cb)


# ---------------------------------------------------------------------------
# Single SEM run
# ---------------------------------------------------------------------------

def _sem_single_run(
    model: PMCModel,
    Y: np.ndarray,
    cfg: dict,
    run_tag: str = "",
    progress_cb=None,
) -> tuple[PMCModel, SemTrace]:
    """Single SEM run — extracted from :func:`sem` to support multistart.

    ``cfg`` must already be the merged dict returned by :func:`_parse_sem_cfg`.
    """
    max_iter    = int(cfg["max_iter"])
    candidates  = list(cfg["candidates"])
    fit_margins = bool(cfg["fit_margins"])
    selection_criterion   = str(cfg.get(
        "selection_criterion",   DEFAULT_SELECTION_CRITERION,
    ))
    margin_selection_rule = str(cfg.get(
        "margin_selection_rule", DEFAULT_MARGIN_SELECTION_RULE,
    ))
    sem_seed    = int(cfg.get("sem_seed", 0))
    rng_ffbs    = np.random.default_rng(sem_seed)
    log_prefix  = f"SEM[{run_tag}]" if run_tag else "SEM"

    K = model.K
    N = len(Y)

    log_liks:    list[float]            = []
    tau_buf:     list[np.ndarray]       = []
    fam_buf:     list[list[list[str]]]  = []
    p_buf:       list[np.ndarray]       = []
    margin_buf:  list[list[dict]]       = []
    X_buf:       list[np.ndarray]       = []

    raw     = model.raw
    current = PMCModel.from_dict(raw)

    logger.info(
        "%s: variant=%s  K=%d  N=%d  max_iter=%d  candidates=%s  seed=%d",
        log_prefix, current.variant.value, K, N, max_iter, candidates, sem_seed,
    )

    for it in range(max_iter):
        # ── E-step (filter) ───────────────────────────────────────────────
        W, f_pdf            = precompute_weights(current, Y)
        alpha_hat, log_lik  = forward(current, Y, W=W, f_pdf=f_pdf)

        log_liks.append(log_lik)
        # Snapshot of current model state, matching IceTrace semantics.
        tau_kk, fam_kk = snapshot_tau_family(current)
        tau_buf.append(tau_kk)
        fam_buf.append(fam_kk)
        p_buf.append(snapshot_prior_p(current))
        margin_buf.append(snapshot_margins(current))

        logger.info("%s iter %d: log-lik = %.4f", log_prefix, it, log_lik)
        if progress_cb is not None:
            try:
                progress_cb(it, max_iter, float(log_lik), run_tag)
            except Exception as exc:                            # pragma: no cover
                logger.debug("SEM progress_cb raised %s; ignoring.", exc)

        # ── S-step (stochastic completion) ────────────────────────────────
        X_tilde = sample_posterior(
            current, Y, rng_ffbs,
            W=W, f_pdf=f_pdf, alpha_hat=alpha_hat,
        )
        X_buf.append(X_tilde.copy())

        # Build hard one-hot posteriors.
        gamma = np.zeros((N, K), dtype=float)
        gamma[np.arange(N), X_tilde] = 1.0
        xi    = np.zeros((N - 1, K, K), dtype=float)
        xi[np.arange(N - 1), X_tilde[:-1], X_tilde[1:]] = 1.0

        # ── M-step ───────────────────────────────────────────────────────
        m_step(
            raw, current, Y, xi, gamma,
            fit_margins=fit_margins,
            candidates=candidates,
            selection_criterion=selection_criterion,
            margin_selection_rule=margin_selection_rule,
        )
        current = PMCModel.from_dict(raw)

    trace = SemTrace(
        log_liks          = log_liks,
        tau_history       = (np.stack(tau_buf, axis=0) if tau_buf
                              else np.empty((0, K, K), dtype=float)),
        family_history    = fam_buf,
        p_history         = (np.stack(p_buf, axis=0) if p_buf
                              else np.empty((0, K, K), dtype=float)),
        margin_history    = margin_buf,
        sampled_X_history = (np.stack(X_buf, axis=0) if X_buf
                              else np.empty((0, N), dtype=int)),
        run_tag           = run_tag,
        candidates        = list(candidates),
    )
    return current, trace


# ---------------------------------------------------------------------------
# Quick smoke-test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

    from prg.pmc.logging_setup import configure as _cfg_log
    _cfg_log(level=logging.INFO)

    from prg.pmc.inference import classify, error_rate
    from prg.pmc.simulate  import simulate

    models_dir = pathlib.Path(__file__).parent / "models"
    for name in ("hmc_in_gauss_k2.toml", "pmc_gauss_k2.toml"):
        toml_file = models_dir / name
        if not toml_file.exists():
            continue
        print(f"\n{'='*60}\n  {name}\n{'='*60}")
        mdl = PMCModel(toml_file)
        X_ref, Y = simulate(mdl, N=1000, seed=42)

        # Perturb the initial model so SEM has work to do.
        raw_init = mdl.raw
        for blk in raw_init.get("margins", []):
            p = blk.get("params", {})
            if "loc" in p:
                p["loc"] = p["loc"] + 0.3
        init_mdl = PMCModel.from_dict(raw_init)

        fitted, trace = sem(
            init_mdl, Y,
            sem_cfg={"max_iter": 10, "sem_seed": 7, "fit_margins": True},
        )
        print(f"  log-lik history: {[f'{ll:.2f}' for ll in trace.log_liks]}")
        X_hat, _, _ = classify(fitted, Y)
        er = error_rate(X_ref, X_hat)
        print(f"  error rate (fitted): {er:.4f}  ({er*100:.1f} %)")
