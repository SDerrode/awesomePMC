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
    delegated to :func:`pmcprg.pmc._estim_common.m_step` so the implementation
    stays identical to ICE's M-step body. State margins f_i are fitted on
    {y_n : X̃_n = i}; pair margins f_ij (general PMC, DerrodePieczynski_CSDA2013
    Eqs. 12–14) on the dual-view sub-sample {y_n : (X̃_n, X̃_{n+1}) = (i, j)} ∪
    {y_{n+1} : (X̃_n, X̃_{n+1}) = (j, i)}, and the copula c_ij on the
    pseudo-observations (F_ij(y_n), F_ji(y_{n+1})) of the pairs drawn in
    (i, j) — the model's F, or with ``copula_margins = "empirical"`` the
    rescaled empirical CDFs of those drawn sub-samples. A pair that does not
    occur in the draw keeps its margin (and copula) unchanged.

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
  semantics as ICE — see :func:`pmcprg.pmc._estim_common.warmstart_from_kmeans`.

Missing observations
--------------------
A row of ``Y`` with a non-finite value is missing (ignorable missingness,
:mod:`pmcprg.pmc.gaps`); a ``Y`` without missing rows runs the algorithm
above unchanged. Otherwise SEM is exact data augmentation (Tanner & Wong
1987): the S-step draws the states **and the missing values** jointly,
(X̃, ỹ_mis) ~ P(x, y_mis | y_obs, θ^q), by forward-filter backward-sample on
the missing-data chain of :mod:`pmcprg.pmc.gaps` (the exact K-state shortcut
for HMC-IN, HMC-IN2 and PMC-IN with state margins — then ỹ_n ~ f_{X̃_n}; the
quadrature grid of ``gap_nodes`` nodes otherwise — then ỹ_n is the drawn
node). The M-step is the complete-data one on the completed series
(ỹ_mis filled in), GICE margin selection included; ``log_liks`` is the
observed-data log-likelihood log p(y_obs) of the forward pass. The k-means
warm start clusters the observed rows only.

Non-ignorable missingness (``model.missingness`` not None,
:mod:`pmcprg.pmc.missingness`): the mechanism is carried unchanged to every
iterate and to the returned model (held fixed, not estimated in this
version); the S-step draws from P(x, y_mis | y_obs, m, θ^q) and ``log_liks``
is log p(y_obs, m) — the missingness factors are in the forward messages the
draw comes from, for a complete Y (mask m = 0) too.

Drawing ỹ_n on the quadrature nodes rather than from the continuous law is
an approximation of the grid variants; measured against a continuous
within-cell jitter and a 4× finer grid, it leaves the estimates unbiased
where the jitter is not: see :data:`pmcprg.pmc.ice._GAP_DRAW`.

Relation to the papers' ICE
---------------------------
SEM is the closest estimator in the package to the ICE of the papers with
L = 1 (DerrodePieczynski_CSDA2013 §4.2, DerrodePieczynski_SP2016 §3): both
estimate the copula and margin parameters on **one** posterior draw x^(q) ~
p(x | y, θ^q). It differs in taking the prior p_ij from the draw too, where
the papers use its conditional expectation.
The :mod:`pmcprg.pmc.ice` module docstring ("Relation to the papers' ICE")
sets the two side by side.

Reference
---------
Celeux, G. & Diebolt, J. (1985). The SEM algorithm: a probabilistic teacher
algorithm derived from the EM algorithm for the mixture problem.
*Computational Statistics Quarterly* 2, 73–82.

Tanner, M. A. & Wong, W. H. (1987). The calculation of posterior
distributions by data augmentation. *Journal of the American Statistical
Association* 82(398), 528–540.

The awesomePMC
implementation reuses the shared M-step / warm-start helpers in
:mod:`pmcprg.pmc._estim_common` to avoid duplicating the update logic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from pmcprg.pmc._estim_common import (
    BestIterate,
    IceResult,
    IceTrace,
    best_iter_of,
    check_copula_margins,
    check_init_strategy,
    check_missing_cfg,
    check_multistart_families,
    check_return_best_iterate,
    copula_margin_defaults,
    evaluate_log_lik,
    gap_e_step,
    linearise_image,
    m_step,
    record_copula_margins,
    report_degenerate_states,
    run_multistart,
    sem_estim_defaults,
    sem_missing_defaults,
    snapshot_margins,
    snapshot_prior_p,
    snapshot_tau_family,
    warmstart_from_kmeans,
)
from pmcprg.pmc.inference import forward, precompute_weights, sample_posterior
from pmcprg.pmc.model     import PMCModel

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
class SemTrace(IceTrace):
    """Per-iteration SEM history — :class:`pmcprg.pmc.ice.IceTrace` plus the
    SEM-specific stochastic completions.

    Inherits every :class:`IceTrace` field (``log_liks``, ``tau_history``,
    ``family_history``, ``p_history``, ``margin_history``, ``multistart_runs``,
    ``run_tag``, ``candidates``, ``best_iter``, ``returned_iter``,
    ``degenerate``) and the ``n_iters`` / ``__len__`` helpers, so
    the two share all GUI/plotting code. Adds:

    * ``sampled_X_history`` : (T, N) int — the FFBS draw X̃ used at each M-step
      (no counterpart in :class:`IceTrace`).
    """
    sampled_X_history: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=int)
    )


@dataclass
class SemResult(IceResult):
    """Bundle returned by GUI/diagnostics layers around an SEM run.

    Identical structure to :class:`pmcprg.pmc.ice.IceResult` (``initial_model``,
    ``fitted_model``, ``Y``, ``trace``); kept as a distinct class so callers
    can tell an SEM result from an ICE one by type.
    """


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------

def _parse_sem_cfg(model: PMCModel, sem_cfg: dict | None) -> dict:
    """Merge TOML ``[sem]`` (or ``[ice]`` fallback) section with defaults.

    SEM reuses most ICE keys (``fit_margins``, ``candidates``,
    ``selection_criterion``, ``margin_selection_rule``, ``init``,
    ``kmeans_seed``, ``n_starts``, ``multistart_seed``, ``multistart_jitter``,
    ``multistart_workers``, ``multistart_families``, ``return_best_iterate``)
    and adds:

    * ``max_iter``  (int, default 30) — fixed number of SEM iterations
                                        (no early stop — SEM does not converge
                                        in the deterministic sense).
    * ``sem_seed``  (int, default 0)  — base RNG seed for the per-iteration
                                        :func:`sample_posterior` draws.
    * ``gap_nodes`` (int, default 64) — quadrature nodes of the grid variants,
                                        read only when Y has missing rows.

    ``copula_margins`` (default ``"parametric"``) is ICE's key
    (:func:`pmcprg.pmc.ice._parse_ice_cfg`), applied to SEM's shared M-step:
    with ``"empirical"`` the copula pseudo-observations come from the
    empirical margins of the drawn sub-samples, the one-hot weights of the
    draw X̃ in place of ICE's posteriors — F̂_i(y) = #{n : X̃_n = i, y_n ≤ y} /
    (n_i + 1) for a state margin, the ½-weighted dual-view sub-sample for a
    pair margin (:func:`pmcprg.pmc.ice._empirical_margin_cdfs`). With missing
    rows those sub-samples contain the drawn ỹ_mis (on the quadrature nodes
    for the grid variants, hence ties, handled by ``≤``). A non-default value
    is recorded in the fitted model's ``[sem]`` table.

    Resolution order (lowest → highest priority):
      1. defaults (``sem_estim_defaults()`` — shared + SEM-specific keys)
      2. TOML ``[ice]`` section (``model.ice_config()`` — shared keys)
      3. TOML ``[sem]`` section (``model.sem_config()`` — overrides ``[ice]``)
      4. caller-supplied ``sem_cfg`` (highest priority)
    """
    # Defaults come from a single source of truth so that ICE, SEM and the
    # GUI cannot silently drift apart (audit Q-9).
    defaults: dict = {**sem_estim_defaults(), **sem_missing_defaults(),
                      **copula_margin_defaults()}
    # TOML [ice] section is the fallback for keys not duplicated under [sem];
    # a dedicated [sem] section then overrides the shared keys.
    cfg = {**defaults, **model.ice_config(), **model.sem_config()}
    if sem_cfg:
        cfg.update(sem_cfg)
    check_multistart_families(cfg["multistart_families"])
    check_return_best_iterate(cfg["return_best_iterate"])
    check_missing_cfg(cfg)
    check_copula_margins(cfg["copula_margins"])
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
    :func:`pmcprg.pmc.ice.ice`, ``multistart_families`` included.

    Parameters
    ----------
    model        : PMCModel — initial parameter values.
    Y            : np.ndarray, shape (N,) or (N, d) — observation sequence;
                   non-finite rows are missing observations (module
                   docstring, "Missing observations").
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

    Best iterate
    ------------
    With ``return_best_iterate = True`` the model returned is the iterate θ^q
    with the highest observed-data log-likelihood ``log_liks[q]`` (computed
    with θ^q before the S- and M-steps that give θ^(q+1); after the last
    M-step, θ^max_iter is evaluated once more, so the trace has
    ``max_iter + 1`` log-likelihoods and ``max_iter`` draws). This is a
    heuristic, not an SEM estimator: SEM's iterates are a Markov chain whose
    stationary spread is the Monte-Carlo noise of one draw, and the
    likelihood-best iterate of that chain is not its mean — averaging the
    post-burn-in iterates is the usual estimate. It picks the point of the
    chain that fits best, and inherits its noise. ``trace.best_iter`` and
    ``trace.returned_iter`` locate it; multistart ranks each start by the
    log-likelihood of the model it returns.

    Degenerate states
    -----------------
    Checked on the returned model as for :func:`pmcprg.pmc.ice.ice`
    (``trace.degenerate``, one WARNING line).
    """
    cfg = _parse_sem_cfg(model, sem_cfg)
    from pmcprg.pmc.gaps import missing_mask
    miss = missing_mask(Y)
    if miss.any():
        logger.info(
            "SEM: %d of %d observations missing (%.1f %%) — data augmentation "
            "(states and missing values drawn jointly), observed-data likelihood.",
            int(miss.sum()), miss.size, 100.0 * miss.mean(),
        )

    # Optional K-means warm-start (shared with ICE).
    init_strategy = check_init_strategy(str(cfg["init"]))
    if init_strategy == "kmeans":
        logger.info(
            "SEM init='kmeans': clustering Y (N=%d, K=%d, seed=%d) "
            "before estimation.",
            len(Y), model.K, int(cfg["kmeans_seed"]),
        )
        model = warmstart_from_kmeans(
            model, Y,
            random_state=int(cfg["kmeans_seed"]),
            fit_margins=bool(cfg["fit_margins"]),
            candidates=list(cfg["candidates"]),
            selection_criterion=str(cfg["selection_criterion"]),
            margin_selection_rule=str(cfg["margin_selection_rule"]),
            copula_margins=str(cfg["copula_margins"]),
        )

    # Multistart driver is shared with ICE (see _estim_common.run_multistart).
    # SEM threads the start index into ``seed_offset`` so each run draws an
    # independent FFBS stream (audit N-5); the unperturbed run uses offset 0.
    def _single(init_mdl, run_tag, start_index):
        return _sem_single_run(
            init_mdl, Y, cfg, run_tag=run_tag, progress_cb=progress_cb,
            seed_offset=start_index,
        )

    fitted, trace = run_multistart(
        _single, model, cfg, label="SEM", worker=(_multistart_worker, Y),
    )
    # Diagnostic only: reads the returned model, changes nothing in it.
    report_degenerate_states(fitted, trace, Y, label="SEM")
    return fitted, trace


def _multistart_worker(raw: dict, Y, cfg: dict, run_tag: str, start_index: int):
    """Top-level (picklable) multistart worker for ``multistart_workers > 1``.

    Rebuilds the model from its raw dict inside the subprocess and runs one
    SEM fit; ``start_index`` keeps the per-start FFBS seed offset (audit N-5)
    identical to the sequential path. ``progress_cb`` cannot cross process
    boundaries.
    """
    return _sem_single_run(
        PMCModel.from_dict(raw), Y, cfg, run_tag=run_tag, seed_offset=start_index,
    )


def sem_image(
    model: PMCModel,
    img: np.ndarray,
    sem_cfg: dict | None = None,
    progress_cb=None,
) -> tuple[PMCModel, SemTrace]:
    """SEM on a 2D image — linearises along the gilbert path then calls :func:`sem`.

    Same API as :func:`pmcprg.pmc.ice.ice_image`.
    """
    Y = linearise_image(model, img, what="sem_image")
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
    *,
    seed_offset: int = 0,
) -> tuple[PMCModel, SemTrace]:
    """Single SEM run — extracted from :func:`sem` to support multistart.

    ``cfg`` must already be the merged dict returned by :func:`_parse_sem_cfg`.
    ``seed_offset`` is added to ``sem_seed`` so each multistart run draws an
    *independent* FFBS stream (audit N-5). The primary run uses ``0``, so a
    single-start fit stays bit-for-bit reproducible at a given ``sem_seed``.

    ``log_liks[q]`` is computed with θ^q before the S- and M-steps that give
    θ^(q+1). With ``return_best_iterate`` the model of the last M-step is
    evaluated once more (``log_liks[max_iter]``, no draw — hence one more
    log-likelihood than ``sampled_X_history`` rows) and the first θ^q
    maximising ``log_liks`` is returned, kept by reference (see
    :func:`pmcprg.pmc.ice._ice_single_run`).
    """
    return_best = bool(cfg.get("return_best_iterate", False))
    best        = BestIterate()
    max_iter    = int(cfg["max_iter"])
    candidates  = list(cfg["candidates"])
    fit_margins = bool(cfg["fit_margins"])
    selection_criterion   = str(cfg["selection_criterion"])
    margin_selection_rule = str(cfg["margin_selection_rule"])
    copula_margins = check_copula_margins(cfg.get("copula_margins", "parametric"))
    sem_seed    = int(cfg["sem_seed"])
    rng_ffbs    = np.random.default_rng(sem_seed + int(seed_offset))
    log_prefix  = f"SEM[{run_tag}]" if run_tag else "SEM"

    K = model.K
    N = len(Y)
    from pmcprg.pmc.gaps import missing_mask
    miss = missing_mask(Y)
    miss = miss if miss.any() else None
    gap_nodes = cfg.get("gap_nodes")

    log_liks:    list[float]            = []
    tau_buf:     list[np.ndarray]       = []
    fam_buf:     list[list[list[str]]]  = []
    p_buf:       list[np.ndarray]       = []
    margin_buf:  list[list[dict]]       = []
    X_buf:       list[np.ndarray]       = []

    raw     = model.raw
    # A non-default copula_margins is recorded in the [sem] table (no-op on
    # the default path).
    record_copula_margins(raw, copula_margins, section="sem")
    current = PMCModel.from_dict(raw)

    logger.info(
        "%s: variant=%s  K=%d  N=%d  max_iter=%d  candidates=%s  seed=%d",
        log_prefix, current.variant.value, K, N, max_iter, candidates, sem_seed,
    )

    for it in range(max_iter + 1 if return_best else max_iter):
        # ── E-step (filter) ───────────────────────────────────────────────
        Y_fit = Y
        if it == max_iter:
            # return_best_iterate: evaluate θ^max_iter (no draw, no M-step).
            log_lik = evaluate_log_lik(current, Y, miss, gap_nodes)
        elif miss is None:
            W, f_pdf            = precompute_weights(current, Y)
            alpha_hat, log_lik  = forward(current, Y, W=W, f_pdf=f_pdf)
        else:
            # Missing rows: forward pass of the missing-data chain and one
            # joint draw of (states, missing values) from its messages.
            gap = gap_e_step(current, Y, miss, gap_nodes=gap_nodes, posterior=False,
                             n_draws=1, rng=rng_ffbs)
            log_lik, Y_fit = gap.log_lik, gap.Y_draws[0]

        log_liks.append(log_lik)
        # Snapshot of current model state, matching IceTrace semantics.
        tau_kk, fam_kk = snapshot_tau_family(current)
        tau_buf.append(tau_kk)
        fam_buf.append(fam_kk)
        p_buf.append(snapshot_prior_p(current))
        margin_buf.append(snapshot_margins(current))
        if return_best:
            best.offer(it, log_lik, current)
        if it == max_iter:
            logger.info("%s model of the last M-step (iterate %d): log-lik = %.4f",
                        log_prefix, it, log_lik)
            break

        logger.info("%s iter %d: log-lik = %.4f", log_prefix, it, log_lik)
        if progress_cb is not None:
            try:
                progress_cb(it, max_iter, float(log_lik), run_tag)
            except Exception as exc:                            # pragma: no cover
                logger.debug("SEM progress_cb raised %s; ignoring.", exc)

        # ── S-step (stochastic completion) ────────────────────────────────
        if miss is None:
            X_tilde = sample_posterior(
                current, Y, rng_ffbs,
                W=W, f_pdf=f_pdf, alpha_hat=alpha_hat,
            )
        else:
            X_tilde = gap.X_draws[0]
        X_buf.append(X_tilde.copy())

        # Build hard one-hot posteriors.
        gamma = np.zeros((N, K), dtype=float)
        gamma[np.arange(N), X_tilde] = 1.0
        xi    = np.zeros((N - 1, K, K), dtype=float)
        xi[np.arange(N - 1), X_tilde[:-1], X_tilde[1:]] = 1.0

        # ── M-step ───────────────────────────────────────────────────────
        # (on the completed series ỹ when Y has missing rows)
        m_step(
            raw, current, Y_fit, xi, gamma,
            fit_margins=fit_margins,
            candidates=candidates,
            selection_criterion=selection_criterion,
            margin_selection_rule=margin_selection_rule,
            copula_margins=copula_margins,
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
        best_iter         = best_iter_of(log_liks),
    )
    if return_best and best.model is not None:
        trace.returned_iter = best.iter
        if best.iter < len(log_liks) - 1:
            logger.info(
                "%s: returning the best iterate %d (log-lik %.4f) instead of the "
                "last one %d (log-lik %.4f).",
                log_prefix, best.iter, log_liks[best.iter],
                len(log_liks) - 1, log_liks[-1],
            )
        return best.model, trace
    # SEM always ends on an M-step: θ^max_iter, not evaluated.
    trace.returned_iter = len(log_liks)
    return current, trace


# ---------------------------------------------------------------------------
# Quick smoke-test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

    from pmcprg.pmc.logging_setup import configure as _cfg_log
    _cfg_log(level=logging.INFO)

    from pmcprg.pmc.inference import classify, error_rate
    from pmcprg.pmc.simulate  import simulate

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
