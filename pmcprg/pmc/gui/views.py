"""gui/views.py — matplotlib renderers for the PMCMainWindow right-hand canvas.

Extracted from ``main_window.py`` (audit Q-10) so each view is a plain
function of a :class:`matplotlib.figure.Figure` plus its data — unit-testable
headless (no QMainWindow) and keeping ``main_window`` to layout + state +
dispatch. ``PMCMainWindow`` clears the figure (``_begin_figure``), calls the
matching ``plot_*`` here, then draws the canvas.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

from pmcprg.copulas._fit import _empirical_copula
from pmcprg.diagnostics.pseudo import margin_cdfs
from pmcprg.numerics     import EPS
from pmcprg.pmc          import IceTrace
from pmcprg.pmc.gui._margins import is_pair, state_law
from pmcprg.pmc.inference import classify
from pmcprg.pmc.model    import PMCModel

def _channel(Y, j: int = 0):
    """Column ``j`` of a multivariate observation sequence, or ``Y`` if scalar.

    Observations are ``(N,)`` for scalar models and ``(N, d)`` for
    multivariate ones. The scalar-only panels below plotted ``Y`` directly,
    which raised ``x and y must be the same size`` for every d ≥ 2 model —
    the two main views were unusable on the multivariate fixtures the package
    ships.
    """
    Y = np.asarray(Y)
    return Y if Y.ndim == 1 else Y[:, j]


def _dim(Y) -> int:
    Y = np.asarray(Y)
    return 1 if Y.ndim == 1 else Y.shape[1]


def _draw_gaps(ax, Y) -> int:
    """Shade the missing rows of ``Y`` on a time axis; return how many there are.

    A row is missing when any component is NaN. The scatter panels already
    leave a hole there (matplotlib drops NaN points), but an isolated hole in
    a dense sequence is invisible; a grey band per run of missing rows makes
    the gaps readable. One collection for all runs, drawn behind the data,
    spanning the full height (x in data, y in axes coordinates) and kept out
    of autoscaling. Nothing is drawn when no row is missing.
    """
    from matplotlib.collections import PolyCollection

    miss = np.isnan(np.asarray(Y, dtype=float))
    if miss.ndim == 2:
        miss = miss.any(axis=1)
    n_miss = int(miss.sum())
    if not n_miss:
        return 0
    edges = np.diff(np.concatenate(([0], miss.astype(np.int8), [0])))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    verts = [[(a - 0.5, 0.0), (b - 0.5, 0.0), (b - 0.5, 1.0), (a - 0.5, 1.0)]
             for a, b in zip(starts, stops)]
    ax.add_collection(
        PolyCollection(verts, transform=ax.get_xaxis_transform(),
                       facecolor="0.85", edgecolor="none", zorder=0,
                       label=f"missing ({n_miss})"),
        autolim=False,
    )
    return n_miss


def plot_simulation(fig: Figure, X, Y, mdl):
    K = mdl.K
    colors = _class_colors(K)
    d      = _dim(Y)
    y0     = _channel(Y, 0)
    sfx    = r"^{(0)}" if d > 1 else ""

    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 1, 2)

    # Scatter: index vs Y, coloured by X
    nn = np.arange(len(y0))
    for k in range(K):
        mask = X == k
        ax1.scatter(nn[mask], y0[mask], s=4, alpha=0.5, color=colors[k],
                    label=rf"$X_n={k}$")
    ax1.set_xlabel(r"$n$")
    ax1.set_ylabel(rf"$Y_n{sfx}$")
    ax1.set_title("Simulated sequence" + (f"  (channel 0 of {d})" if d > 1 else ""))
    ax1.legend(markerscale=3)

    # Histogram per class
    for k in range(K):
        vals = y0[X == k]
        if len(vals):
            ax2.hist(vals, bins=40, alpha=0.5, color=colors[k],
                     label=rf"$X_n={k}$", density=True)
    ax2.set_xlabel(rf"$Y_n{sfx}$")
    ax2.set_title("Marginal histograms")
    ax2.legend()

    if d > 1:
        # Multivariate: the informative pair plot is *across channels* at the
        # same n — that is where the emission covariance shows up — rather
        # than across time on an arbitrary channel.
        y1 = _channel(Y, 1)
        for k in range(K):
            mask = X == k
            ax3.scatter(y0[mask], y1[mask], s=4, alpha=0.4,
                        color=colors[k], label=rf"$X_n={k}$")
        ax3.set_xlabel(r"$Y_n^{(0)}$")
        ax3.set_ylabel(r"$Y_n^{(1)}$")
        ax3.set_title(r"Channel scatter $(Y_n^{(0)},\, Y_n^{(1)})$")
    else:
        # Successive pairs (Y_n, Y_{n+1}) — the copula's own scatter.
        for k in range(K):
            mask = (X[:-1] == k)
            ax3.scatter(y0[:-1][mask], y0[1:][mask], s=4, alpha=0.4,
                        color=colors[k], label=rf"$X_n={k}$")
        ax3.set_xlabel(r"$Y_n$")
        ax3.set_ylabel(r"$Y_{n+1}$")
        ax3.set_title(r"Successive-pair scatter $(Y_n,\, Y_{n+1})$")
    ax3.legend(markerscale=3)


def plot_classification(fig: Figure, Y, X_hat, gamma, X_ref, mdl):
    K = mdl.K
    colors = _class_colors(K)

    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 1, 2)
    d   = _dim(Y)
    y0  = _channel(Y, 0)
    sfx = r"^{(0)}" if d > 1 else ""
    nn  = np.arange(len(y0))

    # Sequence coloured by X_hat
    for k in range(K):
        mask = X_hat == k
        ax1.scatter(nn[mask], y0[mask], s=4, alpha=0.5, color=colors[k],
                    label=rf"$\hat{{X}}={k}$")
    n_miss = _draw_gaps(ax1, Y)
    ax1.set_xlabel(r"$n$")
    ax1.set_ylabel(rf"$Y_n{sfx}$")
    ax1.set_title("MPM classification" + (f"  (channel 0 of {d})" if d > 1 else "")
                  + (f"  — {n_miss} missing ({n_miss / len(y0):.1%})"
                     if n_miss else ""))
    ax1.legend(markerscale=3)

    # Posterior marginals γ
    for k in range(K):
        ax2.plot(nn, gamma[:, k], lw=0.7, alpha=0.8, color=colors[k],
                 label=rf"$\gamma(X{{=}}{k}\mid Y)$")
    ax2.set_xlabel(r"$n$")
    ax2.set_ylabel(r"$P(X_n = k \mid Y)$")
    ax2.set_title("Posterior marginals")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend()

    # Errors vs reference (if available)
    if X_ref is not None:
        errors = (X_hat != X_ref).astype(float)
        ax3.fill_between(nn, errors, alpha=0.3, color="red", label="errors")
        ax3.plot(nn, errors, lw=0.5, color="red")
        ax3.set_xlabel(r"$n$")
        ax3.set_ylabel("error")
        ax3.set_title(f"Classification errors  (rate={errors.mean():.4f})")
        ax3.set_ylim(-0.1, 1.1)
        ax3.legend()
    else:
        # Show histogram of max posterior
        max_gamma = gamma.max(axis=1)
        ax3.hist(max_gamma, bins=50, color="steelblue", edgecolor="white")
        ax3.set_xlabel(r"$\max_{k}\,\gamma_n(k)$")
        ax3.set_ylabel("count")
        ax3.set_title("Confidence histogram (max posterior)")


def plot_loglik(fig: Figure, lls: list):
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(lls, "o-", color="steelblue", lw=2, markersize=6)
    ax.set_xlabel("ICE iteration")
    ax.set_ylabel(r"Log-likelihood $\log p(Y\mid\theta)$")
    ax.set_title("ICE convergence")
    ax.grid(True, alpha=0.3)


def plot_image_raw(fig: Figure, img: np.ndarray):
    """Display the loaded grayscale or RGB image."""
    ax  = fig.add_subplot(1, 1, 1)
    if img.ndim == 3:
        ax.imshow(np.clip(img, 0, 1), interpolation="nearest")
        ch = f", d={img.shape[2]}"
    else:
        ax.imshow(img, cmap="gray", interpolation="nearest")
        ch = ""
    ax.set_title(
        f"Input image  ($H \\times W{ch}$ = "
        f"{img.shape[0]} × {img.shape[1]}{ch.replace(', d=', ' × ') if ch else ''})"
    )
    ax.set_xticks([]); ax.set_yticks([])


def plot_image_seg(fig: Figure, img: np.ndarray, cls_state, ref_2d):
    """Show input image, segmentation map, and (if available) error map.

    Parameters
    ----------
    img       : (H, W) — original grayscale image.
    cls_state : tuple from :attr:`_cls_state` —
                ``(Y, X_hat, gamma, X_ref, mdl)`` where ``X_hat`` is the
                1D MPM label sequence along the gilbert path.
    ref_2d    : (H, W) int reference labels, or ``None``.
    """
    from pmcprg.pmc.peano import signal_to_image
    Y, X_hat, gamma, X_ref, mdl = cls_state
    H, W = img.shape[:2]
    K = mdl.K
    seg = signal_to_image(X_hat, (H, W))

    ncols = 3 if ref_2d is not None else 2
    ax_img = fig.add_subplot(1, ncols, 1)
    if img.ndim == 3:
        ax_img.imshow(np.clip(img, 0, 1), interpolation="nearest")
    else:
        ax_img.imshow(img, cmap="gray", interpolation="nearest")
    ax_img.set_title("Input image")
    ax_img.set_xticks([]); ax_img.set_yticks([])

    ax_seg = fig.add_subplot(1, ncols, 2)
    cmap_name = "tab10" if K <= 10 else "viridis"
    ax_seg.imshow(
        seg, cmap=plt.get_cmap(cmap_name),
        vmin=0, vmax=max(K - 1, 1), interpolation="nearest",
    )
    ax_seg.set_title(rf"Segmentation  ($K = {K}$, MPM)")
    ax_seg.set_xticks([]); ax_seg.set_yticks([])

    if ref_2d is not None:
        from pmcprg.pmc.inference import error_rate
        from scipy.optimize import linear_sum_assignment
        # Reuse the Hungarian assignment used by error_rate so the colours
        # of the predicted classes line up with the reference's.
        K_ref = int(max(ref_2d.max(), seg.max())) + 1
        C = np.zeros((K_ref, K_ref), dtype=int)
        for t, h in zip(ref_2d.ravel(), seg.ravel()):
            C[t, h] += 1
        row_ind, col_ind = linear_sum_assignment(-C)
        mapping = {int(c): int(r) for r, c in zip(row_ind, col_ind)}
        seg_aligned = np.vectorize(lambda v: mapping.get(int(v), int(v)))(seg)
        err_map = (seg_aligned != ref_2d).astype(np.uint8)
        er = error_rate(ref_2d.ravel(), seg.ravel())

        ax_err = fig.add_subplot(1, ncols, 3)
        ax_err.imshow(err_map, cmap="gray_r", interpolation="nearest")
        ax_err.set_title(f"Misclassified  ({er * 100:.1f} %)")
        ax_err.set_xticks([]); ax_err.set_yticks([])


def plot_gof_heatmap(fig: Figure, results: list[dict]):
    """K×K p-value heatmap for the GoF test results.

    Errored pairs are shown as grey ‘×’ markers; valid pairs are
    coloured from red (p≈0, reject) through yellow (p≈0.05, edge) to
    green (p≈1, accept). The 0.05 contour is highlighted.
    """
    if not results:
        return
    # Recover K from the indices.
    K = max(max(r["i"], r["j"]) for r in results) + 1
    P = np.full((K, K), np.nan)
    labels = np.full((K, K), "", dtype=object)
    errored: list[tuple[int, int]] = []
    for r in results:
        i, j = int(r["i"]), int(r["j"])
        if "error" in r:
            errored.append((i, j))
            labels[i, j] = f"{r['name']}\nERR"
        else:
            P[i, j] = float(r["p"])
            # n_eff belongs on the cell: the K² p-values are no longer
            # interchangeable (audit S-6), and a pair the chain rarely
            # visits must not read like one it visits constantly.
            n_eff = r.get("n_eff")
            tail  = f"\nn={n_eff:.0f}" if n_eff is not None else ""
            labels[i, j] = f"{r['name']}\np={r['p']:.2f}{tail}"

    ax = fig.add_subplot(1, 1, 1)

    # Colour map: red → yellow → green via "RdYlGn"; clip vmin/vmax for
    # numerical stability.
    cmap = plt.get_cmap("RdYlGn")
    im = ax.imshow(P, cmap=cmap, vmin=0.0, vmax=1.0,
                   aspect="equal", origin="upper")
    # The 0.05 acceptance threshold as a contour line.
    if np.isfinite(P).any():
        ax.contour(P, levels=[0.05], colors="black", linewidths=1.0)

    for i in range(K):
        for j in range(K):
            ax.text(j, i, labels[i, j],
                    ha="center", va="center", fontsize=8,
                    color="black")
    for (ie, je) in errored:
        ax.plot(je, ie, marker="x", markersize=22, color="dimgrey",
                markeredgewidth=2)

    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels([rf"$j={k}$" for k in range(K)])
    ax.set_yticklabels([rf"$i={k}$" for k in range(K)])
    ax.set_title("Goodness-of-fit $p$-values  (CvM, parametric bootstrap)")

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label(r"$p$-value")
    # Mark the 0.05 reject/accept threshold on the colour bar.
    cbar.ax.axhline(0.05, color="black", linewidth=1.0)



# ==================================================================
# ICE-trace visualisations (views A–K)
# ==================================================================

# ----- internal drawing helpers (shared with the dashboard / playback) -----


def draw_tau_trajectories(ax, trace: IceTrace, vline: int | None = None):
    """Draw τ_{ij}(iter) curves with family-change markers on ``ax``."""
    if trace.tau_history is None or trace.tau_history.size == 0:
        ax.text(0.5, 0.5, "no copulas in this variant",
                transform=ax.transAxes, ha="center", va="center")
        ax.set_axis_off()
        return
    T, K, _ = trace.tau_history.shape
    nn = np.arange(T)
    for i in range(K):
        for j in range(K):
            series = trace.tau_history[:, i, j]
            if not np.any(np.isfinite(series)):
                continue
            ax.plot(nn, series, "-",
                    label=rf"$({i},{j})$ {trace.family_history[-1][i][j]}")
            # ★ markers where family changed
            fam_series = [trace.family_history[t][i][j] for t in range(T)]
            for t in range(1, T):
                if fam_series[t] and fam_series[t] != fam_series[t - 1]:
                    ax.plot(t, series[t], "*", markersize=10,
                            color="black", markerfacecolor="yellow")
    if vline is not None:
        ax.axvline(vline, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("ICE iteration")
    ax.set_ylabel(r"Kendall $\tau_K$")
    ax.set_title(r"$\tau_K$ trajectories per pair  (★ = family change)")
    ax.grid(True, alpha=0.3)
    # Skip the legend if every τ-series was non-finite (HMC variant
    # carrying a stub tau_history): matplotlib otherwise prints a
    # ``No artists with labels found`` UserWarning.
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7, ncol=2)


def draw_family_ribbon(ax, trace: IceTrace, t_max: int | None = None):
    """Draw the family-selection ribbon on ``ax``.

    ``t_max`` truncates the ribbon to that iteration (used by the
    playback view); pass None to show the full history.
    """
    if not trace.family_history:
        ax.text(0.5, 0.5, "no copulas", transform=ax.transAxes,
                ha="center", va="center")
        ax.set_axis_off()
        return
    T = len(trace.family_history)
    K = len(trace.family_history[0])
    # Build colour map from the *candidate* list (stable order across runs).
    candidates = trace.candidates or []
    # Plus any family that ended up selected even if not in candidates
    # (shouldn't happen, but defensive).
    seen = sorted({fam for row in trace.family_history
                   for r in row for fam in r if fam})
    all_fams = candidates + [f for f in seen if f not in candidates]
    cmap = plt.get_cmap("tab10")
    fam_color = {f: cmap(k % 10) for k, f in enumerate(all_fams)}

    # Map each (i, j) to a display row (skip pairs that never have a copula).
    row_idx = []
    for i in range(K):
        for j in range(K):
            if any(trace.family_history[t][i][j]
                   for t in range(T)):
                row_idx.append((i, j))
    if not row_idx:
        ax.text(0.5, 0.5, "no copulas", transform=ax.transAxes,
                ha="center", va="center")
        ax.set_axis_off()
        return

    upper = T if t_max is None else min(t_max + 1, T)
    for r, (i, j) in enumerate(row_idx):
        for t in range(upper):
            fam = trace.family_history[t][i][j]
            if not fam:
                continue
            ax.broken_barh([(t - 0.5, 1.0)], (r - 0.4, 0.8),
                           facecolors=fam_color.get(fam, "lightgrey"),
                           edgecolor="white", linewidth=0.3)
    ax.set_yticks(range(len(row_idx)))
    ax.set_yticklabels([rf"$({i},{j})$" for i, j in row_idx])
    ax.set_xlabel("ICE iteration")
    ax.set_xlim(-0.5, T - 0.5)
    ax.set_ylim(-0.5, len(row_idx) - 0.5)
    ax.set_title("Copula family selection ribbon")
    # Build a legend from the families actually present.
    handles = [
        plt.matplotlib.patches.Patch(color=fam_color[f], label=f)
        for f in all_fams if any(
            trace.family_history[t][i][j] == f
            for t in range(upper) for i, j in row_idx
        )
    ]
    if handles:
        ax.legend(handles=handles, fontsize=7, loc="upper right",
                  ncol=min(len(handles), 4))

# ----- view A: τ trajectories per pair ---------------------------------


def plot_ice_tau(fig: Figure, trace: IceTrace):
    ax = fig.add_subplot(1, 1, 1)
    draw_tau_trajectories(ax, trace)

# ----- view B: family selection ribbon ---------------------------------


def plot_ice_family(fig: Figure, trace: IceTrace):
    ax = fig.add_subplot(1, 1, 1)
    draw_family_ribbon(ax, trace)

# ----- view L: GICE margin family ribbon -------------------------------


def plot_ice_margin_family(fig: Figure, trace: IceTrace):
    """Per-state ribbon of the margin family selected at each ICE iter.

    This is the SP-2016 GICE counterpart of view~B (which tracks
    copula families). One row per margin block — per state ``i`` for state
    margins f_i, per pair ``(i, j)`` for pair margins f_ij
    (DerrodePieczynski_CSDA2013 Eq. 12); each row is a horizontal ribbon
    coloured by the ``dist`` field of
    ``trace.margin_history[t][block_idx]`` across iterations. Useful when the
    user enables ``candidates`` per margin block to visualise family
    identification convergence.
    """
    ax  = fig.add_subplot(1, 1, 1)

    history = trace.margin_history
    if not history:
        ax.text(0.5, 0.5, "no margin history",
                transform=ax.transAxes, ha="center", va="center")
        ax.set_axis_off()
        return

    T  = len(history)
    n_blocks = len(history[0])
    # Collect every distinct family that appeared across the trace.
    seen = sorted({
        history[t][k]["dist"] for t in range(T) for k in range(n_blocks)
    })
    cmap = plt.get_cmap("tab10")
    fam_color = {f: cmap(k % 10) for k, f in enumerate(seen)}

    # Label rows with the pair ``(i, j)`` or the state index ``i``; fall
    # back to the block index when blocks are not ``i``-keyed.
    labels = []
    for k in range(n_blocks):
        blk0 = history[0][k]
        if "i" in blk0 and "j" in blk0:
            labels.append(rf"$f_{{{blk0['i']},{blk0['j']}}}$")
        elif "i" in blk0:
            labels.append(rf"$i={blk0['i']}$")
        else:
            labels.append(f"#{k}")

    for r in range(n_blocks):
        for t in range(T):
            fam = history[t][r]["dist"]
            ax.broken_barh([(t - 0.5, 1.0)], (r - 0.4, 0.8),
                           facecolors=fam_color[fam],
                           edgecolor="white", linewidth=0.3)
    ax.set_yticks(range(n_blocks))
    ax.set_yticklabels(labels)
    ax.set_xlabel("ICE iteration")
    ax.set_xlim(-0.5, T - 0.5)
    ax.set_ylim(-0.5, n_blocks - 0.5)
    ax.set_title("Margin family ribbon (GICE — SP-2016 §3)")
    handles = [
        plt.matplotlib.patches.Patch(color=fam_color[f], label=f)
        for f in seen
    ]
    if handles:
        ax.legend(handles=handles, fontsize=7, loc="upper right",
                  ncol=min(len(handles), 4))

# ----- view C: pseudo-observations + fitted PDF contours ---------------


def plot_ice_pseudos(fig: Figure, trace: IceTrace, model: PMCModel,
                     Y: np.ndarray):
    """K×K small-multiples: pseudo-observations + fitted copula log-PDF.

    Two things this panel used to get wrong, both about whether a reader can
    actually decode it.

    **The contours carried no scale** (audit G-9): eight unlabelled levels per
    panel, each panel on its own automatic range, so nothing told you which
    contour was dense — nor whether panel (0,0) was denser than panel (1,1).
    Levels are now shared across the whole grid and shown once in a figure
    colourbar, which makes the panels comparable as well as readable. The
    range is taken from percentiles, not the extremes: several families have
    a density that diverges in a corner, and one singular cell would
    otherwise swallow the entire scale.

    **Every panel showed every transition.** The scatter drew all N−1 points
    in all K² panels with equal weight, though the pair posterior says most
    of them barely belong to a given pair — an independent pair may carry an
    effective sample of 20 while the panel displays 1500 dots. This is the
    visual form of the defect S-6 fixed in the GoF test. Point opacity is now
    proportional to ξ_n(i, j), so each panel shows the transitions it is
    actually about, and the panel title reports the effective count.
    """
    from pmcprg.pmc.inference import (
        backward, forward, joint_posteriors, precompute_weights,
    )

    K = model.K
    N = len(Y)
    f_cdf = compute_marginal_cdfs(model, Y)

    # Pair posteriors — the same quantity the ICE M-step weights with.
    try:
        W, f_pdf = precompute_weights(model, Y)
        alpha, _ = forward(model, Y, W=W, f_pdf=f_pdf)
        xi = joint_posteriors(alpha, W, backward(model, Y, W=W))
    except Exception:                                      # pragma: no cover
        xi = None

    ug = np.linspace(0.05, 0.95, 40)
    UU, VV = np.meshgrid(ug, ug)
    grid_uv = np.column_stack((UU.ravel(), VV.ravel()))

    # First pass: every panel's surface, so the levels can be shared.
    surfaces: dict[tuple[int, int], np.ndarray] = {}
    for i in range(K):
        for j in range(K):
            try:
                Z = model.copula(i, j).pdf_array(grid_uv).reshape(UU.shape)
                surfaces[(i, j)] = np.log10(np.clip(Z, 1e-6, None))
            except Exception:                              # pragma: no cover
                pass

    levels = None
    if surfaces:
        pooled = np.concatenate([z.ravel() for z in surfaces.values()])
        pooled = pooled[np.isfinite(pooled)]
        if pooled.size:
            lo, hi = np.percentile(pooled, [5, 99])
            if hi > lo:
                levels = np.linspace(lo, hi, 9)

    cs = None
    for i in range(K):
        for j in range(K):
            ax = fig.add_subplot(K, K, i * K + j + 1)
            cop = model.copula(i, j)
            u = f_cdf[: N - 1, i, j]
            v = f_cdf[1:,      j, i]

            n_eff = None
            if xi is not None:
                w = xi[:, i, j]
                tot = float(w.sum())
                if tot > 0:
                    # Σw — the same effective size the GoF heatmap and the
                    # τ-CI report (pmcprg.pmc.ice._effective_n); this panel used
                    # to show Kish's (Σw)²/Σw² under the same name (audit K-5).
                    n_eff = tot
                    # Opacity ∝ membership, floored so a faint pair is still
                    # visible and capped so a dominant one does not blot out
                    # the contours underneath.
                    a = w / max(w.max(), 1e-300)
                    ax.scatter(u, v, s=3, color="black",
                               alpha=np.clip(0.05 + 0.55 * a, 0.02, 0.6),
                               linewidths=0)
            if n_eff is None:
                ax.scatter(u, v, s=2, alpha=0.3, color="black", linewidths=0)

            Z = surfaces.get((i, j))
            if Z is not None:
                cs = ax.contour(UU, VV, Z, levels=levels, cmap="viridis",
                                linewidths=1.0)

            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            title = (rf"$({i},{j})$  {cop.copula_enum.value.SHORT_NAME}"
                     rf"  $\hat{{\tau}}={cop.params['tau_k']:.3f}$")
            if n_eff is not None:
                title += rf"  $n_{{\mathrm{{eff}}}}={n_eff:.0f}$"
            ax.set_title(title, fontsize=8)
            if i == K - 1:
                ax.set_xlabel(r"$u$")
            if j == 0:
                ax.set_ylabel(r"$v$")

    if cs is not None and levels is not None:
        cbar = fig.colorbar(cs, ax=fig.axes, fraction=0.025, pad=0.02)
        # Kept short: on a narrow GUI canvas a long rotated label is the
        # first thing to collide with the panel grid.
        cbar.set_label(r"$\log_{10} c(u,v)$", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    # Short on purpose: a long suptitle is clipped on both sides once the GUI
    # canvas narrows, and the single shared colourbar already says the levels
    # are common to the grid.
    fig.suptitle("Pseudo-observations (opacity $\\propto \\xi$) "
                 "+ copula $\\log$-PDF", fontsize=10)


def plot_posterior_draws(fig: Figure, Y, X_hat, gamma, mdl, *,
                         n_draws: int = 24, seed: int = 0):
    """FFBS draws from P(X | Y) beside the MPM estimate — audit G-7.

    The GUI reported classification as a single label sequence and nothing
    else, so nothing on screen distinguished a segmentation the posterior is
    sure of from one it is guessing at. ``pmcprg.pmc.inference.sample_posterior``
    — the exact Forward-Filter Backward-Sample the SEM M-step already relies
    on — was in the package but had no entry point.

    Three panels:

    * the raster of ``n_draws`` posterior realisations, one row each, so
      instability is visible directly as vertical texture;
    * the MPM estimate on the same axis, for comparison;
    * per-position disagreement — the share of draws differing from MPM —
      which localises *where* the uncertainty sits rather than summarising it
      into one number.

    Note the two are answering different questions: MPM maximises the
    marginal posterior position by position, so it can differ from every
    single draw, and a high disagreement rate is not an error but a statement
    about how flat the posterior is there.
    """
    from pmcprg.pmc.inference import (
        forward, precompute_weights, sample_posterior,
    )

    K = mdl.K
    N = len(Y)
    rng = np.random.default_rng(seed)

    # Share the heavy tensors across draws — they do not depend on the draw.
    W, f_pdf = precompute_weights(mdl, Y)
    alpha_hat, _ = forward(mdl, Y, W=W, f_pdf=f_pdf)
    draws = np.empty((n_draws, N), dtype=int)
    for m in range(n_draws):
        draws[m] = sample_posterior(mdl, Y, rng, W=W, f_pdf=f_pdf,
                                    alpha_hat=alpha_hat)

    cmap = ListedColormap(_class_colors(K))
    # The MPM strip is a single row: given a third of the figure it reads as a
    # block of colour rather than as a label sequence.
    gs  = fig.add_gridspec(3, 1, height_ratios=[3.0, 0.45, 2.0])
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(draws, aspect="auto", interpolation="nearest",
               cmap=cmap, vmin=-0.5, vmax=K - 0.5)
    ax1.set_ylabel("draw")
    ax1.set_title(f"{n_draws} posterior draws (FFBS) from $P(X \\mid Y)$",
                  fontsize=9)
    ax1.set_xticklabels([])

    ax2 = fig.add_subplot(gs[1])
    ax2.imshow(X_hat[None, :], aspect="auto", interpolation="nearest",
               cmap=cmap, vmin=-0.5, vmax=K - 0.5)
    ax2.set_yticks([])
    ax2.set_ylabel("MPM")
    ax2.set_title("MPM estimate — the marginal mode, position by position",
                  fontsize=9)
    ax2.set_xticklabels([])

    disagree = np.mean(draws != X_hat[None, :], axis=0)
    ax3 = fig.add_subplot(gs[2])
    ax3.fill_between(np.arange(N), disagree, step="mid", alpha=0.7,
                     color="firebrick", linewidth=0)
    ax3.set_xlim(0, N - 1)
    ax3.set_ylim(0, 1)
    ax3.set_xlabel("$n$")
    ax3.set_ylabel("disagreement")
    ax3.grid(True, alpha=0.3)
    ax3.set_title(
        f"share of draws differing from MPM — mean {disagree.mean():.3f}, "
        f"{float(np.mean(disagree > 0.25)):.1%} of positions above 0.25",
        fontsize=9,
    )
    fig.suptitle("Classification uncertainty — posterior draws vs MPM")


def plot_tau_ci(fig: Figure, rows: list[dict]):
    """Forest plot of τ̂ with its bootstrap interval, one line per pair.

    A forest plot rather than a K×K grid because the question these
    intervals answer is a comparison: *which pairs carry dependence the data
    actually support?* That reads off a shared τ axis with a rule at zero,
    and does not read off separate panels.

    Pairs whose interval straddles 0 are drawn in grey: their dependence is
    not distinguishable from independence at this sample size, which is
    precisely what a point estimate quoted to three decimals concealed.
    """
    if not rows:
        return
    ax = fig.add_subplot(1, 1, 1)
    ordered = sorted(rows, key=lambda r: (r["i"], r["j"]))
    labels, ypos = [], []
    for k, r in enumerate(ordered):
        y = len(ordered) - 1 - k
        ypos.append(y)
        labels.append(rf"$({r['i']},{r['j']})$  {r['name']}")
        if "lo" not in r:
            ax.plot([0], [y], "x", color="grey", markersize=7)
            ax.annotate(r.get("error", "n/a"), (0.02, y), fontsize=7,
                        color="grey", va="center",
                        xycoords=("axes fraction", "data"))
            continue
        spans_zero = r["lo"] <= 0.0 <= r["hi"]
        colour = "grey" if spans_zero else "steelblue"
        tau_hat = r.get("tau_hat", r["tau"])
        ax.plot([r["lo"], r["hi"]], [y, y], "-", color=colour, lw=2.0)
        ax.plot([r["lo"], r["hi"]], [y, y], "|", color=colour, markersize=8)
        ax.plot([tau_hat], [y], "o", color=colour, markersize=6)
        # The value the model declares, when it is not the refitted one: a
        # hollow marker, so a gap between "what the model says" and "what the
        # data support" is visible instead of being read as uncertainty.
        if abs(tau_hat - r["tau"]) >= 5e-4:
            ax.plot([r["tau"]], [y], "D", mfc="none", mec="darkorange",
                    markersize=6, mew=1.4)
        ax.annotate(
            rf"$\hat\tau={tau_hat:+.3f}$  "
            rf"[{r['lo']:+.3f}, {r['hi']:+.3f}]   "
            rf"$n_{{\mathrm{{eff}}}}={r['n_eff']:.0f}$",
            (1.02, y), fontsize=7, va="center",
            xycoords=("axes fraction", "data"),
        )
    ax.axvline(0.0, color="black", lw=0.8, alpha=0.6)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_ylim(-0.6, len(ordered) - 0.4)
    ax.set_xlabel(r"Kendall $\tau$")
    ax.grid(True, axis="x", alpha=0.3)
    B     = next((r["B"] for r in ordered if "B" in r), None)
    alpha = next((r["alpha"] for r in ordered if "alpha" in r), 0.05)
    # Kept to two short lines: a long title is clipped on a narrow canvas.
    ax.set_title(
        f"ξ-weighted bootstrap {100 * (1 - alpha):.0f}% intervals on τ̂ "
        f"(B={B}, at each pair's effective size)\n"
        "grey = straddles 0;  ◇ = model's τ where it differs from the refit",
        fontsize=9,
    )


def plot_margin_ks(fig: Figure, rows: list[dict], mdl: PMCModel,
                   Y: np.ndarray, X_draw: np.ndarray):
    """Per-state ECDF against the fitted margin, with the bootstrap verdict.

    A bar of p-values would say *whether* a margin fits; the ECDF overlay
    says *how* it misses — a shift, a spread, a tail — which is what one acts
    on. The KS statistic is the largest vertical gap between the two curves,
    so the plot shows exactly the quantity the test summarises.

    On a pair model the curve of state k is the law of y_n given x_n = k,
    g_k = Σ_j A[k, j] f_kj (DerrodePieczynski_CSDA2013 Eq. 12 summed over
    x_{n+1}), the law the per-state sample follows.
    """
    if not rows:
        return
    pair = is_pair(mdl)
    K = len(rows)
    ncol = min(K, 3)
    nrow = int(np.ceil(K / ncol))
    for idx, r in enumerate(sorted(rows, key=lambda x: x["state"])):
        k  = int(r["state"])
        ax = fig.add_subplot(nrow, ncol, idx + 1)
        sel = np.asarray(Y)[np.asarray(X_draw) == k]
        if sel.size:
            s = np.sort(_channel(sel, 0))
            ecdf = np.arange(1, len(s) + 1) / len(s)
            ax.step(s, ecdf, where="post", color="steelblue", lw=1.4,
                    label="empirical (posterior draw)")
            try:
                ax.plot(s, state_law(mdl, k).cdf_vec(s), color="firebrick",
                        lw=1.2, ls="--",
                        label=(r"fitted $\sum_j A_{kj} f_{kj}$" if pair
                               else "fitted margin"))
            except Exception:                                  # pragma: no cover
                pass
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        if "p" in r:
            verdict = "✓" if r["p"] >= 0.05 else "✗"
            head = f"state {k} — {r['dist']}   p={r['p']:.3f} {verdict}  n={r['n']}"
            comps = r.get("components") or []
            if comps:
                # The signed components, not just the verdict: heavy tails
                # read as V₂ negative with V₄ positive, and a name alone
                # would lose half of that.
                body = "  ".join(
                    f"$V_{j}$={v:+.1f}" for j, v in enumerate(comps, start=1))
                title = f"{head}\n{r.get('component', '')}:  {body}"
            else:
                title = head + f"\nKS={r.get('statistic_value', float('nan')):.4f}"
        else:
            title = f"state {k} — {r['dist']}\n{r.get('error', 'n/a')}"
        ax.set_title(title, fontsize=8)
        if idx == 0:
            ax.legend(fontsize=6, loc="lower right")
        ax.set_xlabel("$y$")
        if idx % ncol == 0:
            ax.set_ylabel("CDF")
    B    = next((r["B"] for r in rows if "B" in r), None)
    kind = next((r.get("statistic") for r in rows if r.get("statistic")), "")
    label = ("Neyman smooth components, Bonferroni-corrected"
             if kind == "neyman" else "multivariate KS")
    fig.suptitle(f"Margin adequacy — {label}, bootstrapped null (B={B})",
                 fontsize=10)


# ----- view D: multistart comparison -----------------------------------


def plot_ice_multistart(fig: Figure, trace: IceTrace):
    ax = fig.add_subplot(1, 1, 1)
    # Best run (the trace itself) in steel-blue, others in grey.
    for losing in trace.multistart_runs:
        ax.plot(losing.log_liks, "-", color="grey", alpha=0.6, lw=1.0,
                label=f"{losing.run_tag}  final={losing.log_liks[-1]:.2f}")
    ax.plot(trace.log_liks, "o-", color="steelblue", lw=2.0, markersize=4,
            label=f"WINNER  ({trace.run_tag or 'unperturbed'})  "
                  f"final={trace.log_liks[-1]:.2f}")
    ax.set_xlabel("ICE iteration")
    ax.set_ylabel(r"Log-likelihood $\log p(Y\mid\theta)$")
    ax.set_title(f"Multistart comparison "
                 f"({1 + len(trace.multistart_runs)} runs)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

# ----- view E: joint prior p[i,j] evolution ----------------------------


def plot_ice_prior(fig: Figure, trace: IceTrace):
    if trace.p_history is None or trace.p_history.size == 0:
        ax = fig.add_subplot(1, 1, 1)
        ax.text(0.5, 0.5, "no prior history",
                transform=ax.transAxes, ha="center", va="center")
        ax.set_axis_off()
        return
    T, K, _ = trace.p_history.shape
    nn = np.arange(T)

    ax1 = fig.add_subplot(2, 1, 1)
    for i in range(K):
        for j in range(K):
            # Under SR-PMC, only show upper-triangle entries.
            if j < i:
                continue
            ax1.plot(nn, trace.p_history[:, i, j], "-",
                     label=rf"$p_{{{i}{j}}}$")
    ax1.set_ylabel(r"$p_{ij}$")
    ax1.set_title("Joint-prior evolution  (upper triangle under SR-PMC)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=7, ncol=K)

    ax2 = fig.add_subplot(2, 1, 2)
    if T >= 2:
        diffs = np.linalg.norm(
            trace.p_history[1:] - trace.p_history[:-1], axis=(1, 2),
        )
        ax2.semilogy(nn[1:], diffs, "o-", color="firebrick")
    ax2.set_xlabel("ICE iteration")
    ax2.set_ylabel(r"$\Vert p^{(t)} - p^{(t-1)} \Vert_{F}$")
    ax2.set_title("Convergence of the joint prior")
    ax2.grid(True, alpha=0.3, which="both")

# ----- view F: margin parameter evolution ------------------------------


_MAX_PARAM_LINES = 6            # per panel, before we start capping


def _param_component_series(margin_history, k: int, pname: str, T: int):
    """``(labels, array of shape (T, m))`` for one margin parameter.

    A scalar parameter yields a single column; a vector or matrix parameter
    yields one column per flattened component, so the panel shows how each
    moved across iterations. Entries that cannot be read as numbers become
    NaN rather than raising.
    """
    frames = []
    for t in range(T):
        v = margin_history[t][k].get("params", {}).get(pname, np.nan)
        try:
            frames.append(np.ravel(np.asarray(v, dtype=float)))
        except (TypeError, ValueError):
            frames.append(np.array([np.nan]))
    m = max((f.size for f in frames), default=1)
    out = np.full((T, m), np.nan)
    for t, f in enumerate(frames):
        out[t, : f.size] = f
    if m == 1:
        return [rf"${pname}$"], out
    return [rf"${pname}_{{{c}}}$" for c in range(m)], out


def plot_ice_margins(fig: Figure, trace: IceTrace):
    if not trace.margin_history:
        fig.clear()
        return
    # margin_history[t] is a list of dicts (in declaration order).
    T = len(trace.margin_history)
    n_blocks = len(trace.margin_history[0]) if T else 0
    if n_blocks == 0:
        fig.clear()
        return
    # Layout: one subplot per block, in a square-ish grid.
    ncol = int(np.ceil(np.sqrt(n_blocks)))
    nrow = int(np.ceil(n_blocks / ncol))
    for k in range(n_blocks):
        ax = fig.add_subplot(nrow, ncol, k + 1)
        blk0 = trace.margin_history[0][k]
        i, j = blk0.get("i", "?"), blk0.get("j", "")
        dist = blk0.get("dist", "?")
        param_names = list(blk0.get("params", {}).keys())
        capped: list[str] = []
        for pname in param_names:
            # A multivariate margin records a mean vector / covariance matrix
            # here, not a scalar: the bare float() raised and took the whole
            # view down for every d ≥ 2 model. Flatten and plot one line per
            # component.
            labels, series = _param_component_series(
                trace.margin_history, k, pname, T,
            )
            shown = min(len(labels), _MAX_PARAM_LINES)
            for c in range(shown):
                ax.plot(series[:, c], "-o", markersize=3, label=labels[c])
            if len(labels) > shown:
                capped.append(f"{pname}: {shown}/{len(labels)}")
        title = rf"$({i},{j})$  {dist}"
        if capped:
            # Never cap silently — a 3×3 covariance is 9 lines and would
            # drown the panel, but the reader has to know what is not drawn.
            title += "  [" + "; ".join(capped) + "]"
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Margin-parameter evolution")

# ----- view G: ICE dashboard (multi-panel summary) ---------------------


def plot_ice_dashboard(fig: Figure, trace: IceTrace, ice_Y):
    # 3×2 grid: LL, τ, family, ‖Δp‖, AIC/BIC, status.
    # ``constrained_layout`` (rcParam) handles the cross-row spacing,
    # but multi-axes legends would still steal too much space — keep
    # legend.fontsize small via per-call overrides where needed.
    ax_ll  = fig.add_subplot(3, 2, 1)
    ax_tau = fig.add_subplot(3, 2, 2)
    ax_fam = fig.add_subplot(3, 2, 3)
    ax_dp  = fig.add_subplot(3, 2, 4)
    ax_ic  = fig.add_subplot(3, 2, 5)
    ax_st  = fig.add_subplot(3, 2, 6); ax_st.set_axis_off()

    lls = trace.log_liks
    ax_ll.plot(lls, "o-", color="steelblue", markersize=3)
    ax_ll.set_title("Log-likelihood")
    ax_ll.set_xlabel("iter")
    ax_ll.set_ylabel(r"$\log p(Y\mid\theta)$")
    ax_ll.grid(True, alpha=0.3)
    ax_ll.tick_params(labelsize=7)

    draw_tau_trajectories(ax_tau, trace)
    # Tighten the τ panel — within the dashboard, full per-pair labels
    # are noise. Drop the legend; the family ribbon below carries the
    # same information visually.
    leg = ax_tau.get_legend()
    if leg is not None:
        leg.remove()
    ax_tau.set_title(r"$\tau_K$ trajectories")
    ax_tau.set_xlabel("iter")
    ax_tau.tick_params(labelsize=7)

    draw_family_ribbon(ax_fam, trace)
    ax_fam.set_xlabel("iter")
    ax_fam.tick_params(labelsize=7)
    # The per-axes legend is too cramped inside a 6-panel grid.
    # Promote it to a figure-level legend (constrained_layout reserves
    # the space below) so users can still decode the ribbon colours.
    fam_legend = ax_fam.get_legend()
    ribbon_handles, ribbon_labels = [], []
    if fam_legend is not None:
        ribbon_handles = list(fam_legend.legend_handles)
        ribbon_labels  = [t.get_text() for t in fam_legend.get_texts()]
        fam_legend.remove()

    if trace.p_history is not None and trace.p_history.shape[0] >= 2:
        diffs = np.linalg.norm(
            trace.p_history[1:] - trace.p_history[:-1], axis=(1, 2),
        )
        ax_dp.semilogy(diffs, "o-", color="firebrick", markersize=3)
        ax_dp.set_title(r"$\Vert\Delta p\Vert_{F}$  (joint-prior change)")
        ax_dp.set_xlabel("iter")
        ax_dp.grid(True, alpha=0.3, which="both")
        ax_dp.tick_params(labelsize=7)

    # Approximate AIC/BIC: requires a parameter count.
    # We use a coarse upper bound: K² × (#copula params + #margin params).
    if ice_Y is not None:
        n = len(ice_Y)
        # Heuristic: τ + extras per block + margin params per block
        n_params = 0
        if trace.tau_history is not None:
            n_params += int(np.isfinite(trace.tau_history[-1]).sum())
        if trace.margin_history:
            n_params += sum(
                len(blk.get("params", {})) for blk in trace.margin_history[-1]
            )
        aic = 2 * n_params - 2 * np.array(lls)
        bic = n_params * np.log(max(n, 1)) - 2 * np.array(lls)
        ax_ic.plot(aic, "o-", label="AIC", color="darkorange", markersize=3)
        ax_ic.plot(bic, "s-", label="BIC", color="purple",     markersize=3)
        ax_ic.set_title(rf"AIC / BIC  ($k={n_params}$)")
        ax_ic.set_xlabel("iter")
        ax_ic.legend(fontsize=7)
        ax_ic.grid(True, alpha=0.3)
        ax_ic.tick_params(labelsize=7)
    else:
        ax_ic.text(0.5, 0.5, "AIC/BIC unavailable\n(no observation sequence)",
                   transform=ax_ic.transAxes, ha="center", va="center",
                   fontsize=8)
        ax_ic.set_axis_off()

    # Summary box — uses LaTeX ΔLL via mathtext.
    n_iters = trace.n_iters
    ll_init = lls[0] if lls else float("nan")
    ll_fin  = lls[-1] if lls else float("nan")
    msg = (
        f"Iterations: {n_iters}\n"
        f"LL initial: {ll_init:.3f}\n"
        f"LL final:   {ll_fin:.3f}\n"
        f"ΔLL:        {ll_fin - ll_init:+.3f}\n"
        f"Multistart: {1 + len(trace.multistart_runs)} run(s)\n"
        f"Candidates: {', '.join(trace.candidates) or '—'}"
    )
    ax_st.text(0.05, 0.95, msg, transform=ax_st.transAxes,
               ha="left", va="top", family="monospace", fontsize=8)
    ax_st.set_title("Summary", loc="left")

    # Figure-level legend that decodes the family-ribbon colours.
    # ``loc="outside lower center"`` (matplotlib >= 3.6) lets
    # constrained_layout reserve the strip below the grid, so the
    # legend never overlaps a panel.
    if ribbon_handles:
        fig.legend(
            ribbon_handles, ribbon_labels,
            loc="outside lower center",
            ncol=min(len(ribbon_handles), 6),
            fontsize=7, frameon=False,
            title="Copula family (ribbon colour)",
            title_fontsize=7,
        )

    fig.suptitle("ICE Dashboard")

# ----- view H: PP/QQ plot of fitted copulas ----------------------------


def plot_ice_pp(fig: Figure, model: PMCModel, Y: np.ndarray):
    K = model.K
    N = len(Y)
    f_cdf = compute_marginal_cdfs(model, Y)

    for i in range(K):
        for j in range(K):
            ax = fig.add_subplot(K, K, i * K + j + 1)
            cop = model.copula(i, j)
            u   = f_cdf[: N - 1, i, j]
            v   = f_cdf[1:,      j, i]
            uv  = np.column_stack((u, v))
            try:
                Cn = _empirical_copula(uv, uv)
                Ct = np.array([cop.cdf(list(row)) for row in uv])
                ax.scatter(Cn, Ct, s=4, alpha=0.4)
                ax.plot([0, 1], [0, 1], "r--", lw=1)
                ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
                ax.set_title(
                    rf"$({i},{j})$  {cop.copula_enum.value.SHORT_NAME}",
                    fontsize=9,
                )
            except NotImplementedError:
                ax.text(0.5, 0.5, "no CDF", transform=ax.transAxes,
                        ha="center", va="center")
                ax.set_axis_off()
            if i == K - 1:
                ax.set_xlabel(r"$C_n$  (empirical)")
            if j == 0:
                ax.set_ylabel(r"$C_\theta$  (fitted)")
    fig.suptitle(r"PP plots — empirical vs. fitted copula CDF")

# ----- view I: tail dependence per pair --------------------------------


def plot_ice_tail(fig: Figure, model: PMCModel):
    K = model.K
    lam_L = np.zeros((K, K))
    lam_U = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            try:
                lL, lU = model.copula(i, j).tail_dependence()
            except NotImplementedError:
                lL, lU = np.nan, np.nan
            lam_L[i, j] = lL
            lam_U[i, j] = lU
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    for ax, mat, label in (
        (ax1, lam_L, r"$\lambda_L$ (lower tail)"),
        (ax2, lam_U, r"$\lambda_U$ (upper tail)"),
    ):
        im = ax.imshow(mat, cmap="magma", vmin=0.0, vmax=1.0)
        for i in range(K):
            for j in range(K):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.2f}",
                            ha="center", va="center", fontsize=9,
                            color="white" if mat[i, j] < 0.6 else "black")
        ax.set_title(label)
        ax.set_xticks(range(K)); ax.set_yticks(range(K))
        ax.set_xticklabels([rf"$j={k}$" for k in range(K)])
        ax.set_yticklabels([rf"$i={k}$" for k in range(K)])
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    fig.suptitle("Tail dependence per pair (final ICE model)")

# ----- view M: parameter comparison — true (init) vs. fitted ----------


def plot_ice_param_compare(fig: Figure, init_mdl: PMCModel, fitted_mdl: PMCModel):
    """At-a-glance side-by-side of the seed model and the ICE-fitted one.

    Three stacked panels:

    * **Margins** — one row per margin block, i.e. per state ``i`` (state
      margins f_i) or per pair ``(i, j)`` (pair margins f_ij,
      DerrodePieczynski_CSDA2013 Eq. 12): true family + params vs. fitted
      family + params. Family changes (GICE) are flagged.
    * **Copulas** (only when ``variant.uses_copula``) — one row per
      ``(i, j)`` pair: true family + Kendall ``τ`` vs. fitted, with
      a ``Δτ`` column.
    * **Joint prior** — three K×K imshow panels: ``true p``,
      ``fitted p`` and ``fitted − true``.

    Designed for the typical workflow "simulate from a known model,
    run ICE, eyeball the recovery". Self-contained: a single Export
    click yields a publication-grade figure.
    """

    K           = init_mdl.K
    uses_copula = init_mdl.variant.uses_copula and fitted_mdl.variant.uses_copula

    # GridSpec layout. Each section gets a thin "header" sub-row above
    # the body so the section title can never overlap the table headers
    # (the bug visible in the previous iteration was caused by
    # ``ax.set_title`` falling inside a ``set_axis_off`` axes whose
    # table extended to the very top edge). A dedicated bottom-row
    # holds the colour-legend footer, so it never collides with the
    # joint-prior heatmap tick labels.
    #
    # Row heights are tuned per K so the tables don't claim a tall
    # slot they only half-fill (the issue HMC variants showed before
    # — margin table sat at the top of an oversized slot, leaving
    # blank space between it and the heatmaps).
    max_m_params = max(
        (len(blk.get("params", {}))
         for blk in init_mdl.margin_blocks() + fitted_mdl.margin_blocks()),
        default=2,
    )
    # Heuristic: data-rows × per-line cost + header/footer overhead.
    # Tables now use ``loc="upper center"`` so empty body space below
    # the last row appears as a visible gap; tune the slot heights
    # to match the actual content as closely as possible.
    n_m_rows = max(len(init_mdl.margin_blocks()),
                   len(fitted_mdl.margin_blocks()), 1)
    margin_h = 0.25 + 0.16 * n_m_rows * max(2, max_m_params)
    copula_h = 0.30 + 0.16 * (K * K)
    heatmap_h = 1.6
    footer_h  = 0.10

    n_rows  = (4 if uses_copula else 3)
    heights = ([margin_h, copula_h, heatmap_h, footer_h] if uses_copula
               else [margin_h, heatmap_h, footer_h])
    gs      = fig.add_gridspec(n_rows, 1, height_ratios=heights, hspace=0.10)

    # Helper that returns (header_axis, body_axis) for one section.
    def _section(slot, header_height: float = 0.12):
        sub = slot.subgridspec(2, 1, height_ratios=[header_height, 1.0])
        ax_h = fig.add_subplot(sub[0]); ax_h.set_axis_off()
        ax_b = fig.add_subplot(sub[1]); ax_b.set_axis_off()
        return ax_h, ax_b

    ax_m_h, ax_m = _section(gs[0])
    if uses_copula:
        ax_c_h, ax_c = _section(gs[1])
        priors_g     = gs[2].subgridspec(1, 3, wspace=0.30)
        ax_footer    = fig.add_subplot(gs[3]); ax_footer.set_axis_off()
    else:
        priors_g     = gs[1].subgridspec(1, 3, wspace=0.30)
        ax_footer    = fig.add_subplot(gs[2]); ax_footer.set_axis_off()

    # ---- margin table ------------------------------------------------
    # Aligned by key — ``i`` for a state block, ``(i, j)`` for a pair block
    # — as the copula table below is: zipping two lists paired blocks by
    # position, and labelled a pair block with ``i`` alone.
    def _m_key(blk: dict) -> tuple[int, ...]:
        if "j" in blk:
            return (int(blk.get("i", -1)), int(blk["j"]))
        return (int(blk.get("i", -1)),)

    m_init_by   = {_m_key(b): b for b in init_mdl.margin_blocks()}
    m_fitted_by = {_m_key(b): b for b in fitted_mdl.margin_blocks()}
    m_keys = list(m_init_by) + [k for k in m_fitted_by if k not in m_init_by]
    pair_rows = any(len(k) == 2 for k in m_keys)
    m_rows: list[list[str]] = []
    m_cell_colors: list[list] = []
    for key in m_keys:
        blk_t = m_init_by.get(key, {})
        blk_f = m_fitted_by.get(key, {})
        label = f"({key[0]},{key[1]})" if len(key) == 2 else str(key[0])
        fam_t   = blk_t.get("dist", "—")
        fam_f   = blk_f.get("dist", "—")
        # Multi-line params: each "key=value" on its own line. Lets
        # the cell stay narrow while remaining fully legible.
        par_t   = _fmt_params(blk_t.get("params", {}), multiline=True)
        par_f   = _fmt_params(blk_f.get("params", {}), multiline=True)
        row     = [label, fam_t, par_t, fam_f, par_f]
        colours = ["white"] * len(row)
        if fam_t != fam_f:
            # Highlight family changes (GICE often picks a different family).
            colours[3] = "#FFE9A8"  # warm yellow for "fitted family"
            colours[1] = "#FFF6D8"
        m_rows.append(row)
        m_cell_colors.append(colours)

    m_tbl = ax_m.table(
        cellText  = m_rows,
        colLabels = ["(i,j)" if pair_rows else "i",
                     "true family", "true params",
                     "fitted family", "fitted params"],
        cellColours = m_cell_colors if m_cell_colors else None,
        colWidths = ([0.07, 0.19, 0.27, 0.20, 0.27] if pair_rows
                     else [0.05, 0.20, 0.275, 0.20, 0.275]),
        loc       = "upper center",   # anchor at top of body axes
        cellLoc   = "left",
    )
    m_tbl.auto_set_font_size(False)
    m_tbl.set_fontsize(8)
    # Multi-line params need taller rows. Reuse the gridspec
    # heuristic so a 4-param ``betaprime`` fixture isn't cramped.
    m_tbl.scale(1.0, max(1.4, 0.85 * max_m_params))
    ax_m_h.text(
        0.0, 0.5, ("Margins f_ij — true vs fitted" if pair_rows
                   else "Margins — true vs fitted"),
        transform=ax_m_h.transAxes, fontsize=11, fontweight="bold",
        va="center", ha="left",
    )

    # ---- copula table (only for copula-using variants) ---------------
    if uses_copula:
        c_init   = init_mdl.copula_blocks()
        c_fitted = fitted_mdl.copula_blocks()

        # Index both lists by (i, j) so a re-ordered fitted block list
        # still aligns with the seed.
        def _key(blk: dict) -> tuple[int, int]:
            return (int(blk.get("i", -1)), int(blk.get("j", -1)))
        c_init_by   = {_key(b): b for b in c_init}
        c_fitted_by = {_key(b): b for b in c_fitted}
        keys = sorted(set(c_init_by) | set(c_fitted_by))

        c_rows: list[list[str]] = []
        c_cell_colors: list[list] = []
        for k in keys:
            bt = c_init_by  .get(k, {})
            bf = c_fitted_by.get(k, {})
            fam_t = bt.get("name", "—")
            fam_f = bf.get("name", "—")
            tau_t = float(bt.get("tau", float("nan")))
            tau_f = float(bf.get("tau", float("nan")))
            dtau  = tau_f - tau_t if (math.isfinite(tau_t)
                                      and math.isfinite(tau_f)) else float("nan")
            row = [
                f"({k[0]},{k[1]})",
                fam_t, _fmt_tau(tau_t),
                fam_f, _fmt_tau(tau_f),
                _fmt_signed(dtau),
            ]
            colours = ["white"] * len(row)
            if fam_t != fam_f:
                colours[3] = "#FFE9A8"
                colours[1] = "#FFF6D8"
            if math.isfinite(dtau) and abs(dtau) > 0.10:
                # Flag big τ shifts so the eye lands there first.
                colours[5] = "#F7B7B7"
            c_rows.append(row)
            c_cell_colors.append(colours)

        c_tbl = ax_c.table(
            cellText  = c_rows,
            colLabels = ["pair (i,j)",
                         "true family", r"true $\tau$",
                         "fitted family", r"fitted $\tau$",
                         r"$\Delta\tau$"],
            cellColours = c_cell_colors if c_cell_colors else None,
            colWidths = [0.10, 0.22, 0.13, 0.22, 0.13, 0.13],
            loc       = "upper center",
            cellLoc   = "center",
        )
        c_tbl.auto_set_font_size(False)
        c_tbl.set_fontsize(8)
        c_tbl.scale(1.0, 1.3)
        ax_c_h.text(
            0.0, 0.5, "Copulas — true vs fitted",
            transform=ax_c_h.transAxes, fontsize=11, fontweight="bold",
            va="center", ha="left",
        )

    # ---- joint prior heatmaps ----------------------------------------
    ax_pt  = fig.add_subplot(priors_g[0, 0])
    ax_pf  = fig.add_subplot(priors_g[0, 1])
    ax_pd  = fig.add_subplot(priors_g[0, 2])

    p_t = init_mdl  .prior_p
    p_f = fitted_mdl.prior_p
    p_d = p_f - p_t

    # Common colour scale for the two raw priors so they're visually
    # comparable; symmetric scale for the difference panel.
    # ``aspect="auto"`` lets the heatmap fill its grid slot (otherwise
    # ``aspect="equal"`` keeps cells square and leaves a vertical
    # gap above the row whenever the upper sections are short).
    vmax_p = max(p_t.max(), p_f.max(), 1e-9)
    for ax, mat, ttl, cmap, vmin, vmax in (
        (ax_pt, p_t, "true",   "viridis", 0.0,  vmax_p),
        (ax_pf, p_f, "fitted", "viridis", 0.0,  vmax_p),
        (ax_pd, p_d, "fitted − true",
                            "RdBu_r",  -np.max(np.abs(p_d)) - 1e-12,
                                        np.max(np.abs(p_d)) + 1e-12),
    ):
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax,
                       aspect="auto")
        for i in range(K):
            for j in range(K):
                val = mat[i, j]
                # Pick text colour by background luminance.
                txt_col = "white" if (cmap == "viridis"
                                      and val > 0.5 * vmax) else "black"
                ax.text(j, i, f"{val:+.3f}" if cmap == "RdBu_r"
                                            else f"{val:.3f}",
                        ha="center", va="center", fontsize=7,
                        color=txt_col)
        ax.set_xticks(range(K))
        ax.set_yticks(range(K))
        ax.set_xticklabels([rf"$j={k}$" for k in range(K)])
        ax.set_yticklabels([rf"$i={k}$" for k in range(K)])
        ax.set_title(ttl, fontsize=10, pad=2)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        r"Parameter comparison: true (init) vs ICE-fitted   "
        rf"$K={K}$, variant: {init_mdl.variant.value}"
    )
    # Colour-legend footer in its own gridspec row so it can't
    # overlap heatmap tick labels under constrained_layout. Unicode
    # "▮" renders as a vertical-bar swatch on every Qt-shipping
    # font — no mathtext gymnastics required.
    if uses_copula:
        footer = ("Cell tint:  ▮ yellow = family change   "
                  "·   ▮ pink = |Δτ| > 0.10")
    else:
        footer = "Cell tint:  ▮ yellow = family change"
    ax_footer.text(0.5, 0.5, footer,
                   transform=ax_footer.transAxes,
                   ha="center", va="center",
                   fontsize=8, color="#444444")

# ----- view J: γ before vs. after ICE ----------------------------------


def plot_ice_gamma_compare(fig: Figure, init_mdl: PMCModel, fitted_mdl: PMCModel, Y: np.ndarray):
    K  = init_mdl.K
    nn = np.arange(len(Y))

    ax1 = fig.add_subplot(2, 1, 1)
    _, gamma_i, ll_i = classify(init_mdl, Y)
    for k in range(K):
        ax1.plot(nn, gamma_i[:, k], lw=0.7, alpha=0.8,
                 label=rf"$\gamma(X{{=}}{k}\mid Y)$")
    ax1.set_title(rf"BEFORE ICE  ($\log L = {ll_i:.2f}$)")
    ax1.set_ylabel(r"$P(X_n = k \mid Y)$")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(2, 1, 2)
    _, gamma_f, ll_f = classify(fitted_mdl, Y)
    for k in range(K):
        ax2.plot(nn, gamma_f[:, k], lw=0.7, alpha=0.8,
                 label=rf"$\gamma(X{{=}}{k}\mid Y)$")
    ax2.set_title(
        rf"AFTER ICE  ($\log L = {ll_f:.2f}$, "
        rf"$\Delta\log L = {ll_f - ll_i:+.2f}$)"
    )
    ax2.set_xlabel(r"$n$")
    ax2.set_ylabel(r"$P(X_n = k \mid Y)$")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(); ax2.grid(True, alpha=0.3)

    fig.suptitle(r"Posterior marginals $\gamma$ — before vs. after ICE")




def compute_marginal_cdfs(model: PMCModel, Y: np.ndarray) -> np.ndarray:
    """Return ``f_cdf[n, i, j] = F_{ij}(Y[n])`` clipped to ``(EPS, 1-EPS)``.

    The views' clipping of :func:`pmcprg.diagnostics.pseudo.margin_cdfs`
    (writable array; a state model evaluates its K CDFs once and copies them
    over j — the same floats as evaluating ``margin(i, j)`` K² times).
    """
    return margin_cdfs(model, Y, clip=EPS)


# ---------------------------------------------------------------------------
# Colour palette for K classes
# ---------------------------------------------------------------------------

def _class_colors(n_classes: int) -> list:
    """Return a list of ``n_classes`` distinct RGBA colours from tab10.

    ``tab10`` is a discrete 10-colour palette — we index it modulo 10 so
    callers always get the canonical colours rather than interpolated
    intermediates (which would happen with a non-integer index).
    """
    cmap = plt.get_cmap("tab10")
    return [cmap(k % 10) for k in range(n_classes)]


# ---------------------------------------------------------------------------
# Formatting helpers used by the parameter-comparison view
# ---------------------------------------------------------------------------

def _fmt_value(v: float) -> str:
    """Compact float repr that uses a Unicode minus sign for readability."""
    if v != v:                              # NaN
        return "—"
    s = f"{v:.4g}" if abs(v) < 1e4 else f"{v:.3e}"
    return s.replace("-", "−")


def _fmt_params(params: dict, *, multiline: bool = False) -> str:
    """``{'loc': -3.05, 'scale': 0.98}`` → ``'loc=−3.05, scale=0.98'``.

    ``multiline=True`` joins with ``\\n`` so that long param lists fit
    in narrow table cells without truncation. The cell row grows
    vertically instead of overflowing horizontally.
    """
    if not params:
        return "—"
    sep = "\n" if multiline else ", "
    return sep.join(f"{k}={_fmt_param_value(v)}" for k, v in params.items())


def _fmt_param_value(v, *, max_items: int = 4) -> str:
    """Format one parameter value — scalar, vector or matrix.

    A multivariate margin carries its mean as a vector and its covariance as
    a nested list. The bare ``float(v)`` this replaced raised ``TypeError``
    on those, so *Parameters: true vs fitted* was permanently dead for every
    d ≥ 2 model: ``_render_view`` caught it, painted the red "render failed"
    card, and the failure was never diagnosed.

    A full covariance matrix cannot fit a table cell, so a 2-D value is
    reported by shape plus its diagonal — the part a reader checks first.
    """
    try:
        arr = np.asarray(v, dtype=float)
    except (TypeError, ValueError):
        return str(v)
    if arr.ndim == 0:
        return _fmt_value(float(arr))
    if arr.ndim == 1:
        head = ", ".join(_fmt_value(float(x)) for x in arr[:max_items])
        return f"[{head}{', …' if arr.size > max_items else ''}]"
    shape = "×".join(str(k) for k in arr.shape)
    diag  = np.diag(arr) if arr.ndim == 2 else arr.ravel()
    head  = ", ".join(_fmt_value(float(x)) for x in diag[:max_items])
    return f"{shape} diag[{head}{', …' if diag.size > max_items else ''}]"


def _fmt_tau(tau: float) -> str:
    """3-decimal τ with a Unicode minus and ``'—'`` for NaN."""
    if tau != tau:
        return "—"
    return f"{tau:.3f}".replace("-", "−")


def _fmt_signed(delta: float) -> str:
    """``+0.034`` / ``−0.012`` / ``—``  — explicit sign on Δ values."""
    if delta != delta:
        return "—"
    sign = "+" if delta >= 0 else "−"
    return f"{sign}{abs(delta):.3f}"

