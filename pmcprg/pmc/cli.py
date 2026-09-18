"""
cli.py — Command-line interface for PMC/HMC simulation, classification, and GUI.

Usage
-----
  python -m pmcprg.pmc simulate       --model MODEL.toml [--N 5000] [--seed 42] [--out seq.csv]
  python -m pmcprg.pmc classify       --model MODEL.toml --data seq.csv  [--out res.csv] [--ref ref_col]
  python -m pmcprg.pmc estimate       --model INIT.toml  --data seq.csv  [--out fitted.toml]
  python -m pmcprg.pmc classify-image --model MODEL.toml --image img.png [--out seg.png] [--ref-image ref.png]
  python -m pmcprg.pmc estimate-image --model INIT.toml  --image img.png [--out fitted.toml]
  python -m pmcprg.pmc gui [MODEL.toml]
  python -m pmcprg.pmc --help

CSV formats
-----------
  simulate output:   n, X, Y
  classify input:    n, Y  (column 'n' optional; 'Y' required)
  classify output:   n, X_hat, gamma_0, gamma_1, ..., gamma_{K-1}

  If --ref <col> is given on 'classify', the reference labels column is used
  to compute and print the classification error rate.

Margin structure
----------------
  Every command accepts both margin structures of PMCModel: state margins f_i
  and pair margins f_ij, the law of y_n given (x_n = i, x_{n+1} = j) — the
  general PMC of DerrodePieczynski_CSDA2013 Eqs. 12–14 (Derrode & Pieczynski
  2013, CSDA 63:81–98). The model line of the summary says "pair margins f_ij"
  for the latter. Should the estimator fail on a pair model, 'estimate' /
  'estimate-image' print the error on one line and exit with status 2
  (traceback in the log file) rather than raising.

Image I/O
---------
  classify-image / estimate-image accept any format Pillow decodes (PNG,
  JPG, BMP, TIFF, …). Images are converted to grayscale (PIL ``L`` mode)
  and linearised along the Generalized Hilbert ("gilbert") path before
  being passed to the same forward-backward / ICE machinery as 1D signals.
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np

from pmcprg.exceptions import CSVLoadError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared CSV reader
# ---------------------------------------------------------------------------


def _read_observations(
    data_path_str: str,
    ref_col: str | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Read an observation CSV.

    Parameters
    ----------
    data_path_str : str
        Path to the CSV file. Must contain a column named ``Y``.
    ref_col : str, optional
        Name of an integer reference-label column (e.g. ``X``). If the column
        is absent, a warning is printed to stderr and ``None`` is returned.
        Pass ``""`` or ``None`` to skip reference-label loading.

    Returns
    -------
    Y     : np.ndarray, shape (N,) — observations, NaN where missing.
    X_ref : np.ndarray, shape (N,) or None — reference labels if available.

    Missing observations — an empty ``Y`` cell, ``NaN``/``nan`` or ``NA``
    (:data:`pmcprg.missing.cells.MISSING_TOKENS`) — load as float NaN, and a
    warning gives their count and share.

    Raises
    ------
    CSVLoadError
        If the file is missing, empty, has no ``Y`` column, holds a
        non-numeric ``Y`` cell (the row is named), or has no observed ``Y``
        value at all.
    """
    data_path = Path(data_path_str)
    if not data_path.exists():
        raise CSVLoadError(f"data file not found: {data_path}")

    rows: list[dict] = []
    with open(data_path, newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)

    if not rows:
        raise CSVLoadError(f"{data_path} contains no rows.")
    if "Y" not in rows[0]:
        raise CSVLoadError("data CSV must contain a column named 'Y'.")

    from pmcprg.missing.cells import parse_cell

    values: list[float] = []
    for n, r in enumerate(rows):
        try:
            values.append(parse_cell(r["Y"]))
        except (TypeError, ValueError):
            raise CSVLoadError(
                f"non-numeric Y at row {n + 2} (CSV line {n + 2}) of "
                f"{data_path}: {r['Y']!r}"
            ) from None
    Y = np.array(values)

    n_missing = int(np.isnan(Y).sum())
    if n_missing == len(Y):
        raise CSVLoadError(
            f"{data_path}: every Y value is missing ({len(Y)} of {len(Y)} "
            f"rows empty, NaN or NA); at least one observed value is required."
        )
    if n_missing:
        logger.warning(
            "%s: %d of %d Y values missing (%.1f %%) — kept as NaN.",
            data_path, n_missing, len(Y), 100.0 * n_missing / len(Y),
        )

    X_ref: np.ndarray | None = None
    if ref_col:
        if ref_col in rows[0]:
            X_ref = np.array([int(r[ref_col]) for r in rows])
        else:
            print(
                f"WARNING: --ref column {ref_col!r} not found in data; ignored.",
                file=sys.stderr,
            )

    return Y, X_ref


# ---------------------------------------------------------------------------
# Model summary and estimator guard (pair margins — DerrodePieczynski_CSDA2013 Eqs. 12–14)
# ---------------------------------------------------------------------------

#: Exit status of 'estimate' / 'estimate-image' when the estimator fails on a
#: model with pair-indexed margins.
EXIT_PAIR_ESTIMATION_FAILED = 2


def _is_pair(mdl) -> bool:
    return getattr(mdl, "margin_structure", "state") == "pair"


def _model_tag(mdl, **extra) -> str:
    """``"PMC, K=2"`` — plus ``", pair margins f_ij"`` on a pair model."""
    parts = [mdl.variant.value, f"K={mdl.K}"]
    parts += [f"{k}={v}" for k, v in extra.items()]
    if _is_pair(mdl):
        parts.append("pair margins f_ij")
    return ", ".join(parts)


def _run_estimator(label: str, mdl, run, Y=None):
    """Call ``run()``, turning expected failures into a one-line error.

    * Observations with missing values (NaN): the estimators handle them
      (observed-data likelihood — see :func:`pmcprg.pmc.ice.ice` and
      :func:`pmcprg.pmc.sem.sem`, section "Missing observations"); a one-line
      WARNING with their count is printed before ``run()``.
    * On a model with pair margins f_ij any failure is reported in one line
      with the model structure named, the traceback going to the log file
      only. State models otherwise propagate the exception as before.

    Returns ``run()``'s result, or ``None`` after printing the error.
    """
    if Y is not None:
        n_missing = int(np.count_nonzero(~np.isfinite(np.asarray(Y, dtype=float))))
        if n_missing:
            sys.stdout.flush()
            print(
                f"WARNING: {label} on {mdl.name!r}: {n_missing} missing value(s) out "
                f"of {np.asarray(Y).size} — estimated from the observed-data "
                "likelihood (missing values integrated out).",
                file=sys.stderr,
            )
    if not _is_pair(mdl):
        return run()
    try:
        return run()
    except Exception as exc:
        logger.debug("%s failed on a pair-margin model", label, exc_info=True)
        sys.stdout.flush()                     # keep "Running …" above the error
        print(
            f"ERROR: {label} failed on {mdl.name!r} ({_model_tag(mdl)}): "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return None


def _print_fit_summary(trace) -> None:
    """Iterations and log-likelihoods of an ICE/SEM trace, on stdout.

    Adds a ``Returned iter`` line when the model returned is an earlier
    iterate than the last one (``return_best_iterate``), and a
    ``Degenerate`` line when the fitted model has degenerate states — whose
    detail is the estimator's WARNING on stderr.
    """
    lls = trace.log_liks
    print(f"  Iterations    : {len(lls)}")
    print(f"  Final log-lik : {lls[-1]:.4f}")
    returned = getattr(trace, "returned_iter", -1)
    if 0 <= returned < len(lls) - 1:
        print(f"  Returned iter : {returned}  (best iterate, log-lik {lls[returned]:.4f})")
    print(f"  LL history    : {[round(ll, 2) for ll in lls]}")
    degenerate = getattr(trace, "degenerate", None)
    if degenerate:
        where = ", ".join(dict.fromkeys(f.where for f in degenerate))
        print(f"  Degenerate    : {where}  (see the WARNING on stderr)")


# ---------------------------------------------------------------------------
# Sub-command: simulate
# ---------------------------------------------------------------------------

def cmd_simulate(args: argparse.Namespace) -> int:
    from pmcprg.pmc.model    import PMCModel
    from pmcprg.pmc.simulate import simulate

    mdl = PMCModel(args.model)
    N   = args.N if args.N is not None else mdl.N_default
    X, Y = simulate(mdl, N=N, seed=args.seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["n", "X", "Y"])
        for n in range(N):
            writer.writerow([n + 1, int(X[n]), float(Y[n])])

    print(f"Simulated {N} samples  →  {out}")
    print(f"  Model   : {mdl.name}  ({_model_tag(mdl)})")
    print(f"  X unique : {np.unique(X).tolist()}")
    print(f"  Y mean   : {Y.mean():.4f}   Y std : {Y.std():.4f}")
    return 0


# ---------------------------------------------------------------------------
# Sub-command: classify
# ---------------------------------------------------------------------------

def cmd_classify(args: argparse.Namespace) -> int:
    from pmcprg.pmc.model     import PMCModel
    from pmcprg.pmc.inference import classify, error_rate

    mdl = PMCModel(args.model)
    K   = mdl.K

    try:
        Y, X_ref = _read_observations(args.data, args.ref)
    except CSVLoadError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    N = len(Y)

    # ── classify ──────────────────────────────────────────────────────────
    X_hat, gamma, log_lik = classify(mdl, Y)

    # ── print summary ─────────────────────────────────────────────────────
    print(f"Model     : {mdl.name}  ({_model_tag(mdl)})")
    print(f"Sequence  : N={N}")
    print(f"Log-lik   : {log_lik:.4f}")
    if X_ref is not None:
        er = error_rate(X_ref, X_hat)
        print(f"Error rate: {er:.4f}  ({er * 100:.1f} %)")

    # ── write results ─────────────────────────────────────────────────────
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = ["n", "X_hat"] + [f"gamma_{k}" for k in range(K)]
    with open(out, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for n in range(N):
            row_ = [n + 1, int(X_hat[n])] + [f"{gamma[n, k]:.8f}" for k in range(K)]
            writer.writerow(row_)

    print(f"Results   : {out}")
    return 0


# ---------------------------------------------------------------------------
# Sub-command: estimate  (ICE unsupervised)
# ---------------------------------------------------------------------------

def cmd_estimate(args: argparse.Namespace) -> int:
    from pmcprg.pmc.ice       import ice
    from pmcprg.pmc.inference import classify, error_rate
    from pmcprg.pmc.model     import PMCModel
    from pmcprg.pmc.sem       import sem

    mdl = PMCModel(args.model)

    try:
        Y, X_ref = _read_observations(args.data, args.ref)
    except CSVLoadError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # Build estimator config from CLI overrides
    cfg: dict = {}
    if args.max_iter is not None:
        cfg["max_iter"] = args.max_iter
    if args.candidates:
        cfg["candidates"] = args.candidates.split(",")
    if args.fit_margins:
        cfg["fit_margins"] = True
    if getattr(args, "sem_seed", None) is not None:
        cfg["sem_seed"] = args.sem_seed
    if getattr(args, "best_iterate", False):
        cfg["return_best_iterate"] = True

    algorithm = getattr(args, "algorithm", "ice")
    label     = algorithm.upper()
    print(f"Running {label} on {mdl.name}  ({_model_tag(mdl, N=len(Y))}) …")
    if algorithm == "sem":
        result = _run_estimator(label, mdl, lambda: sem(mdl, Y, sem_cfg=cfg), Y=Y)
    else:
        result = _run_estimator(label, mdl, lambda: ice(mdl, Y, ice_cfg=cfg), Y=Y)
    if result is None:
        return EXIT_PAIR_ESTIMATION_FAILED
    fitted, trace = result
    _print_fit_summary(trace)

    if X_ref is not None:
        X_hat, _, _ = classify(fitted, Y)
        er = error_rate(X_ref, X_hat)
        print(f"  Error rate    : {er:.4f}  ({er * 100:.1f} %)")

    fitted.save(args.out)
    print(f"  Fitted model  : {args.out}")
    return 0


# ---------------------------------------------------------------------------
# Sub-command: classify-image
# ---------------------------------------------------------------------------

def cmd_classify_image(args: argparse.Namespace) -> int:
    from pmcprg.pmc.model     import PMCModel
    from pmcprg.pmc.inference import classify_image, error_rate
    from pmcprg.pmc.peano     import load_grayscale, load_color, save_segmentation

    mdl = PMCModel(args.model)
    K   = mdl.K

    if mdl.d > 1 or args.color:
        img = load_color(args.image)
    else:
        img = load_grayscale(args.image)
    H, W = img.shape[:2]

    X_hat_2d, gamma_2d, log_lik = classify_image(mdl, img)

    print(f"Model      : {mdl.name}  ({_model_tag(mdl)})")
    print(f"Image      : {args.image}  ({H}×{W} = {H * W} px)")
    print(f"Log-lik    : {log_lik:.4f}")

    if args.ref_image:
        from pmcprg.pmc.peano import load_labels
        ref_labels = load_labels(args.ref_image)
        if ref_labels.shape != (H, W):
            print(
                f"WARNING: --ref-image shape {ref_labels.shape} ≠ image spatial "
                f"shape ({H}, {W}); ignored.",
                file=sys.stderr,
            )
        else:
            er = error_rate(ref_labels.ravel(), X_hat_2d.ravel())
            print(f"Error rate : {er:.4f}  ({er * 100:.1f} %)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_segmentation(out, X_hat_2d.astype(np.int32), K=K)
    print(f"Segmentation : {out}")
    return 0


# ---------------------------------------------------------------------------
# Sub-command: estimate-image  (ICE on an image)
# ---------------------------------------------------------------------------

def cmd_estimate_image(args: argparse.Namespace) -> int:
    from pmcprg.pmc.ice       import ice_image
    from pmcprg.pmc.inference import classify_image, error_rate
    from pmcprg.pmc.model     import PMCModel
    from pmcprg.pmc.peano     import load_color, load_grayscale
    from pmcprg.pmc.sem       import sem_image

    mdl = PMCModel(args.model)
    if mdl.d > 1 or args.color:
        img = load_color(args.image)
    else:
        img = load_grayscale(args.image)
    H, W = img.shape[:2]

    cfg: dict = {}
    if args.max_iter is not None:
        cfg["max_iter"] = args.max_iter
    if args.candidates:
        cfg["candidates"] = args.candidates.split(",")
    if args.fit_margins:
        cfg["fit_margins"] = True
    if getattr(args, "sem_seed", None) is not None:
        cfg["sem_seed"] = args.sem_seed
    if getattr(args, "best_iterate", False):
        cfg["return_best_iterate"] = True

    algorithm = getattr(args, "algorithm", "ice")
    label     = algorithm.upper()
    print(
        f"Running {label} on image {args.image} ({H}×{W}) "
        f"with {mdl.name} ({_model_tag(mdl)}) …"
    )
    if algorithm == "sem":
        result = _run_estimator(label, mdl,
                                lambda: sem_image(mdl, img, sem_cfg=cfg), Y=img)
    else:
        result = _run_estimator(label, mdl,
                                lambda: ice_image(mdl, img, ice_cfg=cfg), Y=img)
    if result is None:
        return EXIT_PAIR_ESTIMATION_FAILED
    fitted, trace = result
    _print_fit_summary(trace)

    if args.ref_image:
        from pmcprg.pmc.peano import load_labels
        ref_labels = load_labels(args.ref_image)
        if ref_labels.shape == (H, W):
            X_hat_2d, _, _ = classify_image(fitted, img)
            er = error_rate(ref_labels.ravel(), X_hat_2d.ravel())
            print(f"  Error rate    : {er:.4f}  ({er * 100:.1f} %)")
        else:
            print(
                f"WARNING: --ref-image shape {ref_labels.shape} ≠ image "
                f"spatial shape ({H}, {W}); ignored.",
                file=sys.stderr,
            )

    fitted.save(args.out)
    print(f"  Fitted model  : {args.out}")
    return 0


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    # Usage lines name the command the user typed: the ``pmc`` console script,
    # or ``python -m pmcprg.pmc`` (argv[0] is then the package's __main__.py).
    invoked_as_module = Path(sys.argv[0]).name == "__main__.py"
    parser = argparse.ArgumentParser(
        prog="python -m pmcprg.pmc" if invoked_as_module else "pmc",
        description="PMC/HMC model simulation, classification, and unsupervised estimation.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # ── simulate ──────────────────────────────────────────────────────────
    p_sim = sub.add_parser(
        "simulate",
        help="Generate a synthetic (X, Y) sequence from a TOML model.",
    )
    p_sim.add_argument(
        "--model", "-m", required=True, metavar="MODEL.toml",
        help="Path to the TOML model file.",
    )
    p_sim.add_argument(
        "--N", type=int, default=None, metavar="N",
        help="Sequence length (default: N_default from TOML).",
    )
    p_sim.add_argument(
        "--seed", type=int, default=None, metavar="SEED",
        help="Random seed for reproducibility.",
    )
    p_sim.add_argument(
        "--out", "-o", default="simulation.csv", metavar="OUT.csv",
        help="Output CSV path (default: simulation.csv).",
    )
    p_sim.set_defaults(func=cmd_simulate)

    # ── classify ──────────────────────────────────────────────────────────
    p_cls = sub.add_parser(
        "classify",
        help="Supervised MPM classification of an observation sequence.",
    )
    p_cls.add_argument(
        "--model", "-m", required=True, metavar="MODEL.toml",
        help="Path to the TOML model file.",
    )
    p_cls.add_argument(
        "--data", "-d", required=True, metavar="DATA.csv",
        help="CSV file with the observation sequence (must contain column 'Y').",
    )
    p_cls.add_argument(
        "--out", "-o", default="classification.csv", metavar="OUT.csv",
        help="Output CSV path for (X_hat, gamma) (default: classification.csv).",
    )
    p_cls.add_argument(
        "--ref", default="X", metavar="COL",
        help="Column name for reference labels (used to compute error rate). "
             "Default: 'X'. Pass '' to skip.",
    )
    p_cls.set_defaults(func=cmd_classify)

    # ── estimate (ICE / SEM) ──────────────────────────────────────────────
    p_est = sub.add_parser(
        "estimate",
        help="Unsupervised parameter estimation from an observation sequence "
             "(ICE — deterministic — or SEM — stochastic).",
    )
    p_est.add_argument(
        "--model", "-m", required=True, metavar="INIT.toml",
        help="Path to the initial TOML model file (starting point).",
    )
    p_est.add_argument(
        "--data", "-d", required=True, metavar="DATA.csv",
        help="CSV file with the observation sequence (must contain column 'Y').",
    )
    p_est.add_argument(
        "--out", "-o", default="fitted.toml", metavar="OUT.toml",
        help="Output TOML path for the fitted model (default: fitted.toml).",
    )
    p_est.add_argument(
        "--algorithm", choices=("ice", "sem"), default="ice",
        help="Estimation algorithm. 'ice' (default, deterministic) or 'sem' "
             "(Stochastic EM — draws X̃ ~ P(X|Y) at each iteration).",
    )
    p_est.add_argument(
        "--max-iter", dest="max_iter", type=int, default=None, metavar="N",
        help="Maximum iterations (overrides TOML [ice]/[sem] section).",
    )
    p_est.add_argument(
        "--candidates", default=None, metavar="C1,C2,...",
        help="Comma-separated copula SHORT_NAMEs to try (e.g. 'Gauss,Clayton,GH').",
    )
    p_est.add_argument(
        "--fit-margins", dest="fit_margins", action="store_true",
        help="Also re-estimate margin parameters.",
    )
    p_est.add_argument(
        "--sem-seed", dest="sem_seed", type=int, default=None, metavar="N",
        help="RNG seed for SEM's stochastic completion (ignored when --algorithm=ice).",
    )
    p_est.add_argument(
        "--best-iterate", dest="best_iterate", action="store_true",
        help="Return the iterate with the highest log-likelihood instead of the "
             "last one (config key return_best_iterate).",
    )
    p_est.add_argument(
        "--ref", default="X", metavar="COL",
        help="Reference label column for error rate (default: 'X'). Pass '' to skip.",
    )
    p_est.set_defaults(func=cmd_estimate)

    # ── classify-image ────────────────────────────────────────────────────
    p_clsi = sub.add_parser(
        "classify-image",
        help="MPM classification of a 2D image (linearised via gilbert curve).",
    )
    p_clsi.add_argument(
        "--model", "-m", required=True, metavar="MODEL.toml",
        help="Path to the TOML model file.",
    )
    p_clsi.add_argument(
        "--image", "-i", required=True, metavar="IMG",
        help="Input image (PNG/JPG/BMP/TIFF — converted to grayscale).",
    )
    p_clsi.add_argument(
        "--out", "-o", default="segmentation.png", metavar="OUT.png",
        help="Output PNG path for the class-label map (default: segmentation.png).",
    )
    p_clsi.add_argument(
        "--ref-image", default=None, metavar="REF",
        help="Optional reference label image (greyscale; intensities are "
             "rescaled to {0,…,K-1}) used to compute the error rate.",
    )
    p_clsi.add_argument(
        "--color", action="store_true",
        help="Force RGB load (default: auto — RGB if model.d > 1 else grayscale).",
    )
    p_clsi.set_defaults(func=cmd_classify_image)

    # ── estimate-image (ICE / SEM on image) ───────────────────────────────
    p_esti = sub.add_parser(
        "estimate-image",
        help="Unsupervised estimation directly from a 2D image (ICE or SEM).",
    )
    p_esti.add_argument(
        "--model", "-m", required=True, metavar="INIT.toml",
        help="Path to the initial TOML model file (starting point).",
    )
    p_esti.add_argument(
        "--image", "-i", required=True, metavar="IMG",
        help="Input image (PNG/JPG/BMP/TIFF — converted to grayscale).",
    )
    p_esti.add_argument(
        "--out", "-o", default="fitted.toml", metavar="OUT.toml",
        help="Output TOML path for the fitted model (default: fitted.toml).",
    )
    p_esti.add_argument(
        "--algorithm", choices=("ice", "sem"), default="ice",
        help="Estimation algorithm. 'ice' (default) or 'sem' (Stochastic EM).",
    )
    p_esti.add_argument(
        "--max-iter", dest="max_iter", type=int, default=None, metavar="N",
        help="Maximum iterations (overrides TOML [ice]/[sem] section).",
    )
    p_esti.add_argument(
        "--candidates", default=None, metavar="C1,C2,...",
        help="Comma-separated copula SHORT_NAMEs to try (e.g. 'Gauss,Clayton,GH').",
    )
    p_esti.add_argument(
        "--fit-margins", dest="fit_margins", action="store_true",
        help="Also re-estimate margin parameters.",
    )
    p_esti.add_argument(
        "--sem-seed", dest="sem_seed", type=int, default=None, metavar="N",
        help="RNG seed for SEM's stochastic completion (ignored when --algorithm=ice).",
    )
    p_esti.add_argument(
        "--best-iterate", dest="best_iterate", action="store_true",
        help="Return the iterate with the highest log-likelihood instead of the "
             "last one (config key return_best_iterate).",
    )
    p_esti.add_argument(
        "--ref-image", default=None, metavar="REF",
        help="Optional reference label image used to print the post-fit error rate.",
    )
    p_esti.add_argument(
        "--color", action="store_true",
        help="Force RGB load (default: auto — RGB if model.d > 1 else grayscale).",
    )
    p_esti.set_defaults(func=cmd_estimate_image)

    # ── gui ───────────────────────────────────────────────────────────────
    p_gui = sub.add_parser(
        "gui",
        help="Launch the PyQt6 graphical interface.",
    )
    p_gui.add_argument(
        "model", nargs="?", default=None, metavar="MODEL.toml",
        help="Optional TOML model to load at startup.",
    )
    p_gui.set_defaults(func=cmd_gui)

    return parser


def cmd_gui(args: argparse.Namespace) -> int:
    from pmcprg.pmc.gui.app import run_gui
    return run_gui(args.model)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args   = parser.parse_args(argv)

    # Use centralised logging (console level controlled by --verbose;
    # file always at DEBUG so full tracebacks are preserved)
    from pmcprg.pmc.logging_setup import configure
    console_level = logging.DEBUG if args.verbose else logging.WARNING
    log_file = configure(level=console_level)
    if log_file and args.verbose:
        print(f"[logging] Appending to {log_file}", flush=True)

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
