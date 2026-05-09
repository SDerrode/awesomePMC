"""
cli.py — Command-line interface for PMC/HMC simulation, classification, and GUI.

Usage
-----
  python -m prg.pmc simulate       --model MODEL.toml [--N 5000] [--seed 42] [--out seq.csv]
  python -m prg.pmc classify       --model MODEL.toml --data seq.csv  [--out res.csv] [--ref ref_col]
  python -m prg.pmc estimate       --model INIT.toml  --data seq.csv  [--out fitted.toml]
  python -m prg.pmc classify-image --model MODEL.toml --image img.png [--out seg.png] [--ref-image ref.png]
  python -m prg.pmc estimate-image --model INIT.toml  --image img.png [--out fitted.toml]
  python -m prg.pmc gui [MODEL.toml]
  python -m prg.pmc --help

CSV formats
-----------
  simulate output:   n, X, Y
  classify input:    n, Y  (column 'n' optional; 'Y' required)
  classify output:   n, X_hat, gamma_0, gamma_1, ..., gamma_{K-1}

  If --ref <col> is given on 'classify', the reference labels column is used
  to compute and print the classification error rate.

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

from prg.exceptions import CSVLoadError

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
    Y     : np.ndarray, shape (N,) — observations.
    X_ref : np.ndarray, shape (N,) or None — reference labels if available.

    Raises
    ------
    CSVLoadError
        If the file is missing, empty, or has no ``Y`` column.
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

    Y = np.array([float(r["Y"]) for r in rows])

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
# Sub-command: simulate
# ---------------------------------------------------------------------------

def cmd_simulate(args: argparse.Namespace) -> int:
    from prg.pmc.model    import PMCModel
    from prg.pmc.simulate import simulate

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
    print(f"  Model   : {mdl.name}  ({mdl.variant.value}, K={mdl.K})")
    print(f"  X unique : {np.unique(X).tolist()}")
    print(f"  Y mean   : {Y.mean():.4f}   Y std : {Y.std():.4f}")
    return 0


# ---------------------------------------------------------------------------
# Sub-command: classify
# ---------------------------------------------------------------------------

def cmd_classify(args: argparse.Namespace) -> int:
    from prg.pmc.model     import PMCModel
    from prg.pmc.inference import classify, error_rate

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
    print(f"Model     : {mdl.name}  ({mdl.variant.value}, K={K})")
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
    from prg.pmc.model     import PMCModel
    from prg.pmc.ice       import ice
    from prg.pmc.inference import classify, error_rate

    mdl = PMCModel(args.model)

    try:
        Y, X_ref = _read_observations(args.data, args.ref)
    except CSVLoadError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # Build ICE config from CLI overrides
    cfg: dict = {}
    if args.max_iter is not None:
        cfg["max_iter"] = args.max_iter
    if args.candidates:
        cfg["candidates"] = args.candidates.split(",")
    if args.fit_margins:
        cfg["fit_margins"] = True

    print(f"Running ICE on {mdl.name}  ({mdl.variant.value}, K={mdl.K}, N={len(Y)}) …")
    fitted, trace = ice(mdl, Y, ice_cfg=cfg)
    lls = trace.log_liks

    print(f"  Iterations    : {len(lls)}")
    print(f"  Final log-lik : {lls[-1]:.4f}")
    print(f"  LL history    : {[round(ll, 2) for ll in lls]}")

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
    from prg.pmc.model     import PMCModel
    from prg.pmc.inference import classify_image, error_rate
    from prg.pmc.peano     import load_grayscale, load_color, save_segmentation

    mdl = PMCModel(args.model)
    K   = mdl.K

    if mdl.d > 1 or args.color:
        img = load_color(args.image)
    else:
        img = load_grayscale(args.image)
    H, W = img.shape[:2]

    X_hat_2d, gamma_2d, log_lik = classify_image(mdl, img)

    print(f"Model      : {mdl.name}  ({mdl.variant.value}, K={K})")
    print(f"Image      : {args.image}  ({H}×{W} = {H * W} px)")
    print(f"Log-lik    : {log_lik:.4f}")

    if args.ref_image:
        from prg.pmc.peano import load_labels
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
    from prg.pmc.model     import PMCModel
    from prg.pmc.ice       import ice_image
    from prg.pmc.inference import classify_image, error_rate
    from prg.pmc.peano     import load_grayscale, load_color

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

    print(
        f"Running ICE on image {args.image} ({H}×{W}) "
        f"with {mdl.name} ({mdl.variant.value}, K={mdl.K}) …"
    )
    fitted, trace = ice_image(mdl, img, ice_cfg=cfg)
    lls = trace.log_liks

    print(f"  Iterations    : {len(lls)}")
    print(f"  Final log-lik : {lls[-1]:.4f}")
    print(f"  LL history    : {[round(ll, 2) for ll in lls]}")

    if args.ref_image:
        from prg.pmc.peano import load_labels
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
    parser = argparse.ArgumentParser(
        prog="python -m prg.pmc",
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

    # ── estimate (ICE) ────────────────────────────────────────────────────
    p_est = sub.add_parser(
        "estimate",
        help="Unsupervised ICE parameter estimation from an observation sequence.",
    )
    p_est.add_argument(
        "--model", "-m", required=True, metavar="INIT.toml",
        help="Path to the initial TOML model file (starting point for ICE).",
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
        "--max-iter", dest="max_iter", type=int, default=None, metavar="N",
        help="Maximum ICE iterations (overrides TOML [ice] section).",
    )
    p_est.add_argument(
        "--candidates", default=None, metavar="C1,C2,...",
        help="Comma-separated copula SHORT_NAMEs to try (e.g. 'Gauss,Clayton,GH').",
    )
    p_est.add_argument(
        "--fit-margins", dest="fit_margins", action="store_true",
        help="Also re-estimate margin parameters (Gaussian only).",
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

    # ── estimate-image (ICE on image) ─────────────────────────────────────
    p_esti = sub.add_parser(
        "estimate-image",
        help="Unsupervised ICE estimation directly from a 2D image.",
    )
    p_esti.add_argument(
        "--model", "-m", required=True, metavar="INIT.toml",
        help="Path to the initial TOML model file (starting point for ICE).",
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
        "--max-iter", dest="max_iter", type=int, default=None, metavar="N",
        help="Maximum ICE iterations (overrides TOML [ice] section).",
    )
    p_esti.add_argument(
        "--candidates", default=None, metavar="C1,C2,...",
        help="Comma-separated copula SHORT_NAMEs to try (e.g. 'Gauss,Clayton,GH').",
    )
    p_esti.add_argument(
        "--fit-margins", dest="fit_margins", action="store_true",
        help="Also re-estimate margin parameters (Gaussian only).",
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
    from prg.pmc.gui.app import run_gui
    return run_gui(args.model)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args   = parser.parse_args(argv)

    # Use centralised logging (console level controlled by --verbose;
    # file always at DEBUG so full tracebacks are preserved)
    from prg.pmc.logging_setup import configure
    console_level = logging.DEBUG if args.verbose else logging.WARNING
    log_file = configure(level=console_level)
    if log_file and args.verbose:
        print(f"[logging] Appending to {log_file}", flush=True)

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
