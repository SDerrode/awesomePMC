"""
pmcprg.copulas._fit_diagnostics — per-fit convergence diagnostics (audit AUDIT_COPULES FR-12).

What a fit reports about its own optimum, in the spirit of the fits of
Copula.Markov and of GJRM's ``conv.check()``: the gradient of the (weighted)
log-likelihood at the estimate, the eigenvalues of its Hessian, whether the
estimate sits on the boundary of the parameter space, and the optimiser's
iteration and evaluation counts. :attr:`pmcprg.copulas.FitResult.diagnostics`
returns a :class:`FitDiagnostics`.

Cost
----
Nothing is computed during the fit. The optimiser's counts and message are
stored as it returns them; the gradient and the Hessian are computed on the
first access to ``FitResult.diagnostics`` (then cached), with
``1 + 2p + 2p(p − 1)`` evaluations of the log-likelihood for ``p`` free
parameters — 3 for one parameter, 9 for two, 19 for three. ICE's M-step never
builds a ``FitResult`` and pays nothing.

What is differentiated
----------------------
``ℓ(p) = Σᵢ wᵢ log c(uᵢ, vᵢ; p)`` (``wᵢ = 1`` for an unweighted fit) — the
quantity every fit of the package maximises and reports — on the **parameter
scale** of the copula's constructor: ``p = (τ, extra parameters…)`` in the
registry's order (``tau_k``; Student ``df``; BB1 ``delta``; …). Central
differences with the step rule of the FR-4 standard errors
(:func:`pmcprg.copulas._stderr._central_step`: 10⁻⁴ of the distance to the
nearer end of the coordinate's range, capped at 10⁻⁴); the range is the
registered τ-range cut to the τ the family reaches, and the registered box of
each extra parameter (``EXTRA_PARAM_BOUNDS_BY_PARAM``). Where a step leaves
the admissible set (a jointly constrained family next to its constraint, or a
coordinate on an end of its range), the difference is taken one-sided on the
other side; where both sides fail, the entry is NaN.

At an interior maximum the gradient is zero to the optimiser's tolerance and
the Hessian is negative definite. Its eigenvalues depend on the
parametrisation, their **signs** do not (Sylvester's law of inertia), so
``negative_definite`` is a property of the fit. ``newton_decrement`` —
½ gᵀ(−H)⁻¹g, the log-likelihood a Newton step would still gain — is the
parametrisation-free measure of how far the optimiser stopped from the
maximum, in nats.

The inversion of Kendall's τ (``method='tau'``) is not a maximiser of ℓ: its
gradient in τ is not zero, and is reported as it is.

Boundary
--------
``at_boundary`` is set, with one sentence per condition in ``boundary``:

* for the one-parameter families, Student and BB1, by the rules of the FR-4
  standard errors (:func:`pmcprg.copulas._stderr._spec_of`): τ̂ within two
  optimiser pads of an end of its registered range, on the τ its family
  reaches (Frank, Plackett), Student's ν̂ within a relative 10⁻⁴ of an end of
  its fitting box, BB1 δ̂ = 1 or θ̂ on its floor — so that ``at_boundary``
  agrees with :meth:`FitResult.standard_errors`;
* for every other family: τ̂ as above; an extra parameter within a relative
  10⁻⁴ (``_BOUNDARY_REL``) of an end of its registered box; or next to the
  edge of the admissible set at the fitted τ — the constructor refuses the
  parameter moved by that relative 10⁻⁴ (BB6 δ ≤ 1/(1 − τ), BB7 θ < θ_Joe(τ),
  Tawn ψ > τ, …);
* for every family, when a finite-difference step had to be one-sided or
  failed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["FitDiagnostics", "fit_diagnostics"]


@dataclass(frozen=True)
class FitDiagnostics:
    """Convergence diagnostics of one fit — see the module docstring.

    Fields
    ------
    names       : the free parameters, on the constructor's scale
                  (``('tau_k',)``, ``('tau_k', 'df')``, …); empty for the
                  Product copula.
    estimate    : their fitted values.
    gradient    : ∂ℓ/∂p at the estimate (ℓ the weighted log-likelihood).
    hessian     : ∂²ℓ/∂p∂pᵀ at the estimate (NaN entries where no difference
                  could be taken).
    eigenvalues : eigenvalues of ``hessian``, ascending (NaN if it has a NaN).
    at_boundary : the estimate is on, or within the documented tolerance of,
                  the boundary of its admissible range (module docstring).
    boundary    : one sentence per boundary condition met.
    converged   : the optimiser's own verdict (``FitResult.converged``).
    message     : the optimiser's message (``FitResult.message``).
    n_iter      : optimiser iterations (``None`` when not reported; 0 for a
                  moment estimate).
    n_eval      : log-likelihood evaluations by the optimiser (same).
    steps       : the finite-difference step of each coordinate.
    """

    names: tuple
    estimate: np.ndarray
    gradient: np.ndarray
    hessian: np.ndarray
    eigenvalues: np.ndarray
    at_boundary: bool
    boundary: tuple
    converged: bool
    message: str
    n_iter: int | None
    n_eval: int | None
    steps: np.ndarray

    @property
    def negative_definite(self) -> bool:
        """Every eigenvalue of the Hessian is finite and < 0 — an interior maximum."""
        eig = self.eigenvalues
        return bool(eig.size > 0 and np.all(np.isfinite(eig)) and np.all(eig < 0.0))

    @property
    def max_abs_gradient(self) -> float:
        """``max |∂ℓ/∂p|`` — GJRM's ``conv.check()`` headline number (NaN if undefined)."""
        g = self.gradient
        return float(np.max(np.abs(g))) if g.size else 0.0

    @property
    def newton_decrement(self) -> float:
        """½ gᵀ(−H)⁻¹g in nats: the log-likelihood a Newton step would still gain.

        Invariant under reparametrisation. NaN unless the Hessian is negative
        definite.
        """
        if not self.gradient.size:
            return 0.0
        if not self.negative_definite:
            return float("nan")
        step = np.linalg.solve(-self.hessian, self.gradient)
        return float(0.5 * self.gradient @ step)

    def summary(self) -> str:
        """A few lines in the manner of GJRM's ``conv.check()``."""
        lines = [f"converged: {self.converged} ({self.message})" if self.message
                 else f"converged: {self.converged}"]
        if self.n_iter is not None or self.n_eval is not None:
            lines.append(f"iterations: {self.n_iter}, likelihood evaluations: {self.n_eval}")
        if self.names:
            lines.append("maximum absolute gradient: " f"{self.max_abs_gradient:.3g} "
                         f"({', '.join(f'{n}: {g:.3g}' for n, g in zip(self.names, self.gradient))})")
            lines.append("Hessian eigenvalues: "
                         + ", ".join(f"{e:.4g}" for e in self.eigenvalues)
                         + (" (negative definite)" if self.negative_definite
                            else " (not negative definite)"))
            lines.append(f"Newton decrement: {self.newton_decrement:.3g} nat")
        lines.append("at boundary: " + ("; ".join(self.boundary) if self.at_boundary else "no"))
        return "\n".join(lines)

    def __repr__(self) -> str:
        flag = ", at_boundary" if self.at_boundary else ""
        return (f"FitDiagnostics(max|grad|={self.max_abs_gradient:.3g}, "
                f"eigenvalues={np.array2string(self.eigenvalues, precision=4)}, "
                f"converged={self.converged}{flag})")


def _ranges(copula, names: list[str]) -> list[tuple[float, float]]:
    """The range of each coordinate: τ's registered range cut to the reachable τ; each extra's box."""
    from pmcprg.copulas._base import EXTRA_PARAM_BOUNDS_BY_PARAM

    out = []
    for name in names:
        if name == "tau_k":
            lo, hi = float(copula.tau_min), float(copula.tau_max)
            reach = type(copula).reachable_tau_bounds()
            if reach is not None:
                lo, hi = max(lo, reach[0]), min(hi, reach[1])
            out.append((lo, hi))
        else:
            lo, hi, _ = EXTRA_PARAM_BOUNDS_BY_PARAM[name]
            out.append((float(lo), float(hi)))
    return out


def _generic_boundary(copula, names, x0, ranges, build) -> list[str]:
    """Boundary notes for the families outside FR-4's ``_spec_of`` (module docstring)."""
    from pmcprg.copulas._stderr import _BOUNDARY_REL, _reachable_note, _tau_boundary

    tau = float(x0[0])
    notes = _tau_boundary(copula, tau, independence_note=False)
    reach = type(copula).reachable_tau_bounds()
    if reach is not None and not reach[0] < tau < reach[1]:
        notes.append(_reachable_note(type(copula), tau, reach[1] if tau > 0.0 else reach[0]))
    for j in range(1, len(names)):
        name, x = names[j], float(x0[j])
        lo, hi = ranges[j]
        for end in (lo, hi):
            if abs(x - end) <= _BOUNDARY_REL * max(abs(end), 1.0):
                notes.append(f"{name} = {x:.6g} at the end {end:g} of its fitting box.")
        d = _BOUNDARY_REL * max(abs(x), 1.0)
        for sgn in (-1.0, 1.0):
            y = x + sgn * d
            if not lo <= y <= hi:
                continue
            trial = x0.copy()
            trial[j] = y
            if build(trial) is None:
                notes.append(
                    f"{name} = {x:.6g} at the edge of the admissible set at τ = {tau:.6g}: "
                    f"the family refuses {name} = {y:.6g}.")
                break
    return notes


def fit_diagnostics(copula, uv, weights=None, *, converged: bool = True, message: str = "",
                    n_iter: int | None = None, n_eval: int | None = None) -> FitDiagnostics:
    """Diagnostics of ``copula`` as an estimate on ``uv`` with ``weights`` (module docstring).

    ``uv`` and ``weights`` are those of the fit (``FitResult.uv`` and
    ``FitResult.weights``); the optimiser's verdict and counts are passed
    through.
    """
    from pmcprg.copulas._fit import _weighted_log_density_sum
    from pmcprg.copulas._stderr import _central_step, _spec_of

    cls = type(copula)
    entry = copula.copula_enum
    uv = np.asarray(uv, dtype=float)
    w = None if weights is None else np.asarray(weights, dtype=float)
    tau_min, tau_max = entry.value.TAU_MIN_MAX
    if tau_max - tau_min < 1e-8:                     # the Product copula: no free parameter
        empty = np.empty(0)
        return FitDiagnostics((), empty, empty, np.empty((0, 0)), empty, False, (),
                              bool(converged), str(message), n_iter, n_eval, empty)

    names = list(entry.value.PARAMETERS_SET_NAME)
    x0 = np.array([float(copula.params[name]) for name in names])
    ranges = _ranges(copula, names)

    def build(x):
        try:
            return cls(**{name: float(val) for name, val in zip(names, x)})
        except Exception:
            return None

    def ll(x) -> float:
        cop = build(x)
        if cop is None:
            return math.nan
        try:
            val = _weighted_log_density_sum(cop.logpdf_array(uv), w)
        except Exception:
            return math.nan
        return val if math.isfinite(val) else math.nan

    p = len(names)
    f0 = ll(x0)
    steps = np.empty(p)
    grad = np.full(p, np.nan)
    hess = np.full((p, p), np.nan)
    one_sided: list[str] = []
    central = np.zeros(p, dtype=bool)
    eye = np.eye(p)
    for j in range(p):
        lo, hi = ranges[j]
        h = _central_step(float(x0[j]), lo, hi)
        if h <= 0.0:                                 # on an end: one-sided step inward
            h = 1e-6 * min(hi - lo, 1.0)
        steps[j] = h
        fp, fm = ll(x0 + h * eye[j]), ll(x0 - h * eye[j])
        if math.isfinite(fp) and math.isfinite(fm):
            central[j] = True
            grad[j] = (fp - fm) / (2.0 * h)
            hess[j, j] = (fp - 2.0 * f0 + fm) / (h * h)
            continue
        # One-sided three-point formulas, on the side that stays admissible.
        for sgn, f1 in ((1.0, fp), (-1.0, fm)):
            if not math.isfinite(f1):
                continue
            f2 = ll(x0 + 2.0 * sgn * h * eye[j])
            if math.isfinite(f2):
                grad[j] = sgn * (-3.0 * f0 + 4.0 * f1 - f2) / (2.0 * h)
                hess[j, j] = (f0 - 2.0 * f1 + f2) / (h * h)
                one_sided.append(f"{names[j]}: a step of {h:.3g} leaves the admissible set on one "
                                 "side — one-sided differences.")
                break
        else:
            one_sided.append(f"{names[j]}: no admissible step of {h:.3g} on either side — "
                             "no derivative.")
    for a in range(p):
        for c in range(a + 1, p):
            if not (central[a] and central[c]):
                continue
            ha, hc = steps[a] * eye[a], steps[c] * eye[c]
            vals = [ll(x0 + ha + hc), ll(x0 + ha - hc), ll(x0 - ha + hc), ll(x0 - ha - hc)]
            if all(math.isfinite(v) for v in vals):
                hess[a, c] = hess[c, a] = ((vals[0] - vals[1] - vals[2] + vals[3])
                                           / (4.0 * steps[a] * steps[c]))
    if not math.isfinite(f0):
        grad[:] = np.nan
        hess[:] = np.nan
    eig = (np.linalg.eigvalsh(0.5 * (hess + hess.T)) if np.all(np.isfinite(hess))
           else np.full(p, np.nan))

    try:
        notes = list(_spec_of(copula).boundary)      # FR-4's rules (one-parameter, Student, BB1)
    except (NotImplementedError, ValueError):
        notes = _generic_boundary(copula, names, x0, ranges, build)
    notes += one_sided
    return FitDiagnostics(
        names=tuple(names), estimate=x0, gradient=grad, hessian=hess, eigenvalues=eig,
        at_boundary=bool(notes), boundary=tuple(notes), converged=bool(converged),
        message=str(message), n_iter=n_iter, n_eval=n_eval, steps=steps,
    )
