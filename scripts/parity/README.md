# Interior parity references (audit FR-11)

Reference tables of pdf, cdf, both h-functions, both h-inverses, Kendall's τ
and the tail-dependence coefficients λ_L, λ_U, computed **offline** by three
independent implementations and read by
`pmcprg/tests/test_parity_interior.py`. Neither R nor pyvinecopulib is needed
at test time.

| file | produced by | reference |
|------|-------------|-----------|
| `pmcprg/tests/data/parity/pyvinecopulib.json` | `gen_pyvinecopulib.py` | pyvinecopulib 1.0.0 (vinecopulib C++) |
| `pmcprg/tests/data/parity/vinecopula.json` | `gen_r.R` | R VineCopula 2.6.1 |
| `pmcprg/tests/data/parity/rcopula.json` | `gen_r.R` | R copula 1.1-7 |

Each file records its provenance: package and version, Python/R version and
platform, generation date (UTC), the repository version at generation, the
exact command, the definitions of its columns, the native call of every case,
and the convention checks measured on that reference (below).

## Regenerating

From the repository root:

```sh
# pyvinecopulib — its own environment; the generator never imports pmcprg
.venv-parity/bin/python scripts/parity/gen_pyvinecopulib.py

# VineCopula and copula (no JSON package needed)
Rscript scripts/parity/gen_r.R

# then
QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q pmcprg/tests/test_parity_interior.py
```

The environments used for the committed tables:

- `.venv-parity`: Python 3.13.15, numpy 2.5.3, pyvinecopulib 1.0.0
  (`python3.13 -m venv .venv-parity && .venv-parity/bin/pip install pyvinecopulib==1.0.0`);
- R 4.6.1 with VineCopula 2.6.1 and copula 1.1-7
  (`install.packages(c("VineCopula", "copula"))`).

Regenerate after a new version of a reference or a change of the grid, and
review the diff: the test module keeps a register of the measured
discrepancies (`REFERENCE_LIMITED`), each pinned by an `mpmath` computation,
which a new reference version may make obsolete.

## Grid

- `cases.csv` — 60 parameter sets, in a reference-agnostic key: `family`
  (gaussian, student, frank, clayton, gumbel, joe), `rotation` in degrees,
  `par1` = ρ or θ **of the unrotated family** (θ > 0 for every rotation),
  `par2` = ν (Student only). Weak to strong dependence, both signs where the
  family allows them (Gaussian, Student, Frank); every rotation of Clayton,
  Gumbel and Joe.
- `points.csv` — 25 points of the interior [0.01, 0.99]², including both
  orders of each asymmetric pair, so that a transposition error cannot
  cancel. The tails are excluded: references that clamp or cap there are no
  oracle (audit, *Verdict*); FR-1's high-precision references cover them
  (`pmcprg/tests/test_copula_limits.py`).

Student's CDF is not recorded: pmcprg's Student copula has none, and
copula's `pCopula` refuses a non-integer ν.

## Conventions (each measured by the generators, not assumed)

**Parametrisation.** pmcprg is parametrised by Kendall's τ (`tau_k`); the
references by ρ, θ (and ν). The tests build pmcprg at the τ of θ given by
pmcprg's own τ(θ) and check that its constructor returns θ.

| key | pmcprg (`CopulaEnum`) | τ(θ) | pyvinecopulib | VineCopula | copula |
|-----|------------------------|------|---------------|------------|--------|
| gaussian ρ | `GAUSSIAN` | (2/π) asin ρ | `gaussian`, [ρ] | 1, par = ρ | `normalCopula(ρ)` |
| student ρ, ν | `STUDENT`, `df` = ν | (2/π) asin ρ | `student`, [ρ, ν] | 2, par = ρ, par2 = ν | `tCopula(ρ, df = ν, df.fixed = TRUE)` |
| frank θ | `FRANK` | `kendall_tau_frank(θ)` | `frank`, [θ] | 5, θ | `frankCopula(θ)` |
| clayton θ, 0° | `CLAYTON` | θ/(θ+2) | `clayton`, rotation 0 | 3, θ | `claytonCopula(θ)` |
| clayton θ, 180° | `SURVIVAL_CLAYTON` | θ/(θ+2) | rotation 180 | 13, θ | `rotCopula(·, flip = c(TRUE, TRUE))` |
| clayton θ, 90° | `CLAYTON90` | −θ/(θ+2) | rotation 90, θ > 0 | 23, **−θ** | `rotCopula(·, flip = c(TRUE, FALSE))` |
| clayton θ, 270° | `CLAYTON270` | −θ/(θ+2) | rotation 270, θ > 0 | 33, **−θ** | `rotCopula(·, flip = c(FALSE, TRUE))` |
| gumbel θ | `GH`, `SURVIVAL_GH`, `GH90`, `GH270` | ±(1 − 1/θ) | `gumbel` | 4, 14, 24 (−θ), 34 (−θ) | `gumbelCopula(θ)` + the same flips |
| joe θ | `JOE`, `SURVIVAL_JOE`, `JOE90`, `JOE270` | ±τ_Joe(θ) | `joe` | 6, 16, 26 (−θ), 36 (−θ) | `joeCopula(θ)` + the same flips |

**Rotations.** In all three references and in pmcprg, 90° reflects the
*first* argument and 270° the *second*: c₉₀(u, v) = c₀(1 − u, v),
c₂₇₀(u, v) = c₀(u, 1 − v), c₁₈₀(u, v) = c₀(1 − u, 1 − v); τ changes sign
for 90° and 270° only. Measured on each reference's densities
(`convention_checks.rotations` of every file). For these exchangeable bases
c₀(1 − u, v) = c₀(v, 1 − u), so the reflection is identified up to that
symmetry — which is all a density can tell.

**h-functions.** h1(u, v) = ∂C/∂u = P(V ≤ v | U = u) and
h2(u, v) = ∂C/∂v = P(U ≤ u | V = v); hinv1 inverts h1 in its **second**
argument (the y with h1(u, y) = w) and hinv2 inverts h2 in its **first**
(the x with h2(x, v) = w). Measured: finite differences of each reference's
CDF and h-functions, and round trips (`convention_checks.per_family`).

| | pyvinecopulib | VineCopula | copula | pmcprg |
|--|---------------|------------|--------|--------|
| h1(u, v) | `hfunc1([[u, v]])` | `BiCopHfunc1(u, v)` | `cCopula(cbind(u, v), indices = 2)` | `conditional_cdf(v, u)` |
| h2(u, v) | `hfunc2([[u, v]])` | `BiCopHfunc2(u, v)` | h1 at (v, u) ¹ | transposed copula's `conditional_cdf(u, v)` ² |
| hinv1(u, w) | `hinv1([[u, w]])` | `BiCopHinv1(u, w)` | `cCopula(cbind(u, w), indices = 2, inverse = TRUE)` ³ | `inv_h(w, u)` |
| hinv2(w, v) | `hinv2([[w, v]])` | `BiCopHinv2(w, v)` | hinv1 at (v, w) ¹ | transposed copula's `inv_h(w, v)` ² |

¹ Exchangeable (unrotated) families only. copula 1.1-7 gives no h-values for
the rotated families: `cCopula(inverse = TRUE)` is not implemented for
`rotCopula` ("Not yet implemented for copula class rotExplicitCopula"), and
the forward `cCopula` of a `rotCopula` returns the base's value at the
flipped point **without** the `1 −` a flipped second margin requires — the
Rosenblatt transform of the flipped vector, uniform but not ∂C/∂u (Clayton
θ = 2, `flip = c(FALSE, TRUE)`, (0.3, 0.8): 0.178 against ∂C/∂u = 0.822).

² The transposed copula is the copula itself for every exchangeable family,
and swaps 90° and 270° (c₉₀(u, v) = c₂₇₀(v, u) for an exchangeable base).

³ copula inverts h by `uniroot()` for every Archimedean family but Clayton,
with uniroot's default `tol = .Machine$double.eps^0.25` = 1.2·10⁻⁴ — a
median round-trip error of 3.3·10⁻⁵ on this grid. `iRosenblatt()` passes
`...` on to `uniroot()`, so the generator hands over `tol = 1e-15`.

**VineCopula clamps its h-functions** to [10⁻¹², 1 − 10⁻¹²] (observed: it
returns exactly 1e-12 and 0.999999999999 where the true value lies beyond);
the tests compare pmcprg clamped the same way.

**Tail dependence.** pmcprg's `tail_dependence()` returns the diagonal
(λ_L, λ_U), 0 for the 90°/270° rotations; pyvinecopulib's `taildep[0, 0]`,
`taildep[1, 1]`; VineCopula's `BiCopPar2TailDep`; copula's `lambda()`
(unrotated only: not implemented for `rotCopula`).

## Output format

JSON. Header: provenance, `definitions`, `convention_checks`, `points` (the
25 (u, v) pairs, shared by all files). `cases`: one object per line with
`family`, `rotation`, `pars` (the key above), `native` (the reference's own
call), the six arrays over the points (`null` where the reference has no
value), and `tau`, `lambda_L`, `lambda_U`. Doubles are written exactly:
Python's shortest round-trip `repr`, R's `%.17g`.
