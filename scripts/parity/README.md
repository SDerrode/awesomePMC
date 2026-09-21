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

---

# The families beyond the pilot (FR-11, wave 1)

The extreme-value families (Galambos, Hüsler–Reiss, t-EV, the asymmetric
logistic Tawn model), Plackett, AMH, FGM, and Nelsen's Archimedean families
4.2.12 (A12) and 4.2.14 (A14) with their 90°/270° rotations, read by
`pmcprg/tests/test_parity_extra.py`.

| file | produced by | reference | families |
|------|-------------|-----------|----------|
| `extra_pyvinecopulib.json` | `gen_pyvinecopulib_extra.py` | pyvinecopulib 1.0.0 | Tawn (every quantity) |
| `extra_vinecopula.json` | `gen_r_extra.R` | VineCopula 2.6.1 | Tawn types 1 and 2 (every quantity) |
| `extra_rcopula.json` | `gen_r_extra.R` | copula 1.1-7 | Galambos, Hüsler–Reiss, t-EV (pdf, cdf, τ, λ, A); Tawn (pdf, cdf); Plackett (pdf, cdf, τ, λ); AMH (every quantity); FGM (pdf, cdf, τ) |
| `extra_copbasic.json` | `gen_r_extra.R` | copBasic 2.2.16 | every family but A14: the closed-form CDF (A12 rotated too) |
| `extra_fcopulae.json` | `gen_r_extra.R` | fCopulae 4052.86 | Galambos, Hüsler–Reiss, Tawn (pdf, cdf, τ, λ, A); AMH, A12, A14 (pdf, cdf, τ) |

No package has the h-functions of the EV families, Plackett, FGM, A12 or
A14, nor anything of the A14 rotations: the tests' **mpmath oracle**
evaluates every quantity of every case on every point from the textbook CDF
(pdf and h by `mp.diff`, h⁻¹ by root finding, τ by the Genest–MacKay or
Hoeffding integrals, λ by the diagonal limits or 2 − 2A(½), A from ℓ), so no
quantity rests on one implementation.

```sh
.venv-parity/bin/python scripts/parity/gen_pyvinecopulib_extra.py
Rscript scripts/parity/gen_r_extra.R      # also needs copBasic and fCopulae
QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q pmcprg/tests/test_parity_extra.py
```

Environment of the committed tables: R 4.6.1 with VineCopula 2.6.1,
copula 1.1-7, copBasic 2.2.16 and fCopulae 4052.86
(`install.packages(c("copBasic", "fCopulae"))` besides the pilot's two);
`.venv-parity` as above.

## Grid

- `cases_extra.csv` — 42 parameter sets (also written, whole, under `grid`
  in every file): galambos θ ∈ {0.3, 1.5, 5}; husler_reiss λ ∈ {0.4, 1.5, 5};
  tev (ρ, ν) ∈ {(−0.3, 2), (0.5, 4), (0.8, 10)}; tawn (θ, ψ_u, ψ_v): two
  cases with ψ_u = 1, two with ψ_v = 1, two with both free; plackett
  θ ∈ {0.15, 3, 25}; amh θ ∈ {−0.9, 0.4, 0.95}; fgm θ ∈ {−0.8, 0.35, 1};
  a12 and a14 θ ∈ {1.3, 3, 8}, each at 0°, 90° and 270°.
- `points.csv` — the pilot's 25 interior points.
- `pickands_points.csv` — t ∈ {0.05, 0.2, …, 0.95} for A(t).

## Parametrisation (each measured by the generators)

Every package's CDF is compared with the textbook formula of the key below
(`tb_*` in `gen_r_extra.R`, `textbook_cdf` in `gen_pyvinecopulib_extra.py`),
median absolute error < 10⁻⁹ required (measured ≤ 4.4·10⁻¹⁵ everywhere):
that fixes which native parameter is which. w = −log u, z = −log v.

| key | textbook C | pmcprg (τ-based) | copula | copBasic | fCopulae | VineCopula / pyvinecopulib |
|-----|------------|------------------|--------|----------|----------|----------------------------|
| galambos [θ] | uv·exp((w^−θ + z^−θ)^−1/θ) | `GALAMBOS`, τ by quadrature | `galambosCopula(θ)` | `GLcop(θ)` | `"galambos"`, θ | — |
| husler_reiss [λ] | exp(−wΦ(1/λ + λ/2·log(w/z)) − zΦ(1/λ + λ/2·log(z/w))) | `HUSLER_REISS` | `huslerReissCopula(λ)` | `HRcop(λ)` | `"husler.reiss"`, λ | — |
| tev [ρ, ν] | exp(−wT_{ν+1}(a) − zT_{ν+1}(b)), a = k((w/z)^{1/ν} − ρ), b = k((z/w)^{1/ν} − ρ), k = √((ν+1)/(1−ρ²)) | `TEV`, `nu` = ν | `tevCopula(ρ, df = ν)` | `tEVcop(c(ρ, ν))` | — | — |
| tawn [θ, ψ_u, ψ_v] | exp(−(1−ψ_u)w − (1−ψ_v)z − ((ψ_u w)^θ + (ψ_v z)^θ)^{1/θ}) | `TAWN3`; `TAWN1` (ψ_u = 1, `psi` = ψ_v); `TAWN2` (ψ_v = 1, `psi` = ψ_u) | `khoudrajiCopula(indepCopula(), gumbelCopula(θ), shapes = c(ψ_u, ψ_v))` | `khoudrajiPCOP(cop = GHcop, para = θ, alpha = 1−ψ_u, beta = 1−ψ_v)` | `"tawn"`, c(α, β, r) = c(ψ_u, ψ_v, θ) | VineCopula **104** (ψ_v = 1, par2 = ψ_u) and **204** (ψ_u = 1, par2 = ψ_v); pyvinecopulib `tawn`, [ψ_u, ψ_v, θ] |
| plackett [θ] | (S − √(S² − 4θ(θ−1)uv))/(2(θ−1)), S = 1 + (θ−1)(u+v) | `PLACKETT`, τ by quadrature, θ(τ) by a table | `plackettCopula(θ)` | `PLcop(θ)` | — | — |
| amh [θ] | uv/(1 − θ(1−u)(1−v)) | `AMH` | `amhCopula(θ)` | `AMHcop(θ)` | archm `"3"`, α = θ | — |
| fgm [θ] | uv(1 + θ(1−u)(1−v)) | `FGM`, τ = 2θ/9 | `fgmCopula(θ)` | `FGMcop(θ)` | — | — |
| a12 [θ] | 1/(1 + ((1/u−1)^θ + (1/v−1)^θ)^{1/θ}) | `A12`, τ = 1 − 2/(3θ); `A1290`, `A12270` at −τ | — | `N4212cop(θ)`; `COP(reflect = "acute")` 90°, `"grave"` 270° | archm `"12"`, α = θ | — |
| a14 [θ] | (1 + ((u^−1/θ−1)^θ + (v^−1/θ−1)^θ)^{1/θ})^−θ | `A14`, τ = 1 − 2/(1+2θ); `A1490`, `A14270` | — | — (no `N4214cop`) | archm `"14"`, α = θ | — |

**VineCopula numbers the Tawn types the other way round from pmcprg**:
its 104 frees ψ_u, which is pmcprg's `TAWN2`, and its 204 frees ψ_v,
pmcprg's `TAWN1` (`convention_checks.tawn_codes`). The two are transposes
of each other; `tawn.py` used to claim the opposite from a recollection of
VineCopula's sources.

copula's own `tawnCopula` is Tawn's one-parameter *mixed* model
A(t) = 1 − θt + θt², not the asymmetric logistic one: not recorded (the
Khoudraji construction above is). copula's `A()` uses the v-share
t = log v/log(uv); the tables hold A in the u-share t = log u/log(uv)
(pmcprg's and fCopulae's, measured), so copula's is recorded at 1 − t.

## Package defects found (each measured; see the tests' registers)

- **fCopulae**: its default Archimedean density (`.darchm1Copula`) is wrong
  for AMH (type 3) — `.invPhiFirstDer`/`.invPhiSecondDer` write e^y − 1
  where ψ(y) = (1 − θ)/(e^y − θ) needs e^y − θ (relative error up to 1,
  `convention_checks.fcopulae_amh_default_density_rel_error_max`); the table
  holds its per-type formula (`alternative = TRUE`). Its Tawn
  `pevCopula(alternative = TRUE)` swaps α and β relative to `Afunc`. Its EV
  densities lose all digits in the far corners (Hüsler–Reiss λ = 5 at
  (0.99, 0.01): −1.1·10⁻¹⁶ for 5.7·10⁻⁵¹; Galambos θ = 5: 3.9·10⁻³
  relative); `evTau` integrates to ~10⁻⁷.
- **copula**: the τ of Galambos, Hüsler–Reiss, t-EV and Plackett are
  splines (`*TauFun`): 1.4·10⁻⁸, 2.3·10⁻⁸, 8.3·10⁻⁸ and up to 6.7·10⁻⁴;
  `tevTauFun` moreover ignores `df` and the sign of ρ (a spline in ρ² of the
  ν = 4 curve: τ(−0.3, ν = 2) reads τ(0.3, 4) = 0.1225 for 0.0720). Its
  symbolic Galambos density cancels in the corner (9·10⁻² relative at
  (0.01, 0.99), θ = 5). `cCopula` is not implemented for the EV families,
  Plackett or FGM.
- **VineCopula / pyvinecopulib**: the Tawn inverses are numerical
  (5.7·10⁻¹², 2.9·10⁻¹¹), the densities 2–3·10⁻¹⁰ off in the far corner,
  VineCopula's τ 9.7·10⁻⁸ (numerical integration).
- **copBasic**: only the CDFs are closed forms; `derCOP`, `densityCOP`
  (finite differences) and `tauCOP` (numerical, 1.8·10⁻⁵ off for Plackett,
  a double fallback for `N4212cop`) are not recorded.
