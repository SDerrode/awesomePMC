"""
gaps.py — supervised inference with missing observations (NaN) for PMC/HMC models.

Public API
----------
gap_posterior(model, Y, *, gap_nodes=64) -> GapPosterior
    Filtering/smoothing quantities of a NaN-bearing sequence: log-likelihood of
    the observed data, α̂, β̂, γ, ξ and, for the grid variants, the posterior of
    (x_n, y_n) on the quadrature nodes at every missing position.
impute(model, Y, *, gap_nodes=64, quantiles=(0.05, 0.5, 0.95), n_samples=0, rng=None)
    -> Imputation
    Posterior mean, standard deviation and quantiles of every missing y_n given
    all the observed data; optionally joint FFBS draws of the missing values.
forecast(model, Y, h, *, gap_nodes=64, quantiles=(0.05, 0.5, 0.95)) -> Forecast
    h-step predictive law of (x_{N+k}, y_{N+k}), k = 1..h, given the observed
    part of Y (Y may itself contain NaN).
missing_mask(Y), needs_grid(model), reference_grid(model, gap_nodes)
    Helpers (mask of missing rows, whether a model needs the quadrature grid,
    the quadrature grid itself).

The NaN-aware entry points of :mod:`pmcprg.pmc.inference` (``classify``,
``forward``, ``backward``, ``precompute_weights``, ``sample_posterior``)
dispatch here; a Y without missing values never reaches this module.

Missing observations
--------------------
A row of Y is *missing* when it holds a non-finite value (NaN, ±inf):
``miss = ~np.isfinite(Y)`` for d = 1; for d > 1 a row with any non-finite
component is missing as a whole (partially observed rows are not supported in
this version — their finite components are ignored). The mask is m_n = 1 iff
row n is missing.

Missingness mechanisms
----------------------
The law of the mask belongs to the model (``model.missingness``, TOML table
``[missingness]``, :mod:`pmcprg.pmc.missingness`):

* **Ignorable** (MCAR/MAR — ``model.missingness is None``, the default): the
  mask carries no information beyond y_obs, and the quantities below are
  those of the observed-data likelihood p(y_obs) = ∫ p(y) dy_miss.
* **State-dependent, non-ignorable** (``"state"``, ``"state-markov"``):
  given the states the mask is independent of Y and p(m | x) = Π_n e_n(x_n),
  with e_n(i) = π_i^{m_n} (1 − π_i)^{1 − m_n} for ``"state"`` (independent
  masks, P(m_n = 1 | x_n = i) = π_i) and e_n(i) = p(m_n | m_{n-1}, x_n = i)
  for ``"state-markov"`` (a two-state Markov mask whose onset and
  persistence probabilities depend on the current state — bursts of gaps).
  Then

      p(y_obs, m) = Σ_x ∫ p(x, y) Π_n e_n(x_n) dy_miss,

  a selection model for data missing not at random (Little & Rubin 2019):
  the mask itself is evidence on the states (e.g. sensor dropouts more
  frequent during some activities). e_n depends on x_n only, so it
  multiplies the message at position n componentwise — the initial message
  by e_0, the columns of the transition into n + 1 by e_{n+1}, e_n(i) for
  every node g of an augmented state (i, g). It is a likelihood factor, not
  part of the transition kernel: it is applied after the Tauchen–Hussey
  block renormalisation below (and in ``precompute_weights``' W and f_pdf,
  after the transition, for the exact shortcut). log p(y_obs, m) =
  Σ_n log C_n as before, and γ, ξ, MPM, FFBS draws and imputations are given
  (y_obs, m). Given x, y_miss does not depend on m: the laws of the missing
  values given the states are unchanged, only the state posteriors move. A
  complete Y has the mask m = 0 and still gets its factors (1 − π_i for
  ``"state"``). ``forecast`` gives no factor to its h appended rows, whose
  mask is unknown (Σ_m p(m | x) = 1).

Z = (X, Y) is a Markov chain with transition q(j, y' | i, y) = W[n, i, j] for
y = y_n, y' = y_{n+1} (see :mod:`pmcprg.pmc.inference`) and initial density
μ(i, y) = α_1(i) evaluated at y = y_1.

1. **Exact shortcut** (HMC-IN, HMC-IN2, PMC-IN with state margins). The
   transition q(j, y' | i, y) = A_ij f_j(y') does not depend on y, and
   ∫ f_j(y') dy' = 1: a missing y_{n+1} contributes the factor 1 in place of
   f_j(y_{n+1}), and a missing y_1 gives α_1(j) = Σ_i p_ij. This is exact and
   stays a K-state recursion, so :func:`pmcprg.pmc.inference.precompute_weights`
   returns the marginalised weights (f_pdf row = 1 at a missing n) and every
   K-state function of :mod:`pmcprg.pmc.inference` is exact on them. Given the
   states, the missing y_n are independent with law f_{x_n}: imputation and
   forecasting use the exact mixtures Σ_j P(x_n = j | y_obs) f_j.

2. **Quadrature grid** (HMC-DN, PMC, PMC-IN with pair margins). The
   transition depends on the value of y_n through the copula and/or the pair
   margins, so a missing y_n is integrated out on an augmented state (i, g)
   over quadrature nodes y_g.

   * Reference law g_ref(y) = Σ_i μ(i, y) = Σ_{i,j} p_ij f_ij(y) (pair margins)
     or Σ_j (Σ_i p_ij) f_j(y) (state margins; π_j f_j for the HMC variants) —
     the law of y_1, the stationary law of y_n for a stationary chain.
   * Nodes y_g = G_ref^{-1}(t_g), t_g = ψ(s_g), with (s_g, ws_g) the G-point
     Gauss–Legendre rule on (0, 1) and ψ(s) = s² / (s² + (1 − s)²) an
     endpoint transform; the weights w_g = ws_g ψ'(s_g) integrate over t, and
     ω_g = w_g / g_ref(y_g), so ∫ h(y) dy ≈ Σ_g ω_g h(y_g). Without ψ (plain
     Gauss–Legendre in t = G_ref(y)) the integrands, a conditional density
     over g_ref, behave like t^β at t → 0, 1 (e.g. β = 2ρ²/(1 − ρ²) for a
     Gaussian copula ρ between N(0, 1) margins): an endpoint singularity that
     makes the rule converge only algebraically — 1e-4 errors at G = 64 for
     ρ = 0.5 — whereas ψ restores fast convergence (1e-7 at G = 64, see the
     accuracy tests). G_ref^{-1} is a vectorised bisection on the mixture CDF
     (on the survival function in the upper half), bracketed by the component
     quantiles. Default G = 64 (``gap_nodes``). This *reference grid* serves
     every missing position whose laws it resolves; the others get a local
     grid (below). Each missing position n has its own grid, nodes y_{n,g}
     and weights ω_{n,g}.
   * Transitions of the augmented chain (augmented index u = i·G + g):

       observed n → missing n+1  E[i, (j, g)]        = q(j, y_{n+1,g} | i, y_n) ω_{n+1,g}
       missing n → missing n+1   Q[(i, g), (j, g')]  = q(j, y_{n+1,g'} | i, y_{n,g}) ω_{n+1,g'}
       missing n → observed n+1  X[(i, g), j]        = q(j, y_{n+1} | i, y_{n,g})
       observed → observed       W[n, i, j]          (unchanged)
       missing y_1               α̃_1(i, g)           = μ(i, y_{1,g}) ω_{1,g}

     Q between two reference grids does not depend on the data and is built
     once per call; a step touching a local grid gets its own Q.
   * Renormalisation (Tauchen & Hussey 1991). The quadrature does not
     integrate the transition density exactly, but two of its integrals are
     known: Σ_g of the block (i, y) → (j, ·) must be P(x_{n+1} = j | x_n = i,
     y_n = y) — A_ij, or p_ij f_ij(y) / Σ_k p_ik f_ik(y) with pair margins —
     and the block (·, g) of α̃_1 must be P(x_1 = i). Every block of E, Q and
     α̃_1 is rescaled to that mass (then every row to 1, a no-op unless a
     block vanished). A trailing gap then contributes exactly log C = 0, a
     sequence with no observed value has log-likelihood 0, and for state
     margins (X Markov) the state marginals across a gap are exact.
     Measured on the references of the test-suite, this also lowers the
     likelihood and moment errors 2–5× against plain row renormalisation.
   * Forward and backward run the Devijver/Rabiner-scaled recursions of
     :mod:`pmcprg.pmc.inference` on this chain of variable width (K at an
     observed position, K·G at a missing one); log p(y_obs) = Σ_n log C_n.
     γ_n(i) = Σ_g α̃_n(i, g) β̃_n(i, g) at a missing n, ξ_n by summing the
     augmented pair posterior over the nodes. Copula arguments are clipped to
     [EPS, 1 − EPS] as in ``precompute_weights``.
   * Log space. When a linear step underflows (a normaliser not finite or
     below MIN_POSITIVE) or a kernel value overflows (e.g. a copula density
     beyond 1.8e308), the whole pass is redone in log space from the exact
     log-densities (``logpdf_array`` of the copulas). A single block whose
     linear weights all underflow (quadrature mass below 1e-250: a kernel
     narrower than the distance to the nearest node) is rebuilt in log space
     and rescaled there (:func:`_normalise_blocks`); it used to keep 0 and
     lose its mass to the other blocks of its row, and the state filter of
     the linear pass drifted from the log pass's by up to 0.26. The linear
     and log passes now build the same chain (measured: 1.5e-11 nats and
     1e-14 in α̂ on 28 800 rows of Intel mote 48).
   * Laws of the missing values. Moments of y_n are the quadratures
     Σ_g mass_g y_g^k of the node posterior. Quantiles need the CDF between
     the nodes: the density of y_n at any y is interpolated from the messages
     of the neighbouring positions through the exact kernels (Nyström
     interpolation, :func:`_nystrom_density`) on the same grid with four
     times more nodes, and the CDF of its Legendre interpolant is inverted
     (on a local grid, panel by panel: the CDF at a panel boundary is the
     mass of the panels below). (Interpolating the G node masses directly
     loses half of the polynomial degree: quantile errors 1e-4 where the
     moments are at 1e-6.) ``forecast`` uses the normalised forward messages
     of a trailing gap of length h. The Nyström density propagates the
     discrete message of the previous position through the exact kernel, not
     through the chain's renormalised rows: a row whose kernel the grid
     misses has a block factor of up to 1e30, which the former interpolation
     applied off the nodes — the P3 failure of ``report/forecasting`` (2-step
     forecast quantiles 13.4 / 15.8 / 16.7 for a law of mean 4.26, sd 1.42;
     now 3.04 / 4.00 / 10.47 at every G ≥ 128, within 4e-3 at G = 32).
     Safety net: where the interpolated CDF is not monotone or leaves the
     Markov–Stieltjes bracket [Σ_{g'<g} m_g', Σ_{g'≤g} m_g'] of the node
     masses by more than 1e-6, the piecewise-linear CDF of the cumulative
     masses at the cell edges is inverted instead (first-order accurate) and
     a WARNING is logged. The ``density`` of :class:`Imputation` and
     :class:`Forecast` is given at the reference nodes (mass / ω there, the
     normalised Nyström density for a position on a local grid);
     ``grid_nodes`` / ``grid_mass`` hold each position's own discrete law.

Local grids
-----------
The reference grid places its nodes in the quantiles of the *stationary* law.
Under strong serial dependence — copulas at τ ≈ 0.99, as for a sensor sampled
every 30 s — the law of a missing y given its neighbours is far narrower than
the node spacing and the quadrature fails silently: a Gaussian AR(1) at ρ =
0.9999 written as a PMC gets a 2-step forecast sd of 0.0000 for 0.0200, 843
of 3000 rows flagged by :func:`pmcprg.pmc.outliers.flag_outliers` for 2, and
the log-likelihood of Intel Lab mote 48 moves by 13 600 nats between G = 64
and 512 (``report/erroneous_data/intel_lab``). Every missing position whose
laws the reference grid does not resolve gets its own grid instead:

* **Proposal.** Each run of missing rows [a, b] is summarised by Gaussian
  pieces (m, s): forward pieces from y_{a−1} (one per transition i → j: the
  exact conditional law of y_a, through the copula's h-inverse
  y = F_ji^{-1}(h_ij^{-1}(t | F_ij(y_{a−1}))), then propagated along the gap
  by a Gaussian-sum filter whose pieces merge only when they share state,
  location and scale, the K² heaviest kept: a single switch and a switch
  back stay two narrow pieces), backward pieces from y_{b+1} (the law of y_b
  given (i → j, y_{b+1}) under μ(i, y_b) q(j, y_{b+1} | i, y_b): the
  transposed copula's h-inverse), and their products — the *bridges*, where a
  y pinned by both neighbours sits even across a jump of 20 innovations.
  Location Q(1/2), core scale (Q(Φ(1)) − Q(Φ(−1)))/2 and a tail extent from
  Q(Φ(±2)), Q(Φ(±3)) (skewed or heavy-tailed laws, Clayton and Gumbel in
  their dependent tail). Weights: the one-observation state law of the
  neighbour times the transition; bridges ½, forward and backward pieces ¼
  each; near-identical pieces merged, six kept. These are the proposals of
  a first pass; the runs they serve badly are rebuilt from proposals
  weighted by the filter ("Filter-weighted proposals", below).
* **Grid.** A composite rule on disjoint panels: every narrow piece owns a
  *core* panel m ± 8 e s (e the tail extent), cut where cores meet and given
  to the piece whose density dominates there — or to a piece 4 times
  narrower that peaks higher inside it or holds more of its mass (judged at
  the middle alone, a narrow piece whose centre other cores' ends put off
  the middle drove no panel: Intel runs 0.2–2 nats off at G = 64) — with
  Gauss–Legendre in u for
  y = m + 2s·sinh(u) (32 nodes on m ± 8s: 1e-14 for N(m, s²), 5e-10 for
  N(m + 1.5s, (0.7s)²), exact for a constant — narrow and flat integrands
  alike); the rest of the support of g_ref is covered by *background* panels,
  ψ-Gauss–Legendre in its probability scale restricted to them. 35 % of the
  nodes go to the background (the broad paths: state switches under weak
  copulas, the prior in a leading gap), the rest to the cores ∝ √(weight).
  Disjoint panels, not a single mixture proposal: a mixture puts steps into
  the transformed integrand where a component's mass ends (1e-3 errors with
  a g_ref defensive component), and a multiple-importance rule leaks each
  narrow piece's tails into the coarse rules (5.6e-4); a panel only ever
  integrates a function that is smooth on it.
* **Selection.** A position keeps the reference grid unless a piece of weight
  ≥ 1e-4 has fewer than 4 reference nodes within one scale, the reference
  grid misintegrates the pieces and the transition kernels into the position
  (their mass and second moment, :func:`_proposal_error`) by more than 1e-6,
  and the local grid's total error — that proxy plus its error on the
  integrals known in closed form next to an observed row
  (:func:`_neighbour_error`: the entry masses T_ij(y_{a−1}), the exit masses
  ∫ μ(i, y) q(k, y_{b+1} | i, y) dy, which see support boundaries and skewed
  laws the Gaussian pieces miss) — is 3 times smaller than the reference
  grid's; inside a run, a position next to a local grid of the run takes
  its own local grid as soon as that total error is smaller at all (a
  reference grid between two local grids relays a narrow law through its
  coarse nodes: a 25-row gap of the Intel robust-estimation window was
  12.5 nats off). A local grid of a missing y_1 must also carry the prior
  (within 1e-2). (Until this version the local grid had to be 3 times
  better on the proxy alone and no worse on the closed-form integrals: a
  reference grid that integrated each kernel of a single gap to 6e-5 but
  their product, the bridge, with an 11 % error was kept, 2.6 % off.)
  Weak and moderate dependence keep the reference grid everywhere, bit for
  bit.
* **Exactness kept.** The block renormalisation applies unchanged: a trailing
  gap still contributes log C = 0 and the state marginals of state margins
  stay exact; the log-space fallback rebuilds the same grids. The filter of
  :mod:`pmcprg.pmc.outliers` takes the grids of every run of missing rows
  from :func:`_gap_grids` (a run of gated rows: :func:`_run_grids`, built as
  the filter enters it) and uses the transitions of the batch pass, leading
  gaps included: its PITs, log-likelihood and state filter equal
  ``gap_posterior``'s (measured: 2.3e-13 nats and 1.7e-15 in α̂ on leading,
  interior and trailing gaps of regime AR(1) series at τ = 0.997–0.999,
  where they differed by 2.4 nats and 0.26 before). The grids of the last 4
  (model, series, G) are cached (they are half the cost of a pass).
* **Measured** (exact references; ``test_gaps_local_grids.py``): AR(1) at
  ρ = 0.998, 0.999, 0.9999, every gap pattern, G = 64: log-likelihood
  2.4e-7–5.0e-7, means, sds and quantiles 3e-8–9e-7 of the conditional sd
  (reference grid: 0.24–35 nats, sds off by 41–180 %); forecasts to 2e-9;
  the PIT after a gap to 3e-12 (0.85 before). Pair-margin PMCs at τ = 0.9–0.99
  against brute force (the ten cases of the test): 7e-8–3.8e-2 at G = 64
  and 2e-13–2.4e-4 at G = 128, where the reference grid gave 5.7e-3–5.6 and
  2.6e-5–1.0; on every case measured, a local grid is never worse than the
  reference grid it replaces. Intel mote 48: 65 641.05 / 65 640.28 /
  65 640.245 / 65 640.245 nats at G = 64 / 128 / 256 / 512 (reference grid:
  51 798 / 57 574 / 63 096 / 65 364). The grid still has G nodes: a gap
  longer than ~20 rows at ρ = 0.9999 (the law of the middle rows broadens as
  √L while the kernel stays narrow) or six separated narrow pieces need a
  larger G — the diagnostic below says so.
* **Where G = 64 is not enough.** Regime switching under near-deterministic
  copulas: a switch moves y by whole margins, and a run can need several
  narrow pieces far apart. Regime AR(1) references (y = m_x + s_x z, z an
  AR(1) of coefficient ρ; ``test_gaps_leading.py``), 18 series of 300 rows
  with 25–30 runs of 1–6 rows: at ρ = 0.99 the sum over the runs of the
  |log-likelihood errors| is ≤ 1.3e-2 nats at G = 64 and ≤ 2.5e-8 at
  G = 128; at ρ = 0.999–0.9999 with switches between states of different
  means or scales, 0.02–0.57 nats at G = 64 and 1.8e-3–0.44 at G = 128
  (identical states, the AR(1) itself: ≤ 3.1e-7 at G = 64).
  The Intel Lab clean windows (the step-1 fits of motes 48, 47 and 22, days
  1–7; G = 512 references), the same sum before the filter-weighted
  proposals: 0.80, 0.87 and 1.1 nats at G = 64, 6.8e-2, 1.5e-2 and 0.14 at
  G = 128, 2.8e-4, 1.4e-3 and 7.1 at G = 256; now 3.6e-2, 7.9e-2 and 0.18,
  4.0e-3, 4.0e-4 and 1.1e-2, 4.6e-5, 1.2e-4 and 5e-4 ("Filter-weighted
  proposals").
* **Cost** (forward pass, N = 3000, 10 % single gaps, Gaussian copulas,
  G = 64): τ = 0.6, no candidate: 38 against 34 ms (K = 2), 64 against
  52 ms (K = 3); τ = 0.9, candidates screened, few or none switched: 101
  against 36 ms, 152 against 53 ms; τ = 0.99, every gap local: 112 against
  34 ms, 175 against 52 ms. A missing → missing block between two local
  grids is a (K·G)² kernel: :func:`_kernel_outer` evaluates it from
  per-node terms (the quantiles, logarithms and exponentials of the
  Gaussian, Clayton, Gumbel and Frank copulas once per node, the same values
  as on the expanded pairs); the layout drivers of a grid's elementary
  pieces are judged in one array pass. Intel mote 48, days 1–7 (20 160
  rows, 2 503 missing), under the same load: an ICE E-step 1.8 s against
  4.7 s at 39f249f (2.3 s there on a quiet machine); ``predictive_pit``
  3.5 s against 35.7 s (27.7 s quiet: grids built run by run, scalar
  h-functions). The robust-estimation window (14 401 rows, 8 275 missing,
  4 962 blocks between local grids): 14.1 s against 17.4 s (16.0 s quiet)
  and 16.1 s against 91.4 s (72.6 s quiet).

Filter-weighted proposals
-------------------------
The pieces above are weighted by the state law of the neighbours given y
alone, μ(i, y_{a−1}) and p_ij f_ji(y_{b+1}). Where the filter sits in a
state whose margin puts y far in its tail — Intel mote 22, a state 5.4 sds
out under a Gaussian copula at τ = 0.975 — that law gives the path that
stays in it ~1e-7, no piece represents it and the grid misses the
integrand: mote 22 was 1.1 nats off at G = 64, 0.16 at 128 and 7.1 at 256
(not monotone in G). The grids are therefore built in two steps
(:func:`_gap_grids_pass`):

* a pass on the grids of those proposals gives α̂ at the row before every
  run and β̂ at the row after it (none is run where no one-step kernel is
  narrower than the reference grid resolves: bit for bit the results
  before);
* the proposals are recomputed with forward pieces weighted by α̂_{a−1}(i)
  T_ij(y_{a−1}), backward pieces by p_ij f_ji(y_{b+1}) β̂_{b+1}(j), and
  bridges divided by the prior density μ(k, m) of their state at their
  location — the forward piece stands for p(x_n, y_n | past), the backward
  one for p(future | x_n, y_n) μ(x_n, y_n), and their product counted the
  prior twice (:func:`_bridges`); the transition kernels into a position
  and the closed-form entry / exit checks are weighted by the bridges
  through them (a kernel no bridge goes through has no posterior mass:
  counted at its prior weight, it made the local grid lose to the
  reference grid at G = 512, 3.5e-4 nats off where it is 2.9e-10);
* a run is rebuilt from them (:func:`_refine_grids`) when its grid
  integrates their pieces or kernels worse than 1e-3, or worse than 1e-6 and
  3 times what it did on the proposal it was built for (a candidate on the
  reference grid: worse than 1e-6); its new grids are kept when their error
  on the new proposal is smaller. Rebuilt grids merge panels only while
  the merged sinh map resolves every piece (_MERGE_RES), or as before where
  that integrates the proposal better (:func:`_better_layouts`): a panel
  centred on a light piece with a heavier one 5.5 scales away put the
  means of a 17-row leading gap 0.16 sd off. One pass (_FILTER_PASSES:
  a second, from the filter of the refined grids, left the Intel windows
  unchanged to the last digit); the final pass takes the transitions
  between unchanged grids from it (:class:`_Chain`, ``reuse``). A run of
  gated rows of :mod:`pmcprg.pmc.outliers` is weighted by the running
  filter on its left.

A pass rather than grids built as the forward recursion reaches each run:
run by run, the grids of Intel mote 48 cost 26 s instead of 1.4 s
(measured), and the backward messages of the exits would be missing.
Measured, sum over the runs of |log-likelihood error| at G = 64 / 128 /
256 / 512, before → now: Intel mote 22 (against G = 512 of this version,
whose runs that other passes dispute agree with G = 2048 to 5e-6),
1.1 / 0.14 / 7.1 / 4e-3 → 0.18 / 1.1e-2 / 5e-4 / ~1e-5, monotone; motes 48
and 47 (against G = 512), 0.80 / 6.8e-2 / 2.8e-4 → 3.6e-2 / 4.0e-3 / 4.6e-5
and 0.87 / 1.5e-2 / 1.4e-3 → 7.9e-2 / 4.0e-4 / 1.2e-4. (At G = 1024 a few
multi-row runs of mote 22 keep the reference grid, whose proxy error is
7e-7 for a run 4e-4 nats off, before and after this version.) Regime AR(1)
(the exact references of ``test_gaps_leading.py``; 162 cases: the regime
models and the AR(1) itself, ρ = 0.99–0.9999, two series, nine gap
patterns): |log-likelihood error| monotone in G = 64 → 512 in every case
(2 were not); a 10-row gap at ρ = 0.999, 0.11 / 1.8e-3 → 2.0e-3 / 1.9e-5
nats at G = 64 / 128; states of different scales at ρ = 0.9999, 0.26 /
0.21 / 0.18 / 4e-2 → 3.1e-3 / 8.9e-6 / 1.1e-6 / 4e-11; at G = 64 no case
worse beyond quadrature noise (one: means 4e-3 → 1.4e-2 sd for a path of
weight 1e-3 on 5 nodes, its log-likelihood 14 times better); the AR(1)
itself bit for bit. A filter 5.4 sds into its state's tail
(``test_gaps_filter_weights.py``): 0.48 nats → 8.3e-6 at G = 64 and 1.3e-3
→ 3e-11 at G = 256, the sds of the missing values 90 % → 3e-3 off, where the
diagnostic stayed quiet; the 17-row leading gap, means 0.16 → 8.6e-3 sd.
Cost (CPU time, the E-steps in one process): an ICE E-step on mote 48
(days 1–7, G = 64) 1.37 times (1.70 → 2.32 s; 1.43 at the k-means start of
ICE), ``predictive_pit`` 1.32, a full ICE start there 1.23 (31.5 → 38.7 s);
the robust-estimation window at ICE iterate 10, 781 runs rebuilt, 1.62
(12.9 → 21.0 s) and 1.92, peak memory 2.2 → 2.9 GB.

Leading gaps
------------
Before the first observed row there is no left neighbour: the forward message
is the prior, the law of (x_n, y_n) given nothing. The rows of an interior
gap (rescaled to the transition probabilities out of each source node)
propagate that broad law onto the local grids of the gap — narrow, pinned by
the first reading — by relocating the mass of every source node onto the
destination nodes nearest to it, and the law of y is lost: −4.3 and −22 nats
at τ = 0.997 and 0.999, G = 64, on the 17-row leading gap of
``report/erroneous_data/intel_lab/repro_leading_gap.py``. When the prior is
known in closed form (:func:`_prior_known`: state margins, where y_n given
the states has the law f_{x_n}; pair margins with a symmetric p, the
stationary SR-PMC, and state-independent missingness factors), the
transitions of a leading gap that contains a local grid are rescaled by
column instead (:func:`_lead_transition`): the entry (i, g) → (j, g') is
q(j, y_g' | i, y_g) ω_g' c_ij(g'), c_ij such that the discrete prior of grid
n is carried exactly onto the prior of grid n + 1. The forward message is
then the prior at every position of the gap, log C = 0 inside it (the
reverse of a trailing gap), and the posterior inside the gap is the
reverse-time forecast from the first reading, which the grids — built from
the backward pieces and the prior (background panels) — resolve. The
Nyström densities of those positions use the prior as incoming density and
the same column factors. Measured: the reproduction's leading gaps (L = 2,
5, 17; τ = 0.997, 0.999) within 2.5e-4 nats of the gap-free pass at G = 64
and 4.8e-8 at G = 256, the state filter equal to π to 1e-15; regime AR(1)
leading gaps (τ = 0.99–0.999, L = 5, 17): log-likelihood within 2.4e-3 /
3.3e-5 nats at G = 64 / 128, state posteriors 2.7e-4 / 8.8e-7, means and
sds of the missing values 2e-2 / 2.8e-3 exact sds (before: 0.65 nats, γ off
by 0.11). Otherwise (asymmetric p with pair margins, state-dependent
missingness with pair margins) and on reference grids alone the rows of an
interior gap are kept (bit for bit the results of the reference grid).

Convergence diagnostic
----------------------
:func:`_quadrature_report` checks, after each pass, integrals every grid must
reproduce and whose exact values are known, and sums them over the runs of
missing rows (``GapPosterior.quad_error``; ``per_run=True`` returns the
parts):

* the raw block masses before the renormalisation, weighted by the posterior
  mass through them (rows from the background panels of a local grid, which
  carry a broad law, left out), and the exit masses of the local grids,
  weighted by the posterior of the exit transition (until this version by
  their prior masses: with the filter-weighted proposals that reported
  0.07–0.50 on regime-AR(1) passes 5e-6–4e-5 nats off);
* inside a leading gap, the reverse masses of the prior's transitions, at
  the destinations of the core panels, weighted by the posterior and scaled
  by 0.02 (the column rescaling makes each reverse mass exact, so a relative
  error r moves the posterior inside the gap by at most 0.02 r, measured).
  Until this version the rows of a leading gap were not checked at all: a
  leading gap off by 22 nats reported 3e-5.

Each block's relative error is capped at 1: a rescaled block carries its
posterior weight whatever its raw mass (uncapped, the k-means start of an
Intel ICE reported 590 for a pass off by 6.7 nats; 27 now, for 4.2). The
cap does not remove a false alarm of that start: a state 8 sds from y,
where Φ rounds to 1 so that the copula term is constant, has raw block
masses 32 times the exact ones, which the renormalisation makes exact — 22
of the 27, and 22 at G = 256 for a pass 0.056 nats off. The sum
estimates Σ_runs |error of the run's log-likelihood factor| in nats, to
within a factor of about 15 on the references below (that start aside). A
WARNING ("Missing-data quadrature not converged … increase gap_nodes") is
logged above ``QUAD_WARN`` = 0.05, whatever the number of runs
(:func:`quad_warn_limit`; it was 1e-3 · √R, R the number of runs). It is a
screen, not an error bound.

Calibration: 67 passes, a pass "off" when the sum of |per-run
log-likelihood errors| exceeds 0.01 nats — against the exact references of
``test_gaps_leading.py`` (18 regime-AR(1) series at G = 64 and 128,
9 leading gaps of the reproduction at G = 64 and 256) and the G = 512–1024
references of the Intel Lab windows (10 passes of the step-1 fits at G =
64–512, 3 of ICE iterates). Off: 11, of which 3 missed (0.010–0.015 nats
off, reports 0.015–0.030). Not off: 56, no false alarm. The report is 0.8–3
times the error on the regime-AR(1) passes off by more than 1e-3, 2–15
times on the Intel ones (860 on the k-means start of ICE at G = 256, a pass
0.027 nats off: the state 8 sds from y above). Before the filter-weighted
proposals and this exit weighting (67 passes of that code): 25 off, 5
missed, 2 false alarms.

Memory
------
A missing → missing transition that touches a local grid is a dense
(K·G)² block — 295 kB at K = 3, G = 64, 4.7 MB at G = 256 — one per step of
a gap. Until this version a chain kept every one, and the pass that weighted
the proposals (:func:`_gap_grids_pass`) kept its own while the final one was
built: on the Intel Lab mote-48 detection window (46 860 rows, 16 271
missing, K = 3) ``gap_posterior`` peaked at 2.3 GB at G = 64 and above 8 GB
at G = 128 (a G = 256 detection check: 21–23 GB per process). A chain now
builds these transitions when a pass first needs them and keeps at most
_TRANSITION_BUDGET bytes of them (2 GiB); the backward pass rebuilds the
others, and computes on the way the parts of ξ and of the quadrature report
that need them; FFBS rebuilds them again. The weighting pass is released
when the final one takes over; the temporaries of the grid construction and
of the Nyström densities are computed by blocks (_CHUNK), and the grid
cache stops at _GRID_CACHE_BYTES. Besides the kept transitions a pass holds
about 0.3 kB per missing row and per node (K = 3: the messages α̃ and β̃,
the grids, the node posteriors, and the node margins while transitions are
rebuilt). A rebuilt transition is the same operations on the same inputs:
every result is the same bit for bit whatever the budget
(``test_gaps_memory.py``); beyond it only the time grows.
Measured (peak footprint / maximum resident size, one process; the version
before → this one): the mote-48 window, ``gap_posterior`` at G = 64, 2.3 /
5.2 → 2.1 / 3.8 GB in the same 61 s; at G = 128, stopped above 8 GB → 2.3 /
4.5 GB in 245 s (it builds 11 252 such transitions, 13 GB, and keeps 1 792);
at G = 256, 3.5 / 5.1 GB in 818 s; its third quarter at G = 256, 6.7 / 6.6
→ 2.3 / 3.0 GB, 23 → 36 s. Mote-20 forecasts at G = 256: 3.9 / 6.3 → 2.3 /
4.1 GB, 26 → 38 s (state margins, K = 3), 1.9 / 3.3 → 1.6 / 2.8 GB in 17 s
(pair margins, K = 2). The 16 other probes (G ≤ 128) kept all their
transitions: 0.68–1.03 times the peak footprint, 0.85–1.06 times the time.

Returned α̂ and β̂ (K-state view of the augmented chain)
-------------------------------------------------------
``alpha_hat[n, i] = P(x_n = i | y_obs ∩ 1:n)`` at every n (missing or not).
``beta_hat[n]`` is the normalised backward message at an observed n; at a
missing n there is no K-state backward message independent of the forward
(y_n couples x_{n-1} and x_{n+1}), so it is defined as
``beta_hat[n, i] ∝ Σ_g α̃_n(i, g) β̃_n(i, g) / Σ_g α̃_n(i, g)`` — the
likelihood of the future observations given x_n = i and the past ones. With
this definition ``smooth(alpha_hat, beta_hat)`` is exact at every n. No K×K
tensor W makes ``joint_posteriors(alpha_hat, W, beta_hat)`` exact for the grid
variants (the posterior of X given y_obs is not Markov across a gap): use
``gap_posterior(model, Y).xi``.

References
----------
* Tauchen, G. & Hussey, R. (1991). Quadrature-based methods for obtaining
  approximate solutions to nonlinear asset pricing models. *Econometrica*
  59(2), 371–396 — row-renormalised quadrature discretisation of a Markov
  kernel.
* Kitagawa, G. (1987). Non-Gaussian state-space modeling of nonstationary
  time series. *J. Amer. Statist. Assoc.* 82(400), 1032–1041 — numerical
  integration of filtering densities on a grid.
* Nyström, E. J. (1930). Über die praktische Auflösung von Integralgleichungen
  mit Anwendungen auf Randwertaufgaben. *Acta Math.* 54, 185–204 — the
  interpolation of a quadrature solution through its kernel.
* Alspach, D. L. & Sorenson, H. W. (1972). Nonlinear Bayesian estimation
  using Gaussian sum approximations. *IEEE Trans. Automat. Control* 17(4),
  439–448 — the Gaussian-sum propagation of the local proposals.
* Davis, P. J. & Rabinowitz, P. (1984). *Methods of Numerical Integration*,
  2nd ed., Academic Press — composite Gauss–Legendre rules and variable
  transformations (the sinh map of the core panels).
* Little, R. J. A. & Rubin, D. B. (2019). *Statistical Analysis with Missing
  Data*, 3rd ed., Wiley — ignorable missingness, observed-data likelihood;
  selection models for data missing not at random (the ``[missingness]``
  mechanisms).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import warnings
from collections import OrderedDict
from dataclasses import dataclass, replace

import numpy as np
from numpy.polynomial import legendre as _leg
from scipy import stats as _ss
from scipy.special import ndtr as _sp_ndtr

from pmcprg.copulas.archimedean.clayton import CopulaClayton, _clayton_logpdf
from pmcprg.copulas.archimedean.frank import CopulaFrank
from pmcprg.copulas.archimedean.gumbel import CopulaGH, _gh_logpdf
from pmcprg.copulas.elliptical.gaussian import CopulaGaussian, _norm_quantile
from pmcprg.exceptions import IncompatibleObservationError
from pmcprg.numerics import EPS, ONE_MINUS_EPS, MIN_POSITIVE
from pmcprg.pmc import inference as _inf
from pmcprg.pmc.model import PMCModel

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_GAP_NODES",
    "DEFAULT_QUANTILES",
    "QuadratureGrid",
    "GapPosterior",
    "Imputation",
    "Forecast",
    "missing_mask",
    "needs_grid",
    "reference_grid",
    "gap_posterior",
    "impute",
    "forecast",
]

#: Default number of Gauss–Legendre nodes of the quadrature grid.
DEFAULT_GAP_NODES = 64
#: Default quantile levels reported by :func:`impute` and :func:`forecast`.
DEFAULT_QUANTILES = (0.05, 0.5, 0.95)


# ---------------------------------------------------------------------------
# Masks and dispatch helpers
# ---------------------------------------------------------------------------

def missing_mask(Y) -> np.ndarray:
    """Boolean mask (N,) of the missing rows of Y (any non-finite component)."""
    fin = np.isfinite(np.asarray(Y, dtype=float))
    if fin.ndim > 1:
        fin = fin.reshape(fin.shape[0], -1).all(axis=1)
    return ~fin


def needs_grid(model: PMCModel) -> bool:
    """True when a missing y must be integrated on the quadrature grid.

    The transition depends on the value of y_n for HMC-DN and PMC (copula) and
    for PMC-IN with pair margins (A_ij(y_n) ∝ p_ij f_ij(y_n)); HMC-IN, HMC-IN2
    and PMC-IN with state margins admit the exact shortcut.
    """
    return bool(model.variant.uses_copula or model.margin_structure == "pair")


def _resolve_nodes(gap_nodes) -> int:
    G = DEFAULT_GAP_NODES if gap_nodes is None else int(gap_nodes)
    if G < 2:
        raise ValueError(f"gap_nodes must be an integer ≥ 2, got {gap_nodes!r}.")
    return G


def _check_quantiles(quantiles) -> tuple[float, ...]:
    qs = tuple(float(q) for q in quantiles)
    if any(not (0.0 < q < 1.0) for q in qs):
        raise ValueError(f"quantiles must lie in (0, 1); got {qs}.")
    return qs


def _check_grid_supported(model: PMCModel) -> None:
    if getattr(model, "d", 1) != 1:
        raise NotImplementedError(
            f"Missing observations for variant {model.variant.value} with "
            f"{model.margin_structure} margins need the quadrature grid, which "
            f"is implemented for scalar observations only (model.d = {model.d})."
        )


# ---------------------------------------------------------------------------
# Mixtures of scalar laws: CDF and quantile by vectorised bisection
# ---------------------------------------------------------------------------

def _frozen(margin):
    fr = getattr(margin, "_frozen", None)
    if fr is None:
        raise TypeError(f"margin {margin!r} exposes no frozen scipy.stats law.")
    return fr


def _mixture_ppf(weights: np.ndarray, dists: list, q: np.ndarray,
                 qc: np.ndarray | None = None) -> np.ndarray:
    """Quantiles of the mixtures Σ_c weights[m, c] · dists[c].

    ``weights`` (M, C) rows summing to 1, ``q`` (M, Q) levels in (0, 1) and
    optionally ``qc`` = 1 − q computed without cancellation (upper tail).
    Bracket [min_c F_c^{-1}(q), max_c F_c^{-1}(q)] over the components of
    positive weight (the mixture CDF is ≤ q below every component quantile and
    ≥ q above them), then bisection to relative width 1e-13 — on the CDF for
    q ≤ 1/2 and on the survival function against ``qc`` above, so that levels
    within 1e-16 of 1 keep their precision.
    """
    weights = np.asarray(weights, dtype=float)
    q = np.asarray(q, dtype=float)
    qc = (1.0 - q) if qc is None else np.asarray(qc, dtype=float)
    upper = q > 0.5
    q = np.clip(q, MIN_POSITIVE, 1.0)
    qc = np.clip(qc, MIN_POSITIVE, 1.0)
    pos = weights > 0.0
    lo = np.full(q.shape, np.inf)
    hi = np.full(q.shape, -np.inf)
    for c, dist in enumerate(dists):
        with np.errstate(all="ignore"):
            yc = np.where(upper, dist.isf(qc), dist.ppf(q))
        active = pos[:, c][:, None]
        lo = np.where(active & (yc < lo), yc, lo)
        hi = np.where(active & (yc > hi), yc, hi)
    target = np.where(upper, qc, q)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        F = np.zeros(q.shape)
        for c, dist in enumerate(dists):
            F += weights[:, c][:, None] * np.where(upper, dist.sf(mid), dist.cdf(mid))
        left = np.where(upper, F > target, F < target)      # the root lies above mid
        lo = np.where(left, mid, lo)
        hi = np.where(left, hi, mid)
        if np.all(hi - lo <= 1e-13 * np.maximum(1.0, np.abs(mid))):
            break
    return 0.5 * (lo + hi)


def _ref_ppf(weights, dists, t: np.ndarray, tc: np.ndarray) -> np.ndarray:
    """Quantiles of g_ref = Σ_c weights[c] dists[c] at levels t (tc = 1 − t), for local grids.

    Safeguarded Newton inside the bracket of the component quantiles (on the
    survival function for t > 1/2), each element stopping on its own — its
    value does not depend on the other elements of the batch, so the filter
    of :mod:`pmcprg.pmc.outliers` and the batch passes build the same grids.
    """
    t = np.asarray(t, dtype=float)
    tc = np.asarray(tc, dtype=float)
    upper = t > 0.5
    target = np.where(upper, np.maximum(tc, MIN_POSITIVE), np.maximum(t, MIN_POSITIVE))
    lo = np.full(t.shape, np.inf)
    hi = np.full(t.shape, -np.inf)
    for w_, dist in zip(weights, dists):
        if w_ <= 0.0:
            continue
        with np.errstate(all="ignore"):
            yc = np.where(upper, dist.isf(target), dist.ppf(target))
        lo = np.minimum(lo, yc)
        hi = np.maximum(hi, yc)
    y = 0.5 * (lo + hi)
    todo = np.nonzero(hi > lo)[0]
    for _ in range(200):
        if not todo.size:
            break
        yy, up, tg = y[todo], upper[todo], target[todo]
        F = np.zeros(yy.shape)
        f = np.zeros(yy.shape)
        for w_, dist in zip(weights, dists):
            with np.errstate(all="ignore"):
                F += w_ * np.where(up, dist.sf(yy), dist.cdf(yy))
                f += w_ * dist.pdf(yy)
        diff = F - tg
        above = np.where(up, diff > 0.0, diff < 0.0)          # the root lies above yy
        l_ = np.where(above, yy, lo[todo])
        h_ = np.where(above, hi[todo], yy)
        with np.errstate(all="ignore"):
            yn = np.where(up, yy + diff / f, yy - diff / f)
        yn = np.where(np.isfinite(yn) & (yn > l_) & (yn < h_), yn, 0.5 * (l_ + h_))
        conv = (np.abs(diff) <= 1e-14 * tg) | (h_ - l_ <= 1e-13 * np.maximum(1.0, np.abs(yy)))
        y[todo] = np.where(conv, yy, yn)
        lo[todo], hi[todo] = l_, h_
        todo = todo[~conv]
    return y


# ---------------------------------------------------------------------------
# Reference law and quadrature grid
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Panels:
    """Panel structure of a local grid (module docstring, "Local grids").

    The real line is cut into disjoint panels, each carrying a Gauss–Legendre
    rule. A core panel (``loc`` finite) is the interval y = loc + sc·sinh(u),
    u ∈ [t_lo, t_hi]; a background panel (``loc`` NaN) is the probability
    range [t_lo, t_hi] of g_ref with the endpoint transform ψ (``tc_lo``,
    ``tc_hi`` the complements, kept without cancellation). ``sizes`` holds
    the nodes of every panel and ``panel`` (G,) the panel of every node,
    nodes sorted by value.
    """

    loc: np.ndarray
    sc: np.ndarray
    t_lo: np.ndarray
    t_hi: np.ndarray
    tc_lo: np.ndarray
    tc_hi: np.ndarray
    sizes: np.ndarray
    panel: np.ndarray


@dataclass(frozen=True)
class QuadratureGrid:
    """Quadrature rule of a missing y (module docstring).

    The reference grid is a single ψ–Gauss–Legendre rule in the quantiles of
    g_ref. A *local* grid (``mix`` not None) is a composite rule on disjoint
    panels (:class:`_Panels`; module docstring, "Local grids"); its per-node
    ``s``, ``ws``, ``t``, ``w`` are those of the panel of the node.

    Attributes
    ----------
    s       : (G,) Gauss–Legendre nodes on (0, 1) (per panel).
    ws      : (G,) Gauss–Legendre weights on (0, 1) (per panel).
    power   : exponent p of the endpoint transform t = ψ_p(s).
    t       : (G,) the nodes in the probability scale of their law.
    w       : (G,) ws_g ψ_p'(s_g) (× the panel's t-width for a local grid)
              — weights of ∫ · dt.
    nodes   : (G,) the node values y_g, increasing.
    ref_pdf : (G,) the density of the law of the node's panel at y_g —
              g_ref(y_g) for the reference grid.
    omega   : (G,) ω_g = w_g / ref_pdf_g — ∫ h(y) dy ≈ Σ_g ω_g h(y_g).
    weights : (C,) mixture weights of the reference law g_ref.
    dists   : list of the C frozen component laws of g_ref.
    mix     : None for the reference grid, the :class:`_Panels` of a local
              grid.
    exit_rel : for a local grid next to an observed row after it, the
              relative errors (K, K) of its exit integrals
              (:func:`_neighbour_error`), kept for the diagnostic.
    """

    s: np.ndarray
    ws: np.ndarray
    power: int
    t: np.ndarray
    w: np.ndarray
    nodes: np.ndarray
    ref_pdf: np.ndarray
    omega: np.ndarray
    weights: np.ndarray
    dists: list
    mix: _Panels | None = None
    exit_rel: np.ndarray | None = None

    @property
    def G(self) -> int:
        return int(self.s.size)

    @property
    def local(self) -> bool:
        """True for a local grid (module docstring, "Local grids")."""
        return self.mix is not None

    def y_of_s(self, s: np.ndarray, panel: np.ndarray | None = None) -> np.ndarray:
        """The point of GL variable s: R^{-1}(ψ_p(s)) (upper tail without cancellation).

        R is g_ref's CDF for the reference grid; for a local grid, the
        restricted CDF of each ``panel`` (same shape as ``s``).
        """
        s = np.asarray(s, dtype=float)
        if self.mix is None:
            t, tc, _ = _psi(s.reshape(1, -1), self.power)
            return _mixture_ppf(self.weights[None, :], self.dists, t, tc).reshape(s.shape)
        k = np.asarray(panel, dtype=int).ravel()
        y, _, _ = _panel_points(self.mix, k, s.ravel(), self.power, self.weights, self.dists)
        return y.reshape(s.shape)


def _psi(s: np.ndarray, p: int):
    """Endpoint transform t = s^p / (s^p + (1-s)^p): (t, 1 − t, dt/ds).

    p = 1 is the identity. For p ≥ 2, ψ' vanishes to order p − 1 at both ends,
    which turns an integrand behaving like t^β at t → 0 or 1 (an algebraic
    endpoint singularity, typical of a conditional density divided by g_ref)
    into one behaving like s^{p(β+1)−1}, restoring fast Gauss–Legendre
    convergence.
    """
    s = np.asarray(s, dtype=float)
    if p == 1:
        return s, 1.0 - s, np.ones_like(s)
    a, b = s ** p, (1.0 - s) ** p
    den = a + b
    dt = p * (s * (1.0 - s)) ** (p - 1) / (den * den)
    return a / den, b / den, dt


def _reference_mixture(model: PMCModel) -> tuple[np.ndarray, list]:
    """Weights and frozen components of g_ref = Σ_i μ(i, ·)."""
    p = model.prior_p
    K = model.K
    if model.margin_structure == "pair":
        pairs = [(p[i, j], model.margin(i, j)) for i in range(K) for j in range(K)]
    else:
        col = p.sum(axis=0)
        pairs = [(col[j], model.margin(j)) for j in range(K)]
    w = np.array([max(float(c[0]), 0.0) for c in pairs])
    keep = w > 0.0
    if not keep.any():
        raise ValueError("The prior gives zero probability to every state.")
    dists = [_frozen(mg) for (_, mg), k in zip(pairs, keep) if k]
    return w[keep] / w[keep].sum(), dists


#: Exponent p of the endpoint transform ψ_p of the Gauss–Legendre nodes (see
#: :func:`_psi`). p = 2 measured best overall on the Gaussian AR(1) reference
#: (pmcprg/tests/test_gaps.py): p = 1 converges only algebraically for weak or
#: moderate dependence (|ρ| ≲ 0.7), p = 3 under-resolves the interior for
#: ρ ≈ 0.99. Private switch kept for accuracy studies.
_T_POWER = 2


def reference_grid(model: PMCModel, gap_nodes: int | None = None) -> QuadratureGrid:
    """The G-node quadrature grid of the reference law g_ref of ``model``."""
    _check_grid_supported(model)
    G = _resolve_nodes(gap_nodes)
    x, W = _leg.leggauss(G)
    s = 0.5 * (x + 1.0)
    ws = 0.5 * W
    t, tc, dt = _psi(s, _T_POWER)
    w = ws * dt
    weights, dists = _reference_mixture(model)
    nodes = _mixture_ppf(weights[None, :], dists, t[None, :], tc[None, :])[0]
    ref_pdf = np.zeros(G)
    for c, dist in enumerate(dists):
        ref_pdf += weights[c] * dist.pdf(nodes)
    if not np.all(np.isfinite(ref_pdf) & (ref_pdf > 0.0)):
        raise ValueError(
            "reference_grid: the reference density g_ref vanishes at a "
            "quadrature node; the margins' supports leave a gap the grid "
            "cannot represent."
        )
    return QuadratureGrid(s=s, ws=ws, power=_T_POWER, t=t, w=w, nodes=nodes,
                          ref_pdf=ref_pdf, omega=w / ref_pdf, weights=weights,
                          dists=dists)


# ---------------------------------------------------------------------------
# Local grids: proposals adapted to the conditional laws around each gap
# ---------------------------------------------------------------------------

#: Private switch kept for accuracy studies: ``False`` puts every missing
#: position on the reference grid (the quadrature before local grids).
_LOCAL_GRIDS = True
#: Shares of the bridge, forward and backward groups of a local proposal
#: when both neighbours of the gap are observed (module docstring).
_W_GROUPS = (0.5, 0.25, 0.25)
#: Gaussian components kept per proposal, after merging.
_MAX_COMPONENTS = 6
#: Two components merge when their locations differ by < _MERGE_LOC times
#: the smaller scale and their scales by a factor < _MERGE_SCALE.
_MERGE_LOC = 0.25
_MERGE_SCALE = 1.25
#: A position keeps the reference grid when every proposal component of
#: weight ≥ _RESOLVE_MIN_WEIGHT has at least _RESOLVE_NODES reference nodes
#: within one scale of its location (module docstring, "Local grids").
_RESOLVE_NODES = 4
_RESOLVE_MIN_WEIGHT = 1e-4
#: Positions with a component spanning fewer reference nodes than this within
#: one scale are candidates for a local grid (:func:`_local_grid_list`).
_CANDIDATE_NODES = 4
#: A candidate switches to its local grid when the reference grid's error on
#: the proposal exceeds _SWITCH_TOL and the local grid's is _SWITCH_GAIN
#: times smaller (:func:`_local_grid_list`).
_SWITCH_TOL = 1e-6
#: A local grid of a missing y_1 must integrate the prior law to this
#: relative accuracy (:func:`_local_grid_list`).
_PRIOR_GUARD = 1e-2
_SWITCH_GAIN = 3.0
#: Private switches kept for accuracy studies (:func:`_local_grid_list`):
#: require the 3× gain on the proxy alone too (the rule before this
#: version), and let a position next to a local grid of its run switch as
#: soon as its local grid is better at all.
_PROXY_PREFILTER = False
_RUN_CONSISTENT = True
#: Share of the G nodes of a local grid on the background (g_ref) panels,
#: and the half-width of a core panel, in scales of its Gaussian.
_BACKGROUND_NODES = 0.35
_CORE_WIDTH = 8.0
#: Scale c of the sinh map y = m + c·s·sinh(u) of a core panel, in scales s
#: of its Gaussian: plain Gauss–Legendre in u integrates both a Gaussian
#: of scale s and a flat integrand accurately (32 nodes on m ± 8s: 1e-14
#: for N(m, s²), 5e-10 for N(m + 1.5s, 0.7²s²), exact for a constant).
_SINH_SCALE = 2.0
#: Adjacent core pieces share one panel when their Gaussians' scales differ
#: by a factor ≤ _NEST_RATIO and the panel spans at most _MERGE_SPAN scales
#: of the narrower; a narrower core inside a wider one, or a chain of cores
#: strung along the line, gets several panels.
_NEST_RATIO = 4.0
_MERGE_SPAN = 3.0 * _CORE_WIDTH
#: ... and when the sinh map of the merged panel (centred on the narrower,
#: scale c = _SINH_SCALE s) spaces its nodes at every merged piece's centre at
#: most _MERGE_RES times wider than the piece's own panel would: √(c² + D²)
#: ≤ _MERGE_RES c_piece, D the distance between the centres. Without it
#: (``None``, the rule before this version) a panel was centred on a light
#: piece with a heavier one 5.5 scales away (a spacing 2.9 times too wide):
#: 0.16 sd off in the means of a 17-row leading gap of the regime AR(1) at
#: τ = 0.999, G = 64.
_MERGE_RES = 1.5
#: Where the two differ, keep of the layouts with and without _MERGE_RES the
#: one whose grid integrates the proposal better (:func:`_better_layouts`):
#: merging nearby pieces of comparable weight into one wide panel integrates
#: their smooth mixture 300 times better on the Clayton τ = 0.9 brute-force
#: case, separating a narrow piece from its neighbours 20 times better on the
#: Gaussian τ = 0.99 one. Private switch kept for accuracy studies.
_LAYOUT_CHOICE = True
#: Fewest nodes of a core panel and of a background panel.
_MIN_CORE_NODES = 8
_MIN_BACKGROUND_NODES = 4
#: Standard normal levels of the location / scale summaries of a law:
#: m = Q(1/2), core scale s = (Q(Φ(1)) − Q(Φ(−1))) / 2 and tail extent
#: e = max_k (Q(Φ(k)) − Q(Φ(−k))) / (2k s) ≥ 1, k = 1, 2, 3: a skewed or
#: heavy-tailed law (a Clayton or Gumbel conditional law in its dependent
#: tail) keeps its narrow core scale — what the grid must resolve — and gets
#: a core panel wide enough for its tails; e = 1 for a Gaussian law.
_LEVELS = np.array([float(_ss.norm.cdf(k)) for k in (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)])
_MID = 3
_SQRT_2PI = float(np.sqrt(2.0 * np.pi))

_GL_CACHE: dict = {}


def _gl(n: int):
    """Gauss–Legendre rule of n nodes on (0, 1): (s, ws), cached."""
    r = _GL_CACHE.get(n)
    if r is None:
        x, W = _leg.leggauss(n)
        r = _GL_CACHE[n] = (0.5 * (x + 1.0), 0.5 * W)
    return r


def _ref_pdf(weights, dists, y: np.ndarray) -> np.ndarray:
    """g_ref(y), elementwise."""
    out = np.zeros(np.shape(y))
    for wr, dist in zip(weights, dists):
        with np.errstate(all="ignore"):
            out = out + wr * dist.pdf(y)
    return out


def _ref_cdf_sf(weights, dists, y: np.ndarray):
    """G_ref(y) and 1 − G_ref(y) (from the survival functions), elementwise."""
    c = np.zeros(np.shape(y))
    sf = np.zeros(np.shape(y))
    for wr, dist in zip(weights, dists):
        with np.errstate(all="ignore"):
            c = c + wr * dist.cdf(y)
            sf = sf + wr * dist.sf(y)
    return c, sf


def _panel_points(pan: _Panels, k: np.ndarray, s: np.ndarray, power: int, weights, dists):
    """Points of GL variable ``s`` in panels ``k``: (y, pdf, dw) with ω = ws · dw / pdf.

    Core panel: u = t_lo + (t_hi − t_lo) s, y = loc + sc sinh(u), dw = t_hi −
    t_lo and pdf = 1 / (sc cosh u). Background panel: t = t_lo + Δ ψ(s) (its
    complement from ψ's, without cancellation), y = G_ref^{-1}(t), pdf =
    g_ref(y) and dw = Δ ψ'(s).
    """
    tlo, thi, tclo, tchi = pan.t_lo[k], pan.t_hi[k], pan.tc_lo[k], pan.tc_hi[k]
    loc, sc = pan.loc[k], pan.sc[k]
    core = np.isfinite(loc)
    y = np.empty(s.shape)
    pdf = np.empty(s.shape)
    dw = np.empty(s.shape)
    if core.any():
        du = thi[core] - tlo[core]
        u = tlo[core] + du * s[core]
        y[core] = loc[core] + sc[core] * np.sinh(u)
        pdf[core] = 1.0 / (sc[core] * np.cosh(u))
        dw[core] = du
    bg = ~core
    if bg.any():
        psi, psic, dpsi = _psi(s[bg], power)
        D = np.where(tlo[bg] <= 0.5, thi[bg] - tlo[bg], tclo[bg] - tchi[bg])
        t = tlo[bg] + D * psi
        tc = tchi[bg] + D * psic
        yy = _ref_ppf(weights, dists, t, tc)
        y[bg] = yy
        pdf[bg] = _ref_pdf(weights, dists, yy)
        dw[bg] = D * dpsi
    return y, pdf, dw


def _margin_law(model: PMCModel, i: int, j: int):
    """Frozen law of f_ij, the margin of y_n in the pair (x_n, x_{n+1}) = (i, j)."""
    return _frozen(model.margin(i, j) if model.margin_structure == "pair" else model.margin(i))


def _cond_ppf(model: PMCModel, i: int, j: int, y, t, *, reverse: bool = False) -> np.ndarray:
    """Quantiles (M, L) of the one-step conditional laws of the transition.

    Forward: y_{n+1} given (x_n, x_{n+1}, y_n) = (i, j, y), of density
    f_ji(y') c_ij(F_ij(y), F_ji(y')) — y' = F_ji^{-1}(h_ij^{-1}(t | F_ij(y))).
    ``reverse``: y_n given (x_n, x_{n+1}, y_{n+1}) = (i, j, y) under
    μ(i, y_n) q(j, y_{n+1} | i, y_n), of density f_ij(y_n) c_ij(F_ij(y_n),
    F_ji(y)) — the h-inverse of the transposed copula. Without a copula
    (PMC-IN) the law is the margin itself.
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    t = np.atleast_1d(np.asarray(t, dtype=float))
    if reverse:
        unk, cond = _margin_law(model, i, j), _margin_law(model, j, i)
    else:
        unk, cond = _margin_law(model, j, i), _margin_law(model, i, j)
    M, L = y.size, t.size
    with np.errstate(all="ignore"):
        if not model.variant.uses_copula:
            return np.broadcast_to(unk.ppf(t), (M, L)).copy()
        u = np.clip(cond.cdf(y), EPS, ONE_MINUS_EPS)
        cop = model.copula(i, j)
        if reverse:
            cop = cop.transposed()
        w = np.ascontiguousarray(np.broadcast_to(t[None, :], (M, L)).ravel())
        uu = np.ascontiguousarray(np.broadcast_to(u[:, None], (M, L)).ravel())
        v = np.asarray(cop.inv_h_array(w, uu), dtype=float).reshape(M, L)
        return unk.ppf(v)


def _summaries(q: np.ndarray, *, extent: bool = False):
    """Location Q(1/2) and core scale (Q(Φ(1)) − Q(Φ(−1))) / 2 from quantiles at
    _LEVELS; with ``extent`` also the tail extent (the comment of _LEVELS)."""
    loc = q[..., _MID]
    sc = np.maximum(0.5 * (q[..., _MID + 1] - q[..., _MID - 1]),
                    64.0 * EPS * np.maximum(1.0, np.abs(loc)))
    if not extent:
        return loc, sc
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        wide = np.nanmax(np.stack([(q[..., _MID + k] - q[..., _MID - k]) / (2.0 * k)
                                   for k in (2, 3)]), axis=0)
    return loc, sc, np.where(np.isfinite(wide), np.maximum(wide / sc, 1.0), 1.0)


def _entry_components(model: PMCModel, y: np.ndarray, *, reverse: bool, w=None):
    """Components of the law of a missing y next to the observed value y.

    Forward (y = y_{a−1}, the position after it): one per (i, j), the exact
    conditional law of the transition i → j from y, weight a_i T_ij(y) with
    a_i ∝ μ(i, y) the state law given y alone, or a_i = ``w[:, i]`` the
    forward filter α̂_{a−1}(i) (module docstring, "Filter-weighted
    proposals"). ``reverse`` (y = y_{b+1}, the position before it): one per
    (i, j) with i the state at the missing position, the law of y_b given
    (i → j, y), weight ∝ p_ij f_ji(y), times ``w[:, j]`` = β̂_{b+1}(j) when
    given. With ``w`` the weights are computed in log space (a state whose
    margin puts y 40 sds out keeps its piece). Returns (lam, loc, sc) (P,
    K²), the state at the missing position and the stay flag (i == j) of
    every column, and the tail extents (P, K²) (:data:`_LEVELS`).
    """
    K = model.K
    y = np.asarray(y, dtype=float)
    P = y.size
    f, _ = _margin_eval(model, y, log=False)
    lam = np.zeros((P, K * K))
    loc = np.zeros((P, K * K))
    sc = np.ones((P, K * K))
    ex = np.ones((P, K * K))
    state = np.empty(K * K, dtype=int)
    stay = np.empty(K * K, dtype=bool)
    with np.errstate(all="ignore"):
        if w is not None:
            lf, _ = _margin_eval(model, y, log=True)
            lw = np.log(np.asarray(w, dtype=float))
            if reverse:
                lwg = np.log(model.prior_p)[None, :, :] + lf.transpose(0, 2, 1) + lw[:, None, :]
            else:
                lwg = lw[:, :, None] + _x_transition(model, lf, log=True)
            lwg = np.where(np.isnan(lwg), -np.inf, lwg)
            mx = lwg.max(axis=(1, 2), keepdims=True)
            wgt = np.where(np.isfinite(mx), np.exp(lwg - np.where(np.isfinite(mx), mx, 0.0)), 0.0)
        elif reverse:
            wgt = model.prior_p[None, :, :] * f.transpose(0, 2, 1)             # p_ij f_ji(y)
        else:
            a = _initial(model, f, log=False)
            a = a / np.where(a.sum(axis=1, keepdims=True) > 0.0, a.sum(axis=1, keepdims=True), 1.0)
            wgt = a[:, :, None] * _x_transition(model, f, log=False)
    for i in range(K):
        for j in range(K):
            c = i * K + j
            state[c] = i if reverse else j
            stay[c] = i == j
            w = wgt[:, i, j]
            if not np.any(w > 0.0):
                continue
            m, s, e = _summaries(_cond_ppf(model, i, j, y, _LEVELS, reverse=reverse), extent=True)
            ok = np.isfinite(w) & (w > 0.0) & np.isfinite(m) & np.isfinite(s)
            lam[:, c] = np.where(ok, w, 0.0)
            loc[:, c] = np.where(ok, m, 0.0)
            sc[:, c] = np.where(ok, s, 1.0)
            ex[:, c] = np.where(ok, e, 1.0)
    return lam, loc, sc, state, stay, ex


def _push(model: PMCModel, lam, loc, sc, state, stay, *, reverse: bool):
    """One step of the Gaussian-sum propagation of the components.

    Each component (state a, location m, scale s) goes through every
    transition a → k (forward) or k ← a (``reverse``): location Q(1/2 | m),
    scale² s_ak(m)² + (dQ(1/2 | y)/dy)² s² (the slope by central differences
    at m ± s). The K·C results of a position are merged by moments only
    where they have the same state and nearly the same location and scale
    (:func:`_merge_components`' criteria), then the K² heaviest are kept:
    paths that reach one state at different places — a single switch and a
    switch back, under margins whose quantile maps shift y by many
    conditional scales — stay separate narrow pieces. (Merged into 2K classes
    by state, as until this version, they made one broad piece between the
    places: 0.086 for a true scale 0.002 in the leading gap of
    ``report/erroneous_data/intel_lab/repro_leading_gap.py``.) ``stay``
    marks the components that never left their state since the anchor.
    Returns (lam, loc, sc) (P, K²), state and stay (P, K²), the one-step
    scale s_ak(m) of each component (weighted mean when merged): the width
    of the transition kernel into the position, which its grid must resolve
    too; last, the tail extents of the one-step laws (the larger when merged).
    """
    K = model.K
    P, C = lam.shape
    state = np.broadcast_to(np.asarray(state), (P, C))
    stay = np.broadcast_to(np.asarray(stay, dtype=bool), (P, C))
    NC = C * K
    Wt = np.zeros((P, NC))
    L1 = np.zeros((P, NC))
    S2 = np.ones((P, NC))
    SK = np.ones((P, NC))
    SE = np.ones((P, NC))
    ST = np.zeros((P, NC), dtype=int)
    SY = np.zeros((P, NC), dtype=bool)
    if reverse:
        p = model.prior_p
        Rv = p / np.where(p.sum(axis=0, keepdims=True) > 0.0, p.sum(axis=0, keepdims=True), 1.0)
    # every (position, component) of state a at once, per transition a → k
    for a in np.unique(state):
        a = int(a)
        rows, cols = np.nonzero((lam > 0.0) & (state == a))
        if rows.size == 0:
            continue
        n = rows.size
        m, s = loc[rows, cols], sc[rows, cols]
        ys = np.concatenate([m, m - s, m + s])
        if not reverse:
            f, _ = _margin_eval(model, m, log=False)
            Tm = _x_transition(model, f, log=False)
        for k in range(K):
            wt = np.full(n, Rv[k, a]) if reverse else Tm[:, a, k]
            if not np.any(wt > 0.0):
                continue
            i_, j_ = (k, a) if reverse else (a, k)
            q = _cond_ppf(model, i_, j_, ys, _LEVELS, reverse=reverse)
            l1, s1, e1 = _summaries(q[:n], extent=True)
            with np.errstate(all="ignore"):
                slope = (q[2 * n:, _MID] - q[n:2 * n, _MID]) / (2.0 * s)
                s2 = np.sqrt(s1 * s1 + slope * slope * s * s)
            ww = lam[rows, cols] * wt
            ok = np.isfinite(l1) & np.isfinite(s2) & np.isfinite(ww) & (ww > 0.0)
            r, col = rows[ok], cols[ok] * K + k
            Wt[r, col], L1[r, col], S2[r, col] = ww[ok], l1[ok], s2[ok]
            SK[r, col], SE[r, col], ST[r, col] = s1[ok], e1[ok], k
            SY[r, col] = stay[r, cols[ok]] & (a == k)
    S2 = np.maximum(S2, 64.0 * EPS * np.maximum(1.0, np.abs(L1)))
    lam2, loc2, sc2, ex2, ks2, st2, sy2 = _merge_components(Wt, L1, S2, SE, ks=SK, state=ST, stay=SY,
                                                            keep=K * K)
    return lam2, loc2, sc2, st2, sy2, ks2, ex2


def _bridges(F, B, model: PMCModel | None = None):
    """Products of the forward and backward components of a common state.

    F, B = (lam, loc, sc, state, extent) with per-position state arrays (P, C).
    The product of N(m_F, s_F²) and N(m_B, s_B²) — the law of a y pinned by
    its two neighbours — weighted by λ_F λ_B N(m_F − m_B; 0, s_F² + s_B²).
    With ``model`` the weight is also divided by the prior density μ(k, m)
    of the common state k at the product's location m: the forward piece
    stands for p(x_n, y_n | past) and the backward one for p(future | x_n,
    y_n) μ(x_n, y_n), so their product counts the prior of (x_n, y_n) twice
    (module docstring, "Filter-weighted proposals"). Returns (lam, loc, sc,
    extent) (P, C_F·C_B), lam normalised per position (the larger of the two
    extents).
    """
    lF, mF, sF, stF, eF = F
    lB, mB, sB, stB, eB = B
    vF, vB = (sF * sF)[:, :, None], (sB * sB)[:, None, :]
    v = vF + vB
    same = (stF[:, :, None] == stB[:, None, :]) & (lF[:, :, None] > 0.0) & (lB[:, None, :] > 0.0)
    with np.errstate(all="ignore"):
        d = mF[:, :, None] - mB[:, None, :]
        lw = (np.log(lF)[:, :, None] + np.log(lB)[:, None, :]
              - 0.5 * d * d / v - 0.5 * np.log(v))
        lw = np.where(same, lw, -np.inf)
        loc = (mF[:, :, None] * vB + mB[:, None, :] * vF) / v
        sc = np.sqrt(vF * vB / v)
        if model is not None:
            r, cf, cb = np.nonzero(same & np.isfinite(lw))
            if r.size:
                lf, _ = _margin_eval(model, loc[r, cf, cb], log=True)
                lmu = _initial(model, lf, log=True)[np.arange(r.size),
                                                    np.broadcast_to(stF, lF.shape)[r, cf]]
                lw[r, cf, cb] = np.where(np.isfinite(lmu), lw[r, cf, cb] - lmu, -np.inf)
        mx = lw.max(axis=(1, 2), keepdims=True)
        w = np.where(np.isfinite(mx), np.exp(lw - np.where(np.isfinite(mx), mx, 0.0)), 0.0)
    P = lF.shape[0]
    w = w.reshape(P, -1)
    tot = w.sum(axis=1, keepdims=True)
    w = w / np.where(tot > 0.0, tot, 1.0)
    ex = np.maximum(eF[:, :, None], eB[:, None, :]).reshape(P, -1)
    return (w, np.where(w > 0.0, loc.reshape(P, -1), 0.0), np.where(w > 0.0, sc.reshape(P, -1), 1.0),
            np.where(w > 0.0, ex, 1.0))


def _rows(x, keep):
    """Per-position arrays (P, C) restricted to the rows ``keep``; shared (C,) ones as they are."""
    x = np.asarray(x)
    return x[keep] if x.ndim == 2 else x


def _gap_proposals(model: PMCModel, yL, yR, L, *, fwd0=None, w_in=None, w_out=None,
                   with_bridged: bool = False):
    """Local proposals of the positions of P runs of missing rows (module docstring).

    ``yL``, ``yR`` (P,) the observed neighbours (NaN: none), ``L`` (P,) the
    run lengths; ``fwd0`` optional forward components (lam, loc, sc, state,
    stay) at the position before each run (a continued gap, see
    :mod:`pmcprg.pmc.outliers`) used instead of ``yL``. ``w_in``, ``w_out``
    (P, K): the forward filter at the row before each run and the backward
    message at the row after it (module docstring, "Filter-weighted
    proposals"); with either, the entry weights come from them (the state
    law given the neighbour alone where one is None) and the bridges are
    divided by the prior (:func:`_bridges`), and the kernels weighted by the
    bridges through them. ``with_bridged`` adds the mask (ΣL,) of the
    positions with bridges.

    Returns (lam, loc, sc, fwd, kern): the Gaussian components (ΣL, C) of
    every position in run order (λ summing to 1, or to 0 for a position with
    no observed neighbour), ``fwd`` the forward components of every position
    (to continue a run), ``kern`` the (weight, location, one-step scale)
    of the transition kernels into each position (:func:`_push`) and ``ex``
    the tail extents of the components (:data:`_LEVELS`).
    """
    K = model.K
    yL = np.asarray(yL, dtype=float)
    yR = np.asarray(yR, dtype=float)
    L = np.asarray(L, dtype=int)
    P = L.size
    filtered = w_in is not None or w_out is not None
    off = np.concatenate([[0], np.cumsum(L)])
    Mt = int(off[-1])
    CF = K * K
    Fl, Fm, Fs = np.zeros((Mt, CF)), np.zeros((Mt, CF)), np.ones((Mt, CF))
    Fst = np.zeros((Mt, CF), dtype=int)
    Fsy = np.zeros((Mt, CF), dtype=bool)
    Fk = np.ones((Mt, CF))
    Fe = np.ones((Mt, CF))
    Bl, Bm, Bs = np.zeros((Mt, CF)), np.zeros((Mt, CF)), np.ones((Mt, CF))
    Bst = np.zeros((Mt, CF), dtype=int)
    Bk = np.ones((Mt, CF))
    Be = np.ones((Mt, CF))

    def put(dst, rows, vals, width):
        dst[0][rows, :width], dst[1][rows, :width], dst[2][rows, :width] = vals[0], vals[1], vals[2]

    # forward components, depth by depth
    hasF = np.isfinite(yL) if fwd0 is None else np.ones(P, dtype=bool)
    cur, runs = None, np.nonzero(hasF & (L >= 1))[0]
    if runs.size:
        if fwd0 is None:
            lam, loc, sc, st, sy, es = _entry_components(
                model, yL[runs], reverse=False, w=None if w_in is None else np.asarray(w_in)[runs])
            ks = sc
        else:
            lam, loc, sc, st, sy, ks, es = _push(
                model, *[np.asarray(x)[runs] if np.ndim(x) == 2 else x for x in fwd0], reverse=False)
        cur = (lam, loc, sc, st, sy, ks, es)
    d = 1
    while runs.size:
        lam, loc, sc, st, sy, ks, es = cur
        rows = off[runs] + d - 1
        put((Fl, Fm, Fs), rows, (lam, loc, sc), lam.shape[1])
        Fst[rows, :lam.shape[1]] = st
        Fsy[rows, :lam.shape[1]] = sy
        Fk[rows, :lam.shape[1]] = ks
        Fe[rows, :lam.shape[1]] = es
        keep = L[runs] > d
        runs = runs[keep]
        if not runs.size:
            break
        cur = _push(model, lam[keep], loc[keep], sc[keep], _rows(st, keep), _rows(sy, keep),
                    reverse=False)
        d += 1
    # backward components, depth by depth from the right neighbour
    runs = np.nonzero(np.isfinite(yR) & (L >= 1))[0]
    if runs.size:
        lam, loc, sc, st, sy, es = _entry_components(
            model, yR[runs], reverse=True, w=None if w_out is None else np.asarray(w_out)[runs])
        ks = sc
        d = 1
        while True:
            rows = off[runs + 1] - d
            put((Bl, Bm, Bs), rows, (lam, loc, sc), lam.shape[1])
            Bst[rows, :lam.shape[1]] = st
            Bk[rows, :lam.shape[1]] = ks
            Be[rows, :lam.shape[1]] = es
            keep = L[runs] > d
            runs = runs[keep]
            if not runs.size:
                break
            lam, loc, sc, st, sy, ks, es = _push(model, lam[keep], loc[keep], sc[keep],
                                                 _rows(st, keep), _rows(sy, keep), reverse=True)
            d += 1

    def norm(x):
        tot = x.sum(axis=1, keepdims=True)
        return x / np.where(tot > 0.0, tot, 1.0), tot[:, 0] > 0.0

    Fl, hF = norm(Fl)
    Bl, hB = norm(Bl)
    Pl, Pm, Ps, Pe = _bridges((Fl, Fm, Fs, Fst, Fe), (Bl, Bm, Bs, Bst, Be),
                              model if filtered else None)
    hP = Pl.sum(axis=1) > 0.0
    share = np.array(_W_GROUPS, dtype=float)[None, :] * np.stack([hP, hF, hB], axis=1)
    tot = share.sum(axis=1, keepdims=True)
    share = share / np.where(tot > 0.0, tot, 1.0)
    lam = np.concatenate([Pl * share[:, :1], Fl * share[:, 1:2], Bl * share[:, 2:]], axis=1)
    loc = np.concatenate([Pm, Fm, Bm], axis=1)
    sc = np.concatenate([Ps, Fs, Bs], axis=1)
    ex = np.concatenate([Pe, Fe, Be], axis=1)
    lam, loc, sc, ex = _merge_components(lam, loc, sc, ex)
    tot = lam.sum(axis=1, keepdims=True)
    lam = np.where(tot > 0.0, lam / np.where(tot > 0.0, tot, 1.0), 0.0)
    # kernel test pieces: the one-step laws into the position (its grid must
    # resolve each of them, not only their mixture); with filter weights,
    # weighted by the bridges through them where there are bridges (a kernel
    # no bridge goes through carries no posterior mass: counted at its prior
    # weight, an unresolved kernel of weight 8e-3 made the local grid lose to
    # the reference grid at G = 512, 3.5e-4 nats off where it is 2e-10)
    kt = np.concatenate([Fl * share[:, 1:2] + 0.0, Bl * share[:, 2:]], axis=1)
    if filtered:
        Pw = Pl.reshape(Pl.shape[0], Fl.shape[1], Bl.shape[1])
        kt = np.where(hP[:, None], np.concatenate([Pw.sum(axis=2), Pw.sum(axis=1)], axis=1), kt)
    kt = kt / np.where(kt.sum(axis=1, keepdims=True) > 0.0, kt.sum(axis=1, keepdims=True), 1.0)
    kern = (kt, np.concatenate([Fm, Bm], axis=1), np.concatenate([Fk, Bk], axis=1))
    if with_bridged:
        return lam, loc, sc, (Fl, Fm, Fs, Fst, Fsy), kern, ex, hP
    return lam, loc, sc, (Fl, Fm, Fs, Fst, Fsy), kern, ex


def _merge_components(lam, loc, sc, ex, *, ks=None, state=None, stay=None, keep=None):
    """Merge near-identical components, then keep the ``keep`` (default
    _MAX_COMPONENTS) heaviest.

    Components in decreasing weight; each merges (by moments) into the
    heaviest earlier one whose location is within _MERGE_LOC of the smaller
    scale and whose scale is within a factor _MERGE_SCALE — and, with
    ``state``, of the same state (for :func:`_push`, which also passes the
    one-step scales ``ks``, merged by weighted mean, and the ``stay`` flags,
    kept where both are set). Returns (lam, loc, sc, ex) and, with
    ``state``, (ks, state, stay) too.
    """
    keep = _MAX_COMPONENTS if keep is None else keep
    order = np.argsort(-lam, axis=1, kind="stable")

    def take(x):
        return None if x is None else np.take_along_axis(x, order, axis=1).copy()

    lam, loc, sc, ex = take(lam), take(loc), take(sc), take(ex)
    ks, state, stay = take(ks), take(state), take(stay)
    C = lam.shape[1]
    for c in range(1, C):
        live = lam[:, c] > 0.0
        if not live.any():
            break
        lo, so = loc[:, :c], sc[:, :c]
        smin = np.minimum(so, sc[:, c:c + 1])
        close = ((lam[:, :c] > 0.0) & (np.abs(lo - loc[:, c:c + 1]) <= _MERGE_LOC * smin)
                 & (np.maximum(so, sc[:, c:c + 1]) <= _MERGE_SCALE * smin))
        if state is not None:
            close &= state[:, :c] == state[:, c:c + 1]
        rows = np.nonzero(live & close.any(axis=1))[0]
        if not rows.size:
            continue
        tgt = np.argmax(close[rows], axis=1)
        w1, w2 = lam[rows, tgt], lam[rows, c]
        m1, m2 = loc[rows, tgt], loc[rows, c]
        v1, v2 = sc[rows, tgt] ** 2, sc[rows, c] ** 2
        w = w1 + w2
        m = (w1 * m1 + w2 * m2) / w
        v = (w1 * (v1 + m1 * m1) + w2 * (v2 + m2 * m2)) / w - m * m
        lam[rows, tgt], loc[rows, tgt] = w, m
        sc[rows, tgt] = np.sqrt(np.maximum(v, np.minimum(v1, v2)))
        ex[rows, tgt] = np.maximum(ex[rows, tgt], ex[rows, c])
        if ks is not None:
            ks[rows, tgt] = (w1 * ks[rows, tgt] + w2 * ks[rows, c]) / w
        if stay is not None:
            stay[rows, tgt] &= stay[rows, c]
        lam[rows, c] = 0.0
    order = np.argsort(-lam, axis=1, kind="stable")[:, :keep]
    out = (take(lam), take(loc), take(sc), take(ex))
    if state is None:
        return out
    out = out + (take(ks), take(state), take(stay))
    # dead columns: neutral values (as the zero-weight classes before)
    dead = out[0] <= 0.0
    for x, v in zip(out[1:], (0.0, 1.0, 1.0, 1.0, 0, False)):
        x[dead] = v
    return out


def _split(total: int, score: np.ndarray, floor: int) -> np.ndarray:
    """Integers ≥ 1 summing to ``total``: ``floor`` each (at most half of the
    total between them), the rest ∝ ``score`` by largest remainders."""
    n = score.size
    base = np.full(n, max(1, min(floor, total // (2 * n))))
    extra = (total - base.sum()) * score / score.sum() if score.sum() > 0 else np.zeros(n)
    out = base + np.floor(extra).astype(int)
    rem = total - out.sum()
    if rem > 0:
        out[np.argsort(-(extra - np.floor(extra)), kind="stable")[:rem]] += 1
    return out


def _panel_layout(lam, loc, sc, core, G, cdf_at, support=(-np.inf, np.inf), ex=None,
                  res=None, both: bool = False):
    """Panels of one local grid (module docstring, "Local grids").

    Core panels: the intervals m_c ± κ e_c s_c of the ``core`` components cut
    at each other's ends, every piece on the component whose density
    λ_c N(y; m_c, s_c²) dominates at its middle, or on a component
    _NEST_RATIO times narrower than that one when it peaks higher inside the
    piece or holds more of the piece's mass; adjacent pieces merge while
    their scales stay within _NEST_RATIO and the panel within _MERGE_SPAN
    scales — and, with ``res``, while the panel's sinh map resolves each of
    them (_MERGE_RES).
    Background panels (g_ref): the rest of the line. Cores are clipped to
    the ``support`` of g_ref. ``cdf_at(y)`` gives (G_ref(y), 1 − G_ref(y)).
    Returns the :class:`_Panels` fields as lists (sizes summing to G), or
    None when there is no core; with ``both``, the layouts with ``res`` and
    without it (the elementary pieces computed once).
    """
    ex = np.ones_like(sc) if ex is None else ex
    c = np.nonzero(core)[0]
    lo = np.maximum(loc[c] - _CORE_WIDTH * ex[c] * sc[c], support[0])
    hi = np.minimum(loc[c] + _CORE_WIDTH * ex[c] * sc[c], support[1])
    c, lo, hi = c[lo < hi], lo[lo < hi], hi[lo < hi]
    if not c.size:
        return (None, None) if both else None
    pts = np.unique(np.concatenate([lo, hi]))
    # the elementary pieces between consecutive ends, all at once: (E, C)
    # arrays over the pieces and the core components c
    ea, eb = pts[:-1], pts[1:]
    mid = 0.5 * (ea + eb)
    lc, sl, lw = loc[c], sc[c], sc[c] * ex[c]
    cov = (lo[None, :] <= mid[:, None]) & (mid[:, None] <= hi[None, :])
    rows = np.arange(mid.size)
    with np.errstate(divide="ignore"):
        loglam = np.log(lam[c])
    # the component that dominates the mixture density there drives the piece
    z = (mid[:, None] - lc[None, :]) / sl[None, :]
    score = np.where(cov, loglam[None, :] - 0.5 * z * z - np.log(sl)[None, :], -np.inf)
    d = np.argmax(score, axis=1)
    # ... unless a component _NEST_RATIO times narrower, which the driver's
    # panel does not resolve, peaks higher inside the piece or holds more of
    # the mixture's mass on it: judged at the middle alone, a narrow
    # component whose centre other cores' ends put off the middle, or whose
    # core tail fills a piece, drove no panel there (runs of Intel motes 47
    # and 48 off by 0.2 to 2 nats at G = 64)
    inside = (ea[:, None] <= lc[None, :]) & (lc[None, :] <= eb[:, None])
    nest = cov & inside & (_NEST_RATIO * lw[None, :] <= lw[d][:, None])
    peak = np.where(nest, (loglam - np.log(sl))[None, :], score)
    d = np.where(nest.any(axis=1), np.argmax(peak, axis=1), d)
    za, zb = (ea[:, None] - lc[None, :]) / sl[None, :], (eb[:, None] - lc[None, :]) / sl[None, :]
    mass = lam[c][None, :] * np.where(za > 0, _sp_ndtr(-za) - _sp_ndtr(-zb), _sp_ndtr(zb) - _sp_ndtr(za))
    narrow = cov & (_NEST_RATIO * lw[None, :] <= lw[d][:, None])
    mn = np.where(narrow, mass, -np.inf)
    alt = np.argmax(mn, axis=1)
    d = np.where(narrow.any(axis=1) & (mn[rows, alt] > mass[rows, d]), alt, d)
    elem = zip(ea, eb, np.where(cov.any(axis=1), c[d], -1).tolist())
    pieces = []                                 # (a, b, driver or −1)
    for a, b, drv in elem:
        if pieces and pieces[-1][2] == drv:
            pieces[-1] = (pieces[-1][0], b, drv)
        else:
            pieces.append((a, b, drv))
    pieces = [(-np.inf, pts[0], -1)] + pieces + [(pts[-1], np.inf, -1)]

    out = [_merged_layout(pieces, lam, loc, sc, ex, G, cdf_at, r)
           for r in ((res, None) if both else (res,))]
    return tuple(out) if both else out[0]


def _merged_layout(pieces, lam, loc, sc, ex, G, cdf_at, res):
    """:func:`_panel_layout` from its elementary pieces (a, b, driver): merged,
    the sinh maps resolving every merged piece within ``res`` (_MERGE_RES;
    None: no such condition), and the nodes allocated."""
    def resolved(drv, members):
        # the sinh map of ``drv`` resolves every member within ``res`` of its
        # own (_MERGE_RES)
        if res is None:
            return True
        ms = np.fromiter(members, dtype=int)
        span = np.hypot(_SINH_SCALE * sc[drv], loc[ms] - loc[drv])
        return bool(np.all(span <= res * _SINH_SCALE * sc[ms]))

    merged = []                          # (a, b, driver, smallest, largest scale, members)
    for a, b, d in pieces:
        sd = sc[d] * ex[d] if d >= 0 else np.nan
        if merged and merged[-1][2] == -1 and d == -1:
            merged[-1] = (merged[-1][0], b, -1, np.nan, np.nan, ())
            continue
        if merged and merged[-1][2] >= 0 and d >= 0:
            a0, _, d0, lo0, hi0, mem = merged[-1]
            nd = d0 if sc[d0] * ex[d0] <= sd else d
            if (max(hi0, sd) <= _NEST_RATIO * min(lo0, sd)
                    and b - a0 <= _MERGE_SPAN * min(lo0, sd)
                    and resolved(nd, mem + (d,))):
                merged[-1] = (a0, b, nd, min(lo0, sd), max(hi0, sd), mem + (d,))
                continue
        merged.append((a, b, d, sd, sd, (d,) if d >= 0 else ()))
    merged = [(a, b, d) for a, b, d, _, _, _ in merged]
    rows = []
    for a, b, d in merged:
        if d >= 0:
            m, c_ = loc[d], _SINH_SCALE * sc[d]
            ua, ub = np.arcsinh((a - m) / c_), np.arcsinh((b - m) / c_)
            # the share of the component's mass on the panel, for the allocation
            w_ = _sp_ndtr((b - m) / sc[d]) - _sp_ndtr((a - m) / sc[d])
            rows.append((m, c_, ua, ub, w_, 0.0, lam[d]))
        else:
            ca, sa = cdf_at(a)
            cb, sb = cdf_at(b)
            rows.append((np.nan, np.nan, ca, cb, sa, sb, 0.0))
    if len(rows) > G // 2:
        return None
    rows = np.array(rows, dtype=float)
    bg = ~np.isfinite(rows[:, 0])
    width = np.where(bg, np.where(rows[:, 2] <= 0.5, rows[:, 3] - rows[:, 2], rows[:, 4] - rows[:, 5]),
                     rows[:, 4])
    keep = ~bg | (width > 1e-15)
    rows, width, bg = rows[keep], width[keep], bg[keep]
    sizes = np.zeros(len(rows), dtype=int)
    n_bg = int(round(_BACKGROUND_NODES * G)) if bg.any() else 0
    if bg.any():
        sizes[bg] = _split(n_bg, np.sqrt(width[bg]), _MIN_BACKGROUND_NODES)
    if (~bg).any():
        sizes[~bg] = _split(G - n_bg, np.sqrt(rows[~bg, 6] * np.maximum(width[~bg], 0.0)),
                            _MIN_CORE_NODES)
    return rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3], rows[:, 4], rows[:, 5], sizes


def _single_core_layouts(lam, loc, sc, core, G, cdf, sf, sup, ex, res=None) -> dict:
    """:func:`_panel_layout` of the positions whose core intervals form one
    cluster of scales within _NEST_RATIO — one core panel between two
    background panels — computed for all of them at once (the common case).

    ``cdf``, ``sf``: g_ref at the clipped core ends, in the order of
    :func:`_panel_grids` (all lower ends, then all upper ends). Returns
    {position: layout}; the others go through :func:`_panel_layout`.
    """
    P, C = lam.shape
    lo = np.where(core, np.maximum(loc - _CORE_WIDTH * ex * sc, sup[0]), np.inf)
    hi = np.where(core, np.minimum(loc + _CORE_WIDTH * ex * sc, sup[1]), -np.inf)
    nc = core.sum()
    cl = np.full((P, C), np.nan)
    ch = np.full((P, C), np.nan)
    cs = np.full((P, C), np.nan)
    cl[core], ch[core] = cdf[:nc], cdf[nc:]
    sl = np.full((P, C), np.nan)
    sh = np.full((P, C), np.nan)
    sl[core], sh[core] = sf[:nc], sf[nc:]
    del cs
    ok = core.any(axis=1) & np.all(~core | (lo < hi), axis=1)
    # one cluster: sorted by lower end, every lower end below the running upper end
    order = np.argsort(np.where(core, lo, np.inf), axis=1, kind="stable")
    los = np.take_along_axis(lo, order, axis=1)
    his = np.take_along_axis(hi, order, axis=1)
    run = np.maximum.accumulate(np.where(np.isfinite(his), his, -np.inf), axis=1)
    gapped = (los[:, 1:] > run[:, :-1]) & np.isfinite(los[:, 1:])
    ok &= ~gapped.any(axis=1)
    smin = np.where(core, ex * sc, np.inf).min(axis=1)
    smax = np.where(core, ex * sc, -np.inf).max(axis=1)
    ok &= smax <= _NEST_RATIO * smin
    ok &= np.where(core, hi, -np.inf).max(axis=1) - np.where(core, lo, np.inf).min(axis=1) \
        <= _MERGE_SPAN * smin
    d = np.argmin(np.where(core, sc, np.inf), axis=1)
    if res is not None:
        # the panel's sinh map (the narrowest's) resolves every core (resolved()
        # of _panel_layout)
        r = np.arange(P)
        span = np.hypot(_SINH_SCALE * sc[r, d][:, None], loc - loc[r, d][:, None])
        ok &= np.all(~core | (span <= res * _SINH_SCALE * sc), axis=1)
    a = lo.min(axis=1)
    b = hi.max(axis=1)
    ia, ib = np.argmin(lo, axis=1), np.argmax(hi, axis=1)
    r = np.arange(P)
    ca, sa = cl[r, ia], sl[r, ia]
    cb, sb = ch[r, ib], sh[r, ib]
    wl = ca                                                        # g_ref mass below a
    wu = np.where(cb <= 0.5, 1.0 - cb, sb)                         # above b
    ok &= (wl > 1e-15) & (wu > 1e-15)
    out = {}
    idx = np.nonzero(ok)[0]
    if not idx.size:
        return out
    n_bg = int(round(_BACKGROUND_NODES * G))
    for p in idx:
        m, c_ = loc[p, d[p]], _SINH_SCALE * sc[p, d[p]]
        ua, ub = np.arcsinh((a[p] - m) / c_), np.arcsinh((b[p] - m) / c_)
        w_ = _sp_ndtr((b[p] - m) / sc[p, d[p]]) - _sp_ndtr((a[p] - m) / sc[p, d[p]])
        bgs = _split(n_bg, np.sqrt(np.array([wl[p], wu[p]])), _MIN_BACKGROUND_NODES)
        out[int(p)] = (np.array([np.nan, m, np.nan]), np.array([np.nan, c_, np.nan]),
                       np.array([0.0, ua, cb[p]]), np.array([ca[p], ub, 1.0]),
                       np.array([1.0, w_, sb[p]]), np.array([sa[p], 0.0, 0.0]),
                       np.array([bgs[0], G - n_bg, bgs[1]]))
    return out


def _panel_grids(ref: QuadratureGrid, lam, loc, sc, G: int, ex=None, res=None, alt=False):
    """The local grids (G nodes each) of P proposals (module docstring, "Local grids").

    ``res``: the _MERGE_RES of the layouts. With ``alt`` (and ``res``), also
    the grids of the layouts without it, None where they are the same:
    returns (grids, alternative grids).
    """
    P = lam.shape[0]
    ex = np.ones_like(sc) if ex is None else ex
    cnt = (np.searchsorted(ref.nodes, loc + sc, side="right")
           - np.searchsorted(ref.nodes, loc - sc, side="left"))
    core = (lam >= _RESOLVE_MIN_WEIGHT) & (cnt < _RESOLVE_NODES / _BACKGROUND_NODES)
    sup = _support(ref)
    # g_ref CDF at every core end, in one call
    hw = _CORE_WIDTH * ex * sc
    ends = np.clip(np.concatenate([(loc - hw)[core], (loc + hw)[core]]), *sup)
    cdf, sf = _ref_cdf_sf(ref.weights, ref.dists, ends)
    table = dict(zip(ends.tolist(), zip(cdf.tolist(), sf.tolist())))
    table[-np.inf], table[np.inf] = (0.0, 1.0), (1.0, 0.0)

    def cdf_at(y):
        return table[float(y)]

    alt = bool(alt) and res is not None
    fast = _single_core_layouts(lam, loc, sc, core, G, cdf, sf, sup, ex, res)
    fast0 = _single_core_layouts(lam, loc, sc, core, G, cdf, sf, sup, ex, None) if alt else {}
    layouts, others = [], []
    for p in range(P):
        L0 = None
        if p in fast:
            L = fast[p]                  # the same without res (a looser condition)
        elif alt and p in fast0:
            L, L0 = _panel_layout(lam[p], loc[p], sc[p], core[p], G, cdf_at, sup, ex[p], res), fast0[p]
        elif alt:
            L, L0 = _panel_layout(lam[p], loc[p], sc[p], core[p], G, cdf_at, sup, ex[p], res, both=True)
        else:
            L = _panel_layout(lam[p], loc[p], sc[p], core[p], G, cdf_at, sup, ex[p], res)
        same = (L0 is None or (L is not None and len(L[6]) == len(L0[6])
                               and all(np.array_equal(x, y, equal_nan=True) for x, y in zip(L, L0))))
        layouts.append(L)
        others.append(None if same else L0)
    grids = _layout_grids(ref, layouts, G)
    return (grids, _layout_grids(ref, others, G)) if alt else grids


def _layout_grids(ref: QuadratureGrid, layouts: list, G: int) -> list:
    """The grids of panel layouts (:func:`_panel_layout`; None: None)."""
    out = [None] * len(layouts)
    idx = [p for p, L in enumerate(layouts) if L is not None]
    if not idx:
        return out
    npan = np.array([len(layouts[p][6]) for p in idx])
    cat = [np.concatenate([layouts[p][f] for p in idx]) for f in range(7)]
    pan_all = _Panels(loc=cat[0], sc=cat[1], t_lo=cat[2], t_hi=cat[3], tc_lo=cat[4], tc_hi=cat[5],
                      sizes=cat[6].astype(int), panel=np.empty(0, dtype=int))
    sizes = pan_all.sizes
    k = np.repeat(np.arange(sizes.size), sizes)                        # panel of every node
    j = np.arange(k.size) - np.repeat(np.cumsum(sizes) - sizes, sizes)  # index within the panel
    s_ = np.empty(k.size)
    ws = np.empty(k.size)
    for m in np.unique(sizes):
        sel = sizes[k] == m
        sm, wm = _gl(int(m))
        s_[sel], ws[sel] = sm[j[sel]], wm[j[sel]]
    y, pdf, dw = _panel_points(pan_all, k, s_, ref.power, ref.weights, ref.dists)
    with np.errstate(all="ignore"):
        omega = ws * dw / pdf
    p0 = np.concatenate([[0], np.cumsum(npan)])
    for q, p in enumerate(idx):
        kp = slice(p0[q], p0[q + 1])
        sel = slice(int(sizes[:p0[q]].sum()), int(sizes[:p0[q + 1]].sum()))
        yy, oo = y[sel], omega[sel]
        ok = (np.all(np.isfinite(yy)) and np.all(np.isfinite(oo)) and np.all(oo >= 0.0)
              and np.all(np.diff(yy) >= 0.0) and yy.size == G)
        if not ok:
            continue
        pan = _Panels(loc=pan_all.loc[kp], sc=pan_all.sc[kp], t_lo=pan_all.t_lo[kp],
                      t_hi=pan_all.t_hi[kp], tc_lo=pan_all.tc_lo[kp], tc_hi=pan_all.tc_hi[kp],
                      sizes=sizes[kp], panel=k[sel] - p0[q])
        out[p] = QuadratureGrid(s=s_[sel], ws=ws[sel], power=ref.power, t=s_[sel],
                                w=ws[sel] * dw[sel], nodes=yy, ref_pdf=pdf[sel], omega=oo,
                                weights=ref.weights, dists=ref.dists, mix=pan)
    return out


def _support(ref: QuadratureGrid) -> tuple[float, float]:
    """Support (lo, hi) of g_ref."""
    lo, hi = np.inf, -np.inf
    for d in ref.dists:
        a, b = d.support()
        lo, hi = min(lo, float(a)), max(hi, float(b))
    return lo, hi


def _proposal_error(nodes: np.ndarray, omega: np.ndarray, lam, loc, sc,
                    support=(-np.inf, np.inf)) -> np.ndarray:
    """Quadrature error (P,) of a grid on its proposal's components.

    Σ_c λ_c (|∫ N_c − I_0| + |∫ N_c (y − m_c)² / s_c² − I_2|) / I_0, the
    integrals by the rule (``nodes``, ``omega``) of each position and I_0,
    I_2 their exact values on the ``support`` of g_ref: how well the grid
    integrates the Gaussian pieces that approximate the law of the missing y
    (module docstring, "Local grids"). Components with less than 1e-3 of
    their mass on the support are left out.
    """
    e, on = _piece_errors(nodes, omega, loc, sc, support)
    return (np.where((lam > 0.0) & on, lam, 0.0) * e).sum(axis=1)


#: Elements of a whole-sequence temporary of the grid construction — (P, C,
#: G) Gaussian pieces at the nodes, (P·G, K, K) kernels — beyond which it is
#: computed by blocks of rows (32 MB of float64). Row by row the operations
#: are the same (elementwise, and sums along the rows), so are the results.
#: On the Intel mote-48 window at G = 32 they took 232 MB (_piece_errors) and
#: 99 MB (_neighbour_error) at once (tracemalloc); they grow with G.
_CHUNK = 1 << 22


def _row_blocks(P: int, per_row: int):
    """Slices of at most _CHUNK elements (per_row each), covering range(P)."""
    step = max(1, _CHUNK // max(1, per_row))
    return [slice(a, min(P, a + step)) for a in range(0, P, step)]


def _piece_errors(nodes: np.ndarray, omega: np.ndarray, loc, sc, support=(-np.inf, np.inf)):
    """Relative errors (P, C) of :func:`_proposal_error`, component by
    component, and whether each has 1e-3 of its mass on the support
    (by blocks of rows, :data:`_CHUNK`)."""
    with np.errstate(all="ignore"):
        za, zb = (support[0] - loc) / sc, (support[1] - loc) / sc
        I0 = _sp_ndtr(zb) - _sp_ndtr(za)
        pa = np.where(np.isfinite(za), za * np.exp(-0.5 * za * za), 0.0) / _SQRT_2PI
        pb = np.where(np.isfinite(zb), zb * np.exp(-0.5 * zb * zb), 0.0) / _SQRT_2PI
        I2 = I0 - pb + pa
        P, C = loc.shape
        s0 = np.empty((P, C))
        s2 = np.empty((P, C))
        for r in _row_blocks(P, C * nodes.shape[1]):
            z = (nodes[r, None, :] - loc[r, :, None]) / sc[r, :, None]          # (P, C, G)
            f = np.exp(-0.5 * z * z) / (sc[r, :, None] * _SQRT_2PI) * omega[r, None, :]
            s0[r] = f.sum(axis=2)
            s2[r] = (f * z * z).sum(axis=2)
        e = (np.abs(s0 - I0) + np.abs(s2 - I2)) / I0
    return np.where(np.isfinite(e), e, 2.0), I0 > 1e-3


def _better_layouts(ref: QuadratureGrid, new: list, old: list, lam, loc, sc, kern) -> list:
    """Per position, the local grid of ``new`` or of ``old`` (two panel layouts
    of the same proposals; ``old`` None where the same) that integrates the
    proposal's components and kernels better (:func:`_piece_errors`,
    weighted as in :func:`_proposal_error`), the pieces that neither
    resolves (relative error ≥ 1: no node near them) left out of both sums."""
    out = list(new)
    diff = [p for p, b in enumerate(old) if b is not None]
    if not diff:
        return out
    one = [p for p in diff if new[p] is None]
    for p in one:
        out[p] = old[p]
    two = np.array([p for p in diff if new[p] is not None], dtype=int)
    if not two.size:
        return out
    sup = _support(ref)
    tot = []
    parts = [(lam[two], loc[two], sc[two])] + ([] if kern is None else
                                                [tuple(x[two] for x in kern)])
    errs = {}
    for name, gr in (("new", new), ("old", old)):
        nodes = np.array([gr[p].nodes for p in two])
        omega = np.array([gr[p].omega for p in two])
        errs[name] = [_piece_errors(nodes, omega, lo, s, sup) for _, lo, s in parts]
    for name in ("new", "old"):
        t = np.zeros(two.size)
        for k, (w, _, _) in enumerate(parts):
            e, on = errs[name][k]
            seen = (errs["new"][k][0] < 1.0) | (errs["old"][k][0] < 1.0)
            t += (np.where((w > 0.0) & on & seen, w, 0.0) * e).sum(axis=1)
        tot.append(t)
    for q, p in enumerate(two):
        if tot[1][q] < tot[0][q]:
            out[p] = old[p]
    return out


def _neighbour_error(model: PMCModel, nodes: np.ndarray, omega: np.ndarray,
                     yl: np.ndarray, yr: np.ndarray, *, split: bool = False, first=None,
                     w_entry=None, w_exit=None, exit_rel=None):
    """Relative error (P,) of grids on the integrals of their gap they must reproduce.

    Entry, where the row before the missing y is observed (``yl`` finite):
    Σ_g ω_g q(j, y_g | i, y_l) against its exact value T_ij(y_l), weighted by
    a_i T_ij (a_i ∝ μ(i, y_l); ``w_entry`` (P, K, K) instead when given) —
    the raw block masses of the Tauchen–Hussey renormalisation. Exit, where the
    row after is observed (``yr`` finite): Σ_g ω_g μ(i, y_g) q(k, y_r | i,
    y_g) against ∫ μ(i, y) q(k, y_r | i, y) dy = p_ik f_ki(y_r) (pair
    margins; π_i T_ik f_k(y_r) for state margins — the copula density
    integrates to 1 in its first argument), weighted by these exact masses
    (``w_exit`` (P, K, K) instead when given): the exit kernel is as narrow
    as the transition. Both use the exact kernels, so
    they see what the Gaussian proposal does not (support boundaries, skewed
    conditional laws). A non-finite value counts as an error 1. ``first``
    (P,) marks a missing y_1: Σ_g ω_g μ(i, y_g) against P(x_1 = i), counted
    with the entry part. Returns the sum (P,), or with ``split`` the entry
    and exit parts; ``exit_rel`` (P, K, K), when given, receives the
    relative errors of the exit integrals (:func:`_quadrature_report`).
    """
    K = model.K
    P, G = nodes.shape
    blocks = _row_blocks(P, G * K * K)
    if len(blocks) > 1:
        # by blocks of rows (:data:`_CHUNK`): every row is computed alone
        def cut(x, r):
            return None if x is None else x[r]

        e_in, e_out = np.zeros(P), np.zeros(P)
        for r in blocks:
            rel = None if exit_rel is None else exit_rel[r].copy()
            e_in[r], e_out[r] = _neighbour_error(
                model, nodes[r], omega[r], yl[r], yr[r], split=True, first=cut(first, r),
                w_entry=cut(w_entry, r), w_exit=cut(w_exit, r), exit_rel=rel)
            if exit_rel is not None:
                exit_rel[r] = rel
        return (e_in, e_out) if split else e_in + e_out
    parts = {"entry": np.zeros(P), "exit": np.zeros(P)}
    fN, FN = _margin_eval(model, nodes.ravel(), log=False)

    def take(F, idx):
        return None if F is None else F[idx]

    with np.errstate(all="ignore"):
        if first is not None and np.any(first):
            rows = np.nonzero(first)[0]
            iN = (rows[:, None] * G + np.arange(G)[None, :]).ravel()
            mu = _initial(model, fN[iN], log=False).reshape(rows.size, G, K)
            got = (mu * omega[rows][:, :, None]).sum(axis=1)                   # (n, K)
            exact = _initial(model, np.ones((1, K, K)), log=False)[0]
            ok = exact > 0.0
            rel = np.where(ok, np.abs(got / np.where(ok, exact, 1.0) - 1.0), 0.0)
            rel = np.where(np.isfinite(rel), rel, 1.0)
            parts["entry"][rows] += (np.where(ok, exact, 0.0) * rel).sum(axis=1) / exact[ok].sum()
        for side, y in (("entry", yl), ("exit", yr)):
            rows = np.nonzero(np.isfinite(y))[0]
            if not rows.size:
                continue
            fo, Fo = _margin_eval(model, y[rows], log=False)
            rp = np.repeat(np.arange(rows.size), G)
            iN = (rows[:, None] * G + np.arange(G)[None, :]).ravel()
            om = omega[rows][:, :, None, None]
            if side == "entry":
                ker, _ = _kernel(model, fo[rp], take(Fo, rp), fN[iN], take(FN, iN))
                got = (ker.reshape(rows.size, G, K, K) * om).sum(axis=1)       # (n, K, K)
                exact = _x_transition(model, fo, log=False)
                if w_entry is None:
                    wgt = _initial(model, fo, log=False)[:, :, None] * exact
                else:
                    wgt = np.asarray(w_entry)[rows]
            else:
                ker, _ = _kernel(model, fN[iN], take(FN, iN), fo[rp], take(Fo, rp))
                mu = _initial(model, fN[iN], log=False)                          # (nG, K)
                got = ((ker * mu[:, :, None]).reshape(rows.size, G, K, K) * om).sum(axis=1)
                fki = fo.transpose(0, 2, 1)                                     # f_ki(y_r) at [i, k]
                if model.margin_structure == "pair":
                    exact = model.prior_p[None, :, :] * fki
                else:
                    pi = _initial(model, np.ones((1, K, K)), log=False)[0]
                    exact = pi[None, :, None] * _x_transition(model, fo, log=False) * fki
                wgt = exact if w_exit is None else np.asarray(w_exit)[rows]
            ok = np.isfinite(exact) & (exact > 0.0)
            rel = np.where(ok, np.abs(got / np.where(ok, exact, 1.0) - 1.0), 0.0)
            rel = np.where(np.isfinite(rel), rel, 1.0)
            if side == "exit" and exit_rel is not None:
                exit_rel[rows] = rel
            wgt = np.where(ok & np.isfinite(wgt), wgt, 0.0)
            tot = wgt.sum(axis=(1, 2))
            parts[side][rows] += np.where(
                tot > 0.0, (wgt * rel).sum(axis=(1, 2)) / np.where(tot > 0.0, tot, 1.0), 0.0)
    if split:
        return parts["entry"], parts["exit"]
    return parts["entry"] + parts["exit"]


def _grid_error(model, ref, nodes, omega, lam, loc, sc, yl, yr, kern=None, exact=True,
                first=None, w_entry=None, w_exit=None, split_proxy=False, exit_rel=None):
    """(proxy, (entry, exit)) errors (P,) of grids: :func:`_proposal_error` on
    the Gaussian proxies of the law of y and on the transition kernels into
    the position (``kern``: weights, locations, one-step scales; with
    ``split_proxy`` the two parts apart), :func:`_neighbour_error` on the
    exact entry / exit integrals (0 where the gap has no observed neighbour,
    or without ``exact``; ``w_entry``, ``w_exit`` its weights per position,
    ``exit_rel`` (P, K, K) filled with the relative errors of the exits)."""
    first = np.zeros(nodes.shape[0], bool) if first is None else np.asarray(first, dtype=bool)
    near = np.isfinite(yl) | np.isfinite(yr) | first
    ex = (np.zeros(nodes.shape[0]), np.zeros(nodes.shape[0]))
    if exact and near.any():
        rel = None if exit_rel is None else np.full((int(near.sum()),) + exit_rel.shape[1:], np.nan)
        e_in, e_out = _neighbour_error(model, nodes[near], omega[near], yl[near], yr[near],
                                       split=True, first=first[near],
                                       w_entry=None if w_entry is None else w_entry[near],
                                       w_exit=None if w_exit is None else w_exit[near],
                                       exit_rel=rel)
        ex[0][near], ex[1][near] = e_in, e_out
        if exit_rel is not None:
            exit_rel[near] = rel
    sup = _support(ref)
    px = _proposal_error(nodes, omega, lam, loc, sc, sup)
    pk = np.zeros_like(px) if kern is None else _proposal_error(nodes, omega, *kern, sup)
    if split_proxy:
        return (px, pk), ex
    return px + pk, ex


def _local_grid_list(model: PMCModel, ref: QuadratureGrid, lam, loc, sc, yl=None, yr=None,
                     kern=None, first=None, ex=None, run=None, w_entry=None, w_exit=None,
                     err_out=None, res=None) -> list:
    """Grids of the proposals: the reference grid unless a local grid does better.

    Candidates are the positions with a component of weight ≥
    _RESOLVE_MIN_WEIGHT that has fewer than _CANDIDATE_NODES reference nodes
    within one scale of its location. A candidate gets its local grid
    (:func:`_panel_grids`) when the reference grid's error on the Gaussian
    proxies of the law of y (:func:`_proposal_error`, sensitive to the
    resolution of the posterior) exceeds _SWITCH_TOL and the local grid's
    total error — that proxy plus its error on the exact entry / exit
    integrals next to an observed row ``yl`` / ``yr`` (:func:`_neighbour_error`,
    which sees support boundaries and skewed laws the proxies miss) — is
    _SWITCH_GAIN times smaller than the reference grid's (module docstring,
    "Local grids"). The reference grid is kept wherever it resolves the
    laws of the gap — weak and moderate dependence — with results bit for bit
    unchanged. ``w_entry``, ``w_exit`` (Mt, K, K): the weights of the
    exact integrals (:func:`_neighbour_error`) of each position. ``err_out``
    (Mt, 2), when given, receives the proxy errors of the chosen grid of
    every position on the components and on the kernels (0 where no
    component is a candidate), for :func:`_refine_grids`. ``res``: the
    _MERGE_RES of the panel layouts (None: the layouts before this version,
    those of the proposals given the neighbours alone); with it, the better
    of the two layouts (:func:`_better_layouts`).
    """
    Mt = lam.shape[0]
    yl = np.full(Mt, np.nan) if yl is None else np.asarray(yl, dtype=float)
    yr = np.full(Mt, np.nan) if yr is None else np.asarray(yr, dtype=float)
    first = np.zeros(Mt, bool) if first is None else np.asarray(first, dtype=bool)
    out = [ref] * Mt
    heavy = lam >= _RESOLVE_MIN_WEIGHT
    cnt = (np.searchsorted(ref.nodes, loc + sc, side="right")
           - np.searchsorted(ref.nodes, loc - sc, side="left"))
    cand = np.any(heavy & (cnt < _CANDIDATE_NODES), axis=1)
    if kern is not None:
        kl, km, kss = kern
        kc = (np.searchsorted(ref.nodes, km + kss, side="right")
              - np.searchsorted(ref.nodes, km - kss, side="left"))
        cand |= np.any((kl >= _RESOLVE_MIN_WEIGHT) & (kc < _CANDIDATE_NODES), axis=1)
    idx = np.nonzero(cand)[0]
    if not idx.size:
        return out

    def ksel(rows):
        return None if kern is None else tuple(x[rows] for x in kern)

    def wsel(w, rows):
        return None if w is None else w[rows]

    (pp_ref, pk_ref), _ = _grid_error(model, ref, np.broadcast_to(ref.nodes, (idx.size, ref.G)),
                                      np.broadcast_to(ref.omega, (idx.size, ref.G)), lam[idx],
                                      loc[idx], sc[idx], yl[idx], yr[idx], ksel(idx), exact=False,
                                      split_proxy=True)
    p_ref = pp_ref + pk_ref
    if err_out is not None:
        err_out[idx] = np.stack([pp_ref, pk_ref], axis=1)
    keep = p_ref > _SWITCH_TOL
    idx, p_ref = idx[keep], p_ref[keep]
    if not idx.size:
        return out
    exi = None if ex is None else ex[idx]
    if res is not None and _LAYOUT_CHOICE:
        # the panels merged as before this version where they integrate the
        # proposal better (_MERGE_RES, _better_layouts)
        loc_g, alt = _panel_grids(ref, lam[idx], loc[idx], sc[idx], ref.G, exi, res, alt=True)
        loc_g = _better_layouts(ref, loc_g, alt, lam[idx], loc[idx], sc[idx], ksel(idx))
    else:
        loc_g = _panel_grids(ref, lam[idx], loc[idx], sc[idx], ref.G, exi, res)
    ok = np.array([g is not None for g in loc_g], dtype=bool)
    if not ok.any():
        return out
    sel = idx[ok]
    gl = [g for g in loc_g if g is not None]
    gn, go = np.array([g.nodes for g in gl]), np.array([g.omega for g in gl])
    xrel = np.full((sel.size, model.K, model.K), np.nan)
    (pp_loc, pk_loc), (xi_loc, xo_loc) = _grid_error(
        model, ref, gn, go, lam[sel], loc[sel], sc[sel], yl[sel], yr[sel], ksel(sel),
        w_entry=wsel(w_entry, sel), w_exit=wsel(w_exit, sel), split_proxy=True, exit_rel=xrel)
    p_loc = pp_loc + pk_loc
    x_loc = xi_loc + xo_loc
    pr = p_ref[ok]
    total_loc = _SWITCH_GAIN * (p_loc + x_loc)
    keep_ok = np.ones(sel.size, bool)
    # a missing y_1: the grid must carry the prior (the reference grid does, by
    # construction), to within _PRIOR_GUARD — a gross-failure guard
    fs = first[sel]
    if fs.any():
        nf = int(fs.sum())
        e_pr, _ = _neighbour_error(model, gn[fs], go[fs], np.full(nf, np.nan), np.full(nf, np.nan),
                                   split=True, first=np.ones(nf, bool))
        keep_ok[np.nonzero(fs)[0][e_pr > _PRIOR_GUARD]] = False
    # the reference grid's exact check where it can decide: next to an
    # observed row, and where the proxy alone does not already decide
    near = np.isfinite(yl[sel]) | np.isfinite(yr[sel])
    need = np.nonzero(keep_ok & near & ~(total_loc < pr))[0]
    x_ref = np.zeros(sel.size)
    if need.size:
        r = sel[need]
        e_in, e_out = _neighbour_error(model, np.broadcast_to(ref.nodes, (r.size, ref.G)),
                                       np.broadcast_to(ref.omega, (r.size, ref.G)), yl[r], yr[r],
                                       split=True, w_entry=wsel(w_entry, r),
                                       w_exit=wsel(w_exit, r))
        x_ref[need] = e_in + e_out
    switch = keep_ok & (total_loc < pr + x_ref)
    if _PROXY_PREFILTER:
        switch &= _SWITCH_GAIN * p_loc < pr
    if run is not None and _RUN_CONSISTENT:
        # inside a run, a position between local grids takes its local grid as
        # soon as it is better at all: a reference grid there relays a narrow
        # law between two narrow grids through its coarse nodes (a 25-row gap
        # of the Intel robust-estimation window, 12.5 nats off with two such
        # positions)
        run = np.asarray(run)
        idx = {int(k): q for q, k in enumerate(sel)}
        better = keep_ok & (p_loc + x_loc < pr + x_ref)
        local = np.zeros(Mt, bool)
        local[sel[switch]] = True
        changed = True
        while changed:
            changed = False
            for q in np.nonzero(better & ~switch)[0]:
                k = int(sel[q])
                if ((k > 0 and run[k - 1] == run[k] and local[k - 1])
                        or (k + 1 < Mt and run[k + 1] == run[k] and local[k + 1])):
                    switch[q] = local[k] = changed = True
    for q, (k, g) in enumerate(zip(sel, gl)):
        if switch[q]:
            out[k] = replace(g, exit_rel=xrel[q]) if np.isfinite(yr[k]) else g
            if err_out is not None:
                err_out[k] = (pp_loc[q], pk_loc[q])
    return out


def _fine_grid(ref: QuadratureGrid, ref_fine: QuadratureGrid, grid: QuadratureGrid,
               factor: int) -> QuadratureGrid:
    """The same panels with ``factor`` times more nodes each (Nyström grid)."""
    if grid.mix is None:
        return ref_fine
    pan = grid.mix
    sizes = factor * pan.sizes
    k = np.repeat(np.arange(sizes.size), sizes)
    j = np.arange(k.size) - np.repeat(np.cumsum(sizes) - sizes, sizes)
    s_ = np.empty(k.size)
    ws = np.empty(k.size)
    for m in np.unique(sizes):
        sel = sizes[k] == m
        sm, wm = _gl(int(m))
        s_[sel], ws[sel] = sm[j[sel]], wm[j[sel]]
    y, pdf, dw = _panel_points(pan, k, s_, ref.power, ref.weights, ref.dists)
    with np.errstate(all="ignore"):
        omega = ws * dw / pdf
    fpan = _Panels(loc=pan.loc, sc=pan.sc, t_lo=pan.t_lo, t_hi=pan.t_hi, tc_lo=pan.tc_lo,
                   tc_hi=pan.tc_hi, sizes=sizes, panel=k)
    return QuadratureGrid(s=s_, ws=ws, power=ref.power, t=s_, w=ws * dw,
                          nodes=y, ref_pdf=pdf, omega=omega, weights=ref.weights,
                          dists=ref.dists, mix=fpan)


def _runs(miss: np.ndarray):
    """(starts, ends) of the maximal runs of True in ``miss`` (inclusive ends)."""
    m = np.concatenate([[False], np.asarray(miss, dtype=bool), [False]])
    dm = np.diff(m.astype(int))
    return np.nonzero(dm == 1)[0], np.nonzero(dm == -1)[0] - 1


def _run_grids(model: PMCModel, ref: QuadratureGrid, yL: float, yR: float, L: int,
               fwd0=None, start: int = -1, w_in=None) -> tuple[list, list]:
    """Grids of one run of L missing rows, and the forward components of each.

    The step-by-step filter of :mod:`pmcprg.pmc.outliers` builds the grids of
    a run as it enters it: ``yL`` / ``yR`` its observed neighbours (NaN:
    none), ``fwd0`` the forward components of the position before it when
    the run continues a gap (a gated row), ``start`` the row of its first
    position, ``w_in`` (K,) the filter at the row before it (the proposal
    is then filter-weighted on its left side; module docstring,
    "Filter-weighted proposals"). Without ``w_in``, on the same run it gives
    the grids of :func:`_build_gap_grids` without weights.
    """
    if not (_LOCAL_GRIDS and model.variant.uses_copula):
        return [ref] * L, [None] * L
    f0 = None if fwd0 is None else (fwd0[0][None], fwd0[1][None], fwd0[2][None], fwd0[3], fwd0[4])
    wi = None if (w_in is None or fwd0 is not None or not np.isfinite(yL)) else \
        np.asarray(w_in, dtype=float)[None, :]
    lam, loc, sc, (Fl, Fm, Fs, Fst, Fsy), kern, ex, bridged = _gap_proposals(
        model, np.array([yL]), np.array([yR]), np.array([L]), fwd0=f0, w_in=wi, with_bridged=True)
    yl, yr = np.full(L, np.nan), np.full(L, np.nan)
    yl[0] = yL if fwd0 is None else np.nan
    yr[-1] = yR
    first = np.zeros(L, bool)
    first[0] = start == 0
    we_in = None
    if wi is not None:
        we_in, _ = _boundary_weights(model, kern, bridged, yl, Fl, Fst, np.repeat(wi, L, axis=0), None)
    grids = _local_grid_list(model, ref, lam, loc, sc, yl, yr, kern, first, ex, np.zeros(L, int),
                             w_entry=we_in, res=None if wi is None else _MERGE_RES)
    return grids, [(Fl[k], Fm[k], Fs[k], Fst[k], Fsy[k]) for k in range(L)]


#: Grids of the last _GRID_CACHE_SIZE (model, Y, G, missingness factors) of
#: :func:`_gap_grids`. The grids depend on the model's parameters, the
#: observed values, the factors (through the filter that weights the
#: proposals) and G only; the same series is often passed again with the
#: same model (``classify`` and the quadrature check after a fit,
#: ``gap_posterior`` then ``predictive_pit``), and building them is half the
#: cost of a pass. At most _GRID_CACHE_BYTES of grid arrays are kept, the
#: newest entry always (an entry of the Intel mote-48 window holds 38 MB at
#: G = 64, and grows with G); a miss only costs time, the grids are the same.
_GRID_CACHE: "OrderedDict" = OrderedDict()
_GRID_CACHE_SIZE = 4
_GRID_CACHE_BYTES = 1 << 28
_GRID_CACHE_NB: dict = {}                    # key → bytes of its grids


def _grids_nbytes(grids: dict) -> int:
    """Bytes of the node arrays of the distinct grids of {n: grid}."""
    seen = {id(g): g for g in grids.values()}
    return sum(getattr(g, f).nbytes for g in seen.values()
               for f in ("s", "ws", "t", "w", "nodes", "ref_pdf", "omega"))


def _grid_key(model: PMCModel, Y: np.ndarray, miss: np.ndarray, ref: QuadratureGrid, ev=None):
    raw = json.dumps(model.raw, sort_keys=True, default=repr).encode()
    h = hashlib.sha1(raw)
    h.update(np.ascontiguousarray(np.where(miss, np.nan, np.asarray(Y, dtype=float))).tobytes())
    h.update(np.ascontiguousarray(ref.nodes).tobytes())
    h.update(b"-" if ev is None else np.ascontiguousarray(np.asarray(ev, dtype=float)).tobytes())
    return (h.hexdigest(), ref.G, bool(_LOCAL_GRIDS), int(_FILTER_PASSES))


def _gap_grids(model: PMCModel, Y: np.ndarray, miss: np.ndarray, ref: QuadratureGrid,
               *, with_fwd: bool = False, ev=_inf._FROM_MODEL) -> dict:
    """{n: grid} for every missing n (the reference grid where it suffices).

    With ``with_fwd`` also {n: forward components} (those of :func:`_run_grids`):
    the filter of :mod:`pmcprg.pmc.outliers` takes the grids of every run of
    missing rows from one call instead of building them run by run. The
    proposals are weighted by the filter of a first pass (module docstring,
    "Filter-weighted proposals"), with the missingness factors ``ev`` (by
    default those of the model). Results are cached (_GRID_CACHE) by model
    parameters, series, factors and G.
    """
    hit, _ = _gap_grids_pass(model, Y, miss, ref, ev)
    return (dict(hit[0]), dict(hit[1])) if with_fwd else dict(hit[0])


#: Passes run to weight the local proposals by the filter (module docstring,
#: "Filter-weighted proposals"); 0 keeps the state laws given the neighbours
#: alone (the proposals before this version). A second pass (from the filter
#: of the refined grids, runs not rebuilt yet) left the log-likelihood of
#: Intel motes 22, 47, 48 at G = 64 and 128 unchanged to the last digit and
#: rebuilt 5 of 8 275 rows of the robust-estimation window. Private switch
#: kept for accuracy studies.
_FILTER_PASSES = 1
#: A run is rebuilt from its filter-weighted proposal when its grid
#: integrates that proposal's components or kernels worse than this
#: (:func:`_refine_grids`), whatever it did on its own.
_REBUILD_TOL = 1e-3


def _run_weights(alphas, betas, miss: np.ndarray, K: int):
    """(w_in, w_out) (R, K): α̂ at the row before each run of missing rows and β̂
    at the row after it (ones where there is none), from a pass's messages."""
    a, b = _runs(miss)
    N = miss.size
    w_in = np.ones((a.size, K))
    w_out = np.ones((a.size, K))
    for r, (s, e) in enumerate(zip(a, b)):
        if s > 0:
            w_in[r] = alphas[s - 1]
        if e < N - 1 and betas is not None:
            w_out[r] = betas[e + 1]
    return w_in, w_out


def _refine_grids(model: PMCModel, Y: np.ndarray, miss: np.ndarray, ref: QuadratureGrid,
                  grids: dict, fwd: dict, err: dict, w_in, w_out, tried: set):
    """Rebuild the runs whose grids misintegrate their filter-weighted proposals.

    The proposals of every run not yet rebuilt (``tried``) are recomputed
    with the weights ``w_in``, ``w_out`` (:func:`_run_weights`). A run is
    rebuilt from them (:func:`_build_gap_grids`) when, at one of its
    positions, the current grid integrates their components or their
    kernels (:func:`_proposal_error`) worse than _REBUILD_TOL, or worse than
    _SWITCH_TOL and _SWITCH_GAIN times it integrated those of the proposal
    it was built for (``err``; a reference grid: worse than _SWITCH_TOL, the
    local grid is judged afresh); its new grids are kept when their total
    error on the new proposals is smaller. Every other grid is kept, and with
    it the transitions of the previous pass. Returns new (grids, fwd, err)
    and the sorted positions whose grid changed.
    """
    Y = np.asarray(Y, dtype=float)
    N = len(Y)
    a, b = _runs(miss)
    open_ = np.array([r for r in range(a.size) if r not in tried], dtype=int)
    if not open_.size:
        return grids, fwd, err, []
    a, b = a[open_], b[open_]
    w_in, w_out = np.asarray(w_in)[open_], np.asarray(w_out)[open_]
    yL = np.where(a > 0, Y[np.maximum(a - 1, 0)], np.nan)
    yR = np.where(b < N - 1, Y[np.minimum(b + 1, N - 1)], np.nan)
    props = _gap_proposals(model, yL, yR, b - a + 1, w_in=w_in, w_out=w_out, with_bridged=True)
    lam, loc, sc, _, kern, _, _ = props
    pos = np.concatenate([np.arange(s, e + 1) for s, e in zip(a, b)])
    run = np.repeat(np.arange(a.size), b - a + 1)
    heavy = lam >= _RESOLVE_MIN_WEIGHT
    cnt = (np.searchsorted(ref.nodes, loc + sc, side="right")
           - np.searchsorted(ref.nodes, loc - sc, side="left"))
    kl, km, kss = kern
    kc = (np.searchsorted(ref.nodes, km + kss, side="right")
          - np.searchsorted(ref.nodes, km - kss, side="left"))
    cand = (np.any(heavy & (cnt < _CANDIDATE_NODES), axis=1)
            | np.any((kl >= _RESOLVE_MIN_WEIGHT) & (kc < _CANDIDATE_NODES), axis=1))
    local = np.array([grids[int(n)].local for n in pos], dtype=bool)
    rows = np.nonzero(cand | local)[0]
    if not rows.size:
        return grids, fwd, err, []
    sup = _support(ref)
    nodes = np.array([grids[int(pos[r])].nodes for r in rows])
    omega = np.array([grids[int(pos[r])].omega for r in rows])
    e_new = np.stack([_proposal_error(nodes, omega, lam[rows], loc[rows], sc[rows], sup),
                      _proposal_error(nodes, omega, kl[rows], km[rows], kss[rows], sup)], axis=1)
    e_old = np.array([err.get(int(pos[r]), (0.0, 0.0)) for r in rows])
    # a candidate on the reference grid is judged afresh: the local grid of the
    # new proposal may win where that of the old one lost
    e_old = np.where(local[rows][:, None], e_old, 0.0)
    bad = np.any((e_new > _REBUILD_TOL)
                 | (e_new > np.maximum(_SWITCH_TOL, _SWITCH_GAIN * e_old)), axis=1)
    todo = np.unique(run[rows[bad]])
    if not todo.size:
        return grids, fwd, err, []
    tried.update(open_[todo].tolist())
    sel = np.isin(run, todo)
    sub = tuple(tuple(x[sel] for x in p) if isinstance(p, tuple) else p[sel] for p in props)
    g1, f1, e1, _ = _build_gap_grids(model, Y, miss, ref, w_in[todo], w_out[todo],
                                     runs=open_[todo], err=True, props=sub)
    # per run: the old grids' error on the new proposals against the new grids'
    old_tot = np.zeros(a.size)
    np.add.at(old_tot, run[rows], e_new.sum(axis=1))
    new_tot = np.zeros(a.size)
    for n, e in e1.items():
        new_tot[run[np.searchsorted(pos, n)]] += e[0] + e[1]
    keep = {int(r) for r in todo if new_tot[r] <= old_tot[r]}
    grids, fwd, err = dict(grids), dict(fwd), dict(err)
    changed = []
    for n in g1:
        if int(run[np.searchsorted(pos, n)]) in keep:
            if g1[n] is not grids[n]:
                changed.append(n)
            grids[n], fwd[n], err[n] = g1[n], f1[n], e1[n]
    return grids, fwd, err, sorted(changed)


def _gap_grids_pass(model: PMCModel, Y: np.ndarray, miss: np.ndarray, ref: QuadratureGrid,
                    ev=_inf._FROM_MODEL):
    """((grids, fwd), last): the cached result of :func:`_gap_grids` and, when
    this call built it, the last pass run to weight its proposals — (chain,
    forward, backward, changed), ``changed`` the positions whose grid changed
    after that pass (empty: the pass ran on the final grids) — or None.

    The grids of the proposals given the neighbours alone come first; while a
    one-step kernel into a gap is narrower than the reference grid resolves,
    up to _FILTER_PASSES passes on the current grids give the filter
    weights and :func:`_refine_grids` rebuilds the runs that need it.
    """
    if ev is _inf._FROM_MODEL:
        ev = _inf._evidence(model, miss)
    key = _grid_key(model, Y, miss, ref, ev)
    hit = _GRID_CACHE.get(key)
    if hit is not None:
        _GRID_CACHE.move_to_end(key)
        return hit, None
    grids, fwd, err, narrow = _build_gap_grids(model, Y, miss, ref, err=True)
    last = None
    if narrow and bool(np.any(~np.asarray(miss, dtype=bool))):
        K = model.K
        tried = set()
        for _ in range(_FILTER_PASSES):
            chain, fw, bw = _pass(model, Y, miss, ref, grids, ev, backward=True,
                                  reuse=None if last is None else last[0])
            w_in, w_out = _run_weights(fw[0], bw, miss, K)
            grids, fwd, err, changed = _refine_grids(model, Y, miss, ref, grids, fwd, err,
                                                     w_in, w_out, tried)
            last = (chain, fw, bw, changed)
            if not changed:
                break
    hit = (grids, fwd)
    _GRID_CACHE[key] = hit
    for k in [k for k in _GRID_CACHE_NB if k not in _GRID_CACHE]:
        del _GRID_CACHE_NB[k]
    _GRID_CACHE_NB[key] = _grids_nbytes(grids)
    while len(_GRID_CACHE) > 1 and (len(_GRID_CACHE) > _GRID_CACHE_SIZE
                                    or sum(_GRID_CACHE_NB.get(k, 0) for k in _GRID_CACHE)
                                    > _GRID_CACHE_BYTES):
        _GRID_CACHE_NB.pop(_GRID_CACHE.popitem(last=False)[0], None)
    return hit, last


def _boundary_weights(model: PMCModel, kern, bridged, yl, Fl, Fst, w_in, w_out):
    """Weights (P, K, K) of the entry and exit integrals of :func:`_neighbour_error`
    for filter-weighted proposals, or None for the side without weights.

    Where the position has bridges, the posterior of each transition as the
    bridges through its kernel give it (``kern``: at the first position of a
    run its forward kernels i → j, at the last its backward kernels i → k,
    in the order of :func:`_entry_components`). Elsewhere: at an entry,
    α̂_{a−1}(i) T_ij(y_l) (``w_in`` (P, K)); at an exit, the forward state law
    at the position over the prior of i (its forward components ``Fl``,
    states ``Fst``; 1 without) times β̂_{b+1}(k) (``w_out`` (P, K)).
    """
    K = model.K
    K2 = K * K
    kt = kern[0]
    P = kt.shape[0]
    bridged = np.asarray(bridged, dtype=bool)
    w_entry = w_exit = None
    if w_in is not None:
        ok = np.isfinite(yl)
        f, _ = _margin_eval(model, np.where(ok, yl, 0.0), log=False)
        w_entry = np.asarray(w_in, dtype=float)[:, :, None] * _x_transition(model, f, log=False)
        w_entry = np.where(bridged[:, None, None], kt[:, :K2].reshape(P, K, K), w_entry)
    if w_out is not None:
        pi = _initial(model, np.ones((1, K, K)), log=False)[0]
        pf = np.zeros((P, K))
        for i in range(K):
            pf[:, i] = np.where(np.broadcast_to(Fst, Fl.shape) == i, Fl, 0.0).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where((pf.sum(axis=1) > 0.0)[:, None], pf / np.where(pi > 0.0, pi, 1.0)[None, :], 1.0)
        w_exit = r[:, :, None] * np.asarray(w_out, dtype=float)[:, None, :]
        w_exit = np.where(bridged[:, None, None], kt[:, K2:2 * K2].reshape(P, K, K), w_exit)
    return w_entry, w_exit


def _build_gap_grids(model: PMCModel, Y: np.ndarray, miss: np.ndarray, ref: QuadratureGrid,
                     w_in=None, w_out=None, runs=None, *, err: bool = False, props=None):
    """The ({n: grid}, {n: forward components}) of :func:`_gap_grids`.

    ``w_in``, ``w_out`` (R, K): the forward filter before each of the R runs
    of missing rows and the backward message after it (module docstring,
    "Filter-weighted proposals"); without them, the state laws given the
    neighbours alone. ``runs``: indices of the runs to build (default all;
    ``w_in``, ``w_out`` then of those runs), ``props``: their proposals
    (:func:`_gap_proposals` with ``with_bridged``) when already computed.
    With ``err`` also {n: proxy error of its grid} (:func:`_local_grid_list`)
    and whether any one-step kernel into a position (whatever its weight)
    is narrower than the reference grid resolves — without one, no weighting
    of the proposals can call for a local grid.
    """
    idx = np.nonzero(miss)[0]
    if not (_LOCAL_GRIDS and model.variant.uses_copula) or not idx.size:
        g, f = {int(n): ref for n in idx}, {int(n): None for n in idx}
        return (g, f, {int(n): (0.0, 0.0) for n in idx}, False) if err else (g, f)
    Y = np.asarray(Y, dtype=float)
    N = len(Y)
    a, b = _runs(miss)
    if runs is not None:
        a, b = a[runs], b[runs]
    yL = np.where(a > 0, Y[np.maximum(a - 1, 0)], np.nan)
    yR = np.where(b < N - 1, Y[np.minimum(b + 1, N - 1)], np.nan)
    if props is None:
        props = _gap_proposals(model, yL, yR, b - a + 1, w_in=w_in, w_out=w_out, with_bridged=True)
    lam, loc, sc, (Fl, Fm, Fs, Fst, Fsy), kern, ex, bridged = props
    pos = np.concatenate([np.arange(s, e + 1) for s, e in zip(a, b)])
    yl = np.where(np.isin(pos, a) & (pos > 0), Y[np.maximum(pos - 1, 0)], np.nan)
    yr = np.where(np.isin(pos, b) & (pos < N - 1), Y[np.minimum(pos + 1, N - 1)], np.nan)
    run = np.repeat(np.arange(a.size), b - a + 1)
    filtered = w_in is not None or w_out is not None
    we_in = we_out = None
    if filtered:
        we_in, we_out = _boundary_weights(
            model, kern, bridged, yl, Fl, Fst, None if w_in is None else np.asarray(w_in, float)[run],
            None if w_out is None else np.asarray(w_out, float)[run])
    perr = np.zeros((pos.size, 2)) if err else None
    grids = _local_grid_list(model, ref, lam, loc, sc, yl, yr, kern, pos == 0, ex, run,
                             w_entry=we_in, w_exit=we_out, err_out=perr,
                             res=_MERGE_RES if filtered else None)
    g = {int(n): gr for n, gr in zip(pos, grids)}
    f = {int(n): (Fl[k], Fm[k], Fs[k], Fst[k], Fsy[k]) for k, n in enumerate(pos)}
    if not err:
        return g, f
    kl, km, kss = kern
    kc = (np.searchsorted(ref.nodes, km + kss, side="right")
          - np.searchsorted(ref.nodes, km - kss, side="left"))
    narrow = bool(np.any((kl > 0.0) & (kc < _CANDIDATE_NODES))) or any(gr.local for gr in grids)
    return g, f, dict(zip(pos.tolist(), map(tuple, perr.tolist()))), narrow


# ---------------------------------------------------------------------------
# Transition kernel q(j, y' | i, y) on arbitrary pairs — linear and log
# ---------------------------------------------------------------------------

def _margin_eval(model: PMCModel, y: np.ndarray, *, log: bool):
    """f[:, i, j] = f_ij(y) (log f if ``log``), F[:, i, j] = F_ij(y) clipped.

    Shapes (M, K, K); for state margins both are broadcast from the K
    per-state evaluations, as in ``precompute_weights``.
    """
    K = model.K
    M = len(y)
    uses_cop = model.variant.uses_copula
    f = np.empty((M, K, K))
    F = np.empty((M, K, K)) if uses_cop else None
    if model.margin_structure == "pair":
        for ii in range(K):
            for jj in range(K):
                mg = model.margin(ii, jj)
                f[:, ii, jj] = (_inf._margin_logpdf_vec(mg, y) if log else mg.pdf_vec(y))
                if uses_cop:
                    F[:, ii, jj] = mg.cdf_vec(y)
    else:
        for kk in range(K):
            mg = model.margin(kk)
            f[:, kk, :] = (_inf._margin_logpdf_vec(mg, y) if log else mg.pdf_vec(y))[:, None]
            if uses_cop:
                F[:, kk, :] = np.asarray(mg.cdf_vec(y))[:, None]
    if F is not None:
        np.clip(F, EPS, ONE_MINUS_EPS, out=F)
    return f, F


def _frank_log_outer(th: float, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """The Frank copula's ``_frank_logpdf`` on the outer product of u (A,), v (B,)."""
    from pmcprg.copulas.archimedean import frank as _fr
    if abs(th) <= _fr._FRANK_THETA_SERIES:
        return (0.5 * th * (1.0 - 2.0 * u))[:, None] * (1.0 - 2.0 * v)[None, :]
    uu, vv = u[:, None], v[None, :]
    if th > 0.0:
        A = np.expm1(-th * u)[:, None]
        BE = (np.expm1(-th * v) / math.expm1(-th))[None, :]
        x = A * BE
        with np.errstate(divide="ignore"):
            L = np.log1p(x)
        far = x < -0.5
        if np.any(far):
            ia, ib = np.nonzero(far)
            a = np.log(np.expm1(th * (1.0 - u)))[ia]
            b = th * (1.0 - v)[ib] + np.log(-np.expm1(-th * u))[ia]
            L[far] = -th + np.logaddexp(a, b) - float(_fr._log1mexp(th))
    else:
        a = -th
        lBE = (_fr._log_expm1(a * v) - float(_fr._log_expm1(a)))[None, :]
        lx = _fr._log_expm1(a * u)[:, None] + lBE
        L = np.logaddexp(0.0, lx)
    return _fr._frank_g(th) - th * (uu + vv) - 2.0 * L


def _copula_log_outer(cop, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """log c(u_a, v_b) (A, B) on the outer product of u (A,) and v (B,).

    The ``logpdf_array`` of the Gaussian, Clayton, Gumbel and Frank copulas
    computed with their per-argument terms (quantiles, logarithms,
    exponentials) taken once per node instead of once per pair — the same
    operations element by element, so the same values: these terms on the G²
    pairs of a missing → missing block made most of the cost of an E-step.
    Other families: ``logpdf_array`` on the expanded pairs.
    """
    kind = type(cop)
    if kind in (CopulaClayton, CopulaGH, CopulaFrank):
        uc, vc = np.clip(u, EPS, ONE_MINUS_EPS), np.clip(v, EPS, ONE_MINUS_EPS)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            if kind is CopulaFrank:
                return _frank_log_outer(cop.theta, uc, vc)
            f = _clayton_logpdf if kind is CopulaClayton else _gh_logpdf
            return f(np.log(uc)[:, None], np.log(vc)[None, :], cop.theta)
    if kind is CopulaGaussian:
        x = _norm_quantile(np.clip(u, EPS, ONE_MINUS_EPS))[:, None]
        y = _norm_quantile(np.clip(v, EPS, ONE_MINUS_EPS))[None, :]
        r = abs(cop.theta)
        sg = 1.0 if cop.theta >= 0.0 else -1.0
        quad = r * (r * (x - sg * y) ** 2 - 2.0 * sg * (1.0 - r) * x * y)
        return (-0.5 * (math.log1p(-r) + math.log1p(r))
                - quad / (2.0 * cop._one_minus_rho2))
    A, B = u.size, v.size
    uv = np.column_stack((np.repeat(u, B), np.tile(v, A)))
    return np.asarray(cop.logpdf_array(uv), dtype=float).reshape(A, B)


def _copula_pdf_outer(cop, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """c(u_a, v_b) (A, B): the values of ``pdf_array`` on the outer product."""
    if type(cop) in (CopulaGaussian, CopulaClayton, CopulaGH, CopulaFrank):
        with np.errstate(under="ignore", over="ignore"):
            return np.exp(_copula_log_outer(cop, u, v))
    A, B = u.size, v.size
    uv = np.column_stack((np.repeat(u, B), np.tile(v, A)))
    return np.asarray(cop.pdf_array(uv), dtype=float).reshape(A, B)


def _kernel_outer(model: PMCModel, fL, FL, fR, FR, *, log: bool):
    """q(j, y_b | i, y_a) (A, B, K, K) from every node a to every node b.

    :func:`_kernel` (linear, returns (Wk, overflow)) or :func:`_log_kernel`
    (``log``) on the A·B pairs, without expanding the per-node quantities.
    """
    var = model.variant
    K = model.K
    fRT = fR.transpose(0, 2, 1)[None, :, :, :]                          # f_ji(y_b) at [., b, i, j]
    if log:
        with np.errstate(divide="ignore", invalid="ignore"):
            p = model.prior_p
            log_p = np.log(p)
            if var.has_markov_prior:
                L = np.log(model.transition_A)[None, None, :, :] + fRT
            elif model.margin_structure == "pair":
                left = log_p[None, :, :] + fL
                log_d = _inf._lse(left, axis=2)
                L = (left - log_d[:, :, None])[:, None, :, :] + fRT
            else:
                log_kernel = log_p - np.log(p.sum(axis=1, keepdims=True))
                L = log_kernel[None, None, :, :] + fRT
            L = np.array(np.broadcast_to(L, (fL.shape[0], fR.shape[0], K, K)))
            if var.uses_copula:
                for ii in range(K):
                    for jj in range(K):
                        L[:, :, ii, jj] += _copula_log_outer(model.copula(ii, jj), FL[:, ii, jj],
                                                             FR[:, jj, ii])
        L[np.isnan(L) | (L == np.inf)] = -np.inf
        return L
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        if var.has_markov_prior:
            Wk = model.transition_A[None, None, :, :] * fRT
        else:
            p = model.prior_p
            joint = (p[None, :, :] * fL)[:, None, :, :] * fRT
            D = np.einsum("ij,nij->ni", p, fL)
            Wk = joint / np.maximum(D, MIN_POSITIVE)[:, None, :, None]
        Wk = np.array(np.broadcast_to(Wk, (fL.shape[0], fR.shape[0], K, K)))
        if var.uses_copula:
            for ii in range(K):
                for jj in range(K):
                    Wk[:, :, ii, jj] *= _copula_pdf_outer(model.copula(ii, jj), FL[:, ii, jj],
                                                          FR[:, jj, ii])
    bad = ~np.isfinite(Wk)
    overflow = bool(bad.any())
    if overflow:
        Wk[bad] = 0.0
    np.clip(Wk, 0.0, None, out=Wk)
    return Wk, overflow


def _kernel(model: PMCModel, fL, FL, fR, FR) -> tuple[np.ndarray, bool]:
    """q(j, y_R | i, y_L) on (M,) pairs — the formulas of ``precompute_weights``.

    Returns ``(Wk, overflow)`` with Wk (M, K, K) cleaned like
    ``precompute_weights`` (non-finite → 0, negatives → 0) and ``overflow``
    True when a raw value was +inf or NaN (the caller then switches to log
    space).
    """
    var = model.variant
    K = model.K
    f_next_T = fR.transpose(0, 2, 1)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        if var.has_markov_prior:
            Wk = model.transition_A[None, :, :] * f_next_T
        else:
            p = model.prior_p
            joint = p[None, :, :] * fL * f_next_T
            D = np.einsum("ij,nij->ni", p, fL)
            Wk = joint / np.maximum(D, MIN_POSITIVE)[:, :, None]
        if var.uses_copula:
            for ii in range(K):
                for jj in range(K):
                    uv = np.column_stack((FL[:, ii, jj], FR[:, jj, ii]))
                    Wk[:, ii, jj] *= model.copula(ii, jj).pdf_array(uv)
    bad = ~np.isfinite(Wk)
    overflow = bool(bad.any())
    if overflow:
        Wk[bad] = 0.0
    np.clip(Wk, 0.0, None, out=Wk)
    return Wk, overflow


def _log_kernel(model: PMCModel, lfL, FL, lfR, FR) -> np.ndarray:
    """log q(j, y_R | i, y_L) on (M,) pairs — the formulas of ``_log_transition_weights``."""
    var = model.variant
    K = model.K
    lf_next_T = lfR.transpose(0, 2, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = model.prior_p
        log_p = np.log(p)
        if var.has_markov_prior:
            L = np.log(model.transition_A)[None, :, :] + lf_next_T
        elif model.margin_structure == "pair":
            left = log_p[None, :, :] + lfL
            log_d = _inf._lse(left, axis=2)
            L = left - log_d[:, :, None] + lf_next_T
        else:
            log_kernel = log_p - np.log(p.sum(axis=1, keepdims=True))
            L = log_kernel[None, :, :] + lf_next_T
        if var.uses_copula:
            for ii in range(K):
                for jj in range(K):
                    uv = np.column_stack((FL[:, ii, jj], FR[:, jj, ii]))
                    L[:, ii, jj] += model.copula(ii, jj).logpdf_array(uv)
    L[np.isnan(L) | (L == np.inf)] = -np.inf
    return L


def _initial(model: PMCModel, f: np.ndarray, *, log: bool) -> np.ndarray:
    """μ(i, y) = α_1(i) at the rows of f (M, K, K) — the ``forward`` initialisation."""
    p = model.prior_p
    if not log:
        if model.margin_structure == "pair":
            return np.einsum("ij,nij->ni", p, f)
        return np.einsum("ij,nji->nj", p, f)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.log(p)
        if model.margin_structure == "pair":
            out = _inf._lse(log_p[None, :, :] + f, axis=2)
        else:
            out = _inf._lse(log_p, axis=0)[None, :] + f[:, :, 0]
    out[np.isnan(out) | (out == np.inf)] = -np.inf
    return out


def _log_rows(model: PMCModel, src, src_idx, i_idx, j_idx, dst, dst_omega) -> np.ndarray:
    """log q(j, y' | i, y) + log ω' (n, G') of n blocks (i, y → j, ·) at the nodes y' of a grid.

    The log-space weights of blocks whose linear weights all underflowed
    (the ``rescue`` of :func:`_normalise_blocks`): the formulas of
    :func:`_log_kernel`, evaluated for the (i, j) pairs and source values of
    those blocks only (the full kernel doubled the cost of an E-step with
    many underflowing blocks). ``src`` = (log f, F) of the source values,
    ``src_idx`` (n,) the source of every block; ``dst`` = (log f, F) of the
    destination nodes (:func:`_margin_eval`, ``log=True``).
    """
    lfs_all, Fs_all = src
    lfd, Fd = dst
    src_idx = np.asarray(src_idx, dtype=int)
    i_idx, j_idx = np.asarray(i_idx, dtype=int), np.asarray(j_idx, dtype=int)
    with np.errstate(divide="ignore", invalid="ignore"):
        lom = np.log(dst_omega)
        p = model.prior_p
        log_p = np.log(p)
        out = np.empty((src_idx.size, lfd.shape[0]))
        for i, j in set(zip(i_idx.tolist(), j_idx.tolist())):
            sel = np.nonzero((i_idx == i) & (j_idx == j))[0]
            us, inv = np.unique(src_idx[sel], return_inverse=True)
            lfs = lfs_all[us]
            if model.variant.has_markov_prior:
                a = np.full(us.size, np.log(model.transition_A)[i, j])
            elif model.margin_structure == "pair":
                left = log_p[None, i, :] + lfs[:, i, :]
                a = left[:, j] - _inf._lse(left, axis=1)
            else:
                a = np.full(us.size, (log_p - np.log(p.sum(axis=1, keepdims=True)))[i, j])
            L = a[:, None] + lfd[None, :, j, i]
            if model.variant.uses_copula:
                L = L + _copula_log_outer(model.copula(i, j), Fs_all[us, i, j], Fd[:, j, i])
            L[np.isnan(L) | (L == np.inf)] = -np.inf
            out[sel] = L[inv] + lom[None, :]
    return out


#: A linear-space block whose quadrature mass is below this value is rebuilt
#: in log space (:func:`_normalise_blocks`): its weights are subnormal or
#: zero, and the factor target / mass would overflow (2.2e-313 → inf,
#: measured on Intel mote 48). Above it every weight that matters (≥ 1e-16
#: of the mass) is a normal float.
_UNDERFLOW = 1e-250

#: Renormalisation of the quadrature rows of E, Q and α̃_1 (module docstring):
#: ``"block"`` (production) rescales every block (i, g) → (j, ·) to the exact
#: transition probability P(x_{n+1} = j | x_n = i, y_n), then every row to 1;
#: ``"row"`` only rescales rows; ``None`` keeps the raw quadrature. Private
#: switch kept for accuracy studies.
_RENORMALISE = "block"


def _x_transition(model: PMCModel, f: np.ndarray, *, log: bool) -> np.ndarray:
    """P(x_{n+1} = j | x_n = i, y_n) at the rows of f (M, K, K) — log if ``log``.

    A_ij for the HMC variants and for state margins (p_ij / Σ_k p_ik); for
    pair margins p_ij f_ij(y) / Σ_k p_ik f_ik(y) (DerrodePieczynski_CSDA2013
    Eq. 13), with the linear-space guard of ``precompute_weights``.
    """
    M, K = f.shape[0], model.K
    p = model.prior_p
    with np.errstate(divide="ignore", invalid="ignore", under="ignore"):
        if model.variant.has_markov_prior or model.margin_structure != "pair":
            A = model.transition_A if model.variant.has_markov_prior else p / p.sum(axis=1, keepdims=True)
            T = np.broadcast_to(np.log(A) if log else A, (M, K, K))
        elif log:
            left = np.log(p)[None, :, :] + f
            T = left - _inf._lse(left, axis=2)[:, :, None]
        else:
            joint = p[None, :, :] * f
            T = joint / np.maximum(joint.sum(axis=2), MIN_POSITIVE)[:, :, None]
    return T


def _normalise_blocks(B: np.ndarray, target: np.ndarray | None, *, log: bool,
                      with_mass: bool = False, rescue=None):
    """Rescale the quadrature tensor B (..., K, G) to its exact block masses.

    ``B[..., j, g]`` holds the weights of the destination (j, y_g) and
    ``target[..., j]`` the exact mass Σ_g of each block (a transition
    probability). Blocks are rescaled to ``target`` (``"block"`` mode), then the
    flattened rows (..., K·G) to sum to 1 (``"block"`` and ``"row"``).

    A block whose weights all underflow in linear space (a kernel narrower
    than the distance to the nearest node: exp(−z²/2) < 1e-308 beyond
    z ≈ 38) would keep 0 and lose its mass to the other blocks of its row —
    the state law would change, where the log-space pass keeps it and moves
    the mass to the nearest nodes. ``rescue(index)`` (linear space only)
    returns the log-weights (n, G) of those blocks, ``index`` a tuple of
    index arrays over ``B.shape[:-1]``: each is normalised in log space and
    rescaled to its target, so the linear and the log passes build the same
    chain (module docstring, "Log space"). Without ``rescue`` a vanished
    block keeps 0 and its mass goes to the other blocks through the row step.

    Returns ``(rows, factors)``: the (..., K·G) rows and the (..., K) factors
    by which each block was multiplied, in linear scale. ``with_mass`` adds
    the raw block masses Σ_g B (log Σ_g exp B in log space) — the quadrature
    of a known integral, whose distance to ``target`` is the convergence
    diagnostic of :func:`_quadrature_report`.
    """
    shape = B.shape[:-2] + (B.shape[-2] * B.shape[-1],)
    with np.errstate(divide="ignore", invalid="ignore", under="ignore", over="ignore"):
        raw = (_inf._lse(B, axis=-1) if log else B.sum(axis=-1)) if with_mass else None
    if _RENORMALISE is None:
        out = (B.reshape(shape), np.ones(B.shape[:-1]))
        return out + (raw,) if with_mass else out
    # the block masses of B as it is (``raw``, when computed), until the rescue changes B
    sums = raw
    if not log and rescue is not None and _RENORMALISE == "block" and target is not None:
        with np.errstate(invalid="ignore"):
            sb = B.sum(axis=-1) if raw is None else raw
            lost = ~(np.isfinite(sb) & (sb >= _UNDERFLOW)) & (target > 0.0)
        if lost.any():
            sums = None
            index = np.nonzero(lost)
            LB = np.asarray(rescue(index), dtype=float)
            with np.errstate(divide="ignore", invalid="ignore", under="ignore", over="ignore"):
                ls = _inf._lse(LB, axis=-1)
                fin = np.isfinite(ls)
                shp = np.where(fin[:, None], np.exp(LB - np.where(fin, ls, 0.0)[:, None]), 0.0)
            B = np.array(B, dtype=float, copy=True)
            # the log-normalised shape, scaled to the (finite) target below
            B[index] = np.where(np.isfinite(shp), shp, 0.0)
    with np.errstate(divide="ignore", invalid="ignore", under="ignore", over="ignore"):
        if log:
            fac = np.zeros(B.shape[:-1])
            if _RENORMALISE == "block" and target is not None:
                sb = _inf._lse(B, axis=-1) if sums is None else sums
                ok = np.isfinite(sb) & np.isfinite(target)
                fac = np.where(ok, np.where(ok, target, 0.0) - np.where(ok, sb, 0.0), -np.inf)
                B = np.where(ok[..., None], B + np.where(ok, fac, 0.0)[..., None], -np.inf)
            R = B.reshape(shape)
            sr = _inf._lse(R, axis=-1)
            fin = np.isfinite(sr)
            R = np.where(fin[..., None], R - np.where(fin, sr, 0.0)[..., None], R)
            fac = np.exp(fac - np.where(fin, sr, 0.0)[..., None])
            out = (R, np.where(np.isfinite(fac), fac, 0.0))
            return out + (raw,) if with_mass else out
        fac = np.ones(B.shape[:-1])
        if _RENORMALISE == "block" and target is not None:
            sb = B.sum(axis=-1) if sums is None else sums
            ok = np.isfinite(sb) & (sb > 0.0)
            fac = np.where(ok, target / np.where(ok, sb, 1.0), 0.0)
            B = B * fac[..., None]
        R = B.reshape(shape)
        sr = R.sum(axis=-1, keepdims=True)
        ok = np.isfinite(sr) & (sr > 0.0)
        R = np.divide(R, np.where(ok, sr, 1.0), out=np.zeros_like(R), where=ok)
        fac = fac / np.where(ok, sr, 1.0)
        out = (R, np.broadcast_to(fac, B.shape[:-1]).copy())
        return out + (raw,) if with_mass else out


# ---------------------------------------------------------------------------
# Leading gaps: transitions normalised to the exact prior
# ---------------------------------------------------------------------------

#: A pair-margin model is taken as stationary (the law of (x_n, y_n) is μ at
#: every n, module docstring "Leading gaps") when max |p − pᵀ| is at most
#: this fraction of max p.
_SYM_TOL = 1e-12


def _prior_known(model: PMCModel, ev) -> bool:
    """True when the law of (x_n, y_n) before the first observed row is known.

    State margins (HMC-DN, PMC with K margins): y_n given x_{0:n} has the law
    f_{x_n} whatever the path and the missingness factors (T does not depend
    on y), so the forward message is w_n(i) f_i(y). Pair margins: y_n given
    (x_{n−1}, x_n) = (h, i) has the law f_ih only while the state pair has
    the law p — a stationary chain (symmetric p, the SR-PMC of the package)
    whose missingness factors, if any, do not depend on the state.
    Otherwise :class:`_Chain` keeps the row-normalised transitions of an
    interior gap in the leading gap.
    """
    if model.margin_structure != "pair":
        return True
    if ev is not None and not np.all(np.asarray(ev) == np.asarray(ev)[:, :1]):
        return False                   # factors that reweight the states
    p = model.prior_p
    return float(np.max(np.abs(p - p.T))) <= _SYM_TOL * max(float(np.max(p)), MIN_POSITIVE)


def _lead_matrix(model: PMCModel) -> np.ndarray:
    """∫ ρ_ij = P(x_{n+1} = j | x_n = i) in a leading gap: A, or p / Σ_j p_ij."""
    if model.variant.has_markov_prior:
        return model.transition_A
    p = model.prior_p
    r = p.sum(axis=1, keepdims=True)
    return np.divide(p, r, out=np.zeros_like(p), where=r > 0.0)


def _lead_transition(model: PMCModel, A: QuadratureGrid, B: QuadratureGrid, *, log: bool):
    """Transition missing n → missing n+1 inside a leading gap (module docstring).

    Before the first observed row the forward message is the prior, w_n(i)
    φ_i(y) with φ_i = μ(i, ·) / ∫ μ(i, ·) (:func:`_prior_known`), and the
    exact message at n+1 is Σ_i w_n(i) ρ_ij(y'), ρ_ij(y') = ∫ φ_i(y) q(j, y'
    | i, y) dy = A_ij f_ji(y') (A_ij f_j(y') for state margins). The rows of
    an interior gap (rescaled to the transition probabilities out of each
    source node) propagate a broad prior through a narrow kernel onto a
    local grid by relocating the mass of every source node onto the
    destination nodes nearest to it: the law of y is lost (−22 nats at τ =
    0.999, ``report/erroneous_data/intel_lab/repro_leading_gap.py``). Here
    the columns are rescaled instead: the entry (i, g) → (j, g') is q(j,
    y_g' | i, y_g) ω_g' c_ij(g'), with c_ij(g') such that the discrete prior
    ψ_i(g) ∝ φ_i(y_g) ω_g of grid n is carried exactly onto τ_ij(g') ∝ ρ_ij(y_g')
    ω_g' of grid n+1 (Σ_g' τ_ij = A_ij). The forward message is then the
    prior at every position of the gap, log C = 0 there (the reverse of a
    trailing gap), and the posterior inside the gap is the reverse-time
    forecast from the first observed row, which the grids built from the
    backward pieces resolve.

    Returns ``(T, dev, lc)``: T (K·G_A, K·G_B), log T if ``log``; ``dev``
    (log raw, log exact) (K, K, G_B) of the reverse masses Σ_g ψ_i(g) q(j,
    y_g' | i, y_g) ω_g' against ρ_ij(y_g') ω_g' (:func:`_quadrature_report`);
    ``lc`` (K, K, G_B) the log column factors (:func:`_nystrom_density`).
    """
    K = model.K
    GA, GB = A.G, B.G
    lfA, FA = _margin_eval(model, A.nodes, log=True)
    lfB, FB = _margin_eval(model, B.nodes, log=True)
    LK = _kernel_outer(model, lfA, FA, lfB, FB, log=True).transpose(2, 0, 3, 1)  # (K, G_A, K, G_B)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        lpsi = _initial(model, lfA, log=True).T + np.log(A.omega)[None, :]   # (K, G_A)
        s = _inf._lse(lpsi, axis=1)
        lpsi = np.where(np.isfinite(s)[:, None], lpsi - np.where(np.isfinite(s), s, 0.0)[:, None],
                        -np.inf)
        lraw = _inf._lse(lpsi[:, :, None, None] + LK, axis=1)              # (K, K, G_B)
        lom = np.log(B.omega)
        lrho = lfB.transpose(2, 1, 0) + lom[None, None, :]                  # f_ji(y_g') ω_g'
        lA = np.log(_lead_matrix(model))
        lexact = lrho + lA[:, :, None]
        s = _inf._lse(lrho, axis=2)
        lt = np.where(np.isfinite(s)[:, :, None],
                      lrho - np.where(np.isfinite(s), s, 0.0)[:, :, None] + lA[:, :, None], -np.inf)
        lc = lt - lraw - lom[None, None, :]
        lc = np.where(np.isfinite(lc), lc, -np.inf)       # unreachable node or empty block: 0
        LQ = LK + lom[None, None, None, :] + lc[:, None, :, :]
        LQ = np.where(np.isnan(LQ), -np.inf, LQ).reshape(K * GA, K * GB)
        T = LQ if log else np.exp(LQ)
    return T, (lraw + lom[None, None, :], lexact), lc


# ---------------------------------------------------------------------------
# The augmented chain
# ---------------------------------------------------------------------------

#: Bytes of built missing → missing transitions that a chain keeps (module
#: docstring, "Memory"): the (K·G)² blocks between local grids and a leading
#: gap's transitions onto local grids, built when a pass first needs them.
#: Beyond this budget a transition is not kept, and the passes that need it
#: again (backward, ξ, FFBS, the next chain of :func:`_gap_grids_pass`)
#: rebuild it — the same operations on the same inputs, so every result is
#: the same bit for bit whatever the budget; only the time changes. 2 GiB:
#: the G = 64 passes of the Intel Lab windows keep every transition (no
#: rebuild, the time of the version before), a G = 256 pass stays within a
#: few GB. Private switch (0: keep none).
_TRANSITION_BUDGET = 2 << 30

#: Value of ``_Chain.block_dev[n]`` / ``lead_dev[n]`` for a transition into n
#: that the chain did not keep: its part of the quadrature report is computed
#: by the backward pass that rebuilds it (:func:`_backward_chain`).
_DEFERRED = object()


class _Margins:
    """The margins (f, F) at the nodes of the grids of a chain (linear or log).

    The local grids ``gl`` are evaluated in one call (one call per grid made
    a quarter of an E-step on a window with 8 000 missing rows), others one
    by one; ``logs`` gives the log margins a linear chain needs to rescue a
    block whose weights all underflow (:func:`_normalise_blocks`), evaluated
    on first use in one call over ``gl`` too. :meth:`subset` keeps the
    values of a few grids for the transitions another chain takes over.
    With state margins f[:, i, j] and F[:, i, j] repeat the value of state i
    over j (as in ``precompute_weights``): the cache keeps (G, K) and expands
    it when asked — the same values, the same layout, K times less memory
    while a chain rebuilds its transitions.
    """

    def __init__(self, model: PMCModel, grid: QuadratureGrid, ref, log: bool, gl: list):
        self.model, self.grid, self.ref, self.log, self.gl = model, grid, ref, log, gl
        self.packed = model.margin_structure != "pair"
        self.lin = {}
        self.lg = {}
        self._fill(self.lin, log)

    def _pack(self, v):
        if not self.packed:
            return v
        return tuple(None if a is None else np.ascontiguousarray(a[:, :, 0]) for a in v)

    def _unpack(self, v):
        if not self.packed:
            return v
        K = self.model.K
        return tuple(None if a is None else np.repeat(a[:, :, None], K, axis=2) for a in v)

    def _fill(self, cache, as_log):
        if not self.gl:
            return
        G = self.grid.G
        f_all, F_all = _margin_eval(self.model, np.concatenate([g.nodes for g in self.gl]),
                                    log=as_log)
        for k, g in enumerate(self.gl):
            sl = slice(k * G, (k + 1) * G)
            cache[id(g)] = (g, self._pack((f_all[sl], None if F_all is None else F_all[sl])))

    def nodes(self, g):
        """(f, F) at the nodes of g, in the scale of the chain."""
        if g is self.grid:
            return self.ref
        if id(g) not in self.lin:
            self.lin[id(g)] = (g, self._pack(_margin_eval(self.model, g.nodes, log=self.log)))
        return self._unpack(self.lin[id(g)][1])

    def logs(self, g):
        """(log f, F) at the nodes of g."""
        if self.log:
            return self.nodes(g)
        if not self.lg:
            self._fill(self.lg, True)
        if id(g) not in self.lg:
            self.lg[id(g)] = (g, self._pack(_margin_eval(self.model, g.nodes, log=True)))
        return self._unpack(self.lg[id(g)][1])

    def subset(self, grids) -> "_Margins":
        """The values of ``grids`` alone (copies of views of a batch, so that
        the batch arrays can go; packed values are arrays of their own)."""
        out = _Margins.__new__(_Margins)
        out.model, out.grid, out.ref, out.log, out.gl = self.model, self.grid, self.ref, self.log, []
        out.packed = self.packed

        def own(a):
            return None if a is None else (a if a.base is None else a.copy())

        def keep(cache):
            return {id(g): (g, tuple(own(a) for a in cache[id(g)][1]))
                    for g in grids if id(g) in cache}

        out.lin, out.lg = keep(self.lin), keep(self.lg)
        return out


class _Transitions:
    """``_Chain.trans``: ``trans[n]`` for n = 0 … N − 2, each transition kept
    or built when asked for (:meth:`_Chain._step`). A view made on access:
    the chain holds no reference to it (no reference cycle), so a chain that
    is dropped is freed at once, not at the next garbage collection."""

    def __init__(self, chain: "_Chain"):
        self._chain = chain

    def __len__(self) -> int:
        return len(self._chain._fixed)

    def __getitem__(self, n):
        return self._chain._step(int(n))[0]


class _Chain:
    """Transitions of the chain on the augmented state (module docstring).

    ``trans[n]`` maps position n to n+1 and has shape S_n × S_{n+1}, with
    S_n = K at an observed n and K·G at a missing one (index i·G + g); in log
    space when ``log``. ``init`` has shape (S_0,).

    ``ev`` (N, K): the missingness evidence factors e_n(i) (module
    docstring), by default those of ``model.missingness`` on ``miss``;
    ``None`` applies none. They multiply ``init`` and the columns of every
    ``trans[n]`` (e_{n+1}(i) repeated over the nodes g at a missing n+1)
    after the block renormalisation; ``self.ev`` keeps the factors for
    :func:`_nystrom_density`.

    ``grids`` {n: QuadratureGrid}: the grid of every missing n (by default
    :func:`_gap_grids`: the reference grid ``grid`` or a local one).
    ``block_dev`` {n: (raw, target)}: the raw quadrature masses of the blocks
    into n and their exact values, for :func:`_quadrature_report`, which
    sets ``quad_error``. Inside a leading gap whose prior is known
    (:func:`_prior_known`; ``lead_exact``, first observed row ``lead``) the
    transitions are :func:`_lead_transition`'s, with their reverse masses in
    ``lead_dev`` {n: (log raw, log exact)} and column factors in ``lead_c``
    {n: log c}.

    Memory (module docstring, "Memory"): a missing → missing transition that
    touches a local grid is a dense (K·G)² block, one per step, and a long
    gappy series has thousands of them. They are built when a pass first
    needs them (``trans[n]``) and kept up to _TRANSITION_BUDGET bytes; the
    others are rebuilt when needed again — ``block_dev`` / ``lead_dev`` then
    hold _DEFERRED, and the backward pass computes their part of the
    quadrature report and of ξ while it rebuilds them (``_fused``).

    ``reuse``: a chain of the same series, factors and scale on other grids
    (a pass that weighted them, :func:`_gap_grids_pass`); every transition
    between positions whose grids are the same objects in both is taken
    from it (``raw``: the transitions before the factors; a built
    transition between local grids as it is, factors included, or its
    recipe) instead of being recomputed. The reused chain is released: it
    serves one chain only.
    """

    def __init__(self, model: PMCModel, Y: np.ndarray, miss: np.ndarray,
                 grid: QuadratureGrid, *, log: bool, ev=_inf._FROM_MODEL, grids=None,
                 reuse=None):
        self.model = model
        self.K = K = model.K
        self.G = G = grid.G
        self.N = N = len(Y)
        self.miss = miss
        self.log = log
        self.grid = grid
        self.overflow = False
        Y = np.asarray(Y, dtype=float)
        self.Y = Y
        if ev is _inf._FROM_MODEL:
            ev = _inf._evidence(model, miss)
        # Grid of every missing position: the reference grid, or a local one.
        self.grids = _gap_grids(model, Y, miss, grid, ev=ev) if grids is None else grids
        #: Raw quadrature masses of the blocks against their exact values
        #: (entry, inner and initial blocks; see :func:`_quadrature_report`).
        self.block_dev = {}
        self.lead_dev = {}
        self.lead_c = {}
        self.Q_dev = None
        obs = np.nonzero(~miss)[0]
        self.lead = int(obs[0]) if obs.size else N
        # the transitions of the prior once a local grid enters the leading
        # gap (on reference grids alone the rows of an interior gap are
        # accurate, and results stay bit for bit those of the reference grid)
        self.lead_exact = (_prior_known(model, ev)
                           and any(self.grids[n].local for n in range(min(self.lead, N))))
        self.quad_error = None
        # transitions built when first needed: n → (kind, A, B, margins),
        # kind "q" (between local grids) or "lead" (a leading gap);
        # margins None: those of this chain (``_mg``)
        self._lazy = {}
        self._store = {}                 # n → (T, dev) kept (_TRANSITION_BUDGET)
        self._stored = 0
        self._built = set()
        self._fused = None               # (alphas, betas, {n: (ξ part, report part)})
        Yf = np.where(miss, grid.nodes[G // 2], Y)
        # Raw weights: the evidence factors are applied below, uniformly.
        if log:
            Wobs, _ = _inf._log_transition_weights(model, Yf, ev=None)
        else:
            Wobs, _ = _inf._weights(model, Yf, None)
        self.Wobs = Wobs

        fN, FN = _margin_eval(model, grid.nodes, log=log)

        m0, m1 = miss[:-1], miss[1:]
        # transitions taken from ``reuse`` (before the factors), by n; the
        # built transitions between local grids (factors included) or their
        # recipes, by n
        R = {}
        adopted = {}
        usable = (reuse is not None and reuse.log == log and reuse.N == N and reuse.grid is grid
                  and (reuse.ev is ev or (reuse.ev is not None and ev is not None
                                          and np.array_equal(reuse.ev, ev))))
        if usable:
            for n in np.nonzero(m0 | m1)[0].tolist():
                if not all(not miss[k] or reuse.grids[k] is self.grids[k] for k in (n, n + 1)):
                    continue
                if m0[n] and m1[n] and n + 1 < self.lead and reuse.lead_exact != self.lead_exact:
                    continue
                if n in reuse._lazy:
                    adopted[n] = reuse._lazy[n] + (reuse._store.get(n),)
                else:
                    R[n] = reuse.raw[n]
                if n + 1 in reuse.block_dev:
                    self.block_dev[n + 1] = reuse.block_dev[n + 1]
                if n + 1 in reuse.lead_dev:
                    self.lead_dev[n + 1] = reuse.lead_dev[n + 1]
                if n in reuse.lead_c:
                    self.lead_c[n] = reuse.lead_c[n]
        init_kept = usable and (not miss[0] or reuse.grids[0] is self.grids[0])
        # the positions whose grids the transitions still to compute touch
        need = set()
        for n in np.nonzero(m0 | m1)[0].tolist():
            if n not in R and n not in adopted:
                need.update(k for k in (n, n + 1) if miss[k])
        if miss[0] and not init_kept:
            need.add(0)

        def kernel(fl, Fl, fr, Fr):
            if log:
                return _log_kernel(model, fl, Fl, fr, Fr)
            Wk, over = _kernel(model, fl, Fl, fr, Fr)
            self.overflow |= over
            return Wk

        def take(F, idx):
            return None if F is None else F[idx]

        # the margins at the nodes of every local grid, in one call
        mg = _Margins(model, grid, (fN, FN), log,
                      list({id(g): g for n, g in self.grids.items()
                            if g is not grid and n in need}.values()))
        self._mg = mg

        def weigh(ker, om):                              # × ω of the destination nodes
            return (ker + np.log(om)) if log else ker * om

        def is_ref(n):
            return self.grids[int(n)] is grid

        def todo(idx):
            return np.array([n for n in idx.tolist() if n not in R and n not in adopted], dtype=int)

        inner = todo(np.nonzero(m0 & m1)[0])
        entries = todo(np.nonzero(~m0 & m1)[0])
        exits = todo(np.nonzero(m0 & ~m1)[0])

        Q = reuse.Q if usable else None
        Qn = {}
        if self.lead_exact:
            # inside a leading gap: the transitions of the prior (module docstring)
            lead_in = inner[inner + 1 < self.lead]
            inner = inner[inner + 1 >= self.lead]
            shared = None
            for n in lead_in:
                a, b = self.grids[int(n)], self.grids[int(n) + 1]
                if a is grid and b is grid:
                    if shared is None:
                        shared = _lead_transition(model, grid, grid, log=log)
                    T_, dev, lc = shared
                    Qn[int(n)], self.lead_dev[int(n) + 1], self.lead_c[int(n)] = T_, dev, lc
                else:
                    self._lazy[int(n)] = ("lead", a, b, None)
                    self.lead_dev[int(n) + 1] = _DEFERRED
        ref_inner = np.array([is_ref(n) and is_ref(n + 1) for n in inner], dtype=bool)
        if ref_inner.any():
            if Q is None or reuse.Q_dev is None:
                Q, dev = self._between(grid, grid, mg)
            else:
                dev = reuse.Q_dev
            self.Q_dev = dev
            for n in inner[ref_inner]:
                self.block_dev[int(n) + 1] = dev
        for n in inner[~ref_inner]:
            self._lazy[int(n)] = ("q", self.grids[int(n)], self.grids[int(n) + 1], None)
            self.block_dev[int(n) + 1] = _DEFERRED

        E = {}
        ref_e = np.array([is_ref(n + 1) for n in entries], dtype=bool)
        for sel in (entries[ref_e], entries[~ref_e]):
            if not sel.size:
                continue
            fo, Fo = _margin_eval(model, Y[sel], log=log)
            e_rep = np.repeat(np.arange(sel.size), G)
            gs = [self.grids[int(n) + 1] for n in sel]
            if is_ref(sel[0] + 1):
                g_til = np.tile(np.arange(G), sel.size)
                fr, Fr, om = fN[g_til], take(FN, g_til), grid.omega
            else:
                fr, Fr = _margin_eval(model, np.concatenate([g.nodes for g in gs]), log=log)
                om = np.stack([g.omega for g in gs])[:, None, None, :]
            ker = kernel(fo[e_rep], take(Fo, e_rep), fr, Fr)
            ker = weigh(ker.reshape(sel.size, G, K, K).transpose(0, 2, 3, 1), om)  # (n_e, K, K, G)
            T = _x_transition(model, fo, log=log)

            def rescue(idx, sel=sel, gs=gs):             # blocks (e, i → j, ·)
                out = np.empty((idx[0].size, G))
                src = _margin_eval(model, Y[sel], log=True)
                for e in np.unique(idx[0]):
                    r = idx[0] == e
                    out[r] = _log_rows(model, src, np.full(int(r.sum()), e), idx[1][r], idx[2][r],
                                       mg.logs(gs[e]), gs[e].omega)
                return out

            ker, _, raw = _normalise_blocks(ker, T, log=log, with_mass=True, rescue=rescue)
            for e, n in enumerate(sel):
                E[int(n)] = ker[e]
                self.block_dev[int(n) + 1] = (raw[e], T[e])

        X = {}
        ref_x = np.array([is_ref(n) for n in exits], dtype=bool)
        for sel in (exits[ref_x], exits[~ref_x]):
            if not sel.size:
                continue
            fo, Fo = _margin_eval(model, Y[sel + 1], log=log)
            x_rep = np.repeat(np.arange(sel.size), G)
            if is_ref(sel[0]):
                g_til = np.tile(np.arange(G), sel.size)
                fl, Fl = fN[g_til], take(FN, g_til)
            else:
                fl, Fl = _margin_eval(model, np.concatenate([self.grids[int(n)].nodes for n in sel]),
                                      log=log)
            ker = kernel(fl, Fl, fo[x_rep], take(Fo, x_rep))
            ker = ker.reshape(sel.size, G, K, K).transpose(0, 2, 1, 3)     # (n_x, K, G, K)
            ker = ker.reshape(sel.size, K * G, K)
            for x, n in enumerate(sel):
                X[int(n)] = ker[x]

        if miss[0] and init_kept:
            self.init = reuse.init_raw
            self.block_dev[0] = reuse.block_dev[0]
        elif miss[0]:
            g0 = self.grids[0]
            f0, _ = mg.nodes(g0)
            mu = _initial(model, f0, log=log).T                  # (K, G)
            if not log and not np.all(np.isfinite(mu)):
                self.overflow = True
                mu = np.where(np.isfinite(mu), mu, 0.0)
            mu = weigh(mu, g0.omega[None, :])
            prior = _initial(model, np.zeros((1, K, K)) if log else np.ones((1, K, K)), log=log)

            def rescue(idx, g0=g0):                     # blocks (0, i): log μ(i, y_g) ω_g
                lf0, _ = _margin_eval(model, g0.nodes, log=True)
                with np.errstate(divide="ignore"):
                    return _initial(model, lf0, log=True).T[idx[1]] + np.log(g0.omega)[None, :]

            init, _, raw = _normalise_blocks(mu[None], prior, log=log, with_mass=True, rescue=rescue)
            self.init = init[0]
            self.block_dev[0] = (raw[0], prior[0])
        else:
            f0, _ = _margin_eval(model, Y[:1], log=log)
            self.init = _initial(model, f0, log=log)[0]

        # the transitions taken over from ``reuse``: kept as they are, or
        # rebuilt from the margins that built them there
        for n, (kind, A, B, src, ent) in adopted.items():
            if ent is None:
                src = src if src is not None else reuse._mg.subset([g for g in (A, B)
                                                                    if g is not grid])
                self._lazy[n] = (kind, A, B, src)
            else:
                self._lazy[n] = (kind, A, B, None)
                self._built.add(n)
                self._store[n] = ent
                self._stored += sum(a.nbytes for a in (ent[0],) + tuple(ent[1]))

        trans = []
        for n in range(N - 1):
            if n in R:
                trans.append(R[n])
            elif n in self._lazy:
                trans.append(None)
            elif not m0[n] and not m1[n]:
                trans.append(Wobs[n])
            elif not m0[n]:
                trans.append(E[n])
            elif m1[n]:
                trans.append(Qn.get(n, Q))
            else:
                trans.append(X[n])
        # before the factors: what a later chain on other grids reuses
        self.raw = list(trans)
        self.init_raw = self.init
        self.Q = Q
        self.ev = ev
        self._fac = None
        if ev is not None:
            # Missingness evidence: a likelihood factor of the destination
            # state, after the block renormalisation (module docstring).
            with np.errstate(divide="ignore"):
                self._fac = np.log(ev) if log else np.asarray(ev, dtype=float)
            self.init = self._scale(self.init, self._at(0))
            q_scaled = {}                    # Q is shared: one copy per factor
            for n in range(N - 1):
                if trans[n] is None:         # built later, factors included
                    continue
                v = self._at(n + 1)
                if Q is not None and trans[n] is Q:
                    key = v.tobytes()
                    if key not in q_scaled:
                        q_scaled[key] = self._scale(Q, v[None, :])
                    trans[n] = q_scaled[key]
                else:
                    trans[n] = self._scale(trans[n], v[None, :])
        self._fixed = trans
        if usable:
            reuse._release()

    @property
    def trans(self) -> _Transitions:
        """trans[n], n = 0 … N − 2 (kept, or built when asked for)."""
        return _Transitions(self)

    # ---- missingness factors ---------------------------------------------

    def _at(self, n):                        # (S_n,) factor, augmented layout
        fac = self._fac
        return np.repeat(fac[n], self.G) if self.miss[n] else fac[n]

    def _scale(self, T, v):
        return (T + v) if self.log else (T * v)

    # ---- transitions built when needed -------------------------------------

    def _between(self, A: QuadratureGrid, B: QuadratureGrid, mg: _Margins):
        """Q[(i, g), (j, g')] from the nodes of A to those of B, block-normalised,
        and its (raw block masses, exact masses)."""
        model, K, G, log = self.model, self.K, self.G, self.log
        fA, FA = mg.nodes(A)
        fB, FB = mg.nodes(B)
        if log:
            ker = _kernel_outer(model, fA, FA, fB, FB, log=True)
        else:
            ker, over = _kernel_outer(model, fA, FA, fB, FB, log=False)
            self.overflow |= over
        om = B.omega[None, None, None, :]
        Qb = ker.transpose(2, 0, 3, 1)
        Qb = (Qb + np.log(om)) if log else Qb * om                                # (K, G, K, G)
        T = _x_transition(model, fA, log=log).transpose(1, 0, 2)               # (K, G, K)

        def rescue(idx):                                 # blocks (i, g → j, ·)
            return _log_rows(model, mg.logs(A), idx[1], idx[0], idx[2], mg.logs(B), B.omega)

        Qb, _, raw = _normalise_blocks(Qb, T, log=log, with_mass=True, rescue=rescue)
        return Qb.reshape(K * G, K * G), (raw, T)

    def _build(self, n: int):
        """(trans[n], dev) of a transition built when needed, factors included."""
        kind, A, B, mg = self._lazy[n]
        if kind == "q":
            T, dev = self._between(A, B, self._mg if mg is None else mg)
        else:
            T, dev, lc = _lead_transition(self.model, A, B, log=self.log)
            self.lead_c[n] = lc
        self._built.add(n)
        if self._fac is not None:
            T = self._scale(T, self._at(n + 1)[None, :])
        return T, dev

    def _step(self, n: int):
        """(trans[n], dev or None, kept): built if needed, and kept within
        _TRANSITION_BUDGET (``dev`` then goes to ``block_dev`` / ``lead_dev``)."""
        T = self._fixed[n]
        if T is not None:
            return T, None, True
        ent = self._store.get(n)
        if ent is not None:
            return ent[0], ent[1], True
        T, dev = self._build(n)
        size = sum(a.nbytes for a in (T,) + tuple(dev))
        if self._stored + size <= _TRANSITION_BUDGET:
            self._store[n] = (T, dev)
            self._stored += size
            (self.block_dev if self._lazy[n][0] == "q" else self.lead_dev)[n + 1] = dev
            return T, dev, True
        return T, dev, False

    def _built_all(self) -> None:
        """After a full forward pass: the margins are no longer needed when
        every transition built here was kept."""
        if all(n in self._store or self._lazy[n][3] is not None for n in self._lazy):
            self._mg = None

    def _scan_overflow(self) -> None:
        """Set ``overflow`` from the transitions not built yet (a linear pass
        that stopped early names its cause as when every transition was built
        first)."""
        for n, (kind, A, B, mg) in self._lazy.items():
            if self.overflow:
                return
            if kind != "q" or n in self._built:
                continue
            mg = self._mg if mg is None else mg
            fA, FA = mg.nodes(A)
            fB, FB = mg.nodes(B)
            self.overflow |= _kernel_outer(self.model, fA, FA, fB, FB, log=False)[1]

    def _forget(self) -> None:
        """Drop the transitions and what rebuilds them, once no pass needs them
        (:func:`impute`, before the laws of the missing values); ``trans`` is
        then unavailable. The messages, grids and column factors stay."""
        self._store, self._stored, self._mg, self._fused = {}, 0, None, None
        self._fixed = self.raw = None
        self.Q = self.Wobs = None

    def _kept(self, n: int) -> bool:
        """Whether trans[n] is at hand without being rebuilt."""
        return self._fixed[n] is not None or n in self._store

    def _release(self) -> None:
        """Drop what a chain taken over by another one (``reuse``) holds."""
        self._store, self._stored, self._lazy, self._mg = {}, 0, {}, None
        self._fixed = self.raw = None
        self._fused = None

    def _deferred(self, n: int, alphas, betas):
        """(ξ part, report part) of the transition into n that the chain did not
        keep: from the backward pass that rebuilt it (``_fused``), for these
        messages, or rebuilt now."""
        f = self._fused
        if f is not None and f[0] is alphas and f[1] is betas and n in f[2]:
            return f[2][n]
        T, dev = self._build(n - 1)
        return _fused_step(self, n - 1, T, dev, alphas, betas)

    def size(self, n: int) -> int:
        return self.K * self.G if self.miss[n] else self.K

    def nodes_at(self, positions) -> np.ndarray:
        """(P, G) quadrature nodes of the missing ``positions``."""
        return np.array([self.grids[int(n)].nodes for n in positions]).reshape(-1, self.G)


def _joint(log: bool, a: np.ndarray, T: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Posterior mass through a transition, up to a constant: a_u T_uv b_v
    (``log``: T in log space, the result scaled by its maximum)."""
    if log:
        L = np.log(a)[:, None] + T + np.log(b)[None, :]
        mx = np.max(L)
        return np.exp(L - mx) if np.isfinite(mx) else np.zeros_like(L)
    return a[:, None] * T * b[None, :]


def _fused_step(chain: _Chain, n: int, T: np.ndarray, dev, alphas, betas):
    """(ξ part, report part) of the transition n → n + 1 between missing rows:
    its joint posterior summed over the nodes of both ends (K, K), and its
    term of :func:`_quadrature_report` (None: no weight), from the transition
    T and its ``dev``."""
    K, G = chain.K, chain.G
    with np.errstate(divide="ignore", under="ignore", over="ignore", invalid="ignore"):
        J = _joint(chain.log, alphas[n], T, betas[n + 1])
        xi = J.reshape(K, G, -1).sum(axis=1)
        xi = xi.reshape(K, K, G).sum(axis=2)
    with np.errstate(all="ignore"):
        if chain._lazy[n][0] == "q":
            part = _block_part(chain, n + 1, dev[0], dev[1], J)
        else:
            part = _lead_part(chain, n + 1, dev[0], dev[1], J)
    return xi, part


def _forward_chain(chain: _Chain, *, stop_on_overflow: bool = False):
    """Scaled forward pass; ``None`` when a linear step underflows (or, with
    ``stop_on_overflow``, when a transition it builds overflows).

    Returns ``(alphas, log_lik)`` with ``alphas[n]`` the normalised message in
    *linear* scale (S_n,).
    """
    N = chain.N
    alphas = [None] * N
    trans = chain.trans
    if chain.log:
        la = chain.init
        lc = float(_inf._lse(la))
        if not np.isfinite(lc):
            raise IncompatibleObservationError(
                f"Forward pass (missing data): Y[0] has zero density under every "
                f"state even in log space (log C_1 = {lc!r})."
            )
        ll = lc
        la = la - lc
        alphas[0] = np.exp(la)
        for n in range(N - 1):
            r = _inf._lse(la[:, None] + trans[n], axis=0)
            lc = float(_inf._lse(r))
            if not np.isfinite(lc):
                raise IncompatibleObservationError(
                    f"Forward pass (missing data): Y[{n + 1}] has zero density "
                    f"under every state reachable from step {n} (log C = {lc!r}), "
                    f"even in log space. The observation sequence is incompatible "
                    f"with the model."
                )
            ll += lc
            la = r - lc
            alphas[n + 1] = np.exp(la)
        chain._built_all()
        return alphas, float(ll)

    a = chain.init
    C = float(a.sum())
    if not (np.isfinite(C) and C >= MIN_POSITIVE):
        return None
    ll = np.log(C)
    alphas[0] = a / C
    for n in range(N - 1):
        T = trans[n]
        if stop_on_overflow and chain.overflow:
            return None
        raw = alphas[n] @ T
        C = float(raw.sum())
        if not (np.isfinite(C) and C >= MIN_POSITIVE):
            return None
        ll += np.log(C)
        alphas[n + 1] = raw / C
    chain._built_all()
    return alphas, float(ll)


def _backward_chain(chain: _Chain, alphas=None):
    """Scaled backward pass (own normaliser per step); ``None`` on linear underflow.

    ``alphas``: the chain's forward messages; with them, at every transition
    the chain did not keep (rebuilt here), the parts of ξ and of the
    quadrature report that need it are computed on the way (``chain._fused``).
    """
    N = chain.N
    betas = [None] * N
    fused = None if alphas is None else {}
    S = chain.size(N - 1)
    if chain.log:
        lb = np.full(S, -np.log(S))
        betas[N - 1] = np.exp(lb)
        n_void = 0
        for n in range(N - 2, -1, -1):
            T, dev, kept = chain._step(n)
            r = _inf._lse(T + lb[None, :], axis=1)
            ld = float(_inf._lse(r))
            if np.isfinite(ld):
                lb = r - ld
            else:
                n_void += 1
                s = chain.size(n)
                lb = np.full(s, -np.log(s))
            betas[n] = np.exp(lb)
            if fused is not None and not kept:
                fused[n + 1] = _fused_step(chain, n, T, dev, alphas, betas)
        if n_void:
            logger.warning(
                "Backward (missing data, log space): %d step(s) with zero weight "
                "under every state — β̂ reset to uniform there.", n_void,
            )
        if fused is not None:
            chain._fused = (alphas, betas, fused)
        return betas
    b = np.full(S, 1.0 / S)
    betas[N - 1] = b
    for n in range(N - 2, -1, -1):
        T, dev, kept = chain._step(n)
        raw = T @ betas[n + 1]
        D = float(raw.sum())
        if not (np.isfinite(D) and D >= MIN_POSITIVE):
            return None
        betas[n] = raw / D
        if fused is not None and not kept:
            fused[n + 1] = _fused_step(chain, n, T, dev, alphas, betas)
    if fused is not None:
        chain._fused = (alphas, betas, fused)
    return betas


def _pass(model, Y, miss, grid, grids, ev, *, backward: bool, reuse=None):
    """Build the chain on ``grids`` and run forward (and backward), linear
    first, log on failure; ``reuse`` a previous pass's chain whose
    transitions between unchanged grids are taken as they are (a pass that
    needed log space goes straight to log space)."""
    chain = fw = bw = None
    if reuse is None or not reuse.log:
        chain = _Chain(model, Y, miss, grid, log=False, ev=ev, grids=grids, reuse=reuse)
        if not chain.overflow:
            fw = _forward_chain(chain, stop_on_overflow=True)
            if fw is not None and backward and not chain.overflow:
                bw = _backward_chain(chain, fw[0])
        if chain.overflow or fw is None or (backward and bw is None):
            if not chain.overflow:
                chain._scan_overflow()
            logger.warning(
                "Missing-data pass: %s in linear space; recomputing in log space.",
                "a kernel value overflows" if chain.overflow else "a step underflows",
            )
            chain = None
    if chain is None:
        chain = _Chain(model, Y, miss, grid, log=True, ev=ev, grids=grids,
                       reuse=reuse if (reuse is not None and reuse.log) else None)
        fw = _forward_chain(chain)
        bw = _backward_chain(chain, fw[0]) if backward else None
    return chain, fw, bw


def _run_chain(model, Y, miss, grid, *, backward: bool, ev=_inf._FROM_MODEL):
    """Build the chain and run forward (and backward), linear first, log on failure.

    ``ev``: missingness evidence factors (:class:`_Chain`), by default those
    of ``model.missingness`` on ``miss``. When the grids are built by this
    call (:func:`_gap_grids_pass`), the pass that weighted them is reused:
    as it is if it ran on the final grids, else for the transitions between
    grids that did not change.
    """
    if ev is _inf._FROM_MODEL:
        ev = _inf._evidence(model, miss)
    (grids, _), last = _gap_grids_pass(model, Y, miss, grid, ev)
    if last is not None and not last[3]:
        chain, fw, bw = last[:3]
    else:
        # the weighting pass's messages go now; its chain, once taken over
        reuse, last = (None if last is None else last[0]), None
        chain, fw, bw = _pass(model, Y, miss, grid, grids, ev, backward=backward, reuse=reuse)
        reuse = None
    chain.quad_error = _quadrature_report(chain, fw[0], bw)
    limit = quad_warn_limit(miss)
    if chain.quad_error > limit:
        logger.warning(
            "Missing-data quadrature not converged: relative error %.1e > %.1e on the integrals "
            "the grid must reproduce exactly (%d missing rows, gap_nodes = %d). Results that "
            "involve the missing rows may be off; increase gap_nodes.",
            chain.quad_error, limit, int(miss.sum()), chain.G,
        )
    return chain, fw[0], fw[1], bw


#: WARNING threshold of the quadrature diagnostic (:func:`_quadrature_report`),
#: an estimate of Σ_runs |error of the run's log-likelihood factor| (module
#: docstring, "Convergence diagnostic"). Calibrated on 67 passes against exact
#: or G = 512–1024 references, a pass "off" when that sum exceeds 0.01 nats.
QUAD_WARN = 0.05


def quad_warn_limit(miss: np.ndarray) -> float:
    """The WARNING threshold of ``quad_error``: QUAD_WARN, whatever the mask.

    Until this version it was 1e-3 · √R, R the number of runs of missing
    rows: 0.046 on an Intel Lab window of 2 157 runs, 1e-3 on a leading gap
    alone. ``quad_error`` now sums per-run estimates capped at 1, which
    scale with the errors of the runs, not with their number (module
    docstring, "Convergence diagnostic"). The argument is kept for callers.
    """
    return QUAD_WARN


#: Cap of the relative error of one block in :func:`_quadrature_report`. A
#: block rescaled to its exact mass carries its posterior weight whatever its
#: raw mass, so the error it can cause is bounded by that weight; uncapped,
#: blocks whose raw mass was 100 times its target (a narrow kernel on a coarse
#: grid) made the report 590 on a pass off by 6.7 nats (Intel mote 48, the
#: k-means start of ICE).
_REL_CAP = 1.0
#: Weight of the reverse-mass check of a leading gap in :func:`_quadrature_report`.
#: The column rescaling makes every reverse mass exact, so a relative error
#: r of the raw mass moves the posterior inside the gap by far less than r:
#: measured on the regime-AR(1) references (leading gaps of 2, 5 and 17 rows,
#: τ = 0.99–0.999, G = 64–256), the largest error of a posterior mean or sd
#: inside the gap, in units of the exact posterior sd, is at most 0.02 times
#: the sum of the checks over the gap (2e-4 to 2e-2 times it).
_LEAD_SCALE = 0.02


def _block_rel(raw, tgt, log: bool):
    """Capped relative errors of raw block masses against their exact values,
    and those exact values in linear scale (:func:`_quadrature_report`)."""
    if log:
        ok = np.isfinite(raw) & np.isfinite(tgt)
        rel = np.where(ok, np.abs(np.expm1(np.where(ok, raw - tgt, 0.0))), 0.0)
        T = np.where(np.isfinite(tgt), np.exp(tgt), 0.0)
        bad = np.isfinite(tgt) & ~np.isfinite(raw)
    else:
        ok = np.isfinite(raw) & (tgt > 0.0)
        rel = np.where(ok, np.abs(raw / np.where(ok, tgt, 1.0) - 1.0), 0.0)
        T = np.where(np.isfinite(tgt), tgt, 0.0)
        bad = (tgt > 0.0) & ~(np.isfinite(raw) & (raw > 0.0))
    return np.minimum(np.where(bad, 1.0, rel), _REL_CAP), T


def _block_w(chain: _Chain, n: int, J: np.ndarray, shape) -> np.ndarray:
    """Posterior weights of the blocks into n from the joint mass J through
    trans[n − 1], rows from background panels left out."""
    K, G = chain.K, chain.G
    w = J.reshape(J.shape[0], K, G).sum(axis=2).reshape(shape)
    src = chain.grids[n - 1] if chain.miss[n - 1] else None
    if src is not None and src.local:
        bgn = ~np.isfinite(src.mix.loc[src.mix.panel])      # (G,)
        w = np.where(bgn[None, :, None], 0.0, w)
    return w


def _lead_rel(lraw, lex):
    """(finite exact values, capped relative errors) of a leading gap's
    reverse masses (:func:`_quadrature_report`)."""
    ok = np.isfinite(lex)
    rel = np.where(ok & np.isfinite(lraw),
                   np.abs(np.expm1(np.where(ok & np.isfinite(lraw), lraw - lex, 0.0))), 0.0)
    return ok, np.minimum(np.where(ok & ~np.isfinite(lraw), 1.0, rel), _REL_CAP)


def _lead_mask(chain: _Chain, n: int, w: np.ndarray) -> np.ndarray:
    dst = chain.grids[n]
    if dst.local:
        # destinations on background panels: the broad tail of the
        # prior, which the column rescaling keeps (as the rows of the
        # forward blocks from background sources above)
        w = np.where(np.isfinite(dst.mix.loc[dst.mix.panel])[None, None, :], w, 0.0)
    return w


def _weighted(w: np.ndarray, rel: np.ndarray):
    """Σ w·rel / Σ w, or None without weight."""
    tot = w.sum()
    if np.isfinite(tot) and tot > 0.0:
        return float((w * rel).sum() / tot)
    return None


def _block_part(chain: _Chain, n: int, raw, tgt, J: np.ndarray):
    """Term of the blocks into n (``block_dev[n]``), J the joint mass through trans[n − 1]."""
    rel, _ = _block_rel(raw, tgt, chain.log)
    return _weighted(_block_w(chain, n, J, raw.shape), rel)


def _lead_part(chain: _Chain, n: int, lraw, lex, J: np.ndarray):
    """Term of a leading gap's reverse masses into n (``lead_dev[n]``), before _LEAD_SCALE."""
    K, G = chain.K, chain.G
    _, rel = _lead_rel(lraw, lex)
    return _weighted(_lead_mask(chain, n, J.reshape(K, G, K, G).sum(axis=1)), rel)


def _quadrature_report(chain: _Chain, alphas, betas=None, *, per_run: bool = False):
    """Convergence diagnostic of the quadrature of a pass (module docstring).

    Every missing position is checked on integrals its grid must reproduce
    and whose exact value is known, each check a weighted mean of relative
    errors (capped at _REL_CAP), summed over the positions:

    * the raw block masses before the Tauchen–Hussey renormalisation
      (Σ_g ω_g q(j, y_g | u) against P(x_n = j | u) for every source u —
      the observed row before a gap, a node of the previous missing
      position, or the prior for a missing y_1), weighted by the posterior
      mass that goes through them (α̂ × transition × β̂ summed over the
      nodes of the block; a backward pass is run when the caller has none).
      Rows from the background panels of a local grid, which carry a broad
      law that the renormalised rows keep however sparse the nodes, are not
      counted;
    * inside a leading gap carried by the prior's transitions
      (:func:`_lead_transition`), the reverse masses Σ_g ψ_i(g) q(j, y_g' |
      i, y_g) ω_g' against ρ_ij(y_g') ω_g', weighted by the posterior mass
      through each destination node — how well the grid of n resolves the
      reverse kernel into every node of n + 1 (until this version the rows
      of a leading gap were not checked at all, and a leading gap off by 22
      nats went unreported);
    * at the last missing row before an observed y_{n+1}, on a local grid,
      the exit integrals Σ_g ω_g μ(i, y_g) q(k, y_{n+1} | i, y_g) against
      their closed form (:func:`_neighbour_error`, cached per (i, k) as the
      grid's ``exit_rel``), weighted by the posterior of (x_n, x_{n+1})
      through the exit transition: the exit kernel is as narrow as the
      transition, and the renormalisation does not see it.

    Returns the sum (0 with no missing row); with ``per_run`` also a dict
    {first row of the run: its part}. The sum estimates Σ_runs |error of the
    run's log-likelihood factor| in nats, within a factor of about 15 on the
    references of the module docstring ("Convergence diagnostic").
    """
    K, G = chain.K, chain.G
    miss = chain.miss
    a_, b_ = _runs(miss)
    run_of = np.full(chain.N, -1)
    for a, b in zip(a_, b_):
        run_of[a:b + 1] = a
    parts = dict.fromkeys((int(a) for a in a_), 0.0)
    if betas is None and (chain.block_dev or chain.lead_dev):
        betas = _backward_chain(chain, alphas)

    trans = chain.trans

    def joint(n):                          # posterior mass through trans[n - 1] (S_{n-1}, S_n)
        return _joint(chain.log, alphas[n - 1], trans[n - 1], betas[n])

    with np.errstate(all="ignore"):
        for n, dev in chain.block_dev.items():
            if dev is _DEFERRED:           # a transition the chain did not keep
                part = chain._deferred(n, alphas, betas)[1]
            else:
                raw, tgt = dev
                rel, T = _block_rel(raw, tgt, chain.log)
                if betas is None:                            # forward weights only
                    if n == 0 and miss[0]:
                        w = T                                                # (K,)
                    else:
                        a = alphas[n - 1]
                        a = a.reshape(K, G) if miss[n - 1] else a
                        w = a[..., None] * T
                elif n == 0 and miss[0]:
                    w = (alphas[0] * betas[0]).reshape(K, G).sum(axis=1)
                else:
                    w = _block_w(chain, n, joint(n), raw.shape)
                part = _weighted(w, rel)
            if part is not None:
                parts[int(run_of[n])] += part
        for n, dev in chain.lead_dev.items():
            if dev is _DEFERRED:
                part = chain._deferred(n, alphas, betas)[1]
            else:
                lraw, lex = dev
                ok, rel = _lead_rel(lraw, lex)
                if betas is None:
                    w = np.where(ok, np.exp(np.where(ok, lex, -np.inf)), 0.0) * \
                        alphas[n - 1].reshape(K, G).sum(axis=1)[:, None, None]
                else:
                    w = joint(n).reshape(K, G, K, G).sum(axis=1)             # (K_i, K_j, G)
                part = _weighted(_lead_mask(chain, n, w), rel)
            if part is not None:
                parts[int(run_of[n])] += _LEAD_SCALE * part
        exits = [int(n) for n in np.nonzero(miss[:-1] & ~miss[1:])[0] if chain.grids[int(n)].local]
        cached = [chain.grids[n].exit_rel for n in exits]
        todo = np.array([n for n, c in zip(exits, cached) if c is None], dtype=int)
        if todo.size:
            nodes = np.array([chain.grids[int(n)].nodes for n in todo])
            omega = np.array([chain.grids[int(n)].omega for n in todo])
            rel = np.full((todo.size, K, K), np.nan)
            _neighbour_error(chain.model, nodes, omega, np.full(todo.size, np.nan),
                             chain.Y[todo + 1], exit_rel=rel)
            done = dict(zip(todo.tolist(), rel))
            cached = [done[n] if c is None else c for n, c in zip(exits, cached)]
        for n, rel in zip(exits, cached):
            rel = np.minimum(np.where(np.isfinite(rel), rel, 0.0), _REL_CAP)
            if betas is None:
                w = alphas[n].reshape(K, G).sum(axis=1)[:, None] * np.ones((1, K))
            else:
                # the posterior of (x_n, x_{n+1}) through the exit transition
                w = joint(n + 1).reshape(K, G, K).sum(axis=1)
            tot = w.sum()
            if np.isfinite(tot) and tot > 0.0:
                parts[int(run_of[n])] += float((w * rel).sum() / tot)
    total = float(sum(parts.values()))
    return (total, parts) if per_run else total


# ---------------------------------------------------------------------------
# Posterior summaries of the augmented chain
# ---------------------------------------------------------------------------

@dataclass
class GapPosterior:
    """Posterior quantities of a sequence with missing observations.

    Attributes
    ----------
    miss       : (N,) bool — missing rows.
    log_lik    : float — log p(y_obs), the observed-data log-likelihood
                 (log p(y_obs, m) with a non-ignorable ``model.missingness``,
                 every posterior below being then given the mask m too).
    alpha_hat  : (N, K) — P(x_n = i | observations up to n).
    beta_hat   : (N, K) — normalised backward messages (module docstring for
                 their definition at a missing n).
    gamma      : (N, K) — P(x_n = i | y_obs).
    xi         : (N-1, K, K) or None — P(x_n = i, x_{n+1} = j | y_obs).
    method     : ``"exact"`` (K-state shortcut) or ``"grid"``.
    grid       : QuadratureGrid or None (exact shortcut) — the reference grid.
    node_post  : (M, K, G) or None — P(x_n = i, y_n ∈ node g | y_obs) at the
                 M missing positions ``np.nonzero(miss)[0]`` (grid variants).
    nodes      : (M, G) or None — the quadrature nodes y_g of each missing
                 position: the reference nodes, or those of its local grid
                 (module docstring, "Local grids").
    quad_error : float or None — the quadrature diagnostic of
                 :func:`_quadrature_report` (grid variants): the relative
                 errors of the quadrature on the integrals it must reproduce
                 exactly, weighted by the posterior and summed over the runs
                 of missing rows — an estimate of Σ_runs |error of the run's
                 log-likelihood factor| in nats (module docstring,
                 "Convergence diagnostic"). Above ``QUAD_WARN`` (0.05) a
                 WARNING is logged.
    """

    miss: np.ndarray
    log_lik: float
    alpha_hat: np.ndarray
    beta_hat: np.ndarray
    gamma: np.ndarray
    xi: np.ndarray | None
    method: str
    grid: QuadratureGrid | None = None
    node_post: np.ndarray | None = None
    nodes: np.ndarray | None = None
    quad_error: float | None = None

    @property
    def index(self) -> np.ndarray:
        """Positions of the missing rows."""
        return np.nonzero(self.miss)[0]


def _marginal_alpha(chain: _Chain, alphas) -> np.ndarray:
    K, G = chain.K, chain.G
    out = np.empty((chain.N, K))
    for n in range(chain.N):
        a = alphas[n]
        out[n] = a.reshape(K, G).sum(axis=1) if chain.miss[n] else a
    return out


def _chain_posterior(chain: _Chain, alphas, betas, *, want_xi: bool) -> dict:
    K, G, N, miss = chain.K, chain.G, chain.N, chain.miss
    alpha_hat = _marginal_alpha(chain, alphas)
    beta_hat = np.empty((N, K))
    gamma = np.empty((N, K))
    idx = np.nonzero(miss)[0]
    node_post = np.empty((idx.size, K, G))
    obs = np.nonzero(~miss)[0]
    if obs.size:
        A = np.array([alphas[n] for n in obs])
        B = np.array([betas[n] for n in obs])
        beta_hat[obs] = B
        gamma[obs] = _inf.smooth(A, B)
    for m, n in enumerate(idx):
        a = alphas[n].reshape(K, G)
        b = betas[n].reshape(K, G)
        joint = a * b
        tot = joint.sum()
        if np.isfinite(tot) and tot > 0.0:
            node_post[m] = joint / tot
        else:
            node_post[m] = a / max(a.sum(), MIN_POSITIVE)
        gamma[n] = node_post[m].sum(axis=1)
        am = a.sum(axis=1)
        bm = np.where(am > 0.0, joint.sum(axis=1) / np.where(am > 0.0, am, 1.0), b.mean(axis=1))
        s = bm.sum()
        beta_hat[n] = bm / s if (np.isfinite(s) and s > 0.0) else np.full(K, 1.0 / K)

    xi = None
    if want_xi and N > 1:
        xi = np.empty((N - 1, K, K))
        trans = chain.trans
        with np.errstate(divide="ignore", under="ignore", over="ignore", invalid="ignore"):
            for n in range(N - 1):
                if n in chain._lazy and not chain._kept(n):
                    # a transition the chain did not keep: summed by the
                    # backward pass that rebuilt it
                    J = chain._deferred(n + 1, alphas, betas)[0]
                else:
                    J = _joint(chain.log, alphas[n], trans[n], betas[n + 1])
                    if miss[n]:
                        J = J.reshape(K, G, -1).sum(axis=1)
                    if miss[n + 1]:
                        J = J.reshape(K, K, G).sum(axis=2)
                s = J.sum()
                if np.isfinite(s) and s > 0.0:
                    xi[n] = J / s
                else:
                    xi[n] = gamma[n][:, None] * gamma[n + 1][None, :]
    return dict(alpha_hat=alpha_hat, beta_hat=beta_hat, gamma=gamma, xi=xi,
                node_post=node_post, nodes=chain.nodes_at(idx))


def _check_length(Y) -> None:
    if len(Y) < 2:
        raise ValueError(
            f"Inference with missing observations requires N ≥ 2 rows, got N={len(Y)}."
        )


def gap_posterior(
    model: PMCModel,
    Y: np.ndarray,
    *,
    gap_nodes: int | None = None,
    xi: bool = True,
) -> GapPosterior:
    """Exact (up to quadrature) posterior quantities of a NaN-bearing Y.

    Works for every variant and for a Y without missing values too (then
    identical to the functions of :mod:`pmcprg.pmc.inference`). The exact
    shortcut variants use ``precompute_weights`` / ``forward`` / ``backward``
    / ``smooth`` / ``joint_posteriors`` on the marginalised weights; the grid
    variants run the augmented chain (module docstring). With a
    non-ignorable ``model.missingness`` every quantity is given (y_obs, m),
    the missingness factors included (module docstring, "Missingness
    mechanisms").

    Parameters
    ----------
    model     : PMCModel — known parameters.
    Y         : (N,) or (N, d) observations, NaN (any non-finite value) where missing.
    gap_nodes : number G of quadrature nodes (default 64; grid variants only).
                Memory grows with G: about 0.3 kB per missing row and per
                node (K = 3; messages, grids, posteriors), plus 8·(K·G)²
                bytes per step of a gap that touches a local grid (295 kB
                at G = 64, 4.7 MB at G = 256), of which at most 2 GiB are
                kept and the rest rebuilt when needed (module docstring,
                "Memory").
    xi        : also compute the pairwise posteriors ξ (default True).
    """
    return _posterior(model, Y, gap_nodes, xi)[0]


def _posterior(model, Y, gap_nodes, xi, *, ev=_inf._FROM_MODEL):
    """:func:`gap_posterior` plus the run it came from.

    Returns ``(GapPosterior, run)`` with ``run = (W, None, None)`` for the
    exact shortcut (the marginalised weights, missingness factors included)
    and ``(chain, alphas, betas)`` for the grid. ``ev``: evidence factors,
    by default those of ``model.missingness`` on the mask of Y.
    """
    Y = np.asarray(Y, dtype=float)
    _check_length(Y)
    miss = missing_mask(Y)
    if ev is _inf._FROM_MODEL:
        ev = _inf._evidence(model, miss)
    if not needs_grid(model) or not miss.any():
        W, f_pdf = _inf._weights(model, Y, ev)
        a, ll = _inf._forward(model, Y, W, f_pdf, ev=ev)
        b = _inf._backward(model, Y, W, ev=ev)
        g = _inf.smooth(a, b)
        x = _inf.joint_posteriors(a, W, b) if xi else None
        return (GapPosterior(miss=miss, log_lik=float(ll), alpha_hat=a, beta_hat=b,
                             gamma=g, xi=x, method="exact"), (W, None, None))
    _check_grid_supported(model)
    grid = reference_grid(model, gap_nodes)
    chain, alphas, ll, betas = _run_chain(model, Y, miss, grid, backward=True, ev=ev)
    post = _chain_posterior(chain, alphas, betas, want_xi=xi)
    return (GapPosterior(miss=miss, log_lik=float(ll), method="grid", grid=grid,
                         quad_error=chain.quad_error, **post),
            (chain, alphas, betas))


# ---- entry points used by pmcprg.pmc.inference --------------------------------

def _grid_forward(model, Y, miss, gap_nodes, *, ev=_inf._FROM_MODEL):
    _check_length(Y)
    _check_grid_supported(model)
    grid = reference_grid(model, gap_nodes)
    chain, alphas, ll, _ = _run_chain(model, np.asarray(Y, dtype=float), miss, grid,
                                      backward=False, ev=ev)
    return _marginal_alpha(chain, alphas), ll


def _grid_backward(model, Y, miss, gap_nodes, *, ev=_inf._FROM_MODEL):
    return _posterior(model, Y, gap_nodes, False, ev=ev)[0].beta_hat


def _grid_sample(model, Y, miss, rng, gap_nodes, return_y):
    Y = np.asarray(Y, dtype=float)
    _check_length(Y)
    _check_grid_supported(model)
    grid = reference_grid(model, gap_nodes)
    chain, alphas, _, _ = _run_chain(model, Y, miss, grid, backward=False)
    U = _ffbs(chain.trans, alphas, chain.log, rng, 1)[0]
    X, Yc = _decode_path(U, miss, chain, Y)
    return (X, Yc) if return_y else X


# ---------------------------------------------------------------------------
# Forward-filter backward-sample on a chain of variable width (vectorised)
# ---------------------------------------------------------------------------

def _draw_rows(w: np.ndarray, rng: np.random.Generator, fallback: np.ndarray) -> np.ndarray:
    """One index per row of w (S, M), with probability ∝ the row."""
    S, M = w.shape
    tot = w.sum(axis=1)
    bad = ~(np.isfinite(tot) & (tot > 0.0))
    if bad.any():
        fb = np.asarray(fallback, dtype=float)
        s = fb.sum()
        fb = fb / s if (np.isfinite(s) and s > 0.0) else np.full(M, 1.0 / M)
        w = w.copy()
        w[bad] = fb
    cum = np.cumsum(w, axis=1)
    u = rng.random(S) * cum[:, -1]
    idx = np.sum(cum <= u[:, None], axis=1)
    over = idx >= M
    if over.any():
        last_pos = M - 1 - np.argmax(w[over, ::-1] > 0.0, axis=1)
        idx[over] = last_pos
    return idx


def _ffbs(trans, alphas, log: bool, rng: np.random.Generator, S: int) -> np.ndarray:
    """S joint draws of the (augmented) path; (S, N) indices."""
    N = len(alphas)
    U = np.empty((S, N), dtype=int)
    last = np.broadcast_to(alphas[N - 1], (S, alphas[N - 1].size))
    U[:, N - 1] = _draw_rows(np.array(last), rng, alphas[N - 1])
    with np.errstate(divide="ignore", under="ignore", over="ignore", invalid="ignore"):
        for n in range(N - 2, -1, -1):
            T = trans[n][:, U[:, n + 1]].T                 # (S, S_n)
            if log:
                L = np.log(alphas[n])[None, :] + T
                mx = np.max(L, axis=1, keepdims=True)
                w = np.where(np.isfinite(mx), np.exp(L - np.where(np.isfinite(mx), mx, 0.0)), 0.0)
            else:
                w = alphas[n][None, :] * T
            U[:, n] = _draw_rows(w, rng, alphas[n])
    return U


def _node_draws(chain: _Chain, U: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """The node values (…, M) of augmented indices ``U`` (…, M) at the missing ``idx``."""
    nodes = chain.nodes_at(idx)                                      # (M, G)
    return nodes[np.arange(len(idx)), U % chain.G]


def _decode_path(U: np.ndarray, miss: np.ndarray, chain: _Chain, Y: np.ndarray):
    """Augmented indices → (states, Y with the missing rows set to the drawn nodes)."""
    X = np.where(miss, U // chain.G, U).astype(int)
    Yc = np.array(Y, dtype=float, copy=True)
    idx = np.nonzero(miss)[0]
    Yc[idx] = _node_draws(chain, U[idx], idx)
    return X, Yc


# ---------------------------------------------------------------------------
# Nyström interpolation of the posterior density of a missing observation
# ---------------------------------------------------------------------------

#: Size of the fine grid of the Nyström interpolation, in multiples of G.
_NYSTROM_FACTOR = 4


def _nystrom_density(chain: _Chain, alphas, betas, positions: np.ndarray,
                     points: np.ndarray) -> np.ndarray:
    """Unnormalised posterior density of y_n at ``points`` (P, M), for the missing ``positions``.

    The density of y_n = y given the observations follows from the messages
    of the neighbouring positions through the exact kernels, at any y:

        π_n(y) ∝ Σ_i F_n(i, y) B_n(i, y),
        F_n(i, y) = Σ_u α̃_{n-1}(u) q(i, y | u),
        B_n(i, y) = Σ_v T_{(i, y) → v} β̃_{n+1}(v),

    with u, v the (augmented) states at n − 1 and n + 1 on their own grids,
    and T_{(i, y) → v} the chain's transition out of (i, y), built like a row
    of Q (block-normalised to P(x_{n+1} | x_n = i, y_n = y)) or the exit
    density q(j, y_{n+1} | i, y) when y_{n+1} is observed (Nyström 1930).
    F_n propagates the discrete message of n − 1 through the exact kernel:
    ∫ F_n(i, y) dy = Σ_u α̃_{n−1}(u) P(x_n = i | u) exactly, wherever the
    kernel puts its mass. (Until this version F_n also carried the block
    factors of the chain's renormalisation — exact at the nodes, but a factor
    of up to 1e30 off the nodes for a row whose kernel the grid misses, which
    made the quantiles of problem P3 in ``report/forecasting``.)
    ``betas`` None means B ≡ 1 (a trailing gap, for :func:`forecast`).
    With missingness factors (``chain.ev``) F_n is multiplied by e_n(i) and
    β̃_{n+1}(v) by e_{n+1}(v), as the chain's messages are.

    Rows that are not finite and positive somewhere are returned as NaN (the
    caller then uses the coarse node masses).
    """
    model, K, G = chain.model, chain.K, chain.G
    miss, N, Y = chain.miss, chain.N, chain.Y
    positions = np.asarray(positions, dtype=int)
    points = np.asarray(points, dtype=float)
    P, M = points.shape
    ev = chain.ev
    fP, FP = _margin_eval(model, points.ravel(), log=False)
    fP = fP.reshape(P, M, K, K)
    FP = None if FP is None else FP.reshape(P, M, K, K)
    cache = {}

    def node_margins(g):
        key = id(g)
        if key not in cache:
            cache[key] = _margin_eval(model, g.nodes, log=False)
        return cache[key]

    def take(F, idx):
        return None if F is None else F[idx]

    def kern(fl, Fl, fr, Fr):
        return _kernel(model, fl, Fl, fr, Fr)[0]

    with np.errstate(divide="ignore", invalid="ignore", under="ignore", over="ignore"):
        F = np.zeros((P, K, M))
        B = np.ones((P, K, M))
        first = positions == 0
        prev_miss = np.zeros(P, bool)
        prev_miss[~first] = miss[positions[~first] - 1]
        last = positions == N - 1
        next_miss = np.zeros(P, bool)
        next_miss[~last] = miss[positions[~last] + 1]

        # ---- incoming side ------------------------------------------------
        for p in np.nonzero(first)[0]:
            F[p] = _initial(model, fP[p], log=False).T                          # μ(i, y)
        # inside a leading gap of known prior the forward density is the prior
        # itself, w_n(i) φ_i(y) (:func:`_lead_transition`; e_n included in w_n)
        lead_p = ~first & (positions < chain.lead) & chain.lead_exact
        prior0 = _initial(model, np.ones((1, K, K)), log=False)[0]              # ∫ μ(i, ·)
        for p in np.nonzero(lead_p)[0]:
            n = int(positions[p])
            w = alphas[n].reshape(K, G).sum(axis=1)
            phi = _initial(model, fP[p], log=False).T / np.where(prior0 > 0.0, prior0, 1.0)[:, None]
            F[p] = w[:, None] * phi
        prev_miss &= ~lead_p
        sel = np.nonzero(~first & ~prev_miss & ~lead_p)[0]
        if sel.size:
            n_prev = positions[sel] - 1
            fo, Fo = _margin_eval(model, Y[n_prev], log=False)
            rep_ = np.repeat(np.arange(sel.size), M)
            ker = kern(fo[rep_], take(Fo, rep_), fP[sel].reshape(-1, K, K),
                       take(FP, sel).reshape(-1, K, K) if FP is not None else None)
            ker = ker.reshape(sel.size, M, K, K)                                   # [p, m, h, i]
            a = np.array([alphas[int(n)] for n in n_prev])                          # (P', K)
            F[sel] = np.einsum("ph,pmhi->pim", a, ker)
        for p in np.nonzero(~first & prev_miss)[0]:
            n = int(positions[p])
            A = chain.grids[n - 1]
            fA, FA = node_margins(A)
            rep_, til = np.repeat(np.arange(G), M), np.tile(np.arange(M), G)
            ker = kern(fA[rep_], take(FA, rep_), fP[p][til], take(FP, p)[til] if FP is not None else None)
            C = ker.reshape(G, M, K, K).transpose(2, 0, 3, 1).reshape(K * G, K * M)
            F[p] = (alphas[n - 1] @ C).reshape(K, M)
        if ev is not None:
            fe = np.where(lead_p[:, None], 1.0, ev[positions])
            F *= fe[:, :, None]                               # e_n(i) of the transition into n

        # ---- outgoing side ------------------------------------------------
        if betas is not None:
            sel = np.nonzero(~last & ~next_miss)[0]
            if sel.size:
                n_next = positions[sel] + 1
                fo, Fo = _margin_eval(model, Y[n_next], log=False)
                rep_ = np.repeat(np.arange(sel.size), M)
                ker = kern(fP[sel].reshape(-1, K, K),
                           take(FP, sel).reshape(-1, K, K) if FP is not None else None,
                           fo[rep_], take(Fo, rep_)).reshape(sel.size, M, K, K)
                b = np.array([betas[int(n)] for n in n_next])                     # (P', K)
                if ev is not None:
                    b = b * ev[n_next]                          # e_{n+1}(j), as in the chain
                B[sel] = np.einsum("pmij,pj->pim", ker, b)
            for p in np.nonzero(~last & next_miss)[0]:
                n = int(positions[p])
                Bg = chain.grids[n + 1]
                if n in chain.lead_c:
                    # a leading gap: the chain's column factors (:func:`_lead_transition`)
                    lB, FBl = _margin_eval(model, Bg.nodes, log=True)
                    lP, FPl = _margin_eval(model, points[p], log=True)
                    rep_, til = np.repeat(np.arange(M), G), np.tile(np.arange(G), M)
                    LK = _log_kernel(model, lP[rep_], take(FPl, rep_), lB[til],
                                     take(FBl, til)).reshape(M, G, K, K).transpose(2, 0, 3, 1)
                    T = np.exp(LK + np.log(Bg.omega) + chain.lead_c[n][:, None, :, :])
                    T = np.where(np.isfinite(T), T, 0.0).reshape(K * M, K * G)
                    b = betas[n + 1]
                    if ev is not None:
                        b = b * np.repeat(ev[n + 1], G)
                    B[p] = (T @ b).reshape(K, M)
                    continue
                fB, FB = node_margins(Bg)
                rep_, til = np.repeat(np.arange(M), G), np.tile(np.arange(G), M)
                ker = kern(fP[p][rep_], take(FP, p)[rep_] if FP is not None else None,
                           fB[til], take(FB, til)).reshape(M, G, K, K)
                T = ker.transpose(2, 0, 3, 1) * Bg.omega                          # (K, M, K, G)

                def rescue(idx, pts=points[p], Bg=Bg):   # blocks (i, y_m → j, ·)
                    return _log_rows(model, _margin_eval(model, pts, log=True), idx[1], idx[0],
                                     idx[2], _margin_eval(model, Bg.nodes, log=True), Bg.omega)

                T, _ = _normalise_blocks(T, _x_transition(model, fP[p], log=False).transpose(1, 0, 2),
                                         log=False, rescue=rescue)
                b = betas[n + 1]
                if ev is not None:
                    b = b * np.repeat(ev[n + 1], G)             # e_{n+1}(j), per node
                B[p] = (T.reshape(K * M, K * G) @ b).reshape(K, M)
            # the chain rescales β̃ by its own sum at every step: any positive
            # constant per position cancels in the normalisation.

        dens = (F * B).sum(axis=1)
    bad = ~(np.all(np.isfinite(dens), axis=1) & (dens.sum(axis=1) > 0.0))
    dens[bad] = np.nan
    return dens


# ---------------------------------------------------------------------------
# Laws on the grid: moments and quantiles
# ---------------------------------------------------------------------------

#: Tolerance of the quantile safety net on the CDF of a law on the grid: the
#: interpolated CDF must be non-decreasing and stay within this distance of
#: the Markov–Stieltjes bracket [Σ_{g'<g} m_g', Σ_{g'≤g} m_g'] at every node.
_CDF_TOL = 1e-6


def _legendre_cdf(masses: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Coefficients (n+1, R) of the CDF H(s) = ½ Σ_k c_k P_k(2s − 1) of rules.

    ``masses`` (R, n) the masses of R rules of n Gauss–Legendre nodes ``s``
    (n,): the Legendre interpolant of the density h(s) = m_g / ws_g,
    integrated from 0 (spectrally accurate for a smooth h).
    """
    n = s.size
    V = _leg.legvander(2.0 * s - 1.0, n - 1)                         # (n, n)
    c = (masses @ V) * (2.0 * np.arange(n) + 1.0)                   # (R, n)
    return _leg.legint(c, lbnd=-1, axis=1).T                        # (n+1, R)


def _cdf_ok(cint: np.ndarray, masses: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Safety net (R,) of :func:`_legendre_cdf`: monotone, and within _CDF_TOL
    of the Markov–Stieltjes bracket of the masses at the nodes."""
    n = s.size
    dense = np.linspace(0.0, 1.0, 4 * n + 1)
    H = 0.5 * _leg.legval(2.0 * dense - 1.0, cint)                   # (R, 4n+1)
    mono = np.all(np.diff(H, axis=1) >= -_CDF_TOL, axis=1)
    Hn = 0.5 * _leg.legval(2.0 * s - 1.0, cint)                      # (R, n)
    cum = np.cumsum(masses, axis=1)
    inside = np.all((Hn >= cum - masses - _CDF_TOL) & (Hn <= cum + _CDF_TOL), axis=1)
    return mono & inside & np.all(np.isfinite(H), axis=1)


def _monotone_ppf(masses: np.ndarray, ws: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Fallback quantiles in s (Q,) of a rule: the piecewise-linear CDF through
    the cumulative masses at the Gauss–Legendre cell edges Σ_{g'≤g} ws_g',
    inverted — monotone whatever the masses (first-order accurate)."""
    edges = np.concatenate([[0.0], np.cumsum(ws)])
    edges[-1] = 1.0
    cum = np.concatenate([[0.0], np.cumsum(np.maximum(masses, 0.0))])
    cum = cum / cum[-1] if cum[-1] > 0 else np.linspace(0.0, 1.0, cum.size)
    keep = np.concatenate([[True], np.diff(cum) > 0.0])      # strictly increasing knots
    return np.clip(np.interp(np.clip(levels, 0.0, 1.0), cum[keep], edges[keep]), 0.0, 1.0)


def _grid_summary(mass: np.ndarray, nodes: np.ndarray, qs: tuple[float, ...],
                  fine_mass: np.ndarray | None, grids: list, fine: list | None):
    """Mean, sd and quantiles of the laws with node masses ``mass`` (P, G).

    Moments are the quadratures Σ_g mass_g y_g^k on each position's nodes.
    Quantiles invert the CDF of the Legendre interpolant, in the Gauss–
    Legendre variable s of each rule, of the density h(s) = mass_g / ws_g —
    spectrally accurate for a smooth h — then map s to y; with ``fine_mass``
    (the Nyström masses on the grids ``fine``; NaN rows: not available) that
    finer representation is used. A local grid is a composite rule: the CDF
    at a panel boundary is the sum of the masses of the panels below, the
    interpolant is used inside a panel. Safety net (:func:`_cdf_ok`): where
    the interpolated CDF is not monotone or leaves the Markov–Stieltjes
    bracket of the masses by more than _CDF_TOL, the monotone PCHIP CDF of
    the cumulative masses is inverted instead (first-order accurate) and a
    WARNING is logged.
    """
    mass = np.asarray(mass, dtype=float)
    tot = mass.sum(axis=1, keepdims=True)
    mass = mass / np.where(tot > 0.0, tot, 1.0)
    y = np.asarray(nodes, dtype=float)
    if all(not g.local for g in grids):
        y = y[0] if y.size else np.zeros(mass.shape[1])       # the reference nodes, shared
        mean = mass @ y
        var = mass @ (y * y) - mean * mean
    else:
        mean = (mass * y).sum(axis=1)
        var = (mass * y * y).sum(axis=1) - mean * mean
    sd = np.sqrt(np.maximum(var, 0.0))
    P = mass.shape[0]
    if not qs:
        return mean, sd, np.empty((P, 0))
    levels = np.array(qs, dtype=float)
    use_fine = np.zeros(P, bool) if fine_mass is None else np.all(np.isfinite(fine_mass), axis=1)
    qv = np.empty((P, levels.size))
    n_fallback = 0
    # ---- single-rule grids (the reference grid): vectorised --------------
    for fine_rows in (True, False):
        rows = [p for p in range(P) if use_fine[p] == fine_rows and not grids[p].local]
        if not rows:
            continue
        g = fine[rows[0]] if fine_rows else grids[rows[0]]
        m = np.asarray(fine_mass[rows] if fine_rows else mass[rows], dtype=float)
        m = m / m.sum(axis=1, keepdims=True)
        cint = _legendre_cdf(m, g.s)
        target = levels[:, None] * np.ones((1, len(rows)))           # (Q, R)
        lo = np.zeros_like(target)
        hi = np.ones_like(target)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            H = 0.5 * _leg.legval(2.0 * mid - 1.0, cint, tensor=False)
            below = H < target
            lo = np.where(below, mid, lo)
            hi = np.where(below, hi, mid)
        s_q = 0.5 * (lo + hi)                                          # (Q, R)
        ok = _cdf_ok(cint, m, g.s)
        for r in np.nonzero(~ok)[0]:
            s_q[:, r] = _monotone_ppf(m[r], g.ws, levels)
            n_fallback += 1
        qv[rows] = g.y_of_s(s_q).T
    # ---- composite (local) grids ------------------------------------------
    for p in range(P):
        if not grids[p].local:
            continue
        g = fine[p] if use_fine[p] else grids[p]
        m = np.asarray(fine_mass[p] if use_fine[p] else mass[p], dtype=float)
        m = m / m.sum()
        pan = g.mix
        edges = np.concatenate([[0], np.cumsum(pan.sizes)])
        pm = np.array([m[edges[k]:edges[k + 1]].sum() for k in range(pan.sizes.size)])
        cum = np.concatenate([[0.0], np.cumsum(pm)])
        out_s = np.empty(levels.size)
        out_k = np.empty(levels.size, dtype=int)
        for a, q in enumerate(levels):
            k = int(min(np.searchsorted(cum[1:], q, side="left"), pan.sizes.size - 1))
            sl = slice(edges[k], edges[k + 1])
            mk, sk = m[sl], g.s[sl]
            r = q - cum[k]
            if pm[k] <= 0.0:
                out_s[a], out_k[a] = 0.5, k
                continue
            cint = _legendre_cdf(mk[None, :], sk)[:, 0]
            if _cdf_ok(cint[:, None], mk[None, :], sk)[0]:
                lo_, hi_ = 0.0, 1.0
                for _ in range(60):
                    mid = 0.5 * (lo_ + hi_)
                    if 0.5 * _leg.legval(2.0 * mid - 1.0, cint) < r:
                        lo_ = mid
                    else:
                        hi_ = mid
                out_s[a] = 0.5 * (lo_ + hi_)
            else:
                out_s[a] = _monotone_ppf(mk / pm[k], g.ws[sl], np.array([r / pm[k]]))[0]
                n_fallback += 1
            out_k[a] = k
        qv[p] = g.y_of_s(out_s, out_k)
    if n_fallback:
        logger.warning(
            "Quantiles of the missing values: the interpolated CDF of %d law(s) is not "
            "monotone or leaves the bracket of its node masses by more than %g; used the "
            "monotone CDF of the node masses there (first-order accurate).",
            n_fallback, _CDF_TOL,
        )
    return mean, sd, qv


def _grid_laws(chain: _Chain, alphas, betas, positions: np.ndarray, mass: np.ndarray, qs):
    """Mean, sd, quantiles and reference-node density of the laws of y at ``positions``.

    ``mass`` (P, G) the node masses on each position's grid. Quantiles use the
    Nyström masses on the fine grids (:func:`_nystrom_density`), the coarse
    masses where those are not available. The density is returned at the
    reference nodes: mass / ω on the reference grid (a position on it), the
    normalised Nyström density on a local grid.
    """
    G = chain.G
    ref = chain.grid
    grids = [chain.grids[int(n)] for n in positions]
    nodes = chain.nodes_at(positions)
    loc_rows = np.array([g.local for g in grids], dtype=bool)
    fine = fine_mass = None
    if qs or loc_rows.any():
        # by blocks of positions (:data:`_CHUNK`): the Nyström density takes
        # (P, 5G, K, K) margins and kernels, every position alone
        positions = np.asarray(positions)
        ref_fine = reference_grid(chain.model, _NYSTROM_FACTOR * ref.G)
        P, Mf = len(grids), _NYSTROM_FACTOR * G
        fine = [None] * P
        fine_mass = np.empty((P, Mf))
        dref = np.empty((P, G))
        for r in _row_blocks(P, (Mf + G) * chain.K * chain.K):
            fb = [_fine_grid(ref, ref_fine, g, _NYSTROM_FACTOR) for g in grids[r]]
            pts = np.array([g.nodes for g in fb])
            dens = _nystrom_density(chain, alphas, betas, positions[r],
                                    np.concatenate([pts, np.broadcast_to(ref.nodes, (len(fb), G))],
                                                   axis=1))
            om = np.array([g.omega for g in fb])
            with np.errstate(invalid="ignore"):
                fm = dens[:, :Mf] * om
                Z = fm.sum(axis=1, keepdims=True)
                fine_mass[r] = fm / Z
                dref[r] = dens[:, Mf:] / Z
            # _grid_summary needs the rule and the panels of a local fine grid
            # only, not its node arrays
            fine[r] = [g if g.mix is None else
                       replace(g, nodes=None, ref_pdf=None, omega=None, w=None,
                               mix=replace(g.mix, panel=None)) for g in fb]
    mean, sd, qv = _grid_summary(mass, nodes, qs, fine_mass if qs else None, grids, fine)
    density = mass / ref.omega[None, :]
    if loc_rows.any():
        dl = dref[loc_rows]
        density[loc_rows] = np.where(np.isfinite(dl), dl, np.nan)
    return mean, sd, qv, density


def _state_dists(model: PMCModel) -> list:
    return [_frozen(model.margin(k)) for k in range(model.K)]


def _mixture_summary(model: PMCModel, weights: np.ndarray, qs: tuple[float, ...]):
    """Mean, sd, quantiles of Σ_k weights[m, k] f_k (state margins, exact).

    d = 1: shapes (M,), (M,), (M, Q). d > 1 (multivariate normal margins):
    (M, d), (M, d) and per-component quantiles (M, Q, d).
    """
    K = model.K
    d = getattr(model, "d", 1)
    weights = np.asarray(weights, dtype=float)
    M = weights.shape[0]
    if d == 1:
        dists = _state_dists(model)
        mu = np.array([float(dd.mean()) for dd in dists])
        v = np.array([float(dd.var()) for dd in dists])
        mean = weights @ mu
        sd = np.sqrt(np.maximum(weights @ (v + mu * mu) - mean * mean, 0.0))
        qv = (_mixture_ppf(weights, dists, np.broadcast_to(np.array(qs), (M, len(qs))))
              if qs else np.empty((M, 0)))
        return mean, sd, qv
    mus = np.array([model.margin(k).params["mean"] for k in range(K)], dtype=float)       # (K, d)
    vs = np.array([np.diag(np.asarray(model.margin(k).params["cov"], dtype=float))
                   for k in range(K)])                                                     # (K, d)
    mean = weights @ mus
    sd = np.sqrt(np.maximum(weights @ (vs + mus * mus) - mean * mean, 0.0))
    qv = np.empty((M, len(qs), d))
    for c in range(d):
        dists = [_ss.norm(loc=mus[k, c], scale=np.sqrt(vs[k, c])) for k in range(K)]
        if qs:
            qv[:, :, c] = _mixture_ppf(weights, dists, np.broadcast_to(np.array(qs), (M, len(qs))))
    return mean, sd, qv


def _mixture_density(model: PMCModel, weights: np.ndarray, y: np.ndarray) -> np.ndarray:
    dists = _state_dists(model)
    dens = np.stack([dd.pdf(y) for dd in dists], axis=0)          # (K, G)
    return np.asarray(weights, dtype=float) @ dens


# ---------------------------------------------------------------------------
# Imputation
# ---------------------------------------------------------------------------

@dataclass
class Imputation:
    """Posterior law of the missing observations given all the observed ones.

    Attributes
    ----------
    index           : (M,) positions of the missing rows.
    mean, sd        : (M,) — or (M, d) — posterior mean and standard deviation
                      of y_n given y_obs.
    quantiles       : the requested levels.
    quantile_values : (M, Q) — or (M, Q, d), per component — posterior quantiles.
    gamma           : (M, K) — P(x_n = i | y_obs) at the missing positions.
    log_lik         : float — log p(y_obs).
    Y_mean          : Y with every missing row replaced by its posterior mean.
    method          : ``"exact"`` or ``"grid"``.
    nodes           : (G,) or None — the reference nodes y_g (d = 1).
    density         : (M, G) or None — posterior density of y_n at ``nodes``
                      (the exact mixture density for the shortcut variants;
                      mass / ω on the reference grid, the Nyström density on
                      a local grid, see :func:`_grid_laws`).
    grid_nodes      : (M, G) or None — the quadrature nodes of each missing
                      position (grid variants: reference or local grid).
    grid_mass       : (M, G) or None — the posterior masses on ``grid_nodes``,
                      the discrete law whose moments are ``mean`` and ``sd``.
    x_samples       : (S, N) int or None — FFBS draws of the whole state path.
    y_samples       : (S, M) — or (S, M, d) — or None: the matching draws of the
                      missing observations (exact continuous draws for the
                      shortcut variants; drawn on the quadrature nodes — the
                      discrete law whose moments are ``mean`` and ``sd`` — for
                      the grid variants).
    """

    index: np.ndarray
    mean: np.ndarray
    sd: np.ndarray
    quantiles: tuple
    quantile_values: np.ndarray
    gamma: np.ndarray
    log_lik: float
    Y_mean: np.ndarray
    method: str
    nodes: np.ndarray | None = None
    density: np.ndarray | None = None
    x_samples: np.ndarray | None = None
    y_samples: np.ndarray | None = None
    grid_nodes: np.ndarray | None = None
    grid_mass: np.ndarray | None = None


def impute(
    model: PMCModel,
    Y: np.ndarray,
    *,
    gap_nodes: int | None = DEFAULT_GAP_NODES,
    quantiles=DEFAULT_QUANTILES,
    n_samples: int = 0,
    rng=None,
) -> Imputation:
    """Posterior law of every missing y_n given all the observed data.

    Parameters
    ----------
    model     : PMCModel — known parameters.
    Y         : (N,) or (N, d) with NaN (any non-finite value) at missing rows.
    gap_nodes : quadrature nodes G for the grid variants (default 64).
    quantiles : levels in (0, 1) of the reported posterior quantiles.
    n_samples : number S of joint FFBS draws of (x_{1:N}, y_miss) (default 0).
    rng       : seed or :class:`numpy.random.Generator` for the draws.

    Returns
    -------
    Imputation — see its docstring. A Y without missing rows gives empty
    per-position arrays.
    """
    Y = np.asarray(Y, dtype=float)
    qs = _check_quantiles(quantiles)
    n_samples = int(n_samples)
    if n_samples < 0:
        raise ValueError(f"n_samples must be ≥ 0, got {n_samples}.")
    post, run = _posterior(model, Y, gap_nodes, False)
    idx = post.index
    d = getattr(model, "d", 1)
    nodes = density = grid_nodes = grid_mass = None
    x_s = y_s = None
    if post.method == "grid":
        chain, alphas, betas = run
        if n_samples > 0:
            # the draws first (they need the transitions, the laws below do not)
            gen = np.random.default_rng(rng)
            U = _ffbs(chain.trans, alphas, chain.log, gen, n_samples)
            x_s = np.where(post.miss[None, :], U // chain.G, U).astype(int)
            y_s = _node_draws(chain, U[:, idx], idx)
        chain._forget()          # the transitions are not needed any more (module docstring, "Memory")
        mass = post.node_post.sum(axis=1)                              # (M, G)
        mean, sd, qv, density = _grid_laws(chain, alphas, betas, idx, mass, qs)
        nodes, grid_nodes, grid_mass = post.grid.nodes, post.nodes, mass
    elif idx.size == 0:
        # Complete data (any variant): nothing to impute.
        tail = (d,) if d > 1 else ()
        mean, sd = np.empty((0,) + tail), np.empty((0,) + tail)
        qv = np.empty((0, len(qs)) + tail)
    else:
        w = post.gamma[idx]
        mean, sd, qv = _mixture_summary(model, w, qs)
        if d == 1:
            nodes = reference_grid(model, gap_nodes).nodes
            density = _mixture_density(model, w, nodes)
    Y_mean = np.array(Y, copy=True)
    Y_mean[idx] = mean

    if n_samples > 0 and post.method != "grid":
        gen = np.random.default_rng(rng)
        W = run[0]
        trans = [W[n] for n in range(len(Y) - 1)]
        alphas = list(post.alpha_hat)
        x_s = _ffbs(trans, alphas, False, gen, n_samples)
        y_s = _draw_state_margins(model, x_s[:, idx], gen)
    return Imputation(index=idx, mean=mean, sd=sd, quantiles=qs, quantile_values=qv,
                      gamma=post.gamma[idx], log_lik=post.log_lik, Y_mean=Y_mean,
                      method=post.method, nodes=nodes, density=density,
                      x_samples=x_s, y_samples=y_s, grid_nodes=grid_nodes, grid_mass=grid_mass)


def _draw_state_margins(model: PMCModel, states: np.ndarray, rng) -> np.ndarray:
    """y ~ f_{state} elementwise for integer ``states`` of any shape."""
    d = getattr(model, "d", 1)
    out = np.empty(states.shape + ((d,) if d > 1 else ()))
    for k in range(model.K):
        sel = states == k
        cnt = int(sel.sum())
        if cnt:
            draws = np.asarray(model.margin(k).rvs(cnt, rng), dtype=float)
            out[sel] = draws.reshape((cnt, d) if d > 1 else (cnt,))
    return out


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------

@dataclass
class Forecast:
    """h-step predictive law of (x_{N+k}, y_{N+k}) given the observed part of Y.

    Attributes
    ----------
    h               : horizon.
    state_probs     : (h, K) — P(x_{N+k} = j | y_obs), k = 1..h.
    mean, sd        : (h,) — or (h, d) — predictive mean and standard deviation.
    quantiles       : the requested levels.
    quantile_values : (h, Q) — or (h, Q, d) — predictive quantiles.
    log_lik         : float — log p(y_obs) of the conditioning sequence.
    method          : ``"exact"`` or ``"grid"``.
    nodes           : (G,) or None — the reference nodes y_g (d = 1).
    density         : (h, G) or None — predictive density at ``nodes``.
    grid_nodes      : (h, G) or None — the quadrature nodes of each horizon
                      (grid variants: reference or local grid).
    grid_mass       : (h, G) or None — the predictive masses on ``grid_nodes``.
    """

    h: int
    state_probs: np.ndarray
    mean: np.ndarray
    sd: np.ndarray
    quantiles: tuple
    quantile_values: np.ndarray
    log_lik: float
    method: str
    nodes: np.ndarray | None = None
    density: np.ndarray | None = None
    grid_nodes: np.ndarray | None = None
    grid_mass: np.ndarray | None = None


def forecast(
    model: PMCModel,
    Y: np.ndarray,
    h: int,
    *,
    gap_nodes: int | None = DEFAULT_GAP_NODES,
    quantiles=DEFAULT_QUANTILES,
) -> Forecast:
    """Predictive laws of the next h observations, a trailing gap of length h.

    Runs the forward filter on Y followed by h missing rows: the normalised
    filter at N + k is P(x_{N+k}, y_{N+k} | y_obs) (no backward message is
    needed for a trailing gap). Y may contain missing rows.

    With a non-ignorable ``model.missingness`` the laws are given
    (y_obs, m_{1:N}): the N rows of Y carry their missingness factors, the h
    appended rows none — their mask is unknown, and summing p(m_n | m_{n-1},
    x_n) over it gives 1.

    Parameters
    ----------
    model     : PMCModel — known parameters.
    Y         : (N,) or (N, d) conditioning observations (N ≥ 1).
    h         : horizon ≥ 1.
    gap_nodes : quadrature nodes G for the grid variants (default 64).
                Memory, as for :func:`gap_posterior` over the missing rows of
                Y and the h appended ones: about 0.3 kB per missing row and
                per node (K = 3), plus at most 2 GiB of transitions between
                local grids, 8·(K·G)² bytes per step (module docstring,
                "Memory").
    quantiles : levels in (0, 1) of the predictive quantiles.
    """
    Y = np.asarray(Y, dtype=float)
    qs = _check_quantiles(quantiles)
    h = int(h)
    if h < 1:
        raise ValueError(f"forecast horizon h must be ≥ 1, got {h}.")
    if len(Y) < 1:
        raise ValueError("forecast needs at least one row of Y (it may be NaN).")
    tail = np.full((h,) + Y.shape[1:], np.nan)
    Yx = np.concatenate([Y, tail], axis=0)
    N = len(Y)
    miss = missing_mask(Yx)
    # Missingness factors of the N conditioning rows only: the mask of the h
    # future rows is unknown, and Σ_m p(m_n | m_{n-1}, x_n) = 1.
    ev = _inf._evidence(model, miss[:N])
    if ev is not None:
        ev = np.concatenate([ev, np.ones((h, model.K))], axis=0)
    d = getattr(model, "d", 1)
    nodes = density = grid_nodes = grid_mass = None
    if needs_grid(model):
        _check_grid_supported(model)
        grid = reference_grid(model, gap_nodes)
        chain, alphas, ll, _ = _run_chain(model, Yx, miss, grid, backward=False, ev=ev)
        K, G = chain.K, chain.G
        joint = np.array([alphas[N + k].reshape(K, G) for k in range(h)])   # (h, K, G)
        joint /= joint.sum(axis=(1, 2), keepdims=True)
        state_probs = joint.sum(axis=2)
        mass = joint.sum(axis=1)
        pos = np.arange(N, N + h)
        mean, sd, qv, density = _grid_laws(chain, alphas, None, pos, mass, qs)
        method = "grid"
        nodes, grid_nodes, grid_mass = grid.nodes, chain.nodes_at(pos), mass
    else:
        W, f_pdf = _inf._weights(model, Yx, ev)
        a, ll = _inf._forward(model, Yx, W, f_pdf, ev=ev)
        state_probs = a[N:]
        state_probs = state_probs / state_probs.sum(axis=1, keepdims=True)
        mean, sd, qv = _mixture_summary(model, state_probs, qs)
        method = "exact"
        if d == 1:
            nodes = reference_grid(model, gap_nodes).nodes
            density = _mixture_density(model, state_probs, nodes)
    return Forecast(h=h, state_probs=state_probs, mean=mean, sd=sd, quantiles=qs,
                    quantile_values=qv, log_lik=float(ll), method=method,
                    nodes=nodes, density=density, grid_nodes=grid_nodes, grid_mass=grid_mass)
