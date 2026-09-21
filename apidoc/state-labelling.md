# State labelling

A PMC/HMC's hidden states are indices `0, …, K-1`: the package attaches no
semantic meaning to a state index (there is no "low" or "high" state) and
does not sort states by any statistic of their fitted parameters (mean, τ,
transition probability, …). This page documents where the index of a state
comes from, whether it is preserved by estimation, and what to expect when
comparing states across runs or against known labels — the identifiability
question raised for `dcvar`-style models, whose ordered-ρ constraint pins
down an otherwise arbitrary state numbering (audit `AUDIT_COPULES.md`,
FR-12: "règle d'étiquetage des états documentée").

## Where the index comes from

A state's index is fixed by the model's TOML file, not derived from data:

- The prior (`[prior]` block: `A` for the HMC-\* variants, `p` for the
  PMC-\* variants) is a `K×K` matrix whose row/column `k` **is** state `k`.
- A `"state"`-structured model (`PMCModel.margin_structure == "state"`)
  declares one `[[margins]]` block per state, keyed by its state index `i`
  (`pmcprg/pmc/model.py`, the K-format parser around line 837). A
  `"pair"`-structured model (the general PMC,
  DerrodePieczynski_CSDA2013 Eqs. 12-14) declares `K²` blocks keyed by
  `(i, j)`, and its copulas by the same pair.

Nothing in `PMCModel` reorders these blocks: state `k` in the fitted model
is state `k` of whichever TOML file — the user's initial guess, or a
warm-start model built from it (below) — the estimator started from.

## `ice()` / `sem()` do not relabel

`ice()` and `sem()` (`pmcprg/pmc/ice.py`, `pmcprg/pmc/sem.py`) run
Expectation-Conditional-Maximisation-style iterations from a starting
`PMCModel`. Each M-step updates state `k`'s prior row, margin and copula
parameters from posteriors computed against state `k`'s *previous*-iteration
parameters, so the identity of a state does not move across iterations by
construction: **the numbering of the fitted model is the numbering of the
model the estimation was started from.** This is a routine EM/SEM property,
not a labelling algorithm — nothing enforces it against every possible
local optimum, and the initial copula families can still decide which
*mode* the fit converges to (README, "Multistart over copula families": on
the CSDA-2013 Exp. 3 design, ICE started at independence picks Gumbel
instead of the true Gaussian copula on pair `(0, 0)`, regardless of
jitter).

## `init = "kmeans"`: aligning an unlabelled clustering to the declared states

The alternative initialisation clusters the observations with k-means++
(`_kmeans_label_assignment`, `pmcprg/pmc/ice.py:3262`) *before* any state
identity exists — k-means assigns cluster ids `0, …, K-1` in an order that
has nothing to do with the declared model. Renumbering those clusters to
match the user's declared states is the actual "labelling rule" in this
codebase:

`_align_kmeans_labels` (`pmcprg/pmc/ice.py:3290`) renumbers cluster `c` to
the state `k` that maximises the total log-likelihood of the cluster's
points under state `k`'s **declared** law:

```
maximise  Σ_c Σ_{n : cluster(n) = c}  log g_{π(c)}(y_n)   over bijections π
```

where `g_k` is state `k`'s margin density `f_k` (`"state"`-structured
models), or the posterior-weighted mixture `Σ_j (p_kj / p_k) f_kj` of the
pair margins that involve state `k` (`"pair"`-structured models,
DerrodePieczynski_CSDA2013 Eq. 12) — `pmcprg/pmc/ice.py:3315-3325`. The
optimal bijection is found exactly with the Hungarian algorithm
(`scipy.optimize.linear_sum_assignment`, `pmcprg/pmc/ice.py:3332`) on the
`K×K` cost matrix of negative summed log-likelihoods.

Consequences, each covered by a test in
`pmcprg/tests/test_ice_kmeans_alignment.py`:

- If the declared states have **distinct** margins (e.g. `fit_margins =
  false` with informative initial values), the alignment recovers the
  user's intended numbering regardless of how k-means happened to number
  its clusters — including a full label permutation or a cyclic shift
  (`test_permuted_true_labels_are_renumbered_back_k2`,
  `test_cyclic_permutation_is_undone_k3`). Measured on the CSDA 2013 Exp. 3
  design (Gaussian margins, 10 seeds): without this step the true Clayton
  copula of state pair `(1, 1)` was recovered from the k-means start in
  5 runs out of 10 (a coin flip on which physical state got which index);
  with it, 10 out of 10.
- If every declared margin is **identical** across states, every bijection
  scores equally and the assignment falls back to the identity permutation
  — the numbering is then genuinely arbitrary, and the warm-started fit
  does not depend on it (`test_identical_declared_margins_keep_the_numbering`,
  `test_warm_start_does_not_depend_on_cluster_numbering`).
- An empty k-means cluster is tolerated: the corresponding row of the cost
  matrix is all zero, so that state can be assigned any remaining index
  without affecting the objective (`test_empty_cluster_is_tolerated`).

Once aligned, the k-means clustering is used as a hard-labelled warm start
(`_warmstart_from_kmeans`, `pmcprg/pmc/ice.py:3343`) for the same estimator
loop described above — the fitted model's state indices are therefore, once
again, whichever indices the (now-aligned) starting model used.

## Comparing states against known labels: `error_rate` does not need any of this

When ground-truth labels are available (simulation studies, benchmarks),
`error_rate(X_true, X_hat)` (`pmcprg/pmc/inference.py:1162`) reports the
classification error **invariant to any labelling of `X_hat`**: it builds
the confusion matrix and finds the relabelling that maximises agreement,
again via the Hungarian algorithm. It does not touch the fitted model — it
is a metric, not a labelling rule — so it is the right tool to evaluate a
fit's classification quality without relying on the state-alignment
behaviour above.

## What a user should expect

- With the default `init = "model"`, a state keeps the index it had in the
  TOML file the estimation started from — pick informative initial margins
  (even approximate ones) if you need the fitted state `k` to mean a
  specific thing.
- With `init = "kmeans"`, the initial clustering is realigned to the
  declared states before the M-step runs, so the same "state `k` means
  what the TOML said" expectation holds — provided the declared margins
  differ enough across states to break ties (see above).
- The package does **not** implement a canonical, data-driven ordering
  (such as sorting states by a fitted mean, or constraining an ordered ρ as
  some published models do) at any point after fitting. If your downstream
  use needs a canonical order — for aggregating results across many
  Monte-Carlo replicates, for instance — sort or relabel the fitted
  `PMCModel` yourself (e.g. by a margin statistic, or with the same
  Hungarian-assignment idea as `_align_kmeans_labels`/`error_rate` against
  a reference), or compare classifications with `error_rate`, which
  sidesteps the question entirely.
