"""
pmm.py — Pairwise Mixture Model (PMM): i.i.d. (X, Y) pair sampling and MPM classification.

The PMM is the *non-Markovian* counterpart of PMC introduced in Derrode-
Pieczynski (2011) and used as a baseline in CSDA 2013, §3.3. Pairs
(y_1, y_2), (y_3, y_4), … are independent and identically distributed.
Each pair is drawn from the same bivariate joint as in the PMC:

  p(x_1, x_2, y_1, y_2) = p(x_1, x_2) · f_{x_1, x_2}(y_1, y_2)

where (DerrodePieczynski_CSDA2013 Eq. 12)

  f_{i,j}(y_1, y_2) = f_ij(y_1) · f_ji(y_2) · c_ij(F_ij(y_1), F_ji(y_2)).

With pair margins (``model.margin_structure == "pair"``, the general PMC of
DerrodePieczynski_CSDA2013 Eqs. 12–14) the left observation of the pair
``(i, j)`` follows ``f_ij`` and the right one ``f_ji``. With state margins
both reduce to the per-state densities, ``f_i(y_1) · f_j(y_2) ·
c_ij(F_i(y_1), F_j(y_2))``, and the code path is the one of the SR-PMC: same
random stream, same floats.

This module deliberately reuses :class:`PMCModel` for parameter storage
and density access — only the temporal structure differs.

Public API
----------
simulate_pmm(model, n_pairs, seed=None) -> (X, Y)
    Generate an i.i.d. PMM sample of length 2·n_pairs. Returns ``X`` and
    ``Y`` as flat arrays of length 2·n_pairs (pair k spans indices
    ``[2k, 2k+1]``).

classify_pmm(model, Y) -> (X_hat, log_lik)
    Standalone MPM classification per pair. ``Y`` must have an even
    length; pair k = (Y[2k], Y[2k+1]) is classified independently of the
    others. Returns ``X_hat`` of the same shape as ``Y``.

References
----------
Derrode S., Pieczynski W., *Unsupervised restoration in Gaussian
pairwise mixture model*, EUSIPCO 2011.

Derrode S., Pieczynski W., *Unsupervised data classification using
pairwise Markov chains with automatic copulas selection*, CSDA 2013,
§3.3.
"""

import logging

import numpy as np

from pmcprg.numerics  import EPS, ONE_MINUS_EPS, MIN_POSITIVE
from pmcprg.pmc.model import PMCModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate_pmm(
    model:   PMCModel,
    n_pairs: int,
    seed:    int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate an i.i.d. PMM sample.

    Each pair ``k`` is drawn as follows:

    1. Sample ``(x_1, x_2)`` from the joint prior ``p(i, j)``.
    2. Sample ``y_1`` from the left margin ``f_ij`` (``f_i`` for state
       margins).
    3. Sample ``y_2`` from the conditional
       ``p(y_2 | y_1, x_1=i, x_2=j)`` using the copula h-function
       (Rosenblatt inversion — Rosenblatt 1952, *Ann. Math.
       Statist.* 23(3), 470–472; Nelsen 2006, *An Introduction to
       Copulas*, 2nd ed., ch. 2 (random variate generation);
       DerrodePieczynski_CSDA2013 Appendix B uses rejection sampling
       for the same conditional — same law):

           u = F_ij(y_1)
           w ~ U(0, 1)
           v = h_{ij}^{-1}(w | u)
           y_2 = F_ji^{-1}(v)

    For variants without a copula (PMC-IN, HMC-IN2), ``y_2`` is sampled
    independently from the right margin ``f_ji``.
    (DerrodePieczynski_CSDA2013 Eq. 12; with state margins ``F_ij = F_i``
    and ``F_ji = F_j``.)

    Parameters
    ----------
    model   : :class:`PMCModel` — only its parameters are used; the
              temporal Markov structure is bypassed.
    n_pairs : int — number of i.i.d. pairs to generate.
    seed    : int, optional.

    Returns
    -------
    X : ``np.ndarray`` of shape ``(2 * n_pairs,)``, dtype int.
    Y : ``np.ndarray`` of shape ``(2 * n_pairs,)``, dtype float.
    """
    rng = np.random.default_rng(seed)
    K   = model.K
    p   = model.prior_p              # (K, K) joint, sums to 1
    p_flat = p.ravel()
    if not np.isclose(p_flat.sum(), 1.0):
        # Defensive: should always hold by construction but the fallback
        # keeps the rng stable and the tests deterministic.
        p_flat = p_flat / p_flat.sum()

    # Draw all pair labels at once. ``np.random.choice`` with a 1-D pmf
    # returns indices in [0, K²); we then unravel to (i, j).
    pair_idx = rng.choice(K * K, size=n_pairs, p=p_flat)
    i_arr    = pair_idx // K
    j_arr    = pair_idx %  K

    X = np.empty(2 * n_pairs, dtype=int)
    Y = np.empty(2 * n_pairs, dtype=float)
    X[0::2] = i_arr
    X[1::2] = j_arr

    uses_copula = model.variant.uses_copula
    for k in range(n_pairs):
        i = int(i_arr[k])
        j = int(j_arr[k])
        # Left margin f_ij and right margin f_ji of the pair
        # (DerrodePieczynski_CSDA2013 Eq. 12). For state margins
        # ``margin(i, j) is margin(i)``: same objects, same random stream
        # as before pair margins existed.
        f_left  = model.margin(i, j)
        f_right = model.margin(j, i)
        # y_1 ~ f_ij
        y1 = float(f_left.rvs(1, rng)[0])
        # y_2 conditional on y_1 (Rosenblatt for copula variants;
        # i.i.d. otherwise).
        if uses_copula:
            cop = model.copula(i, j)
            u   = float(np.clip(f_left.cdf(y1), EPS, ONE_MINUS_EPS))
            w   = float(rng.uniform(EPS, ONE_MINUS_EPS))
            v   = cop.inv_h(w, u)
            y2  = float(f_right.ppf(v))
        else:
            y2 = float(f_right.rvs(1, rng)[0])
        Y[2 * k]     = y1
        Y[2 * k + 1] = y2

    logger.debug(
        "simulate_pmm: variant=%s K=%d n_pairs=%d (N=%d)  seed=%s",
        model.variant.value, K, n_pairs, 2 * n_pairs, seed,
    )
    return X, Y


# ---------------------------------------------------------------------------
# MPM classification — standalone per pair
# ---------------------------------------------------------------------------

def classify_pmm(
    model: PMCModel,
    Y:     np.ndarray,
) -> tuple[np.ndarray, float]:
    """MPM classification of an i.i.d. PMM sample.

    For each pair ``k = (Y[2k], Y[2k+1])`` we compute the K×K joint
    posterior

        P(x_1=i, x_2=j | y_1, y_2)  ∝  p(i,j) · f_{ij}(y_1, y_2)

    with ``f_{ij}(y_1, y_2) = f_ij(y_1) · f_ji(y_2) · c_ij(F_ij(y_1),
    F_ji(y_2))`` (DerrodePieczynski_CSDA2013 Eq. 12; ``f_i(y_1) · f_j(y_2) ·
    c_ij(F_i(y_1), F_j(y_2))`` for state margins)

    and report the marginal MPM estimates::

        x̂_1 = arg max_i  Σ_j  P(i, j | y_1, y_2)
        x̂_2 = arg max_j  Σ_i  P(i, j | y_1, y_2)

    The log-likelihood ``log p(Y)`` is the sum over pairs of
    ``log Σ_{i,j} p(i,j) · f_{ij}(y_1, y_2)``.

    Parameters
    ----------
    model : :class:`PMCModel`.
    Y     : ``np.ndarray`` shape ``(2 * n_pairs,)``.

    Returns
    -------
    X_hat   : ``np.ndarray`` shape ``(2 * n_pairs,)``, dtype int.
    log_lik : float.
    """
    Y = np.asarray(Y, dtype=float)
    if Y.size % 2 != 0:
        raise ValueError(
            f"PMM classify expects an even-length sequence; got {Y.size}."
        )
    n_pairs = Y.size // 2
    K = model.K
    p = model.prior_p
    uses_copula = model.variant.uses_copula

    # Pre-compute f_pdf[n, i, j] = f_ij(Y[n]) and F_ij(Y[n]) on the full
    # sequence (cheap). State margins evaluate K densities and broadcast them
    # over j, pair margins evaluate the K² of them (DerrodePieczynski_CSDA2013 Eq. 12).
    pair = getattr(model, "margin_structure", "state") == "pair"
    f_pdf = np.zeros((Y.size, K, K))
    f_cdf = np.zeros((Y.size, K, K))
    for s in range(K):
        if pair:
            for t in range(K):
                f_pdf[:, s, t] = model.margin(s, t).pdf_vec(Y)
                if uses_copula:
                    f_cdf[:, s, t] = model.margin(s, t).cdf_vec(Y)
        else:
            f_pdf[:, s, :] = model.margin(s).pdf_vec(Y)[:, None]
            if uses_copula:
                f_cdf[:, s, :] = model.margin(s).cdf_vec(Y)[:, None]
    np.clip(f_cdf, EPS, ONE_MINUS_EPS, out=f_cdf)

    X_hat = np.empty(Y.size, dtype=int)
    log_lik = 0.0
    for k in range(n_pairs):
        a, b = 2 * k, 2 * k + 1
        # y_1 = Y[a], y_2 = Y[b] — values used implicitly via f_pdf/f_cdf.
        # Build the K×K joint pair density matrix (DerrodePieczynski_CSDA2013 Eq. 12):
        # f_{ij}(y_1, y_2) = f_ij(y_1) · f_ji(y_2) [· c_ij(F_ij(y_1), F_ji(y_2))]
        # — the right observation's margin is f_ji, hence the transpose.
        joint = (p
                 * f_pdf[a]
                 * f_pdf[b].T)
        if uses_copula:
            for i in range(K):
                for j in range(K):
                    cop = model.copula(i, j)
                    joint[i, j] *= cop.pdf([f_cdf[a, i, j], f_cdf[b, j, i]])
        s = float(joint.sum())
        if not np.isfinite(s) or s <= 0:
            # Degenerate density; default to uniform posterior.
            X_hat[a] = 0
            X_hat[b] = 0
            log_lik += np.log(MIN_POSITIVE)
            continue
        log_lik += float(np.log(s))
        post  = joint / s
        # Marginal MPM
        post_i = post.sum(axis=1)        # P(x_1 = i | y_1, y_2)
        post_j = post.sum(axis=0)        # P(x_2 = j | y_1, y_2)
        X_hat[a] = int(np.argmax(post_i))
        X_hat[b] = int(np.argmax(post_j))

    return X_hat, log_lik


# ---------------------------------------------------------------------------
# Quick smoke-test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

    from pmcprg.pmc.inference import error_rate

    models_dir = pathlib.Path(__file__).parent / "models"
    for name in ["pmc_gauss_k2.toml", "pmc_pair_gauss_k2.toml",
                 "hmc_in_gauss_k2.toml"]:
        f = models_dir / name
        if not f.exists():
            continue
        print(f"\n--- PMM on {name} ---")
        mdl = PMCModel(f)
        X_ref, Y = simulate_pmm(mdl, n_pairs=1000, seed=0)
        X_hat, ll = classify_pmm(mdl, Y)
        er = error_rate(X_ref, X_hat)
        print(f"  variant={mdl.variant.value}  N={Y.size}  log-lik={ll:.2f}  "
              f"error_rate={er:.4f} ({er*100:.1f} %)")
