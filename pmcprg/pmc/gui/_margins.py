"""pmcprg.pmc.gui._margins — margin helpers that respect the margin structure.

A :class:`~pmcprg.pmc.model.PMCModel` has state margins f_i
(``margin_structure == "state"``) or pair margins f_ij (``"pair"``, the general
PMC of DerrodePieczynski_CSDA2013 Eqs. 12–14). ``model.margin(i)`` raises on a
pair model, so every panel of the GUI that speaks of "the margin of state i"
goes through :func:`state_law` — f_i on a state model, and on a pair model the
law of y_n given x_n = i alone, g_i = Σ_j (p_ij / p_i) f_ij.

The implementation lives in :mod:`pmcprg.diagnostics.pseudo`, shared with the
non-GUI diagnostics; this module keeps the names the GUI imports and the GUI's
``lo``/``hi`` form of :func:`margin_cdfs`.

Reference: Derrode, S. & Pieczynski, W. (2013). Unsupervised data
classification using pairwise Markov chains with automatic copulas selection.
*Computational Statistics & Data Analysis* 63, 81–98 ("DerrodePieczynski_CSDA2013").
"""

from __future__ import annotations

import numpy as np

from pmcprg.diagnostics import pseudo as _pseudo
from pmcprg.diagnostics.pseudo import (
    StateMixture,
    is_pair,
    mixture_label,
    state_law,
    state_law_name,
    state_weights,
)

__all__ = [
    "StateMixture",
    "is_pair",
    "margin_cdfs",
    "mixture_label",
    "state_law",
    "state_law_name",
    "state_weights",
]


def margin_cdfs(mdl, Y, *, lo: float, hi: float) -> np.ndarray:
    """``F[n, i, j] = F_ij(Y[n])`` clipped to ``[lo, hi]``, shape ``(N, K, K)``.

    :func:`pmcprg.diagnostics.pseudo.margin_cdfs` with ``clip=(lo, hi)`` and
    ``broadcast=True``: a state model computes K CDF vectors and returns them
    broadcast over j (read-only view), so its values are those of
    ``margin(i).cdf_vec`` exactly. The pseudo-observations of the pair (i, j)
    are then ``u = F[:-1, i, j]`` and ``v = F[1:, j, i]`` — the left margin
    f_ij and the right margin f_ji of DerrodePieczynski_CSDA2013 Eq. 12.
    """
    return _pseudo.margin_cdfs(mdl, Y, clip=(lo, hi), broadcast=True)
