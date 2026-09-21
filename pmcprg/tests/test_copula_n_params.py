"""Every registered family counts the free parameters it fits (FR-12).

``n_params`` sets the AIC/BIC penalty of ``fit_best`` and of ICE's family
selection. BB1's four rotations (``BB190``, ``BB1270``, ``SBB190``,
``SBB1270``) inherited ``CopulaVirt``'s 1 while fitting (τ, δ), so they were
under-penalised by one parameter; a rotation now takes its base's count.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from pmcprg.copulas import CopulaEnum

FAMILIES = [e for e in CopulaEnum if e.value.AVAILABLE and e.value.MODULE]


def _cls(entry):
    return getattr(importlib.import_module(entry.value.MODULE), entry.value.CLASS_NAME)


@pytest.mark.parametrize("entry", FAMILIES, ids=lambda e: e.name)
def test_n_params_matches_the_registry(entry):
    expected = 0 if entry.name == "PRODUCT" else len(entry.value.PARAMETERS_SET_NAME)
    assert _cls(entry).n_params == expected


@pytest.mark.parametrize("name", ["BB190", "BB1270", "SURVIVAL_BB190", "SURVIVAL_BB1270"])
def test_bb1_rotations_are_penalised_for_two_parameters(name):
    cls = _cls(CopulaEnum[name])
    data = cls(tau_k=-0.4, delta=1.4).sample(300, seed=7)
    fit = cls.fit(data, method="mle")
    assert fit.n_params == 2
    assert fit.aic == pytest.approx(4.0 - 2.0 * fit.log_likelihood, rel=0, abs=1e-12)
    assert fit.bic == pytest.approx(2.0 * np.log(fit.n_obs) - 2.0 * fit.log_likelihood,
                                    rel=0, abs=1e-12)
