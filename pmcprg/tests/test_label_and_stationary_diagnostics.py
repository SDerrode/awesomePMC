"""Diagnostics that used to mislead: label-value warnings in ``error_rate`` and the
non-convergence message of the stationary distribution.

* ``error_rate`` is invariant under relabeling, so a reference coded {0, 255}
  (a binary mask image) against predictions {0, 1} is not a problem; only the
  number of classes is worth a warning.
* Power iteration stalls on periodic or very slowly mixing chains; the
  stationary law is then solved directly, and the old message reported
  ‖Δ‖∞ = 0 because it compared the last iterate with itself.
"""

from __future__ import annotations

import logging

import numpy as np

from pmcprg.pmc.inference import error_rate
from pmcprg.pmc.model import _stationary_distribution


def test_error_rate_ignores_label_values(caplog):
    rng = np.random.default_rng(0)
    truth = rng.integers(0, 2, size=500)
    ref = np.where(truth == 1, 255, 0)
    pred = truth.copy()
    pred[:25] = 1 - pred[:25]
    with caplog.at_level(logging.WARNING, logger="pmcprg.pmc.inference"):
        assert abs(error_rate(ref, pred) - 0.05) < 1e-12
    assert not [r for r in caplog.records if "error_rate" in r.getMessage()]


def test_error_rate_warns_on_class_counts(caplog):
    truth = np.array([0, 0, 1, 1, 2, 2])
    with caplog.at_level(logging.WARNING, logger="pmcprg.pmc.inference"):
        error_rate(truth, np.array([0, 0, 1, 1, 1, 1]))
        error_rate(np.array([0, 0, 1, 1, 1, 1]), truth)
    messages = [r.getMessage() for r in caplog.records]
    assert any("uses 2 classes, the reference 3" in m for m in messages)
    assert any("uses 3 classes, the reference only 2" in m for m in messages)


def test_stationary_distribution_of_a_periodic_chain(caplog):
    # Period 3: from the uniform start power iteration is stationary at once, but
    # from any other start it would cycle; the function must return the uniform law.
    A = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    with caplog.at_level(logging.WARNING, logger="pmcprg.pmc.model"):
        pi = _stationary_distribution(A)
    np.testing.assert_allclose(pi, [1 / 3, 1 / 3, 1 / 3], atol=1e-12)
    assert not caplog.records


def test_stationary_distribution_of_a_slowly_mixing_chain():
    eps = 1e-6
    A = np.array([[1 - eps, eps], [3 * eps, 1 - 3 * eps]])
    pi = _stationary_distribution(A)
    np.testing.assert_allclose(pi, [0.75, 0.25], rtol=1e-9)
    np.testing.assert_allclose(pi @ A, pi, atol=1e-15)


def test_stationary_distribution_is_unchanged_when_power_iteration_converges():
    A = np.array([[0.9, 0.1], [0.2, 0.8]])
    pi = np.full(2, 0.5)
    for _ in range(1000):
        new = pi @ A
        new = np.maximum(new, 0.0)
        new /= new.sum()
        if np.max(np.abs(new - pi)) < 1e-12:
            break
        pi = new
    assert np.array_equal(_stationary_distribution(A), new)
