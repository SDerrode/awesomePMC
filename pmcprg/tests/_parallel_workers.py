"""Top-level replicate functions for ``test_parallel_resampling.py``.

A worker process of :func:`pmcprg._parallel.map_replicates` unpickles the
replicate function by reference, i.e. imports its module. Keeping these in a
module that imports nothing heavy lets the helper-level tests start a pool in
a fraction of a second; the test module itself imports the copula stack.
Not collected by pytest (no ``test_`` prefix).
"""
from __future__ import annotations

import logging
import os
import warnings

import numpy as np

from pmcprg import _parallel

log = logging.getLogger("pmcprg.tests.parallel_workers")


def square(shared, task):
    """``shared * task``, plus what the worker saw of its environment."""
    return (shared * task, os.getpid(), _parallel.in_worker(),
            os.environ.get("OMP_NUM_THREADS"), os.environ.get("VECLIB_MAXIMUM_THREADS"))


def boom(shared, task):
    if task == 5:
        raise ValueError(f"replicate {task} is broken")
    return task


def warn_some(shared, task):
    if task in (1, 2):
        warnings.warn(f"replicate {task} warns", UserWarning)
    if task in (3, 4):
        warnings.warn("the same warning, twice", RuntimeWarning)
    return task


def log_some(shared, task):
    if task == 2:
        log.warning("replicate %d logs", task)
    log.debug("replicate %d debug", task)
    return task


def nested(shared, task):
    """A replicate that asks for a pool of its own: must run in-process."""
    inner = _parallel.map_replicates(square, range(3), n_tasks=3, n_jobs=2, shared=1)
    return os.getpid(), {pid for _, pid, *_ in inner}


def fake_simulate(model, N, seed):
    """A model-free ``simulate_fn``: ``N`` standard normals."""
    return None, np.random.default_rng(seed).standard_normal(N)


def stat_mean_draw(model, Y, rng):
    """A statistic that consumes the generator (one normal draw)."""
    return {"mean": float(np.mean(Y)), "draw": float(rng.standard_normal())}


def stat_warns(model, Y, rng):
    warnings.warn("statistic warning", UserWarning)
    return {"mean": float(np.mean(Y))}
