"""Process pool for independent resampling replicates (audit FR-14). Private.

Six loops of the package — :func:`pmcprg.diagnostics.parametric_bootstrap`,
:meth:`pmcprg.copulas.FitResult.gof_test` and
:meth:`~pmcprg.copulas.FitResult.bootstrap_ci`, and the Gaussian-surrogate
branches of :func:`~pmcprg.diagnostics.exchangeability_test`,
:func:`~pmcprg.diagnostics.radial_symmetry_test` and
:func:`~pmcprg.diagnostics.rosenblatt_gof_test` — are ``for b in range(B)``
over replicates that do not depend on one another. They share one keyword,
``n_jobs``, resolved by :func:`resolve_n_jobs`, and one driver,
:func:`map_replicates`; this module is both.

Reproducibility is decided by each caller, not here: the driver only runs
``fn(shared, task)`` for every task and returns the results **in task
order**, whatever the number of processes and whatever order they finish
in. Every random number a replicate uses must therefore travel inside its
task (a seed, a resampling index, a ``SeedSequence`` child), drawn by the
caller in replicate order; ``tasks`` may be a lazy iterator, consumed in
order, so drawing on the fly costs no memory.

Conventions, those of the multistart pool of
:func:`pmcprg.pmc._estim_common.run_multistart` where it has them:

* **spawn** start method on every platform — the macOS default, and the only
  one that is safe with threads and Accelerate. ``fn`` must therefore be a
  *top-level* function and ``shared`` picklable (checked up front, with an
  error that says so); a script that asks for a pool must guard its entry
  point with ``if __name__ == "__main__":``.
* **One BLAS/OpenMP thread per worker.** ``OMP_NUM_THREADS``,
  ``OPENBLAS_NUM_THREADS``, ``MKL_NUM_THREADS``, ``VECLIB_MAXIMUM_THREADS``
  and ``NUMEXPR_NUM_THREADS`` are set to 1 in the parent's environment only
  while the workers are being started (a spawned child inherits it, and the
  libraries read it once, at load), then restored.
* **No nested pools.** Inside a worker :func:`in_worker` is true;
  :func:`map_replicates` then runs in-process, and so does the ICE/SEM
  multistart (``multistart_workers`` is ignored there).
* **Exceptions** escaping ``fn`` in a worker are re-raised in the parent —
  the original exception, with the worker traceback chained and a note
  naming the replicate — after the pending chunks are cancelled and the
  workers stopped.
* **Warnings** raised in a worker are recorded there under the parent's own
  warning filters (copied in at start-up, so ``"error"`` still raises inside
  the replicate and ``"ignore"`` still drops) and **re-emitted in the parent**
  with their original category, message, file, line and module, against
  that module's ``__warningregistry__`` — so ``"default"`` shows each one
  once, as the serial loop would, and pytest or a GUI log sees them.
* **Log records** at or above the parent's level are forwarded the same way
  (``logging.getLogger(record.name).handle(record)`` in the parent).
* **Orphans.** A worker whose parent has died exits by itself within a
  second (a watchdog thread polls ``os.getppid()``); a killed parent
  otherwise leaves spawn workers blocked on their queue forever.

Cost: starting the workers (a fresh interpreter importing numpy, scipy and
pmcprg each) takes about a second of wall time, paid on every call. It pays
off only when the serial loop takes several seconds — see the measured
table in the CHANGELOG entry of FR-14.
"""
from __future__ import annotations

import logging
import logging.handlers
import math
import multiprocessing
import numbers
import os
import pickle
import queue
import sys
import threading
import time
import warnings
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from typing import Any

__all__ = ["in_worker", "map_replicates", "resolve_n_jobs"]

logger = logging.getLogger(__name__)

#: Thread-count variables forced to 1 in the workers' environment.
THREAD_ENV_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                   "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")

#: Chunks per worker when ``chunksize`` is not given: enough to balance
#: replicates of uneven cost, few enough that the per-chunk round trip
#: (pickling the tasks and the results) stays negligible.
CHUNKS_PER_WORKER = 4

# Worker-side state, set by _init_worker. Module globals of a spawned child:
# they never exist in the parent.
_IN_WORKER = False
_FN: Callable[[Any, Any], Any] | None = None
_SHARED: Any = None
_LOG_QUEUE: queue.SimpleQueue | None = None


def in_worker() -> bool:
    """``True`` inside a worker process of :func:`map_replicates`."""
    return _IN_WORKER


def _cpu_count() -> int:
    count = getattr(os, "process_cpu_count", os.cpu_count)()   # 3.13+: affinity-aware
    return max(1, int(count or 1))


def resolve_n_jobs(n_jobs) -> int | None:
    """Validate the ``n_jobs`` keyword shared by the six resampling loops.

    ``None`` stays ``None`` (the caller's historical serial path); ``-1``
    becomes the number of CPUs available to this process; a positive int is
    returned unchanged. Anything else raises: ``TypeError`` for a non-integer
    (``True`` and ``2.0`` included), ``ValueError`` for ``0`` or a negative
    other than ``-1``.
    """
    if n_jobs is None:
        return None
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, numbers.Integral):
        raise TypeError(f"n_jobs must be None or an int, got {n_jobs!r}.")
    n = int(n_jobs)
    if n == -1:
        return _cpu_count()
    if n < 1:
        raise ValueError(
            f"n_jobs must be None, a positive int or -1 (all CPUs), got {n}.")
    return n


# ---------------------------------------------------------------------------
# Environment, warning filters, log level: what the workers inherit
# ---------------------------------------------------------------------------

@contextmanager
def _single_threaded_env():
    """Set the thread-count variables to 1 for the processes spawned inside."""
    saved = {k: os.environ.get(k) for k in THREAD_ENV_VARS}
    os.environ.update({k: "1" for k in THREAD_ENV_VARS})
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _portable_filters() -> list[tuple]:
    """The parent's warning filters, as they stand, minus the unpicklable.

    Entries are copied raw — compiled patterns, and the plain strings of the
    interpreter's own exact-match defaults — so the worker matches exactly
    what the parent matches. A filter whose category cannot be pickled is
    left out: the worker then treats that warning as its remaining filters
    say, and it is still re-emitted in the parent, where the full list
    applies.
    """
    out = []
    for entry in list(warnings.filters):
        try:
            pickle.dumps(entry)
        except Exception:                                    # pragma: no cover
            continue
        out.append(entry)
    return out


def _log_threshold() -> int:
    return min(logging.getLogger().getEffectiveLevel(),
               logging.getLogger("pmcprg").getEffectiveLevel())


def _watch_parent(parent_pid: int) -> None:                 # pragma: no cover (child)
    while True:
        time.sleep(1.0)
        if os.getppid() != parent_pid:
            os._exit(1)


def _init_worker(fn, shared, filters, log_level, parent_pid) -> None:  # pragma: no cover (child)
    global _IN_WORKER, _FN, _SHARED, _LOG_QUEUE
    _IN_WORKER = True
    _FN, _SHARED = fn, shared
    os.environ.update({k: "1" for k in THREAD_ENV_VARS})
    # resetwarnings() empties the list in place and invalidates the
    # registries; refilling the same list object is then what the C matcher
    # reads — no private API.
    warnings.resetwarnings()
    warnings.filters.extend(filters)
    _LOG_QUEUE = queue.SimpleQueue()
    root = logging.getLogger()
    root.handlers[:] = [logging.handlers.QueueHandler(_LOG_QUEUE)]
    root.setLevel(log_level)
    threading.Thread(target=_watch_parent, args=(parent_pid,), daemon=True).start()


def _run_chunk(chunk: list[tuple[int, Any]]):               # pragma: no cover (child)
    """Run one chunk in a worker: ``(results, warnings, log records)``."""
    results = []
    with warnings.catch_warnings(record=True) as caught:
        for index, task in chunk:
            try:
                results.append(_FN(_SHARED, task))
            except BaseException as exc:
                exc.add_note(f"pmcprg: raised by replicate {index}, in worker "
                             f"process {os.getpid()} (n_jobs > 1).")
                raise
    records = [(w.category, str(w.message), w.filename, w.lineno) for w in caught]
    logs = []
    while _LOG_QUEUE is not None and not _LOG_QUEUE.empty():
        logs.append(_LOG_QUEUE.get_nowait())
    return results, records, logs


# ---------------------------------------------------------------------------
# Parent side: re-emission
# ---------------------------------------------------------------------------

def _module_for(filename: str, cache: dict) -> str | None:
    """The dotted name of the module a warning came from, for its filters.

    ``warnings.warn`` matches a filter's module regex against the caller's
    ``__name__``, which a recorded warning does not carry: it is recovered
    from the file name through ``sys.modules`` (``None`` when not found —
    ``warn_explicit`` then falls back to the file name, as it always does).
    """
    key = os.path.abspath(filename)
    if key not in cache:                  # (re)scan: new modules may have loaded
        for name, mod in list(sys.modules.items()):
            path = getattr(mod, "__file__", None)
            if path:
                cache.setdefault(os.path.abspath(path), name)
        cache.setdefault(key, None)       # a miss is not rescanned
    return cache[key]


def _reemit(records, logs, module_cache: dict) -> None:
    for category, message, filename, lineno in records:
        module = _module_for(filename, module_cache)
        registry = None
        if module is not None and module in sys.modules:
            registry = vars(sys.modules[module]).setdefault("__warningregistry__", {})
        warnings.warn_explicit(message, category, filename, lineno,
                               module=module, registry=registry)
    for record in logs:
        target = logging.getLogger(record.name)
        if target.isEnabledFor(record.levelno):
            target.handle(record)


def _check_picklable(fn, shared) -> None:
    for what, obj in (("the replicate function", fn), ("the shared payload", shared)):
        try:
            pickle.dumps(obj)
        except Exception as exc:
            raise TypeError(
                f"n_jobs > 1 sends {what} to worker processes, and it cannot be "
                f"pickled ({type(exc).__name__}: {exc}). Every callable involved "
                "must be a top-level function of an importable module — not a "
                "lambda, a closure or a function defined in a notebook — or run "
                "with n_jobs=None / 1."
            ) from exc


def _stop_workers(pool: ProcessPoolExecutor) -> None:
    """Cancel what is pending and stop the running workers now.

    ``shutdown(wait=False)`` alone would leave the running chunks computing
    to the end; the processes are terminated and reaped instead. They are
    read from the executor's private ``_processes`` *before* ``shutdown``,
    which clears it (what 3.14's ``terminate_workers`` does too).
    """
    procs = list((getattr(pool, "_processes", None) or {}).values())
    pool.shutdown(wait=False, cancel_futures=True)
    for proc in procs:
        try:
            proc.terminate()
        except Exception:                                    # pragma: no cover
            pass
    for proc in procs:
        proc.join(timeout=5.0)


# ---------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------

def map_replicates(
    fn: Callable[[Any, Any], Any],
    tasks: Iterable,
    *,
    n_tasks: int,
    n_jobs: int,
    shared: Any = None,
    chunksize: int | None = None,
    on_progress: Callable[[int], None] | None = None,
) -> list:
    """``[fn(shared, task) for task in tasks]``, over ``n_jobs`` processes.

    Parameters
    ----------
    fn : callable ``(shared, task) -> result``
        One replicate. Must be top-level (picklable by reference) when
        ``n_jobs > 1``.
    tasks : iterable
        One item per replicate, consumed **lazily and in order** — draw the
        replicate's random numbers here, in the parent, and the sequence is
        the serial one whatever ``n_jobs`` is. At most ``2 · workers`` chunks
        are in flight, so a lazy iterator of large tasks (resampling indices)
        never sits in memory whole.
    n_tasks : int
        ``len(tasks)``, needed up front to size the chunks.
    n_jobs : int
        A resolved worker count (see :func:`resolve_n_jobs`). ``1`` — and any
        value inside a worker, or with a single task — runs in-process.
    shared : picklable, optional
        What every replicate needs and that does not vary (a model, a copula,
        the data); sent once per worker, not once per task.
    chunksize : int, optional
        Replicates per round trip. Default: ``n_tasks`` split into
        ``CHUNKS_PER_WORKER`` chunks per worker.
    on_progress : callable ``(n_done) -> None``, optional
        Called in the calling thread after each replicate in-process, after
        each chunk with a pool.

    Returns
    -------
    list — the results in task order.
    """
    n_tasks = int(n_tasks)
    if n_jobs <= 1 or n_tasks <= 1 or _IN_WORKER:
        out = []
        for task in tasks:
            out.append(fn(shared, task))
            if on_progress is not None:
                on_progress(len(out))
        return out

    if chunksize is None:
        chunksize = math.ceil(n_tasks / (CHUNKS_PER_WORKER * n_jobs))
    chunksize = max(1, int(chunksize))
    n_chunks = math.ceil(n_tasks / chunksize)
    workers = min(int(n_jobs), n_chunks)
    _check_picklable(fn, shared)

    task_iter = iter(enumerate(tasks))

    def _next_chunk():
        chunk = []
        for item in task_iter:
            chunk.append(item)
            if len(chunk) == chunksize:
                break
        return chunk

    results: list = [None] * n_tasks
    done = 0
    module_cache: dict = {}
    ctx = multiprocessing.get_context("spawn")
    with _single_threaded_env():
        pool = ProcessPoolExecutor(
            max_workers=workers, mp_context=ctx, initializer=_init_worker,
            initargs=(fn, shared, _portable_filters(), _log_threshold(), os.getpid()),
        )
        pending = {}
        try:
            # Workers are spawned on submit, one per submit while none is
            # idle: the first ``workers`` submissions start them all, inside
            # the single-threaded environment.
            for _ in range(2 * workers):
                chunk = _next_chunk()
                if not chunk:
                    break
                pending[pool.submit(_run_chunk, chunk)] = chunk[0][0]
        except BaseException:
            _stop_workers(pool)
            raise
    # The serial loop stops at the lowest failing replicate, whatever the
    # timing: after a failure, only the chunks BEFORE it are awaited (their
    # warnings re-emitted, a failure among them taking over), those after it
    # are dropped, then the earliest failure is raised.
    failed = None                                   # (first index, future)
    try:
        while pending:
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in sorted(finished, key=pending.__getitem__):
                first = pending.pop(fut)
                if failed is not None and first > failed[0]:
                    continue
                if fut.exception() is not None:
                    failed = (first, fut)
                    for later in [f for f, i in pending.items() if i > first]:
                        later.cancel()
                        del pending[later]
                    continue
                chunk_results, records, logs = fut.result()
                results[first:first + len(chunk_results)] = chunk_results
                done += len(chunk_results)
                _reemit(records, logs, module_cache)
                if on_progress is not None:
                    on_progress(done)
                if failed is None:
                    chunk = _next_chunk()
                    if chunk:
                        pending[pool.submit(_run_chunk, chunk)] = chunk[0][0]
        if failed is not None:
            failed[1].result()          # the worker's exception, with its note
    except BaseException:
        _stop_workers(pool)
        raise
    pool.shutdown(wait=True)
    if done != n_tasks:                                      # pragma: no cover
        raise RuntimeError(f"map_replicates: {done} results for {n_tasks} tasks "
                           "(the task iterator and n_tasks disagree).")
    return results
