"""Custom exception hierarchy for the awesomePMC package."""


class CopulaError(Exception):
    """Base class for all copula-related errors."""


class CopulaParameterError(CopulaError, ValueError):
    """Invalid copula parameter (tau_k out of range, unknown parameter name)."""


class CopulaNotAvailableError(CopulaError):
    """Requested copula is not yet available."""


class SamplingConvergenceError(CopulaError, RuntimeError):
    """Acceptance-rejection sampler did not converge within the iteration budget."""


# ---------------------------------------------------------------------------
# PMC errors
# ---------------------------------------------------------------------------

class PMCError(Exception):
    """Base class for pmcprg.pmc-specific errors."""


class IncompatibleObservationError(PMCError, ValueError):
    """The observation sequence has zero likelihood under the model.

    Raised at step 1 of the forward pass when the marginal density of the
    first observation is zero for every state (e.g. the data lies completely
    outside the support of every margin). The model and/or data must be
    fixed before inference can proceed.
    """


class CSVLoadError(PMCError, IOError):
    """Raised by the CLI when an observation CSV is missing, empty, or
    lacks a required column. The CLI catches this and exits 1 with a
    user-friendly message; library callers can catch it directly.
    """
