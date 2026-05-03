"""Custom exception hierarchy for the copulasformm package."""


class CopulaError(Exception):
    """Base class for all copula-related errors."""


class CopulaParameterError(CopulaError, ValueError):
    """Invalid copula parameter (tau_k out of range, unknown parameter name)."""


class CopulaNotAvailableError(CopulaError):
    """Requested copula is not yet available."""


class SamplingConvergenceError(CopulaError, RuntimeError):
    """Acceptance-rejection sampler did not converge within the iteration budget."""
