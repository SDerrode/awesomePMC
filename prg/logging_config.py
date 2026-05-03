"""Logging configuration helper for the copulasformm package.

Call ``configure_logging()`` once at application start-up to route the
package's log records to the console (or any handler you provide).
"""
import logging

_DEFAULT_FORMAT = '%(levelname)s %(name)s: %(message)s'


def configure_logging(
    level: int = logging.WARNING,
    fmt: str = _DEFAULT_FORMAT,
    handler: logging.Handler | None = None,
) -> None:
    """Attach a StreamHandler to the ``prg`` logger.

    Parameters
    ----------
    level:
        Minimum log level (default: WARNING).
    fmt:
        Log record format string.
    handler:
        Custom handler; a StreamHandler writing to stderr is used when None.
    """
    pkg_logger = logging.getLogger('prg')
    if not pkg_logger.handlers:
        h = handler or logging.StreamHandler()
        h.setFormatter(logging.Formatter(fmt))
        pkg_logger.addHandler(h)
    pkg_logger.setLevel(level)
