"""Logging configuration with TRACE level support."""

import logging

# Define TRACE log level (below DEBUG)
TRACE = 5
logging.addLevelName(TRACE, "TRACE")


def _trace(self, message, *args, **kwargs):
    """Log at TRACE level."""
    if self.isEnabledFor(TRACE):
        self._log(TRACE, message, args, **kwargs)


# Add trace method to Logger class
logging.Logger.trace = _trace


def get_logger(name: str = "tts-server") -> logging.Logger:
    """Get a logger with TRACE support."""
    return logging.getLogger(name)
