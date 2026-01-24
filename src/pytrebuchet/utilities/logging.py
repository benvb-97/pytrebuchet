"""Logging configuration."""

import logging
from typing import ClassVar

from .paths import ROOT_DIR


class CustomFormatter(logging.Formatter):
    """Custom logging formatter."""

    # All colors are bold (;1m).
    # this makes it easier to distinguish them in the console of the compiled .exe
    grey = "\u001b[37;1m"
    cyan = "\u001b[36;1m"
    green = "\u001b[32;1m"
    yellow = "\u001b[33;1m"
    red = "\u001b[31;1m"
    reset = "\x1b[0m"
    format_string = "%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s"

    FORMATS: ClassVar[dict[int, str]] = {
        logging.DEBUG: cyan + format_string + reset,
        logging.INFO: green + format_string + reset,
        logging.WARNING: yellow + format_string + reset,
        logging.ERROR: red + format_string + reset,
        logging.CRITICAL: red + format_string + reset,
    }

    def get_header_length(self, record: logging.LogRecord) -> int:
        """Get the header length of a given record."""
        return len(
            super().format(
                logging.LogRecord(
                    name=record.name,
                    level=record.levelno,
                    pathname=record.pathname,
                    lineno=record.lineno,
                    msg="",
                    args=(),
                    exc_info=None,
                )
            )
        )

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record."""
        indent = " " * self.get_header_length(record)

        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        head, *trailing = formatter.format(record).splitlines(keepends=True)

        return head + "".join(indent + line for line in trailing)


LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": True,
    "formatters": {
        "standard": {"format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"},
        "custom": {
            "()": CustomFormatter,
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        },
    },
    "handlers": {
        # File handler that logs to console
        "default": {
            "level": "DEBUG",
            "formatter": "custom",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",  # Default is stderr
        },
        # File handler that logs to a file
        "text_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "standard",
            "level": "DEBUG",
            "filename": f"{ROOT_DIR}/pytrebuchet.log",
            "mode": "a",
            "encoding": "utf-8",
            "backupCount": 4,
        },
    },
    "loggers": {
        "": {  # root logger
            "handlers": ["default", "text_file"],
            "level": "DEBUG",
            "propagate": False,
        },
        "__main__": {  # if __name__ == '__main__'
            "handlers": ["default", "text_file"],
            "level": "DEBUG",
            "propagate": False,
        },
        "mp_logger": {  # if __name__ == '__main__'
            "handlers": ["default", "text_file"],
            "level": "DEBUG",
            "propagate": False,
        },
    },
}
