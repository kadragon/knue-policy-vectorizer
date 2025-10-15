import logging
import os
import sys
from typing import Optional

import structlog
from colorama import Fore, Style, init

# Initialize colorama for cross-platform colored output
init(autoreset=True)


def setup_logger(
    log_level: Optional[str] = None, logger_name: Optional[str] = None
) -> structlog.BoundLogger:
    """Set up structured logging with colors."""
    # Default to environment LOG_LEVEL when not provided explicitly
    if log_level is None:
        log_level = os.getenv("LOG_LEVEL", "INFO")

    # Configure standard logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        stream=sys.stdout,
        format="%(message)s",
    )

    # Color mapping for log levels
    colors = {
        "debug": Fore.CYAN,
        "info": Fore.GREEN,
        "warning": Fore.YELLOW,
        "error": Fore.RED,
        "critical": Fore.MAGENTA,
    }

    def add_colors(logger, method_name, event_dict):
        """Add colors to log output based on level."""
        level = event_dict.get("level", "info").lower()
        color = colors.get(level, "")

        if color:
            event_dict["level"] = (
                f"{color}{event_dict['level'].upper()}{Style.RESET_ALL}"
            )

        return event_dict

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="ISO"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            add_colors,
            (
                structlog.processors.JSONRenderer()
                if log_level == "DEBUG"
                else structlog.dev.ConsoleRenderer()
            ),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    return structlog.get_logger(logger_name or "knue-vectorizer")
