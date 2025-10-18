import logging
import os
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler


def configure_logging(level: Optional[str] = None) -> logging.Logger:
    log_level = level or os.getenv("AK_LOGLEVEL", "INFO")
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=Console(), show_time=True, show_level=True, rich_tracebacks=True)],
    )
    logger = logging.getLogger("aldar_kose")
    logger.setLevel(log_level)
    return logger
