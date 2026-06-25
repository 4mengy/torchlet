import logging
import sys
from typing import TextIO


DEFAULT_LOG_FORMAT = (
    "%(asctime)s [%(levelname)s] %(name)s %(filename)s:%(lineno)d - %(message)s"
)
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(
    name: str = "torchlet",
    level: int = logging.INFO,
    stream: TextIO | None = None,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler(stream or sys.stdout)
        handler.setLevel(level)
        handler.setFormatter(
            logging.Formatter(DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT)
        )
        logger.addHandler(handler)

    return logger


logger = get_logger()


__all__ = ["get_logger", "logger"]
