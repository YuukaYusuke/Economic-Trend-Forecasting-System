"""
Logging configuration for Economic Trend Forecasting System.
Add this as modules/logging_config.py
"""

import logging
import logging.handlers
from pathlib import Path
from modules.config import DATA_PATH

# Create logs directory if it doesn't exist
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

# Logger configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        },
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": LOGS_DIR / "app.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
        },
        "error_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "ERROR",
            "formatter": "detailed",
            "filename": LOGS_DIR / "errors.log",
            "maxBytes": 5242880,  # 5MB
            "backupCount": 3,
        },
    },
    "loggers": {
        "": {  # root logger
            "handlers": ["console", "file", "error_file"],
            "level": "DEBUG",
            "propagate": True,
        },
        "modules.trainer": {
            "handlers": ["console", "file"],
            "level": "DEBUG",
            "propagate": False,
        },
        "modules.predict": {
            "handlers": ["console", "file"],
            "level": "DEBUG",
            "propagate": False,
        },
        "modules.pipeline_service": {
            "handlers": ["console", "file"],
            "level": "INFO",
            "propagate": False,
        },
    }
}


def setup_logging():
    """Initialize logging configuration."""
    import logging.config
    logging.config.dictConfig(LOGGING_CONFIG)
    logger = logging.getLogger(__name__)
    logger.info("Logging initialized")
    return logger


# Initialize logger
logger = setup_logging()
