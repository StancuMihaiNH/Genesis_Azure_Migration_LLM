from logging.config import dictConfig

def setup_logging():
    logging_config = {
        "version": 1,
        "formatters": {
            "default": {
                "format": "[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
            },
        },
        "root": {
            "level": "DEBUG",
            "handlers": ["console"],
        },
        "loggers": {
            "uvicorn": {
                "level": "DEBUG",
                "handlers": ["console"],
                "propagate": False,
            },
        },
    }
    dictConfig(logging_config)