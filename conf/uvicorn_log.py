import logging
import logging.handlers
import logging.config
from common import logPath

mainLog = logPath + "_main.log"
errorLog = logPath + "_error.log"
accessLog = logPath + "_access.log"

LOGGING_CONFIG: dict = {
    "version": 1,
    "disable_existing_loggers": True,
    "loggers": {
        "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": False}, # propagate - handler 중복 설정
        "uvicorn.error": {"handlers": ["error"], "level": "INFO", "propagate": False},
        "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
    },
    "formatters": {
        "default": {
             "()": "uvicorn.logging.DefaultFormatter",
            "fmt": "%(levelname)s : %(asctime)s - %(message)s",
            "use_colors": None,
        },
        "access": {
            "()": "uvicorn.logging.AccessFormatter",
            "fmt": "[%(asctime)s] - [PID : %(process)d] - [REQUEST : %(message)s]",  # noqa: E501
        },
    },
    "handlers": {
        "default": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "formatter": "default",
            "filename": mainLog,
            "when": "midnight",
            "interval": 1,
            "backupCount": 30,
            "encoding": "utf-8"
        },
        "error": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "formatter": "default",
            "filename": errorLog,
            "when" : "midnight",
            "interval": 1,
            "backupCount": 30,
            "encoding":"utf-8"
        },
        "access": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "formatter": "access",
            "filename": accessLog,
            "when": "midnight",
            "interval": 1,
            "backupCount": 30,
            "encoding": "utf-8"
        },
    },

}
logging.config.dictConfig(LOGGING_CONFIG)

# 로그에 request, message 표시
class LogConfig:
    def __int__(self):
        pass

    def Log(self, message):
        uvicorn = logging.getLogger("uvicorn")
        uvicorn.info(message)

    def error_log(self, error_message):
        error = logging.getLogger("uvicorn.error")
        error.info(error_message)
