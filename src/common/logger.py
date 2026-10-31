import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any, Final, TextIO

_RESERVED_ATTRS: Final[frozenset[str]] = frozenset(
    set(vars(logging.makeLogRecord({}))) | {"message", "asctime"}
)
_HANDLER_NAME: Final = "omnidoc-json"


class JsonFormatter(logging.Formatter):
    """Render log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            key: value
            for key, value in vars(record).items()
            if key not in _RESERVED_ATTRS and not key.startswith("_")
        }
        payload.update(
            {
                "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
        )
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str = "INFO", *, stream: TextIO | None = None) -> None:
    """Configure the root logger with a single structured JSON handler."""
    root = logging.getLogger()
    if not any(handler.get_name() == _HANDLER_NAME for handler in root.handlers):
        handler = logging.StreamHandler(stream if stream is not None else sys.stdout)
        handler.setFormatter(JsonFormatter())
        handler.set_name(_HANDLER_NAME)
        root.addHandler(handler)
    root.setLevel(logging.getLevelNamesMapping().get(level.upper(), logging.INFO))


def get_logger(name: str) -> logging.Logger:
    """Return the standard library logger for the given module name."""
    return logging.getLogger(name)
