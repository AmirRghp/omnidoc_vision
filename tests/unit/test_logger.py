import io
import json
import logging
import sys

from src.common.logger import JsonFormatter, configure_logging, get_logger


def _make_record(
    message: str = "query_completed",
    *,
    exc_info: tuple[type[BaseException], BaseException, object] | None = None,
) -> logging.LogRecord:
    return logging.LogRecord(
        name="omnidoc.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=(),
        exc_info=exc_info,
    )


def test_json_formatter_includes_core_and_extra_fields() -> None:
    record = _make_record()
    record.query_id = "q-123"
    record.routing_path = "FAST_TEXT_TABLE_PATH"
    record.vlm_images_count = 0
    record.latency_ms = {"total_ms": 12.5}

    payload = json.loads(JsonFormatter().format(record))

    assert payload["message"] == "query_completed"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "omnidoc.test"
    assert payload["query_id"] == "q-123"
    assert payload["routing_path"] == "FAST_TEXT_TABLE_PATH"
    assert payload["vlm_images_count"] == 0
    assert payload["latency_ms"] == {"total_ms": 12.5}
    assert "timestamp" in payload


def test_json_formatter_preserves_persian_text() -> None:
    output = JsonFormatter().format(_make_record("پاسخ نهایی آماده است"))

    assert "پاسخ نهایی آماده است" in output
    assert json.loads(output)["message"] == "پاسخ نهایی آماده است"


def test_json_formatter_redacts_sensitive_extra_fields() -> None:
    record = _make_record()
    record.vlm_api_key = "super-secret"
    record.request = {"authorization": "Bearer secret"}

    payload = json.loads(JsonFormatter().format(record))

    assert payload["vlm_api_key"] == "[REDACTED]"
    assert payload["request"] == {"authorization": "[REDACTED]"}


def test_json_formatter_preserves_stack_info() -> None:
    record = _make_record()
    record.stack_info = "Stack (most recent call last):\n  ..."

    assert json.loads(JsonFormatter().format(record))["stack_info"].startswith("Stack")


def test_json_formatter_serializes_exceptions() -> None:
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        record = _make_record("query_failed", exc_info=sys.exc_info())

    payload = json.loads(JsonFormatter().format(record))

    assert "RuntimeError: boom" in payload["exception"]


def test_configure_logging_is_idempotent_and_writes_json() -> None:
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level
    stream = io.StringIO()

    try:
        configure_logging("DEBUG", stream=stream)
        configure_logging("INFO", stream=stream)

        json_handlers = [
            handler for handler in root.handlers if handler.get_name() == "omnidoc-json"
        ]
        assert len(json_handlers) == 1

        logging.getLogger("omnidoc.test").info("query_completed", extra={"query_id": "q-1"})
        lines = [line for line in stream.getvalue().splitlines() if line]
        assert len(lines) == 1

        payload = json.loads(lines[0])
        assert payload["query_id"] == "q-1"
        assert payload["level"] == "INFO"
    finally:
        for handler in list(root.handlers):
            if handler not in original_handlers:
                root.removeHandler(handler)
        root.setLevel(original_level)


def test_get_logger_returns_named_logger() -> None:
    assert get_logger("omnidoc.test") is logging.getLogger("omnidoc.test")
