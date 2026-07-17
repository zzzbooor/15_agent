"""Local deterministic tools used by the Agent.

Only lightweight compatibility helpers live here.  Individual Skill modules
are loaded lazily through :mod:`skills.core.catalog`.
"""

from __future__ import annotations

from pathlib import Path

from skills.core.context import DEFAULT_DATA_ROOT
from skills.core.filesystem import enforce_file_size, enforce_text_bytes, resolve_under_root
from skills.core.limits import load_limits


_DEFAULT_LIMITS = load_limits()


class ResourceLimits:
    """Backward-compatible view of the central policy values."""

    MAX_EXPRESSION_LENGTH = _DEFAULT_LIMITS.calculator.max_expression_chars
    MAX_EXPONENT = _DEFAULT_LIMITS.calculator.max_exponent
    MAX_FILE_SIZE_MB = _DEFAULT_LIMITS.file_reader.max_file_bytes / (1024 * 1024)
    MAX_READ_CHARS = _DEFAULT_LIMITS.file_reader.max_read_chars
    MAX_SEARCH_FILES = _DEFAULT_LIMITS.search.max_files
    SEARCH_TIMEOUT_SECONDS = _DEFAULT_LIMITS.search.timeout_seconds
    MAX_TABLE_ROWS = _DEFAULT_LIMITS.table.max_rows
    MAX_TABLE_SIZE_MB = _DEFAULT_LIMITS.table.max_file_bytes / (1024 * 1024)
    MAX_OUTPUT_SIZE_MB = _DEFAULT_LIMITS.format_converter.max_output_bytes / (1024 * 1024)


def resolve_data_path(path: str, data_root: str | None = None) -> tuple[Path, Path]:
    return resolve_under_root(path, data_root or DEFAULT_DATA_ROOT)


def check_file_size(path: Path, max_mb: float) -> None:
    enforce_file_size(path, int(max_mb * 1024 * 1024))


def check_output_size(text: str, max_mb: float) -> None:
    enforce_text_bytes(text, int(max_mb * 1024 * 1024), "output")


__all__ = [
    "DEFAULT_DATA_ROOT",
    "ResourceLimits",
    "check_file_size",
    "check_output_size",
    "resolve_data_path",
]
