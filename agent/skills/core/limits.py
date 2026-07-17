from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields
from functools import lru_cache
from pathlib import Path
from typing import Any, TypeVar, get_type_hints

from .errors import ErrorCode, SkillFault


@dataclass(frozen=True)
class CalculatorLimits:
    max_expression_chars: int = 500
    max_exponent: int = 20
    max_integer_bits: int = 4096
    max_abs_result: float = 1e100


@dataclass(frozen=True)
class FileLimits:
    max_file_bytes: int = 10 * 1024 * 1024
    max_read_chars: int = 10_000


@dataclass(frozen=True)
class SearchLimits:
    max_query_chars: int = 500
    max_entries: int = 5000
    max_files: int = 1000
    max_file_bytes: int = 512 * 1024
    max_total_bytes: int = 8 * 1024 * 1024
    max_index_tokens: int = 300_000
    timeout_seconds: float = 5.0
    max_top_k: int = 20
    snippet_radius: int = 80
    max_skipped_records: int = 20


@dataclass(frozen=True)
class TableLimits:
    max_file_bytes: int = 50 * 1024 * 1024
    max_rows: int = 100_000
    max_preview_rows: int = 100


@dataclass(frozen=True)
class FormatLimits:
    max_input_bytes: int = 10 * 1024 * 1024
    max_output_bytes: int = 10 * 1024 * 1024
    max_collision_attempts: int = 1000


@dataclass(frozen=True)
class SandboxLimits:
    max_code_chars: int = 5000
    wall_timeout_seconds: float = 3.0
    cpu_seconds: int = 2
    memory_mb: int = 128
    max_open_files: int = 16
    max_processes: int = 1
    max_operations: int = 50_000
    max_loop_iterations: int = 10_000
    max_output_bytes: int = 32 * 1024
    max_container_items: int = 10_000
    max_string_bytes: int = 128 * 1024
    max_integer_bits: int = 4096


@dataclass(frozen=True)
class SkillLimits:
    calculator: CalculatorLimits = CalculatorLimits()
    file_reader: FileLimits = FileLimits()
    search: SearchLimits = SearchLimits()
    table: TableLimits = TableLimits()
    format_converter: FormatLimits = FormatLimits()
    sandbox: SandboxLimits = SandboxLimits()


DEFAULT_LIMITS_PATH = Path(__file__).resolve().parents[2] / "configs" / "skill_limits.yaml"
T = TypeVar("T")


def _read_config(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise SkillFault(ErrorCode.FILE_NOT_FOUND, f"limits config not found: {path}") from exc
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:
            raise SkillFault(
                ErrorCode.PARAM_INVALID,
                "skill_limits.yaml must use JSON-compatible YAML when PyYAML is unavailable",
            ) from exc
        value = yaml.safe_load(raw)
    if not isinstance(value, dict):
        raise SkillFault(ErrorCode.PARAM_INVALID, "skill limits config must contain an object")
    return value


def _build_section(cls: type[T], raw: Any, section: str) -> T:
    if raw is None:
        return cls()
    if not isinstance(raw, dict):
        raise SkillFault(ErrorCode.PARAM_INVALID, f"limits section {section} must be an object")
    allowed = {item.name for item in fields(cls)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise SkillFault(
            ErrorCode.PARAM_INVALID,
            f"unknown limits in {section}: {', '.join(unknown)}",
        )
    try:
        value = cls(**raw)
    except TypeError as exc:
        raise SkillFault(ErrorCode.PARAM_INVALID, f"invalid limits section {section}: {exc}") from exc
    type_hints = get_type_hints(cls)
    for item in fields(value):
        number = getattr(value, item.name)
        expected_type = type_hints.get(item.name)
        valid_type = (
            type(number) is int
            if expected_type is int
            else isinstance(number, (int, float)) and not isinstance(number, bool)
        )
        if (
            not valid_type
            or number <= 0
            or (isinstance(number, float) and not math.isfinite(number))
        ):
            raise SkillFault(
                ErrorCode.PARAM_OUT_OF_RANGE,
                f"limit {section}.{item.name} must be a positive {getattr(expected_type, '__name__', 'number')}",
            )
    return value


@lru_cache(maxsize=8)
def _load_cached(path_text: str) -> SkillLimits:
    path = Path(path_text)
    config = _read_config(path)
    return SkillLimits(
        calculator=_build_section(CalculatorLimits, config.get("calculator"), "calculator"),
        file_reader=_build_section(FileLimits, config.get("file_reader"), "file_reader"),
        search=_build_section(SearchLimits, config.get("search"), "search"),
        table=_build_section(TableLimits, config.get("table"), "table"),
        format_converter=_build_section(FormatLimits, config.get("format_converter"), "format_converter"),
        sandbox=_build_section(SandboxLimits, config.get("sandbox"), "sandbox"),
    )


def load_limits(path: str | Path | None = None) -> SkillLimits:
    selected = Path(path).expanduser().resolve() if path else DEFAULT_LIMITS_PATH.resolve()
    return _load_cached(str(selected))
