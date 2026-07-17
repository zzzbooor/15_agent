from __future__ import annotations

import csv
import math

from skills.core.context import current_context
from skills.core.errors import ErrorCode, SkillFault
from skills.core.filesystem import enforce_file_size, require_regular_file, resolve_under_root


def table_analyzer(
    path: str,
    max_rows_preview: int = 5,
    describe: bool = True,
    *,
    data_root: str | None = None,
) -> dict:
    """Stream a bounded CSV/TSV table and return preview plus numeric statistics."""

    context = current_context(data_root)
    limits = context.limits.table
    if not isinstance(max_rows_preview, int) or isinstance(max_rows_preview, bool) or max_rows_preview < 0:
        raise SkillFault(ErrorCode.PARAM_INVALID, "max_rows_preview must be a non-negative integer")
    if not isinstance(describe, bool):
        raise SkillFault(ErrorCode.PARAM_INVALID, "describe must be a boolean")
    applied_preview = min(max_rows_preview, limits.max_preview_rows)
    source, root = resolve_under_root(path, context.data_root)
    if source.suffix.casefold() not in {".csv", ".tsv"}:
        raise SkillFault(ErrorCode.UNSUPPORTED_OPERATION, "table_analyzer only supports .csv and .tsv files")
    require_regular_file(source, display_path=path)
    size_bytes = enforce_file_size(source, limits.max_file_bytes)
    delimiter = "\t" if source.suffix.casefold() == ".tsv" else ","

    preview: list[dict[str, str]] = []
    numeric: dict[str, dict[str, float | int | bool]] = {}
    row_count = 0
    truncated = False
    try:
        with source.open("r", encoding="utf-8", errors="strict", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            if not reader.fieldnames:
                raise SkillFault(ErrorCode.PARAM_INVALID, "table must contain a header row")
            columns = list(reader.fieldnames)
            numeric = {
                column: {"valid": True, "count": 0, "min": math.inf, "max": -math.inf, "sum": 0.0}
                for column in columns
            }
            for row in reader:
                if row_count >= limits.max_rows:
                    truncated = True
                    break
                normalized_row = {column: (row.get(column) or "") for column in columns}
                if len(preview) < applied_preview:
                    preview.append(normalized_row)
                row_count += 1
                if not describe:
                    continue
                for column in columns:
                    state = numeric[column]
                    raw_value = normalized_row[column].strip()
                    if not raw_value:
                        state["valid"] = False
                        continue
                    try:
                        value = float(raw_value)
                    except ValueError:
                        state["valid"] = False
                        continue
                    if not math.isfinite(value):
                        state["valid"] = False
                        continue
                    state["count"] = int(state["count"]) + 1
                    state["sum"] = float(state["sum"]) + value
                    state["min"] = min(float(state["min"]), value)
                    state["max"] = max(float(state["max"]), value)
    except UnicodeDecodeError as exc:
        raise SkillFault(ErrorCode.DECODE_ERROR, f"table is not valid UTF-8: {path}") from exc
    except csv.Error as exc:
        raise SkillFault(ErrorCode.PARAM_INVALID, f"invalid delimited table: {exc}") from exc

    statistics: dict[str, dict[str, float | int]] = {}
    if describe:
        for column, state in numeric.items():
            count = int(state["count"])
            if not bool(state["valid"]) or count == 0:
                continue
            statistics[column] = {
                "count": count,
                "min": float(state["min"]),
                "max": float(state["max"]),
                "mean": float(state["sum"]) / count,
            }
    return {
        "path": source.relative_to(root).as_posix(),
        "num_rows": row_count,
        "num_columns": len(columns),
        "columns": columns,
        "preview": preview,
        "describe": statistics,
        "truncated": truncated,
        "size_bytes": size_bytes,
        "applied_max_rows_preview": applied_preview,
    }
