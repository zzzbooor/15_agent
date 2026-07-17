from __future__ import annotations

import json
from pathlib import Path

from skills.core.context import PROJECT_ROOT, current_context
from skills.core.errors import ErrorCode, SkillFault
from skills.core.filesystem import enforce_text_bytes, write_unique_text


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "format_converter_files"
DEFAULT_FILENAMES = {"markdown": "converted.md", "json": "converted.json"}
SUFFIXES = {"markdown": ".md", "json": ".json"}


def _parse_key_value_lines(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in (line.strip() for line in text.splitlines()):
        if not line:
            continue
        if ":" not in line:
            raise SkillFault(ErrorCode.PARAM_INVALID, f"expected 'key: value' line: {line}")
        key, value = (part.strip() for part in line.split(":", 1))
        if not key or key in result:
            raise SkillFault(ErrorCode.PARAM_INVALID, f"invalid or duplicate key: {key}")
        result[key] = value
    if not result:
        raise SkillFault(ErrorCode.PARAM_INVALID, "text contains no convertible content")
    return result


def format_converter(
    text: str,
    target_format: str,
    output_filename: str | None = None,
    output_dir: str | None = None,
) -> dict:
    """Convert text to Markdown or formatted JSON and create one output file."""

    context = current_context(output_dir=output_dir)
    limits = context.limits.format_converter
    if not isinstance(text, str) or not text.strip():
        raise SkillFault(ErrorCode.PARAM_INVALID, "text must be a non-empty string")
    enforce_text_bytes(text, limits.max_input_bytes, "input text")
    target = target_format.strip().casefold() if isinstance(target_format, str) else ""
    if target == "markdown":
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        formatted_text = "\n".join(f"- {line}" for line in lines)
    elif target == "json":
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = _parse_key_value_lines(text)
        formatted_text = json.dumps(parsed, ensure_ascii=False, indent=2)
    else:
        raise SkillFault(ErrorCode.UNSUPPORTED_OPERATION, "target_format must be markdown or json")
    enforce_text_bytes(formatted_text, limits.max_output_bytes, "converted output")
    directory = context.output_dir or DEFAULT_OUTPUT_DIR
    filename = output_filename or DEFAULT_FILENAMES[target]
    generated_path = write_unique_text(
        formatted_text,
        directory,
        filename,
        SUFFIXES[target],
        max_attempts=limits.max_collision_attempts,
    )
    return {
        "formatted_text": formatted_text,
        "generated_file_path": str(generated_path),
        "target_format": target,
        "output_bytes": len(formatted_text.encode("utf-8")),
    }
