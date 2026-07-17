from __future__ import annotations

from skills.core.context import current_context
from skills.core.errors import ErrorCode, SkillFault
from skills.core.filesystem import enforce_file_size, require_regular_file, resolve_under_root


def file_reader(path: str, max_chars: int = 2000, *, data_root: str | None = None) -> dict:
    """Read a bounded UTF-8 ``.txt`` or ``.md`` file below the data root."""

    context = current_context(data_root)
    limits = context.limits.file_reader
    if not isinstance(max_chars, int) or isinstance(max_chars, bool) or max_chars <= 0:
        raise SkillFault(ErrorCode.PARAM_INVALID, "max_chars must be a positive integer")
    applied_max_chars = min(max_chars, limits.max_read_chars)
    source, root = resolve_under_root(path, context.data_root)
    if source.suffix.lower() not in {".txt", ".md"}:
        raise SkillFault(ErrorCode.UNSUPPORTED_OPERATION, "file_reader only supports .txt and .md files")
    require_regular_file(source, display_path=path)
    size_bytes = enforce_file_size(source, limits.max_file_bytes)
    try:
        with source.open("r", encoding="utf-8", errors="strict") as handle:
            content_with_probe = handle.read(applied_max_chars + 1)
    except UnicodeDecodeError as exc:
        raise SkillFault(
            ErrorCode.DECODE_ERROR,
            f"file is not valid UTF-8: {path}",
            details={"start": exc.start, "end": exc.end},
        ) from exc
    truncated = len(content_with_probe) > applied_max_chars
    content = content_with_probe[:applied_max_chars]
    return {
        "content": content,
        "num_chars": len(content),
        "source": source.relative_to(root).as_posix(),
        "truncated": truncated,
        "size_bytes": size_bytes,
        "applied_max_chars": applied_max_chars,
    }
