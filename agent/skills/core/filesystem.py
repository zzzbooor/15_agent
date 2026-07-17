from __future__ import annotations

import os
from pathlib import Path

from .errors import ErrorCode, SkillFault


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_under_root(
    value: str | Path,
    root: str | Path,
    *,
    must_exist: bool = False,
    allow_symlinks: bool = False,
) -> tuple[Path, Path]:
    """Resolve a user path below root and reject symlink-based escapes."""

    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise SkillFault(ErrorCode.PARAM_INVALID, "path must be a non-empty string")
    resolved_root = Path(root).expanduser().resolve()
    raw = Path(value).expanduser()
    candidate = raw if raw.is_absolute() else resolved_root / raw

    if not allow_symlinks:
        absolute = candidate.absolute()
        current = absolute
        while current != resolved_root and _is_within(current, resolved_root):
            if current.exists() and current.is_symlink():
                raise SkillFault(
                    ErrorCode.PERMISSION_DENIED,
                    f"symbolic links are not allowed: {value}",
                )
            current = current.parent

    resolved = candidate.resolve(strict=False)
    if not _is_within(resolved, resolved_root):
        raise SkillFault(
            ErrorCode.PATH_OUTSIDE_ROOT,
            f"path escapes data root: {value}",
        )
    if must_exist and not resolved.exists():
        raise SkillFault(ErrorCode.FILE_NOT_FOUND, f"path not found: {value}")
    return resolved, resolved_root


def require_regular_file(path: Path, *, display_path: str | Path | None = None) -> None:
    if not path.exists():
        raise SkillFault(ErrorCode.FILE_NOT_FOUND, f"file not found: {display_path or path}")
    if not path.is_file():
        raise SkillFault(ErrorCode.FILE_NOT_FOUND, f"not a regular file: {display_path or path}")


def enforce_file_size(path: Path, max_bytes: int) -> int:
    size = path.stat().st_size
    if size > max_bytes:
        raise SkillFault(
            ErrorCode.FILE_TOO_LARGE,
            f"file is too large: {size} bytes (maximum {max_bytes})",
            details={"size_bytes": size, "max_bytes": max_bytes},
        )
    return size


def enforce_text_bytes(text: str, max_bytes: int, label: str) -> int:
    if not isinstance(text, str):
        raise SkillFault(ErrorCode.PARAM_INVALID, f"{label} must be a string")
    size = len(text.encode("utf-8"))
    if size > max_bytes:
        raise SkillFault(
            ErrorCode.RESOURCE_EXHAUSTED,
            f"{label} is too large: {size} bytes (maximum {max_bytes})",
            details={"size_bytes": size, "max_bytes": max_bytes},
        )
    return size


def _validate_output_name(raw_name: str, suffix: str) -> str:
    if not isinstance(raw_name, str) or not raw_name.strip():
        raise SkillFault(ErrorCode.PARAM_INVALID, "output filename must be a non-empty string")
    name = raw_name.strip()
    if name in {".", ".."} or Path(name).name != name or "/" in name or "\\" in name:
        raise SkillFault(ErrorCode.PERMISSION_DENIED, f"output filename must not contain a path: {raw_name}")
    path = Path(name)
    stem = path.stem
    if not stem:
        raise SkillFault(ErrorCode.PARAM_INVALID, "output filename must contain a stem")
    return f"{stem}{suffix}"


def write_unique_text(
    text: str,
    output_dir: str | Path,
    output_filename: str,
    suffix: str,
    *,
    max_attempts: int,
) -> Path:
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    base_name = _validate_output_name(output_filename, suffix)
    base = Path(base_name)
    for index in range(max_attempts):
        name = base.name if index == 0 else f"{base.stem}({index}){base.suffix}"
        candidate = directory / name
        try:
            with candidate.open("x", encoding="utf-8", newline="") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            return candidate
        except FileExistsError:
            continue
    raise SkillFault(
        ErrorCode.RESOURCE_EXHAUSTED,
        f"could not allocate an output filename after {max_attempts} attempts",
    )

