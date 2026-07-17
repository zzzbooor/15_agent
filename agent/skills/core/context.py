from __future__ import annotations

import contextlib
import contextvars
import uuid
from collections.abc import Iterator
from pathlib import Path

from .contracts import SkillContext
from .limits import load_limits


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"

_ACTIVE_CONTEXT: contextvars.ContextVar[SkillContext | None] = contextvars.ContextVar(
    "active_skill_context",
    default=None,
)


def make_context(
    data_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    limits_config: str | Path | None = None,
    request_id: str | None = None,
) -> SkillContext:
    limits_path = Path(limits_config).expanduser().resolve() if limits_config else None
    return SkillContext(
        data_root=(Path(data_root).expanduser().resolve() if data_root else DEFAULT_DATA_ROOT.resolve()),
        output_dir=(Path(output_dir).expanduser().resolve() if output_dir else None),
        limits=load_limits(limits_path),
        request_id=request_id or f"skill_{uuid.uuid4().hex[:12]}",
        limits_config=limits_path,
    )


def current_context(
    data_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    limits_config: str | Path | None = None,
) -> SkillContext:
    active = _ACTIVE_CONTEXT.get()
    if active is not None:
        return active
    return make_context(data_root, output_dir, limits_config)


@contextlib.contextmanager
def bind_context(context: SkillContext) -> Iterator[SkillContext]:
    token = _ACTIVE_CONTEXT.set(context)
    try:
        yield context
    finally:
        _ACTIVE_CONTEXT.reset(token)

