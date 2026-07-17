from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .limits import SkillLimits


@dataclass(frozen=True)
class SkillSpec:
    """Stable public identity and import location of one Skill."""

    name: str
    module: str
    function: str
    category: str = "basic"


@dataclass(frozen=True)
class SkillContext:
    """Execution-scoped paths and policies shared by nested Skills."""

    data_root: Path
    output_dir: Path | None
    limits: "SkillLimits"
    request_id: str
    limits_config: Path | None = None

