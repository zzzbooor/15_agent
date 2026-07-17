from __future__ import annotations

from .contracts import SkillSpec
from .errors import ErrorCode, SkillFault


SKILL_SPECS: dict[str, SkillSpec] = {
    "calculator": SkillSpec("calculator", "skills.calculator", "calculator"),
    "file_reader": SkillSpec("file_reader", "skills.file_reader", "file_reader"),
    "local_file_search": SkillSpec("local_file_search", "skills.local_file_search", "local_file_search"),
    "table_analyzer": SkillSpec("table_analyzer", "skills.table_analyzer", "table_analyzer"),
    "format_converter": SkillSpec("format_converter", "skills.format_converter", "format_converter"),
    "read_and_convert": SkillSpec(
        "read_and_convert", "skills.composite_skills", "read_and_convert", "advanced"
    ),
    "analyze_and_convert": SkillSpec(
        "analyze_and_convert", "skills.composite_skills", "analyze_and_convert", "advanced"
    ),
    "code_executor": SkillSpec("code_executor", "skills.code_executor", "code_executor", "advanced"),
}


def get_skill_spec(name: str) -> SkillSpec:
    try:
        return SKILL_SPECS[name]
    except KeyError as exc:
        raise SkillFault(ErrorCode.UNSUPPORTED_OPERATION, f"unknown skill: {name}") from exc


def list_skill_names(category: str | None = None) -> list[str]:
    return sorted(
        name for name, spec in SKILL_SPECS.items() if category is None or spec.category == category
    )

