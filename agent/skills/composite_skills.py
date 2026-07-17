from __future__ import annotations

import json
from time import perf_counter
from typing import Callable

from skills.core.context import bind_context, current_context
from skills.core.errors import ErrorCode, SkillFault, normalize_exception
from skills.file_reader import file_reader
from skills.format_converter import format_converter
from skills.table_analyzer import table_analyzer


class CompositeSkillError(SkillFault):
    def __init__(self, step_name: str, cause: SkillFault) -> None:
        super().__init__(
            ErrorCode.COMPOSITE_STEP_FAILED,
            f"composite step '{step_name}' failed: {cause.message}",
            details={"failed_step": step_name, "cause_code": cause.code.value},
            retryable=cause.retryable,
            error_type=type(self).__name__,
        )
        self.step_name = step_name
        self.cause_fault = cause


def _run_step(step_name: str, function: Callable, **kwargs) -> tuple[dict, dict]:
    started = perf_counter()
    try:
        output = function(**kwargs)
    except Exception as exc:
        raise CompositeSkillError(step_name, normalize_exception(exc)) from exc
    trace = {
        "step": step_name,
        "status": "success",
        "latency_ms": round((perf_counter() - started) * 1000, 3),
    }
    return output, trace


def read_and_convert(
    path: str,
    target_format: str = "markdown",
    max_chars: int = 5000,
    *,
    data_root: str | None = None,
    output_dir: str | None = None,
) -> dict:
    """Read a local text document, then convert it in the same execution context."""

    context = current_context(data_root, output_dir)
    with bind_context(context):
        read_output, read_trace = _run_step(
            "file_reader",
            file_reader,
            path=path,
            max_chars=max_chars,
            data_root=str(context.data_root),
        )
        convert_output, convert_trace = _run_step(
            "format_converter",
            format_converter,
            text=read_output["content"],
            target_format=target_format,
            output_dir=str(context.output_dir) if context.output_dir else None,
        )
    return {
        "composite_name": "read_and_convert",
        "steps": ["file_reader", "format_converter"],
        "step_results": {
            "file_reader": read_output,
            "format_converter": convert_output,
        },
        "step_trace": [read_trace, convert_trace],
        "final_output": convert_output["formatted_text"],
        "generated_file_path": convert_output["generated_file_path"],
    }


def _format_analysis_as_text(analysis: dict) -> str:
    lines = [
        "# 表格分析报告",
        "",
        "## 基本信息",
        f"- 文件路径: {analysis['path']}",
        f"- 行数: {analysis['num_rows']}",
        f"- 列数: {analysis['num_columns']}",
        f"- 列名: {', '.join(analysis['columns'])}",
        "",
        "## 数据预览",
    ]
    for index, row in enumerate(analysis["preview"][:5], 1):
        lines.append(f"### 第{index}行")
        for column, value in row.items():
            lines.append(f"- {column}: {value}")
        lines.append("")
    if analysis.get("describe"):
        lines.append("## 统计摘要")
        for column, statistics in analysis["describe"].items():
            lines.append(f"### {column}")
            for name, value in statistics.items():
                lines.append(f"- {name}: {value}")
            lines.append("")
    return "\n".join(lines)


def analyze_and_convert(
    path: str,
    target_format: str = "markdown",
    max_rows_preview: int = 10,
    describe: bool = True,
    *,
    data_root: str | None = None,
    output_dir: str | None = None,
) -> dict:
    """Analyze a local table, then convert the generated report."""

    context = current_context(data_root, output_dir)
    with bind_context(context):
        analysis, analyze_trace = _run_step(
            "table_analyzer",
            table_analyzer,
            path=path,
            max_rows_preview=max_rows_preview,
            describe=describe,
            data_root=str(context.data_root),
        )
        report_text = (
            json.dumps(analysis, ensure_ascii=False)
            if isinstance(target_format, str) and target_format.casefold() == "json"
            else _format_analysis_as_text(analysis)
        )
        converted, convert_trace = _run_step(
            "format_converter",
            format_converter,
            text=report_text,
            target_format=target_format,
            output_dir=str(context.output_dir) if context.output_dir else None,
        )
    return {
        "composite_name": "analyze_and_convert",
        "steps": ["table_analyzer", "format_converter"],
        "step_results": {
            "table_analyzer": analysis,
            "format_converter": converted,
        },
        "step_trace": [analyze_trace, convert_trace],
        "final_output": converted["formatted_text"],
        "generated_file_path": converted["generated_file_path"],
    }


COMPOSITE_SKILLS = {
    "read_and_convert": read_and_convert,
    "analyze_and_convert": analyze_and_convert,
}
