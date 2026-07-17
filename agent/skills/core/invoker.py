from __future__ import annotations

import importlib
import inspect
import json
from time import perf_counter
from typing import Any, Callable

from .catalog import get_skill_spec
from .context import bind_context, make_context
from .contracts import SkillContext
from .errors import ErrorCode, SkillFault, normalize_exception


_RESERVED_INPUT_NAMES = {"data_root", "output_dir", "limits_config"}


def _ensure_json_serializable(value: Any) -> None:
    try:
        json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SkillFault(
            ErrorCode.SERIALIZATION_ERROR,
            f"skill output is not JSON serializable: {exc}",
            error_type=type(exc).__name__,
        ) from exc


def _safe_input_snapshot(value: Any) -> tuple[Any, SkillFault | None]:
    try:
        json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError, OverflowError) as exc:
        snapshot = {
            "unserializable_input_type": type(value).__name__,
            "repr": repr(value)[:1000],
        }
        fault = SkillFault(
            ErrorCode.SERIALIZATION_ERROR,
            f"skill input is not JSON serializable: {exc}",
            error_type=type(exc).__name__,
        )
        return snapshot, fault
    return value, None


def invoke_callable(
    skill_name: str,
    function: Callable[..., Any],
    input_data: dict[str, Any],
    context: SkillContext,
) -> dict[str, Any]:
    """Execute one callable and always return the stable SkillResult envelope."""

    started = perf_counter()
    copied_input = dict(input_data) if isinstance(input_data, dict) else input_data
    original_input, input_fault = _safe_input_snapshot(copied_input)
    try:
        if input_fault is not None:
            raise input_fault
        if not isinstance(input_data, dict):
            raise SkillFault(ErrorCode.PARAM_INVALID, "skill input must be a JSON object")
        reserved = sorted(set(input_data) & _RESERVED_INPUT_NAMES)
        if reserved:
            raise SkillFault(
                ErrorCode.PARAM_INVALID,
                f"reserved execution parameters are not accepted as Skill input: {', '.join(reserved)}",
            )
        signature = inspect.signature(function)
        kwargs = dict(input_data)
        if "data_root" in signature.parameters:
            kwargs["data_root"] = str(context.data_root)
        if "output_dir" in signature.parameters:
            kwargs["output_dir"] = str(context.output_dir) if context.output_dir else None
        missing = [
            name
            for name, parameter in signature.parameters.items()
            if parameter.default is inspect.Parameter.empty
            and parameter.kind not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
            and name not in kwargs
        ]
        if missing:
            raise SkillFault(
                ErrorCode.PARAM_MISSING,
                f"missing required parameters: {', '.join(missing)}",
                details={"missing": missing},
            )
        try:
            signature.bind(**kwargs)
        except TypeError as exc:
            raise SkillFault(ErrorCode.PARAM_INVALID, str(exc)) from exc
        with bind_context(context):
            output = function(**kwargs)
        _ensure_json_serializable(output)
        error = None
        status = "success"
    except Exception as exc:
        fault = normalize_exception(exc)
        output = None
        error = fault.to_dict()
        status = "error"
    latency_ms = round((perf_counter() - started) * 1000, 3)
    result = {
        "skill_name": skill_name,
        "status": status,
        "input": original_input,
        "output": output,
        "error": error,
        "latency_ms": latency_ms,
    }
    _ensure_json_serializable(result)
    return result


def invoke_skill(
    skill_name: str,
    input_data: dict[str, Any],
    *,
    data_root: str | None = None,
    output_dir: str | None = None,
    limits_config: str | None = None,
    context: SkillContext | None = None,
) -> dict[str, Any]:
    started = perf_counter()
    try:
        spec = get_skill_spec(skill_name)
        module = importlib.import_module(spec.module)
        try:
            function = getattr(module, spec.function)
        except AttributeError as exc:
            raise SkillFault(
                ErrorCode.UNKNOWN_ERROR,
                f"configured function is missing: {spec.module}.{spec.function}",
            ) from exc
        selected_context = context or make_context(data_root, output_dir, limits_config)
    except Exception as exc:
        original_input, input_fault = _safe_input_snapshot(input_data)
        fault = input_fault or normalize_exception(exc)
        result = {
            "skill_name": skill_name,
            "status": "error",
            "input": original_input,
            "output": None,
            "error": fault.to_dict(),
            "latency_ms": round((perf_counter() - started) * 1000, 3),
        }
        _ensure_json_serializable(result)
        return result
    return invoke_callable(skill_name, function, input_data, selected_context)
