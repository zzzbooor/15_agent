from __future__ import annotations

import argparse
import importlib
import inspect
import json
import sys
import types
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Literal, get_args, get_origin, get_type_hints

from common.io_utils import append_jsonl, read_json, read_yaml, write_json
from common.logging_utils import now_iso
from common.path_utils import bootstrap_project_root, resolve_cli_path, resolve_from_file
from common.schemas import make_tool_message, normalize_tool_call


bootstrap_project_root()

from skills.core.context import make_context  # noqa: E402
from skills.core.errors import ErrorCode, SkillFault  # noqa: E402
from skills.core.invoker import invoke_callable  # noqa: E402


INJECTED_PARAMETERS = {"data_root", "output_dir", "limits_config", "context"}
JSON_TYPE_CHECKS: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "object": dict,
    "array": list,
    "null": type(None),
}


def _load_tools_config(tools_config: str | Path) -> tuple[Path, dict[str, Any]]:
    config_path = Path(tools_config).resolve()
    config = read_yaml(config_path)
    if not isinstance(config, dict):
        raise ValueError("tools.yaml must contain an object")
    if not isinstance(config.get("tools"), dict) or not isinstance(config.get("toolsets"), dict):
        raise ValueError("tools.yaml must define tools and toolsets")
    return config_path, config


def _resolve_toolset(config: dict[str, Any], toolset: str | None) -> tuple[str, list[str]]:
    selected = toolset or config.get("default_toolset")
    if not isinstance(selected, str) or selected not in config["toolsets"]:
        raise ValueError(f"toolset does not exist: {selected}")
    names = config["toolsets"][selected]
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ValueError(f"toolset {selected} must be a list of tool names")
    if len(set(names)) != len(names):
        raise ValueError(f"toolset {selected} contains duplicate tool names")
    return selected, names


def _load_callable(definition: Mapping[str, Any]) -> Any:
    module_name = definition.get("module")
    function_name = definition.get("function")
    if not isinstance(module_name, str) or not isinstance(function_name, str):
        raise ValueError("tool module and function must be strings")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, function_name)
    except AttributeError as exc:
        raise RuntimeError(f"configured function does not exist: {module_name}.{function_name}") from exc


def _json_type_for_python(annotation: Any) -> str | None:
    if annotation is str:
        return "string"
    if annotation is bool:
        return "boolean"
    if annotation is int:
        return "integer"
    if annotation is float:
        return "number"
    if annotation in {dict, Mapping}:
        return "object"
    if annotation in {list, tuple, set, frozenset, Sequence}:
        return "array"
    if annotation is type(None):
        return "null"
    return None


def _annotation_schema(annotation: Any) -> dict[str, Any]:
    """Translate a resolved Python annotation into the supported JSON Schema subset."""

    if annotation in {inspect.Signature.empty, Any}:
        return {}
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is Annotated:
        return _annotation_schema(args[0])
    if origin is Literal:
        values = list(args)
        schema: dict[str, Any] = {"enum": values}
        value_types = {_json_type_for_python(type(value)) for value in values}
        value_types.discard(None)
        if len(value_types) == 1:
            schema["type"] = value_types.pop()
        return schema
    if origin in {types.UnionType, getattr(__import__("typing"), "Union")}:
        variants = [_annotation_schema(item) for item in args]
        unique: list[dict[str, Any]] = []
        for item in variants:
            if item not in unique:
                unique.append(item)
        return unique[0] if len(unique) == 1 else {"anyOf": unique}
    if origin in {list, tuple, set, frozenset, Sequence}:
        item_annotation = args[0] if args else Any
        schema = {"type": "array"}
        item_schema = _annotation_schema(item_annotation)
        if item_schema:
            schema["items"] = item_schema
        return schema
    if origin in {dict, Mapping}:
        schema = {"type": "object"}
        if len(args) == 2 and args[1] is not Any:
            schema["additionalProperties"] = _annotation_schema(args[1])
        return schema
    json_type = _json_type_for_python(annotation)
    if json_type:
        return {"type": json_type}
    if inspect.isclass(annotation) and issubclass(annotation, Mapping):
        return {"type": "object"}
    raise ValueError(f"unsupported type annotation: {annotation!r}")


def _merge_parameter_metadata(
    generated: dict[str, Any], configured: Mapping[str, Any] | None
) -> dict[str, Any]:
    result = dict(generated)
    for key, value in (configured or {}).items():
        if key not in {"type", "items", "anyOf", "default"}:
            result[key] = value
    return result


def _function_parameter_schema(function: Any, definition: Mapping[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(function)
    try:
        hints = get_type_hints(function, include_extras=True)
    except Exception as exc:
        raise ValueError(f"cannot resolve type hints for {function.__module__}.{function.__name__}: {exc}") from exc
    configured = definition.get("parameters", {})
    if not isinstance(configured, dict):
        raise ValueError("tool parameters must be an object")
    properties: dict[str, dict[str, Any]] = {}
    required: list[str] = []
    for name, parameter in signature.parameters.items():
        if name in INJECTED_PARAMETERS or parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue
        annotation = hints.get(name, parameter.annotation)
        generated = _annotation_schema(annotation)
        if not generated:
            raise ValueError(f"public parameter {name} must have a supported type annotation")
        property_schema = _merge_parameter_metadata(generated, configured.get(name))
        if parameter.default is inspect.Parameter.empty:
            required.append(name)
        else:
            try:
                json.dumps(parameter.default)
            except (TypeError, ValueError):
                pass
            else:
                property_schema["default"] = parameter.default
        properties[name] = property_schema
    unknown_configured = sorted(set(configured) - set(properties))
    if unknown_configured:
        raise ValueError(
            "configured parameters are not present in the function signature: "
            + ", ".join(unknown_configured)
        )
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _configured_parameter_schema(definition: Mapping[str, Any]) -> dict[str, Any]:
    raw_parameters = definition.get("parameters", {})
    if not isinstance(raw_parameters, dict):
        raise ValueError("tool parameters must be an object")
    properties: dict[str, dict[str, Any]] = {}
    for name, parameter in raw_parameters.items():
        if not isinstance(parameter, dict):
            raise ValueError(f"invalid parameter schema for {name}")
        properties[name] = dict(parameter)
    required = definition.get("required", [])
    if not isinstance(required, list) or not all(name in properties for name in required):
        raise ValueError("required parameters must reference declared properties")
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _parameter_schema(
    definition: Mapping[str, Any],
    *,
    function: Any | None = None,
    schema_source: str = "configured",
) -> dict[str, Any]:
    if schema_source == "function":
        return _function_parameter_schema(function or _load_callable(definition), definition)
    if schema_source == "configured":
        return _configured_parameter_schema(definition)
    raise ValueError("schema_source must be configured or function")


def _schema_source(config: Mapping[str, Any], definition: Mapping[str, Any]) -> str:
    settings = config.get("settings", {})
    default = settings.get("schema_source", "configured") if isinstance(settings, dict) else "configured"
    return str(definition.get("schema_source", default))


def _tool_schema(name: str, definition: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    for field in ("module", "function", "description", "returns"):
        if field not in definition:
            raise ValueError(f"tool {name} missing {field}")
    returns = definition["returns"]
    if not isinstance(returns, dict):
        raise ValueError(f"tool {name} returns must be an object")
    function = _load_callable(definition)
    parameters = _parameter_schema(
        definition,
        function=function,
        schema_source=_schema_source(config, definition),
    )
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": definition["description"],
            "parameters": parameters,
            "x-returns": {"type": "object", "properties": returns},
        },
    }


def get_tools_schema(
    tools_config: str,
    toolset: str,
    outdir: str | None = None,
) -> list[dict[str, Any]]:
    _, config = _load_tools_config(tools_config)
    selected, tool_names = _resolve_toolset(config, toolset)
    schema: list[dict[str, Any]] = []
    for name in tool_names:
        definition = config["tools"].get(name)
        if not isinstance(definition, dict):
            raise ValueError(f"toolset references missing tool: {name}")
        schema.append(_tool_schema(name, definition, config))
    if outdir:
        output_dir = Path(outdir)
        write_json(schema, output_dir / "tools_schema.json")
        write_json(
            {
                "status": "success",
                "toolset": selected,
                "tool_count": len(schema),
                "tools": tool_names,
                "schema_source": config.get("settings", {}).get("schema_source", "configured"),
            },
            output_dir / "tool_schema_report.json",
        )
    return schema


def get_auto_tools_schema(
    tools_config: str,
    toolset: str,
    outdir: str | None = None,
) -> list[dict[str, Any]]:
    """Generate schemas from resolved Python annotations regardless of configured mode."""

    _, config = _load_tools_config(tools_config)
    selected, tool_names = _resolve_toolset(config, toolset)
    schema: list[dict[str, Any]] = []
    for name in tool_names:
        definition = config["tools"].get(name)
        if not isinstance(definition, dict):
            raise ValueError(f"toolset references missing tool: {name}")
        copied = dict(definition)
        copied["schema_source"] = "function"
        schema.append(_tool_schema(name, copied, config))
    if outdir:
        output_dir = Path(outdir)
        write_json(schema, output_dir / "b3_auto_schema_from_python.json")
        write_json(
            {"status": "success", "toolset": selected, "tool_count": len(schema), "tools": tool_names},
            output_dir / "b3_auto_schema_report.json",
        )
    return schema


def _schema_accepts(schema: Mapping[str, Any], value: Any) -> bool:
    variants = schema.get("anyOf")
    if isinstance(variants, list):
        return any(isinstance(item, dict) and _schema_accepts(item, value) for item in variants)
    expected_name = schema.get("type")
    if expected_name is None:
        valid = True
    elif isinstance(expected_name, list):
        valid = any(_schema_accepts({"type": item}, value) for item in expected_name)
    elif expected_name not in JSON_TYPE_CHECKS:
        raise ValueError(f"unsupported JSON schema type: {expected_name}")
    else:
        expected = JSON_TYPE_CHECKS[expected_name]
        valid = isinstance(value, expected)
        if expected_name in {"integer", "number"} and isinstance(value, bool):
            valid = False
    if not valid:
        return False
    if "enum" in schema and value not in schema["enum"]:
        return False
    if expected_name == "array" and isinstance(value, list) and isinstance(schema.get("items"), dict):
        return all(_schema_accepts(schema["items"], item) for item in value)
    if expected_name == "object" and isinstance(value, dict):
        additional = schema.get("additionalProperties")
        if isinstance(additional, dict) and not all(_schema_accepts(additional, item) for item in value.values()):
            return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            return False
        if "maximum" in schema and value > schema["maximum"]:
            return False
    return True


def _expected_label(schema: Mapping[str, Any]) -> str:
    if isinstance(schema.get("anyOf"), list):
        return " or ".join(_expected_label(item) for item in schema["anyOf"] if isinstance(item, dict))
    value = schema.get("type")
    return "/".join(value) if isinstance(value, list) else str(value or "valid value")


def _validate_args(args: dict[str, Any], parameter_schema: Mapping[str, Any]) -> None:
    properties = parameter_schema.get("properties", {})
    required = parameter_schema.get("required", [])
    missing = [name for name in required if name not in args]
    if missing:
        raise SkillFault(
            ErrorCode.PARAM_MISSING,
            f"missing required parameters: {', '.join(missing)}",
            details={"missing": missing},
        )
    unknown = sorted(set(args) - set(properties))
    if unknown and parameter_schema.get("additionalProperties") is False:
        raise SkillFault(
            ErrorCode.PARAM_INVALID,
            f"unknown parameters: {', '.join(unknown)}",
            details={"unknown": unknown},
        )
    for name, value in args.items():
        schema = properties.get(name)
        if isinstance(schema, dict) and not _schema_accepts(schema, value):
            raise SkillFault(
                ErrorCode.PARAM_INVALID,
                f"parameter {name} must match {_expected_label(schema)}",
                details={"parameter": name, "expected_schema": schema},
            )


def _result_from_exception(
    name: str,
    args: dict[str, Any],
    exc: Exception,
    context: Any,
) -> dict[str, Any]:
    def fail(**_: Any) -> Any:
        raise exc

    return invoke_callable(name, fail, args, context)


def _limits_config(config_path: Path, config: Mapping[str, Any]) -> Path | None:
    settings = config.get("settings", {})
    value = settings.get("limits_config") if isinstance(settings, dict) else None
    return resolve_from_file(value, config_path) if isinstance(value, str) and value else None


def execute_tool_calls(
    tool_calls: list[dict[str, Any]],
    tools_config: str,
    toolset: str | None = None,
    outdir: str | None = None,
) -> list[dict[str, Any]]:
    config_path, config = _load_tools_config(tools_config)
    selected, allowed_tools = _resolve_toolset(config, toolset)
    if not isinstance(tool_calls, list):
        raise ValueError("tool_calls must be a list")
    data_root_setting = config.get("settings", {}).get("data_root", "../data")
    resolved_data_root = resolve_from_file(data_root_setting, config_path)
    output_dir = Path(outdir).resolve() if outdir else None
    limits_config = _limits_config(config_path, config)
    tool_messages: list[dict[str, Any]] = []
    log_records: list[dict[str, Any]] = []
    for index, raw_call in enumerate(tool_calls):
        try:
            call = normalize_tool_call(raw_call, index)
        except Exception as exc:
            call = {"id": f"call_{index + 1:03d}", "name": "unknown", "args": {}}
            context = make_context(resolved_data_root, output_dir, limits_config, request_id=call["id"])
            result = _result_from_exception(call["name"], call["args"], exc, context)
        else:
            name = call["name"]
            args = call["args"]
            context = make_context(resolved_data_root, output_dir, limits_config, request_id=call["id"])
            definition = config["tools"].get(name)
            if name not in allowed_tools or not isinstance(definition, dict):
                exc = SkillFault(
                    ErrorCode.UNSUPPORTED_OPERATION,
                    f"tool is not available in {selected}: {name}",
                    details={"toolset": selected, "tool": name},
                )
                result = _result_from_exception(name, args, exc, context)
            else:
                try:
                    function = _load_callable(definition)
                    parameters = _parameter_schema(
                        definition,
                        function=function,
                        schema_source=_schema_source(config, definition),
                    )
                    _validate_args(args, parameters)
                except Exception as exc:
                    result = _result_from_exception(name, args, exc, context)
                else:
                    result = invoke_callable(name, function, args, context)
        content = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
        message = make_tool_message(call["id"], call["name"], content, result["status"])
        tool_messages.append(message)
        error = result.get("error") or {}
        log_records.append(
            {
                "timestamp": now_iso(),
                "toolset": selected,
                "tool_call_id": call["id"],
                "name": call["name"],
                "status": result["status"],
                "args": call["args"],
                "skill_result": result,
                "error_code": error.get("code"),
                "retryable": bool(error.get("retryable", False)),
                "latency_ms": result["latency_ms"],
            }
        )
    if output_dir:
        write_json(tool_messages, output_dir / "tool_messages.json")
        for record in log_records:
            append_jsonl(record, output_dir / "tool_call_log.jsonl")
    return tool_messages


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate tool schema or execute tool calls.")
    parser.add_argument("--tools_config", required=True)
    parser.add_argument("--toolset", default=None)
    parser.add_argument("--tool_calls")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--export_schema", action="store_true")
    action.add_argument("--export_auto_schema", action="store_true")
    action.add_argument("--execute", action="store_true")
    parser.add_argument("--outdir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config_path = resolve_cli_path(args.tools_config)
        outdir = resolve_cli_path(args.outdir)
        if args.export_schema or args.export_auto_schema:
            if not args.toolset:
                _, config = _load_tools_config(config_path)
                args.toolset = config.get("default_toolset")
            if args.export_auto_schema:
                get_auto_tools_schema(str(config_path), args.toolset, str(outdir))
                print(outdir / "b3_auto_schema_from_python.json")
            else:
                get_tools_schema(str(config_path), args.toolset, str(outdir))
                print(outdir / "tools_schema.json")
        else:
            if not args.tool_calls:
                raise ValueError("--tool_calls is required with --execute")
            payload = read_json(resolve_cli_path(args.tool_calls))
            tool_calls = payload.get("tool_calls") if isinstance(payload, dict) else payload
            execute_tool_calls(tool_calls, str(config_path), args.toolset, str(outdir))
            print(outdir / "tool_messages.json")
        return 0
    except Exception as exc:
        print(f"fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
