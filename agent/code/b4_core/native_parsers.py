from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from common.schemas import make_ai_message, normalize_tool_call


class ModelOutputError(ValueError):
    pass


@dataclass
class ParsedOutput:
    ai_message: dict
    candidate: dict
    reasoning_text: str = ""


_LEADING_TOOL_BLOCK = re.compile(r"\s*<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_THINK_BLOCK = re.compile(r"^\s*<think>\s*(.*?)\s*</think>\s*", re.DOTALL)
_QWEN35_FUNCTION = re.compile(r"^\s*<function=([^>\s]+)>\s*(.*?)\s*</function>\s*$", re.DOTALL)
_QWEN35_PARAMETER = re.compile(r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>", re.DOTALL)


def _strip_full_code_fence(text: str) -> str:
    stripped = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else stripped


def _call_id(prefix: str, index: int) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", prefix).strip("_") or "generation"
    return f"call_{safe}_{index + 1:03d}"


def _canonical_calls(
    raw_calls: list[Any],
    prefix: str,
    *,
    allow_inline_arguments: bool = False,
) -> list[dict]:
    calls = []
    for index, raw in enumerate(raw_calls):
        if (
            allow_inline_arguments
            and isinstance(raw, dict)
            and isinstance(raw.get("name"), str)
            and "args" not in raw
            and "arguments" not in raw
            and "function" not in raw
        ):
            raw = {
                "id": raw.get("id"),
                "name": raw["name"],
                "args": {
                    key: value
                    for key, value in raw.items()
                    if key not in {"id", "name", "type"}
                },
            }
        normalized = normalize_tool_call(raw, index)
        normalized["id"] = _call_id(prefix, index)
        calls.append(normalized)
    return calls


def parse_prompt_json(raw_text: str, call_prefix: str) -> ParsedOutput:
    try:
        candidate = json.loads(_strip_full_code_fence(raw_text))
    except json.JSONDecodeError as exc:
        raise ModelOutputError(f"prompt_json output is not valid JSON: {exc}") from exc
    if not isinstance(candidate, dict):
        raise ModelOutputError("prompt_json output must be an object")
    unknown = sorted(set(candidate) - {"content", "tool_calls"})
    if unknown:
        raise ModelOutputError(f"prompt_json output has unknown keys: {', '.join(unknown)}")
    content = candidate.get("content", "")
    raw_calls = candidate.get("tool_calls", [])
    if not isinstance(content, str) or not isinstance(raw_calls, list):
        raise ModelOutputError("content must be a string and tool_calls must be an array")
    calls = _canonical_calls(raw_calls, call_prefix, allow_inline_arguments=True)
    if bool(content.strip()) == bool(calls):
        raise ModelOutputError("output must contain final content or tool calls, but not both")
    message = make_ai_message(content, calls)
    return ParsedOutput(message, {"content": content, "tool_calls": calls})


def _parse_scalar(text: str) -> Any:
    value = text.strip()
    python_literals = {"True": True, "False": False, "None": None}
    if value in python_literals:
        return python_literals[value]
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _parse_qwen35_block(block: str) -> dict:
    function_match = _QWEN35_FUNCTION.fullmatch(block)
    if not function_match:
        raise ModelOutputError("invalid Qwen3.5 native function block")
    name = function_match.group(1)
    body = function_match.group(2)
    args: dict[str, Any] = {}
    spans: list[tuple[int, int]] = []
    for parameter in _QWEN35_PARAMETER.finditer(body):
        key = parameter.group(1)
        if key in args:
            raise ModelOutputError(f"duplicate native parameter: {key}")
        args[key] = _parse_scalar(parameter.group(2))
        spans.append(parameter.span())
    remainder = body
    for start, end in reversed(spans):
        remainder = remainder[:start] + remainder[end:]
    if remainder.strip():
        raise ModelOutputError("unexpected text inside Qwen3.5 function block")
    return {"name": name, "args": args}


def _parse_qwen3_block(block: str) -> dict:
    try:
        candidate = json.loads(block.strip())
    except json.JSONDecodeError as exc:
        raise ModelOutputError(f"invalid Qwen3 native tool JSON: {exc}") from exc
    if not isinstance(candidate, dict):
        raise ModelOutputError("Qwen3 native tool call must be an object")
    name = candidate.get("name")
    args = candidate.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError as exc:
            raise ModelOutputError("Qwen3 arguments string is not JSON") from exc
    return {"name": name, "args": args}


def _leading_tool_blocks(text: str) -> tuple[list[str], str]:
    """Return the complete tool-call prefix and any model rollout suffix.

    Some local chat templates do not emit an end-of-turn token immediately
    after ``</tool_call>``.  Greedy generation can then simulate later
    user/tool turns.  Only the consecutive complete calls at the beginning of
    the assistant turn belong to the current AIMessage; the simulated suffix
    must never be treated as a real ToolMessage or final answer.
    """

    blocks: list[str] = []
    offset = 0
    while match := _LEADING_TOOL_BLOCK.match(text, offset):
        blocks.append(match.group(1))
        offset = match.end()
    return blocks, text[offset:].strip()


def parse_native_output(raw_text: str, dialect: str, call_prefix: str) -> ParsedOutput:
    text = raw_text.strip()
    thinking = ""
    think_match = _THINK_BLOCK.match(text)
    if think_match:
        thinking = think_match.group(1).strip()
        text = text[think_match.end() :].strip()

    blocks, rollout_suffix = _leading_tool_blocks(text)
    if text.lstrip().startswith("<tool_call>") and not blocks:
        raise ModelOutputError("native output contains an unclosed tool_call block")
    if blocks:
        if dialect == "qwen35_xml":
            raw_calls = [_parse_qwen35_block(block) for block in blocks]
        elif dialect == "qwen3_json":
            raw_calls = [_parse_qwen3_block(block) for block in blocks]
        else:
            raise ModelOutputError(f"unsupported native tool dialect: {dialect}")
        calls = _canonical_calls(raw_calls, call_prefix)
        message = make_ai_message("", calls)
        candidate = {"content": "", "tool_calls": calls}
        if rollout_suffix:
            candidate["ignored_rollout_suffix_chars"] = len(rollout_suffix)
        return ParsedOutput(message, candidate, thinking)

    if "<tool_call>" in text:
        raise ModelOutputError("native tool output contains text before a tool_call block")

    if not text:
        raise ModelOutputError("native model returned neither content nor tool calls")
    message = make_ai_message(text, [])
    return ParsedOutput(message, {"content": text, "tool_calls": []}, thinking)


_JSON_TYPES: dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "object": dict,
    "array": list,
}


def validate_tool_calls(tool_calls: list[dict], tools_schema: list[dict]) -> dict:
    definitions = {
        item.get("function", {}).get("name"): item.get("function", {})
        for item in tools_schema
        if isinstance(item, dict) and isinstance(item.get("function"), dict)
    }
    errors: list[dict[str, Any]] = []
    for index, call in enumerate(tool_calls):
        name = call.get("name")
        args = call.get("args")
        definition = definitions.get(name)
        if definition is None:
            errors.append({"index": index, "name": name, "message": "tool is not in the supplied schema"})
            continue
        if not isinstance(args, dict):
            errors.append({"index": index, "name": name, "message": "arguments must be an object"})
            continue
        parameters = definition.get("parameters") or {}
        properties = parameters.get("properties") or {}
        required = parameters.get("required") or []
        missing = [field for field in required if field not in args]
        unknown = sorted(set(args) - set(properties)) if parameters.get("additionalProperties") is False else []
        if missing:
            errors.append({"index": index, "name": name, "message": f"missing required: {', '.join(missing)}"})
        if unknown:
            errors.append({"index": index, "name": name, "message": f"unknown arguments: {', '.join(unknown)}"})
        for field, value in args.items():
            expected_name = (properties.get(field) or {}).get("type")
            expected = _JSON_TYPES.get(expected_name)
            if expected is None:
                continue
            valid = isinstance(value, expected)
            if expected_name in {"integer", "number"} and isinstance(value, bool):
                valid = False
            if not valid:
                errors.append(
                    {"index": index, "name": name, "message": f"argument {field} must be {expected_name}"}
                )
    return {"valid": not errors, "errors": errors}
