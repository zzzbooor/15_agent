from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .engine import DecisionEngine, RawGeneration
from .native_parsers import validate_tool_calls


class PlanValidationError(ValueError):
    def __init__(self, message: str, attempts: list[dict] | None = None):
        super().__init__(message)
        self.attempts = attempts or []


@dataclass
class ValidatedPlan:
    plan: dict
    attempts: list[dict]


def parse_plan_json(raw_text: str) -> dict:
    text = raw_text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    try:
        candidate = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PlanValidationError(f"planner output is not valid JSON: {exc}") from exc
    if not isinstance(candidate, dict):
        raise PlanValidationError("planner output must be an object")
    return candidate


def validate_plan(plan: dict, tools_schema: list[dict], max_steps: int = 6) -> dict:
    errors: list[dict[str, Any]] = []
    if set(plan) != {"goal", "steps"}:
        errors.append({"path": "$", "message": "plan keys must be exactly goal and steps"})
    if not isinstance(plan.get("goal"), str) or not plan.get("goal", "").strip():
        errors.append({"path": "$.goal", "message": "goal must be a non-empty string"})
    steps = plan.get("steps")
    if not isinstance(steps, list) or not 1 <= len(steps) <= max_steps:
        errors.append({"path": "$.steps", "message": f"steps must contain 1 to {max_steps} items"})
        return {"valid": False, "errors": errors}

    identifiers: list[str] = []
    dependencies: dict[str, list[str]] = {}
    required_step_keys = {"id", "description", "tool_name", "arguments", "depends_on"}
    for index, step in enumerate(steps):
        path = f"$.steps[{index}]"
        if not isinstance(step, dict):
            errors.append({"path": path, "message": "step must be an object"})
            continue
        if set(step) != required_step_keys:
            errors.append({"path": path, "message": "step has missing or unknown keys"})
        identifier = step.get("id")
        if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,39}", identifier):
            errors.append({"path": path + ".id", "message": "id must be a short identifier"})
            continue
        if identifier in identifiers:
            errors.append({"path": path + ".id", "message": "step id must be unique"})
        identifiers.append(identifier)
        if not isinstance(step.get("description"), str) or not step.get("description", "").strip():
            errors.append({"path": path + ".description", "message": "description must be non-empty"})
        depends_on = step.get("depends_on")
        if not isinstance(depends_on, list) or not all(isinstance(item, str) for item in depends_on):
            errors.append({"path": path + ".depends_on", "message": "depends_on must be a string array"})
            depends_on = []
        dependencies[identifier] = depends_on
        call_check = validate_tool_calls(
            [{"name": step.get("tool_name"), "args": step.get("arguments")}],
            tools_schema,
        )
        for item in call_check["errors"]:
            errors.append({"path": path, "message": item["message"]})

    known = set(identifiers)
    for identifier, required in dependencies.items():
        for dependency in required:
            if dependency not in known:
                errors.append({"path": f"$.steps.{identifier}.depends_on", "message": f"unknown step: {dependency}"})
            if dependency == identifier:
                errors.append({"path": f"$.steps.{identifier}.depends_on", "message": "step cannot depend on itself"})

    remaining = {name: set(required) for name, required in dependencies.items()}
    resolved: set[str] = set()
    while remaining:
        ready = sorted(name for name, required in remaining.items() if required <= resolved)
        if not ready:
            errors.append({"path": "$.steps", "message": "step dependencies contain a cycle"})
            break
        resolved.update(ready)
        for name in ready:
            remaining.pop(name)
    return {"valid": not errors, "errors": errors}


def plan_layers(plan: dict) -> list[list[dict]]:
    steps = {step["id"]: step for step in plan["steps"]}
    remaining = dict(steps)
    completed: set[str] = set()
    layers: list[list[dict]] = []
    while remaining:
        ready_ids = sorted(
            identifier
            for identifier, step in remaining.items()
            if set(step["depends_on"]) <= completed
        )
        if not ready_ids:
            raise PlanValidationError("cannot execute cyclic plan")
        layer = [remaining.pop(identifier) for identifier in ready_ids]
        layers.append(layer)
        completed.update(ready_ids)
    return layers


def _attempt_record(raw: RawGeneration, validation: dict | None, parse_error: str | None) -> dict:
    return {
        "source": "local_llm",
        "raw_text": raw.raw_text,
        "metadata": raw.metadata(),
        "validation": validation,
        "parse_error": parse_error,
    }


def generate_validated_plan(
    engine: DecisionEngine,
    task: str,
    tools_schema: list[dict],
    *,
    profile: str | None = None,
    max_steps: int = 6,
) -> ValidatedPlan:
    planner_instruction = engine.prompt("planner")
    system = (
        planner_instruction
        + "\n\nAvailable tools schema:\n"
        + json.dumps(tools_schema, ensure_ascii=False, separators=(",", ":"))
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": task}]
    attempts: list[dict] = []
    for attempt_index in range(2):
        raw = engine.generate_raw(
            messages,
            [],
            binding="plain",
            profile=profile,
            strategy="plan_execute",
        )
        try:
            plan = parse_plan_json(raw.raw_text)
        except PlanValidationError as exc:
            validation = None
            parse_error = str(exc)
        else:
            validation = validate_plan(plan, tools_schema, max_steps)
            parse_error = None
            if validation["valid"]:
                attempts.append(_attempt_record(raw, validation, None))
                return ValidatedPlan(plan, attempts)
        attempts.append(_attempt_record(raw, validation, parse_error))
        if attempt_index == 0:
            detail = parse_error or json.dumps(validation["errors"], ensure_ascii=False)
            messages.extend(
                [
                    {"role": "assistant", "content": raw.raw_text},
                    {
                        "role": "user",
                        "content": (
                            "The plan failed validation. Correct it and return the complete JSON object only. "
                            f"Validation errors: {detail}"
                        ),
                    },
                ]
            )
    raise PlanValidationError("planner failed validation after one correction attempt", attempts)
