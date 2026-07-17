from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common.io_utils import read_text, read_yaml
from common.path_utils import resolve_from_file


@dataclass(frozen=True)
class ProfileChoice:
    name: str
    settings: dict[str, Any]
    reason: str


def load_decision_config(model_config: str | Path) -> tuple[Path, dict[str, Any]]:
    path = Path(model_config).resolve()
    payload = read_yaml(path)
    if not isinstance(payload, dict):
        raise ValueError("model configuration must be a YAML object")
    return path, payload


def normalized_profiles(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    global_generation = config.get("generation") or {}
    if not isinstance(global_generation, dict):
        raise ValueError("generation configuration must be an object")

    declared = config.get("profiles")
    if declared is None:
        legacy = config.get("model")
        if not isinstance(legacy, dict):
            raise ValueError("model configuration must define profiles or model")
        declared = {"default": legacy}
    if not isinstance(declared, dict) or not declared:
        raise ValueError("profiles must be a non-empty object")

    profiles: dict[str, dict[str, Any]] = {}
    for name, raw in declared.items():
        if not isinstance(name, str) or not name or not isinstance(raw, dict):
            raise ValueError("each model profile must have a non-empty name and object value")
        profile = deepcopy(raw)
        merged_generation = dict(global_generation)
        local_generation = profile.get("generation") or {}
        if not isinstance(local_generation, dict):
            raise ValueError(f"profile {name} generation must be an object")
        merged_generation.update(local_generation)
        profile["generation"] = merged_generation
        profiles[name] = profile
    return profiles


def choose_profile(
    config: dict[str, Any],
    messages: list[dict],
    tools_schema: list[dict],
    requested: str | None = None,
    strategy: str = "react",
) -> ProfileChoice:
    profiles = normalized_profiles(config)
    runtime = config.get("runtime") or {}
    routing = config.get("routing") or {}
    if not isinstance(runtime, dict) or not isinstance(routing, dict):
        raise ValueError("runtime and routing must be objects")

    if requested is not None:
        if requested not in profiles:
            raise ValueError(f"unknown model profile: {requested}")
        return ProfileChoice(requested, profiles[requested], "explicit profile override")

    if strategy == "plan_execute":
        selected = routing.get("planner_profile") or runtime.get("default_profile")
        reason = "plan_execute requires the configured planner profile"
    else:
        serialized_chars = sum(len(str(item.get("content", ""))) for item in messages)
        max_chars = int(routing.get("fast_max_input_chars", 1800))
        max_tools = int(routing.get("fast_max_tools", 8))
        fast_profile = routing.get("fast_profile")
        if fast_profile and serialized_chars <= max_chars and len(tools_schema) <= max_tools:
            selected = fast_profile
            reason = (
                f"short react request ({serialized_chars} chars, {len(tools_schema)} tools) "
                "fits the fast profile thresholds"
            )
        else:
            selected = runtime.get("default_profile") or next(iter(profiles))
            reason = "request exceeds fast-profile thresholds; using the default profile"

    if not isinstance(selected, str) or selected not in profiles:
        raise ValueError(f"routing selected an unavailable model profile: {selected}")
    return ProfileChoice(selected, profiles[selected], reason)


def resolve_profile_path(value: str, config_path: Path) -> Path:
    return resolve_from_file(value, config_path)


def load_prompt(config_path: Path, config: dict[str, Any], name: str, default: str = "") -> str:
    prompts = config.get("prompts") or {}
    if not isinstance(prompts, dict):
        raise ValueError("prompts configuration must be an object")
    setting = prompts.get(name)
    if setting is None:
        return default
    if not isinstance(setting, str):
        raise ValueError(f"prompt path for {name} must be a string")
    return read_text(resolve_from_file(setting, config_path)).strip()
