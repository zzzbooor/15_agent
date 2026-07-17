from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Iterable


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def _atomic_write_text(path: str | Path, text: str) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, target)
    return target


def write_text(text: str, path: str | Path) -> Path:
    return _atomic_write_text(path, text)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(obj: Any, path: str | Path) -> Path:
    text = json.dumps(obj, ensure_ascii=False, indent=2) + "\n"
    return _atomic_write_text(path, text)


def read_yaml(path: str | Path) -> Any:
    try:
        import yaml
    except ImportError as exc:
        return _read_simple_yaml(path)
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _strip_yaml_comment(line: str) -> str:
    in_single = False
    in_double = False
    for index, char in enumerate(line):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            return line[:index].rstrip()
    return line.rstrip()


def _split_inline_items(text: str) -> list[str]:
    items = []
    current = []
    depth = 0
    in_single = False
    in_double = False
    for char in text:
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif not in_single and not in_double:
            if char in "[{":
                depth += 1
            elif char in "]}":
                depth -= 1
            elif char == "," and depth == 0:
                items.append("".join(current).strip())
                current = []
                continue
        current.append(char)
    if current:
        items.append("".join(current).strip())
    return items


def _parse_yaml_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return ""
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None", "~"}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        body = value[1:-1].strip()
        return [] if not body else [_parse_yaml_scalar(item) for item in _split_inline_items(body)]
    if value.startswith("{") and value.endswith("}"):
        body = value[1:-1].strip()
        result = {}
        if not body:
            return result
        for item in _split_inline_items(body):
            key, raw = item.split(":", 1)
            result[key.strip().strip('"').strip("'")] = _parse_yaml_scalar(raw)
        return result
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if re.fullmatch(r"-?\d+\.\d+", value):
        return float(value)
    return value


def _read_simple_yaml(path: str | Path) -> Any:
    """Small fallback YAML reader for this project when PyYAML is unavailable."""
    raw_lines = Path(path).read_text(encoding="utf-8").splitlines()
    lines = []
    for raw in raw_lines:
        stripped = _strip_yaml_comment(raw)
        if stripped.strip():
            indent = len(stripped) - len(stripped.lstrip(" "))
            lines.append((indent, stripped.strip()))

    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]
    pending: list[tuple[int, str, dict[str, Any]]] = []

    for indent, text in lines:
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        while pending and indent <= pending[-1][0]:
            pending.pop()
        if pending and isinstance(parent, dict) and text.startswith("- "):
            pending_indent, pending_key, pending_parent = pending[-1]
            if pending_parent.get(pending_key) == {}:
                pending_parent[pending_key] = []
                parent = pending_parent[pending_key]
                stack.append((pending_indent, parent))

        if text.startswith("- "):
            if not isinstance(parent, list):
                raise RuntimeError(f"Unsupported YAML list location in {path}: {text}")
            parent.append(_parse_yaml_scalar(text[2:].strip()))
            continue

        if ":" not in text:
            raise RuntimeError(f"Unsupported YAML line in {path}: {text}")
        key, raw_value = text.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if raw_value:
            value = _parse_yaml_scalar(raw_value)
            if isinstance(parent, dict):
                parent[key] = value
            else:
                raise RuntimeError(f"Unsupported YAML mapping location in {path}: {text}")
        else:
            value = {}
            if isinstance(parent, dict):
                parent[key] = value
            else:
                raise RuntimeError(f"Unsupported YAML mapping location in {path}: {text}")
            pending.append((indent, key, parent))
            stack.append((indent, value))

    return root


def write_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> Path:
    text = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    return _atomic_write_text(path, text)


def append_jsonl(record: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return target
