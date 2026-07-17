from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any


class NativeToolBindingUnsupported(RuntimeError):
    pass


@dataclass
class RenderedPrompt:
    inputs: Any
    prompt_text: str
    binding: str


def normalize_chat_messages(messages: list[dict]) -> list[dict]:
    """Adapt the project's flat message contract to tokenizer chat templates.

    B1 can append a system template after prior turns. Qwen templates expect a
    single leading system message, so all system updates are merged in order.
    """

    system_chunks: list[str] = []
    body: list[dict] = []
    for index, original in enumerate(deepcopy(messages)):
        role = original.get("role")
        if role == "system":
            content = str(original.get("content", "")).strip()
            if content:
                label = "Initial system instruction" if not system_chunks else f"System update {len(system_chunks)}"
                system_chunks.append(f"[{label}]\n{content}")
            continue
        if role == "assistant":
            converted = {"role": "assistant", "content": original.get("content", "")}
            calls = original.get("tool_calls") or []
            if calls:
                converted["tool_calls"] = [
                    {
                        "id": call.get("id"),
                        "type": "function",
                        "function": {"name": call.get("name"), "arguments": call.get("args", {})},
                    }
                    for call in calls
                ]
            body.append(converted)
            continue
        if role == "tool":
            body.append(
                {
                    "role": "tool",
                    "name": original.get("name"),
                    "tool_call_id": original.get("tool_call_id"),
                    "content": original.get("content", ""),
                }
            )
            continue
        if role == "user":
            body.append({"role": "user", "content": original.get("content", "")})
            continue
        raise ValueError(f"message {index} has unsupported role: {role}")

    if system_chunks:
        body.insert(0, {"role": "system", "content": "\n\n".join(system_chunks)})
    return body


def _apply_chat_template(tokenizer: Any, messages: list[dict], tools: list[dict] | None = None) -> tuple[Any, str]:
    options: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_tensors": "pt",
        "return_dict": True,
        "enable_thinking": False,
    }
    text_options = dict(options)
    text_options.update({"tokenize": False})
    text_options.pop("return_tensors", None)
    text_options.pop("return_dict", None)
    if tools is not None:
        options["tools"] = tools
        text_options["tools"] = tools
    try:
        inputs = tokenizer.apply_chat_template(messages, **options)
        prompt_text = tokenizer.apply_chat_template(messages, **text_options)
    except TypeError as exc:
        if "enable_thinking" not in str(exc):
            raise
        options.pop("enable_thinking", None)
        text_options.pop("enable_thinking", None)
        inputs = tokenizer.apply_chat_template(messages, **options)
        prompt_text = tokenizer.apply_chat_template(messages, **text_options)
    return inputs, prompt_text


class PromptJsonBinding:
    name = "prompt_json"

    def render(
        self,
        tokenizer: Any,
        messages: list[dict],
        tools_schema: list[dict],
        instruction: str,
    ) -> RenderedPrompt:
        chat = normalize_chat_messages(messages)
        schema_text = json.dumps(tools_schema, ensure_ascii=False, separators=(",", ":"))
        injected = f"{instruction.strip()}\n\nAvailable tools JSON schema:\n{schema_text}".strip()
        if chat and chat[0].get("role") == "system":
            chat[0]["content"] = chat[0].get("content", "") + "\n\n" + injected
        else:
            chat.insert(0, {"role": "system", "content": injected})
        if chat and chat[-1].get("role") == "tool":
            chat.append(
                {
                    "role": "user",
                    "content": (
                        "Use the ToolMessage results above. Return the required JSON envelope now; "
                        "do not repeat a completed tool call unless another call is necessary."
                    ),
                }
            )
        inputs, prompt_text = _apply_chat_template(tokenizer, chat)
        return RenderedPrompt(inputs, prompt_text, self.name)


class NativeToolsBinding:
    name = "native_tools"

    def render(
        self,
        tokenizer: Any,
        messages: list[dict],
        tools_schema: list[dict],
        instruction: str = "",
    ) -> RenderedPrompt:
        chat = normalize_chat_messages(messages)
        if instruction:
            if chat and chat[0].get("role") == "system":
                chat[0]["content"] += "\n\n" + instruction.strip()
            else:
                chat.insert(0, {"role": "system", "content": instruction.strip()})
        tool_argument = tools_schema if tools_schema else None
        inputs, prompt_text = _apply_chat_template(tokenizer, chat, tool_argument)
        if tools_schema:
            missing = [
                item.get("function", {}).get("name")
                for item in tools_schema
                if item.get("function", {}).get("name") not in prompt_text
            ]
            if missing:
                raise NativeToolBindingUnsupported(
                    "tokenizer chat template ignored the tools argument; missing: " + ", ".join(missing)
                )
        return RenderedPrompt(inputs, prompt_text, self.name)


class PlainChatBinding:
    name = "plain"

    def render(
        self,
        tokenizer: Any,
        messages: list[dict],
        tools_schema: list[dict] | None = None,
        instruction: str = "",
    ) -> RenderedPrompt:
        chat = normalize_chat_messages(messages)
        inputs, prompt_text = _apply_chat_template(tokenizer, chat)
        return RenderedPrompt(inputs, prompt_text, self.name)
