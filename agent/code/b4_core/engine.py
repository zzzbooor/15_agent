from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from .bindings import NativeToolsBinding, PlainChatBinding, PromptJsonBinding
from .model_pool import MODEL_POOL, release_model_cache
from .native_parsers import ParsedOutput, parse_native_output, parse_prompt_json, validate_tool_calls
from .profiles import ProfileChoice, choose_profile, load_decision_config, load_prompt


def _stop_after_tool_call(tokenizer: Any, prompt_tokens: int) -> Any:
    """Build an opt-in stopping rule for one-tool-per-round live demos."""

    from transformers import StoppingCriteria, StoppingCriteriaList

    class CompleteToolCallStoppingCriteria(StoppingCriteria):
        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
            generated = input_ids[0, prompt_tokens:]
            text = tokenizer.decode(generated, skip_special_tokens=True)
            return "</tool_call>" in text

    return StoppingCriteriaList([CompleteToolCallStoppingCriteria()])


@dataclass
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class RawGeneration:
    raw_text: str
    prompt_text: str
    usage: TokenUsage
    profile: str
    route_reason: str
    binding: str
    model_family: str
    native_parser: str | None
    cache_hit: bool
    load_latency_ms: float
    inference_latency_ms: float

    def metadata(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("raw_text", None)
        payload.pop("prompt_text", None)
        return payload


@dataclass
class DecisionResult:
    parsed: ParsedOutput | None
    raw: RawGeneration
    tool_call_validation: dict
    error: dict[str, str] | None = None


class DecisionEngine:
    def __init__(self, model_config: str | Path):
        self.config_path, self.config = load_decision_config(model_config)

    def prompt(self, name: str, default: str = "") -> str:
        return load_prompt(self.config_path, self.config, name, default)

    def _select(
        self,
        messages: list[dict],
        tools_schema: list[dict],
        profile: str | None,
        strategy: str,
    ) -> ProfileChoice:
        return choose_profile(self.config, messages, tools_schema, profile, strategy)

    def generate_raw(
        self,
        messages: list[dict],
        tools_schema: list[dict],
        *,
        binding: str,
        profile: str | None = None,
        strategy: str = "react",
        instruction: str | None = None,
    ) -> RawGeneration:
        choice = self._select(messages, tools_schema, profile, strategy)
        bundle = MODEL_POOL.acquire(choice.settings, self.config_path)
        tokenizer, model = bundle.tokenizer, bundle.model

        if binding == "prompt_json":
            renderer = PromptJsonBinding()
            prompt_instruction = instruction if instruction is not None else self.prompt("prompt_json")
        elif binding == "native_tools":
            renderer = NativeToolsBinding()
            prompt_instruction = instruction if instruction is not None else self.prompt("native_tools")
        elif binding == "plain":
            renderer = PlainChatBinding()
            prompt_instruction = instruction or ""
        else:
            raise ValueError(f"unsupported B4 binding: {binding}")

        rendered = renderer.render(tokenizer, messages, tools_schema, prompt_instruction)
        device = next(model.parameters()).device
        inputs = rendered.inputs.to(device) if hasattr(rendered.inputs, "to") else {
            key: value.to(device) for key, value in rendered.inputs.items()
        }
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        context = self.config.get("context") or {}
        max_input_tokens = int(choice.settings.get("max_input_tokens", context.get("max_input_tokens", 4096)))
        if prompt_tokens > max_input_tokens:
            raise ValueError(f"rendered prompt has {prompt_tokens} tokens, limit is {max_input_tokens}")

        generation = choice.settings.get("generation") or {}
        options: dict[str, Any] = {
            "max_new_tokens": int(generation.get("max_new_tokens", 1024)),
            "do_sample": bool(generation.get("do_sample", False)),
        }
        if options["do_sample"]:
            if generation.get("temperature") is not None:
                options["temperature"] = float(generation["temperature"])
            if generation.get("top_p") is not None:
                options["top_p"] = float(generation["top_p"])
        if binding == "native_tools" and bool(generation.get("stop_after_tool_call", False)):
            options["stopping_criteria"] = _stop_after_tool_call(tokenizer, prompt_tokens)

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("real B4 inference requires torch") from exc
        started = perf_counter()
        with torch.no_grad():
            generated = model.generate(**inputs, **options)
        inference_latency_ms = round((perf_counter() - started) * 1000, 3)
        new_tokens = generated[0][prompt_tokens:]
        raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        completion_tokens = int(new_tokens.numel())
        usage = TokenUsage(prompt_tokens, completion_tokens, prompt_tokens + completion_tokens)
        return RawGeneration(
            raw_text=raw_text,
            prompt_text=rendered.prompt_text,
            usage=usage,
            profile=choice.name,
            route_reason=choice.reason,
            binding=binding,
            model_family=str(choice.settings.get("family", "unknown")),
            native_parser=choice.settings.get("native_parser"),
            cache_hit=bundle.cache_hit,
            load_latency_ms=bundle.load_latency_ms,
            inference_latency_ms=inference_latency_ms,
        )

    def generate_ai_message(
        self,
        messages: list[dict],
        tools_schema: list[dict],
        *,
        binding: str,
        profile: str | None = None,
        strategy: str = "react",
        call_prefix: str = "generation",
        instruction: str | None = None,
    ) -> DecisionResult:
        raw = self.generate_raw(
            messages,
            tools_schema,
            binding=binding,
            profile=profile,
            strategy=strategy,
            instruction=instruction,
        )
        try:
            if binding == "prompt_json":
                parsed = parse_prompt_json(raw.raw_text, call_prefix)
            elif binding == "native_tools":
                if not raw.native_parser:
                    raise ValueError(f"profile {raw.profile} has no native_parser")
                parsed = parse_native_output(raw.raw_text, raw.native_parser, call_prefix)
            else:
                raise ValueError("AIMessage generation requires prompt_json or native_tools binding")
        except Exception as exc:
            return DecisionResult(
                None,
                raw,
                {"valid": False, "errors": [{"message": "model output could not be parsed"}]},
                {"type": type(exc).__name__, "message": str(exc)},
            )
        validation = validate_tool_calls(parsed.ai_message.get("tool_calls", []), tools_schema)
        return DecisionResult(parsed, raw, validation, None)


__all__ = [
    "DecisionEngine",
    "DecisionResult",
    "RawGeneration",
    "TokenUsage",
    "release_model_cache",
]
