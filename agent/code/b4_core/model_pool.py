from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from .profiles import resolve_profile_path


@dataclass
class ModelBundle:
    tokenizer: Any
    model: Any
    cache_hit: bool
    load_latency_ms: float


class SingleModelPool:
    """A one-entry model pool that prevents profile comparisons from filling VRAM."""

    def __init__(self) -> None:
        self._key: tuple[str, ...] | None = None
        self._tokenizer: Any = None
        self._model: Any = None

    @staticmethod
    def _key_for(profile: dict[str, Any], config_path: Path) -> tuple[str, ...]:
        model_setting = profile.get("model_name_or_path")
        tokenizer_setting = profile.get("tokenizer_name_or_path", model_setting)
        if not isinstance(model_setting, str) or not isinstance(tokenizer_setting, str):
            raise ValueError("profile requires model_name_or_path and tokenizer_name_or_path")
        model_path = resolve_profile_path(model_setting, config_path)
        tokenizer_path = resolve_profile_path(tokenizer_setting, config_path)
        return (
            str(model_path),
            str(tokenizer_path),
            str(bool(profile.get("local_files_only", True))),
            str(bool(profile.get("trust_remote_code", False))),
            str(profile.get("torch_dtype", "auto")),
            json.dumps(profile.get("device_map"), sort_keys=True),
            json.dumps(profile.get("max_memory"), sort_keys=True),
            str(profile.get("device", "cuda")),
        )

    def acquire(self, profile: dict[str, Any], config_path: Path) -> ModelBundle:
        key = self._key_for(profile, config_path)
        if self._key == key and self._tokenizer is not None and self._model is not None:
            return ModelBundle(self._tokenizer, self._model, True, 0.0)

        self.release()
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("real B4 inference requires torch and transformers") from exc

        model_path = Path(key[0])
        tokenizer_path = Path(key[1])
        if not model_path.exists() or not tokenizer_path.exists():
            raise FileNotFoundError(f"local model or tokenizer path does not exist: {model_path}")

        dtype_name = str(profile.get("torch_dtype", "auto"))
        dtype_map = {
            "auto": "auto",
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        if dtype_name not in dtype_map:
            raise ValueError(f"unsupported torch_dtype: {dtype_name}")

        started = perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path),
            local_files_only=bool(profile.get("local_files_only", True)),
            trust_remote_code=bool(profile.get("trust_remote_code", False)),
        )
        load_options: dict[str, Any] = {
            "local_files_only": bool(profile.get("local_files_only", True)),
            "trust_remote_code": bool(profile.get("trust_remote_code", False)),
            "dtype": dtype_map[dtype_name],
        }
        device_map = profile.get("device_map")
        if device_map is not None:
            # This optional route requires accelerate. The shipped profiles do
            # not use it because the target Python 3.10 environment has no
            # accelerate installation.
            load_options["device_map"] = device_map
        if device_map is not None and profile.get("max_memory") is not None:
            load_options["max_memory"] = profile["max_memory"]
        model = AutoModelForCausalLM.from_pretrained(str(model_path), **load_options)
        if device_map is None:
            requested_device = str(profile.get("device", "cuda"))
            if requested_device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError(f"profile requests {requested_device}, but CUDA is unavailable")
            model = model.to(requested_device)
        model.eval()

        self._key = key
        self._tokenizer = tokenizer
        self._model = model
        return ModelBundle(tokenizer, model, False, round((perf_counter() - started) * 1000, 3))

    def release(self) -> None:
        model = self._model
        tokenizer = self._tokenizer
        self._model = None
        self._tokenizer = None
        self._key = None
        if model is not None or tokenizer is not None:
            del model, tokenizer
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass


MODEL_POOL = SingleModelPool()


def release_model_cache() -> None:
    MODEL_POOL.release()
