"""Shared types for the LLM pipeline."""

from dataclasses import dataclass, field, asdict
from math import ceil
from typing import Any, Dict, List, Optional
import json

# GGML cache type constants matching llama.cpp
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q5_0 = 6
GGML_TYPE_Q5_1 = 7
GGML_TYPE_Q8_0 = 8


# ---------------------------------------------------------------------------
# Document / Article Data Classes
# ---------------------------------------------------------------------------

@dataclass
class Article:
    """General text and metadata container for pipeline processing."""
    url: str = ""
    title: str = ""
    original_text: str = ""
    summary: str = ""
    keywords: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)
    lang: str = "en"

    def to_dict(self) -> dict:
        return asdict(self)

    def json(self, **kwargs) -> str:
        """Serialize this object to a JSON string."""
        return json.dumps(asdict(self), ensure_ascii=False, **kwargs)

    @classmethod
    def from_dict(cls, data: dict) -> "Article":
        """Build an Article from a dict, ignoring any unknown keys."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in valid_keys})


# ---------------------------------------------------------------------------
# LLM Pipeline Configuration Classes
# ---------------------------------------------------------------------------

@dataclass
class LLMConfig:
    n_gpu_layers: int = -1  # full GPU
    n_ctx: int = 10_000
    n_batch: int = 1024
    n_threads: int = 8
    verbose: bool = False
    flash_attn: bool = True  # Enforce native Flash Attention
    seed: int = 42  # default seed LLAMA_DEFAULT_SEED, -1 for random
    offload_kqv: bool = True
    logits_all: bool = False
    type_k: int = GGML_TYPE_Q4_0
    type_v: int = GGML_TYPE_Q4_0
    n_ubatch: int = 512  # physical batch size, pairs with n_batch=1024
    use_mlock: bool = True  # prevents model from being swapped out of RAM

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "LLMConfig":
        """Build an LLMConfig from a dict, ignoring any unknown keys."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in valid_keys})


@dataclass
class GENConfig:
    max_tokens: int = 8000
    temperature: float = 0.2
    top_k: int = 50
    top_p: float = 0.9
    repeat_penalty: float = 1.1
    stop: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "GENConfig":
        """Build a GENConfig from a dict, ignoring any unknown keys."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in valid_keys})


@dataclass
class GeneralLLMConfig:
    model_path: str = "./models/Ministral-3-8B-Instruct-2512-Q5_K_M.gguf"
    n_ctx: int = 0
    max_input_text_chars: int = 20_000
    chars_per_token_ratio: float = 3.5
    special_chars_token_headroom: int = 40
    headroom_tokens: int = 200
    system_prompt_chars: int = 2000
    user_prompt_chars: int = 2000
    verbose: bool = False

    def prompt_token_update(self, system_prompt: str = "", user_prompt: str = ""):
        if system_prompt != "":
            self.system_prompt_chars = len(system_prompt)
        if user_prompt != "":
            self.user_prompt_chars = len(user_prompt)

    def ceil_k(self, value: float, k: int = 5000) -> int:
        """Ceil a value to the nearest multiple of k."""
        return int(ceil(value / k) * k)

    def calc_max_n_ctx(self) -> int:
        """Calculate the maximum context window size for the LLM."""
        return (
            self.n_ctx
            if self.n_ctx != 0
            else self.ceil_k(
                ceil(self.max_input_text_chars / self.chars_per_token_ratio)
                + ceil(
                    (self.system_prompt_chars + self.user_prompt_chars)
                    / self.chars_per_token_ratio
                )
                + self.headroom_tokens
                + self.special_chars_token_headroom
            )
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "GeneralLLMConfig":
        """Build a GeneralLLMConfig from a dict, ignoring any unknown keys."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in valid_keys})
