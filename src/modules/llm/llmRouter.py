"""
llm_router.py

Backend-dispatch layer for LLM inference: decides between a local
llama.cpp model and hosted OpenAI-compatible API providers (Groq /
OpenRouter / NVIDIA NIM / Gemini / Cerebras / Mistral / etc.), including
key-rolling, provider fallover, and local-fallback-on-exhaustion.

This module is intentionally decoupled from any article/KG-specific
logic - `llm_base.py` builds on top of `LLMRouter` for that.
"""

from dataclasses import dataclass, field
from datetime import datetime
import gc
import json
import logging
import os
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import requests
from src.modules.llm.llama_server import Llama, LlamaGrammar
from src.utils.logger import setup_logger

# ── logging setup (shared by llm_base.py) ────────────────────────────────────

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = os.environ.get("LOG_DIR", None)
if LOG_DIR is None:
    LOG_DIR = f"./logs/MainLLM_{timestamp}"
os.environ["LOG_DIR"] = LOG_DIR
os.makedirs(LOG_DIR, exist_ok=True)  # Ensure the log directory exists
logger = setup_logger(
    f"{LOG_DIR}/llm_logs.log", log_level=logging.DEBUG
)  # Adjust log level as needed


# ── API-backend support (Groq / OpenRouter / NVIDIA NIM / ...) ───────────────
#
# All providers expose an OpenAI-compatible `/chat/completions` route, so a
# single request path is reused for all of them; only base_url, headers, and
# model names differ per provider.


class _RateLimitError(Exception):
    """Raised when a provider/key returns a true rate-limit response (HTTP 429).
    This is what triggers key-rolling (and provider-rolling once every key on
    a provider is exhausted)."""


class _AuthError(Exception):
    """Raised on HTTP 401/403 - the key itself is invalid/revoked. Treated
    like a dead key: skip it permanently, don't count it against the model."""


class _ModelError(Exception):
    """Raised for failures that look specific to the requested model/request
    (400/404/422 bad request, 5xx, malformed response, transport errors,
    empty content, etc.) rather than an account-wide rate limit. These retry
    a few times, then fall through to the next model instead of rolling
    keys."""


def _keys_from_env(var_name: str) -> List[str]:
    """Read a comma-separated list of API keys from an env var, e.g.
    GROQ_API_KEYS="key1,key2,key3" to enable rolling across multiple keys."""
    raw = os.environ.get(var_name, "")
    return [k.strip() for k in raw.split(",") if k.strip()]


@dataclass
class ProviderConfig:
    name: str
    base_url: str
    api_keys: List[str] = field(default_factory=list)
    models: List[str] = field(default_factory=list)
    extra_headers: dict = field(default_factory=dict)
    timeout: int = 60


def default_provider_configs() -> Dict[str, ProviderConfig]:
    """
    Example provider setup for Groq / OpenRouter / NVIDIA NIM / etc.

    API keys are read from comma-separated env vars so multiple keys can be
    rolled through on rate limit:
        GROQ_API_KEYS="key1,key2"
        OPENROUTER_API_KEYS="key1,key2"
        NVIDIA_NIM_API_KEYS="key1,key2"

    NOTE: the model lists below are illustrative examples only. Hosted model
    catalogs change frequently - check each provider's current docs before
    relying on a specific name in production.
    """
    return {
        "llm7io": ProviderConfig(
            name="llm7io",
            base_url="https://api.llm7.io/v1/chat/completions",
            # LLM7.io allows anonymous access (api_key="unused", ~30 rpm) and a
            # free registered token (~120 rpm, from https://dash.llm7.io). If
            # LLM7IO_API_KEYS isn't set, default to anonymous so this provider
            # still works out of the box.
            api_keys=_keys_from_env("LLM7IO_API_KEYS") or ["unused"],
            models=["gpt-oss"],
        ),
        "groq": ProviderConfig(
            name="groq",
            base_url="https://api.groq.com/openai/v1/chat/completions",
            api_keys=_keys_from_env("GROQ_API_KEYS"),
            models=[
                "groq/compound",
            ],
        ),
        "openrouter": ProviderConfig(
            name="openrouter",
            base_url="https://openrouter.ai/api/v1/chat/completions",
            api_keys=_keys_from_env("OPENROUTER_API_KEYS"),
            models=[
                "minimax/minimax-m3:free",
                "nvidia/nemotron-3-super-120b-a12b:free",
            ],
            extra_headers={
                "HTTP-Referer": os.environ.get(
                    "OPENROUTER_SITE_URL", "http://localhost"
                ),
                "X-Title": os.environ.get("OPENROUTER_SITE_NAME", "llmBase"),
            },
        ),

        "gemini": ProviderConfig(
            name="gemini",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
            api_keys=_keys_from_env("GEMINI_API_KEYS"),
            models=[
                "gemini-2.5-flash",
            ],
        ),
        "ministral": ProviderConfig(
            name="ministral",
            base_url="https://api.mistral.ai/v1/chat/completions",
            api_keys=_keys_from_env("MINISTRAL_API_KEYS"),
            models=["ministral-8b-2512"],
        ),
        "xkiro": ProviderConfig(
            name="xkiro",
            base_url="https://api.xkiro.com/v1/chat/completions",
            api_keys=_keys_from_env("XKIRO_API_KEYS"),
            models=["mistralai/ministral-14b", "mistralai/ministral-8b"],
        ),




        # ------------ JUNK --------------#
        
        # "cerebras": ProviderConfig(
        #     name="cerebras",
        #     base_url="https://api.cerebras.ai/v1/chat/completions",
        #     api_keys=_keys_from_env("CEREBRAS_API_KEYS"),
        #     models=["gpt-oss-120b"],
        # ),

        # "nvidia_nim": ProviderConfig(
        #     name="nvidia_nim",
        #     base_url="https://integrate.api.nvidia.com/v1/chat/completions",
        #     api_keys=_keys_from_env("NVIDIA_NIM_API_KEYS"),
        #     models=[
        #     ],
        # ),
        # "coze": ProviderConfig(
        #     name="coze",
        #     base_url="https://api.coze.com/v3/chat",
        #     api_keys=_keys_from_env("COZE_API_KEYS"),
        #     models=["coze/compound"],
        # )
    }


# Passed: 9/9
# Avg latency: 5.24s | Avg throughput: 96.1 tok/s
#
# Throughput by combo (fastest first):
# - groq/groq/compound: 5.59s | 1466 tok | 262.2 tok/s
# - llm7io/gpt-oss:20b: 7.17s | 965 tok | 134.6 tok/s
# - openrouter/nvidia/nemotron-3-nano-30b-a3b:free: 8.54s | 929 tok | 108.8 tok/s
# - openrouter/nvidia/nemotron-3-ultra-550b-a55b:free: 3.61s | 281 tok | 77.9 tok/s
# - xkiro/mistralai/ministral-14b: 1.61s | 107 tok | 66.3 tok/s
# - ministral/ministral-8b-2512: 1.44s | 94 tok | 65.2 tok/s
# - xkiro/mistralai/ministral-8b: 1.95s | 119 tok | 61.0 tok/s
# - openrouter/nvidia/nemotron-3-super-120b-a12b:free: 7.89s | 480 tok | 60.8 tok/s
# - gemini/gemini-2.5-flash: 9.34s | 261 tok | 28.0 tok/s


@dataclass
class APIConfig:
    use_api: bool = False
    # Order providers are tried in. First provider is preferred; the pool
    # only falls through to the next one once every key on the current
    # provider is confirmed rate-limited (not on a one-off model error).
    providers: Dict[str, ProviderConfig] = field(
        default_factory=default_provider_configs
    )
    provider_order: List[str] = field(
        default_factory=lambda: [
            x
            for x in default_provider_configs().keys()
            if default_provider_configs()[x].api_keys
        ]
    )
    # Retries for errors that look model/request-specific (not rate limits)
    # before giving up on that model and moving to the next one.
    max_model_retries: int = 2
    # How long (seconds) a 429'd key is skipped before being tried again.
    key_cooldown_sec: int = 60
    backoff_base: float = 1.5
    backoff_max: float = 20.0
    # If True, once every configured API provider/model is exhausted, fall
    # back to the local llama.cpp model (loaded on demand) instead of giving
    # up on the article. Requires general_config.model_path to point at a
    # valid local model file.
    fallback_to_local: bool = True

    @classmethod
    def from_dict(cls, data: dict) -> "APIConfig":
        """Build an APIConfig from a dict, ignoring any unknown keys."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in valid_keys})


class _KeyState:
    __slots__ = ("key", "cooldown_until", "dead")

    def __init__(self, key: str):
        self.key = key
        self.cooldown_until = 0.0
        self.dead = False


class ProviderKeyPool:
    """
    Round-robins across a single provider's API keys.

    - A key that returns 429 is put on cooldown (temporarily skipped).
    - A key that returns 401/403 is marked dead (permanently skipped).
    - `all_exhausted()` is only True once every key is either dead or still
      cooling down - i.e. a genuine provider-wide rate limit - which is the
      signal used to roll over to the next provider in `provider_order`.
    """

    def __init__(self, provider_cfg: ProviderConfig, cooldown_sec: int):
        self.provider_cfg = provider_cfg
        self.cooldown_sec = cooldown_sec
        self._states = [_KeyState(k) for k in provider_cfg.api_keys]
        self._ptr = 0

    def _available(self) -> List[_KeyState]:
        now = time.time()
        return [s for s in self._states if not s.dead and s.cooldown_until <= now]

    def get_key(self) -> Optional[_KeyState]:
        avail = self._available()
        if not avail:
            return None
        state = avail[self._ptr % len(avail)]
        self._ptr += 1
        return state

    def mark_rate_limited(self, state: _KeyState):
        state.cooldown_until = time.time() + self.cooldown_sec

    def mark_dead(self, state: _KeyState):
        state.dead = True

    def all_exhausted(self) -> bool:
        return len(self._states) == 0 or len(self._available()) == 0


class LLMRouter:
    """
    Routes chat-completion requests to either a local llama.cpp model or a
    pool of hosted API providers, and owns the local model's lifecycle
    (load/unload) since that model may be needed both as the primary
    backend (use_api=False) and as an on-demand fallback (use_api=True +
    fallback_to_local=True).
    """

    def __init__(
        self,
        llm_config,
        gen_config,
        general_config,
        api_config: Optional[APIConfig] = None,
        verbose: bool = True,
        owner_name: str = "LLMRouter",
        n_parallel: int = 1,
    ):
        self.llm_config = llm_config
        self.gen_config = gen_config
        self.general_config = general_config
        self.owner_name = owner_name
        self.n_parallel = n_parallel

        # api_config.use_api is the single switch between local llama.cpp
        # inference and hosted-API inference (Groq / OpenRouter / NVIDIA NIM).
        self.api_config = api_config or APIConfig()
        self.use_api = self.api_config.use_api
        self._key_pools: Dict[str, ProviderKeyPool] = {}

        self.verbose = verbose or self.general_config.verbose
        # Resolved even in API mode: needed if fallback_to_local kicks in.
        self.model_path = Path(self.general_config.model_path)

        self._llm = None

        if self.use_api:
            logger.info(
                f"[{self.owner_name}] API mode enabled. "
                f"Provider order: {self.api_config.provider_order}. "
                f"Fallback to local model: {self.api_config.fallback_to_local}"
            )
            for pname in self.api_config.provider_order:
                pcfg = self.api_config.providers.get(pname)
                if pcfg is None:
                    logger.warning(
                        f"[{self.owner_name}] Provider '{pname}' not configured - skipping."
                    )
                elif not pcfg.api_keys:
                    logger.warning(
                        f"[{self.owner_name}] Provider '{pname}' has no API keys set - skipping."
                    )
                else:
                    logger.info(
                        f"[{self.owner_name}] Provider '{pname}': {len(pcfg.api_keys)} key(s), models={pcfg.models}"
                    )
        else:
            logger.info(
                f"[{self.owner_name}] Initialized with model: {self.model_path.name}"
            )
        logger.info(
            f"[{self.owner_name}] LLM Config: {json.dumps(self.llm_config.to_dict(), indent=2)}"
        )
        logger.info(
            f"[{self.owner_name}] Generation Config: {json.dumps(self.gen_config.to_dict(), indent=2)}"
        )

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def load(self) -> "LLMRouter":
        """Load the model into VRAM. No-op in API mode - there is no local
        model to load up front (it may still be lazy-loaded later if
        fallback_to_local is enabled and every API provider fails)."""
        if self.use_api:
            return self
        return self._load_local_model()

    def _load_local_model(self) -> "LLMRouter":
        """Actually load the local llama.cpp model into VRAM. Used both as
        the normal load path (use_api=False) and as the on-demand fallback
        path (use_api=True + fallback_to_local=True, once every API
        provider/model has been exhausted)."""
        if self._llm is not None:
            return self

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"[{self.owner_name}] Model not found: {self.model_path}"
            )

        if self.verbose:
            logger.info(f"[{self.owner_name}] Loading model: {self.model_path.name}")
        t0 = time.time()

        self._llm = Llama(
            model_path=str(self.model_path),
            n_parallel=self.n_parallel,
            **self.llm_config.to_dict(),
        )

        if self.verbose:
            logger.info(f"[{self.owner_name}] Ready in {time.time() - t0:.1f}s")

        return self

    def unload(self):
        if self._llm is not None:
            if hasattr(self._llm, "close"):
                self._llm.close()  # explicit synchronous teardown
            else:
                self._llm._sampler = None  # drop refs that keep ctx alive on older versions
                self._llm._model = None
                self._llm._ctx = None
            del self._llm
            self._llm = None
        gc.collect()
        gc.collect()  # second pass catches cyclic refs GC misses on first sweep
        if self.verbose:
            logger.info(f"[{self.owner_name}] Model unloaded.")

    def close(self):
        self.unload()

    def __enter__(self) -> "LLMRouter":
        return self.load()

    def __exit__(self, *args):
        self.unload()

    def __del__(self):
        try:
            self.unload()
        except Exception:
            pass

    # ── Routing entry point ──────────────────────────────────────────────────

    def chat(
        self,
        messages: list,
        schema: dict | None,
        gen_overrides: dict | None = None,
    ) -> tuple[str, dict, str, str]:
        """
        Dispatch a chat-completion request to either the API pool or the
        local model, depending on `self.use_api`. On API exhaustion, falls
        back to the local model if `api_config.fallback_to_local` is set.

        Returns (raw_text, usage_dict, used_provider, used_model). Raises
        the underlying exception if every backend (including local
        fallback) fails.
        """
        if self.use_api:
            try:
                raw, usage, provider, model = self._call_api_chat(
                    messages, schema, gen_overrides # type: ignore
                )
                return raw, usage, provider, model
            except Exception as e:
                if self.api_config.fallback_to_local:
                    logger.warning(
                        f"[{self.owner_name}] All API providers/models exhausted "
                        f"({e}); falling back to local model."
                    )
                    self._load_local_model()
                    raw, usage = self._call_local_chat(messages, schema, gen_overrides)
                    return raw, usage, "local-fallback", self.model_path.name
                raise
        else:
            if self._llm is None:
                self.load()
            raw, usage = self._call_local_chat(messages, schema, gen_overrides)
            return raw, usage, "local", self.model_path.name

    # ── generation backends ──────────────────────────────────────────────────

    def _call_local_chat(
        self, messages: list, schema: dict | None, gen_overrides: dict | None = None
    ) -> tuple[str, dict]:
        """Run chat completion on the locally loaded llama.cpp model.
        Returns (raw_text, usage_dict)."""
        kwargs = self.gen_config.to_dict()
        if gen_overrides:
            kwargs.update(gen_overrides)
        grammar_arg = LlamaGrammar.from_json_schema(json.dumps(schema)) if schema else None
        response = self._llm.create_chat_completion(
            messages=messages,
            grammar=grammar_arg,
            **kwargs,
        )
        msg = response["choices"][0]["message"]
        raw = (msg.get("content") or msg.get("reasoning_content") or "").strip()
        usage = response.get("usage", {})
        return raw, usage

    def _gen_params_for_api(self, gen_overrides: dict | None = None) -> dict:
        """Map local GENConfig fields onto the OpenAI-compatible request
        schema shared by Groq / OpenRouter / NVIDIA NIM, dropping anything
        that's local-inference-only (grammar, mirostat, repeat_penalty, ...)."""
        src = self.gen_config.to_dict()
        if gen_overrides:
            src = {**src, **gen_overrides}
        allowed = {
            "temperature",
            "top_p",
            "max_tokens",
            "stop",
            "presence_penalty",
            "frequency_penalty",
            "seed",
            "n",
        }
        return {
            k: v for k, v in src.items() if k in allowed and v not in (None, [], "")
        }

    def _api_request(
        self,
        provider_cfg: ProviderConfig,
        api_key: str,
        model_name: str,
        messages: list,
        schema: dict,
        gen_overrides: dict | None = None,
    ) -> tuple[str, dict]:
        """Single HTTP call to one provider/model/key. Raises _RateLimitError,
        _AuthError, or _ModelError on failure so the caller can decide
        whether to roll the key, drop the key, or retry the model."""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            **provider_cfg.extra_headers,
        }
        payload = {
            "model": model_name,
            "messages": messages,
            **self._gen_params_for_api(gen_overrides),
        }
        if schema:
            # Best-effort JSON mode. Exact support (json_object vs a strict
            # json_schema) varies by provider/model - json_object is the
            # broadest-compatible option across Groq/OpenRouter/NIM.
            payload["response_format"] = {"type": "json_object"}

        try:
            resp = requests.post(
                provider_cfg.base_url,
                headers=headers,
                json=payload,
                timeout=provider_cfg.timeout,
            )
        except requests.exceptions.RequestException as e:
            raise _ModelError(f"transport error calling {provider_cfg.name}: {e}")

        if resp.status_code == 429:
            raise _RateLimitError(resp.text[:300])
        if resp.status_code in (401, 403):
            raise _AuthError(resp.text[:300])
        if resp.status_code >= 500:
            raise _ModelError(f"server error {resp.status_code}: {resp.text[:300]}")
        if resp.status_code >= 400:
            raise _ModelError(f"request error {resp.status_code}: {resp.text[:300]}")

        try:
            data = resp.json()
            if "error" in data:
                raise _ModelError(
                    f"{provider_cfg.name}/{model_name} returned error body: {data['error']}"
                )
            raw = data["choices"][0]["message"]["content"].strip()
        except (ValueError, KeyError, IndexError, AttributeError) as e:
            raise _ModelError(
                f"unexpected response shape from {provider_cfg.name}/{model_name}: {e}"
            )

        if not raw:
            raise _ModelError(f"{provider_cfg.name}/{model_name} returned empty content")

        usage = data.get("usage", {})
        return raw, usage

    def _call_api_chat(
        self, messages: list, schema: dict, gen_overrides: dict | None = None
    ) -> tuple[str, dict, str, str]:
        """
        Try providers in `provider_order`; within a provider, try each
        configured model; within a model, roll across available API keys.

        - HTTP 429 (_RateLimitError) -> rotate to the next key, same model.
          Only once *every* key on the provider is cooling down/dead
          (pool.all_exhausted()) do we move to the next provider - i.e. we
          only roll providers on a true rate limit, not a one-off failure.
        - HTTP 401/403 (_AuthError) -> that key is dead, try next key.
        - Anything else (_ModelError, bad response, transport error) is
          treated as model-specific: retry with backoff up to
          `max_model_retries`, then give up on that model (not the whole
          provider/key pool) and move to the next model.

        Returns (raw_text, usage_dict, provider_name, model_name).
        """
        api_cfg = self.api_config
        last_err: Optional[Exception] = None

        for provider_name in api_cfg.provider_order:
            provider_cfg = api_cfg.providers.get(provider_name)
            if provider_cfg is None or not provider_cfg.api_keys:
                if self.verbose:
                    logger.info(
                        f"[API] Skipping provider '{provider_name}' - not configured / no keys."
                    )
                continue

            pool = self._key_pools.setdefault(
                provider_name, ProviderKeyPool(provider_cfg, api_cfg.key_cooldown_sec)
            )

            for model_name in provider_cfg.models:
                model_fail_count = 0

                while True:
                    if pool.all_exhausted():
                        logger.warning(
                            f"[API] '{provider_name}': every key is rate-limited or dead "
                            f"(true provider-wide limit) - moving to next provider."
                        )
                        break

                    if model_fail_count > api_cfg.max_model_retries:
                        logger.warning(
                            f"[API] '{provider_name}/{model_name}': exceeded model retry budget "
                            f"({api_cfg.max_model_retries}) - trying next model."
                        )
                        break

                    key_state = pool.get_key()
                    if key_state is None:
                        break  # raced with all_exhausted(); re-check on outer loop

                    try:
                        raw, usage = self._api_request(
                            provider_cfg,
                            key_state.key,
                            model_name,
                            messages,
                            schema,
                            gen_overrides,
                        )
                        return raw, usage, provider_name, model_name

                    except _RateLimitError as e:
                        logger.warning(
                            f"[API] '{provider_name}/{model_name}' key ...{key_state.key[-4:]} "
                            f"hit rate limit: {e}"
                        )
                        pool.mark_rate_limited(key_state)
                        continue  # same model, next available key

                    except _AuthError as e:
                        logger.error(
                            f"[API] '{provider_name}' key ...{key_state.key[-4:]} "
                            f"invalid/unauthorized, dropping it: {e}"
                        )
                        pool.mark_dead(key_state)
                        continue  # same model, next available key

                    except _ModelError as e:
                        model_fail_count += 1
                        last_err = e
                        logger.warning(
                            f"[API] '{provider_name}/{model_name}' model-level failure "
                            f"({model_fail_count}/{api_cfg.max_model_retries}): {e}"
                        )
                        time.sleep(
                            min(api_cfg.backoff_base**model_fail_count, api_cfg.backoff_max)
                        )
                        continue  # retry same model/key a few times, then give up on model

                    except Exception as e:
                        model_fail_count += 1
                        last_err = e
                        logger.error(
                            f"[API] Unexpected error on '{provider_name}/{model_name}': {e}"
                        )
                        time.sleep(
                            min(api_cfg.backoff_base**model_fail_count, api_cfg.backoff_max)
                        )
                        continue

        raise RuntimeError(
            f"All configured API providers/models exhausted. Last error: {last_err}"
        )
        