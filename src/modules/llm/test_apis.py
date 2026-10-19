"""
Standalone diagnostic script for llmBase's API backends.

Hits every configured (provider, model, key) combination with a minimal
prompt and reports which ones actually work vs fail, so broken entries can
be pruned or swapped out of your provider configs.

This deliberately bypasses LLMRouter._call_api_chat()'s retry/roll/fallback
logic - that logic is designed to stop at the first success, which is
exactly wrong for a health check. This script calls
LLMRouter._api_request() directly so every combination gets tested, not
just enough to answer one prompt.

By default EVERY (provider, model, key) combination is tested - not just
the first key per model. Use --first-key-only if you only want a quick
sanity check (one key per model) instead of a full sweep.

Usage:
    python test_providers.py                        # test every provider/model/key
    python test_providers.py --provider groq         # just one provider
    python test_providers.py --first-key-only        # only test the first key per model (fast sanity check)
    python test_providers.py --timeout 15            # override per-request timeout (seconds)
    python test_providers.py --delay 0.5             # pause between requests (avoid self-inflicted 429s)
    python test_providers.py --prompt "2+2=?"         # custom test prompt
    python test_providers.py --json-out report.json  # also dump machine-readable results

Exit code: 0 if every tested combo passed, 1 if anything failed (usable in CI).
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import List, Optional
from dotenv import load_dotenv

from src.modules.llm.llmBase import LLM_Base
from src.modules.llm.llmRouter import (
    APIConfig,
    ProviderConfig,
    _RateLimitError,
    _AuthError,
    _ModelError,
)

from src.modules.llm.types import GENConfig, GeneralLLMConfig

load_dotenv()  # load .env file if present, so API keys are available in os.environ


class _ProbeLLM(LLM_Base):
    """Bare-bones concrete LLM_Base subclass - just enough to reach
    self.router._api_request(); _build_prompt/_apply_to_article are never
    actually used since this script calls _api_request() directly."""

    def __init__(self, api_config: APIConfig):
        super().__init__(
            api_config=api_config,
            gen_config=GENConfig(max_tokens=2048, temperature=0.1),
            general_config=GeneralLLMConfig(n_ctx=2048, verbose=True),
        )

    def _build_prompt(self, article):
        return "You are a test probe.", "ping"

    @staticmethod
    def _apply_to_article(article, parsed):
        return article


@dataclass
class ProbeResult:
    provider: str
    model: str
    key_suffix: str
    ok: bool
    latency_s: float
    tokens_out: Optional[int] = None
    tok_per_s: Optional[float] = None
    error_type: Optional[str] = None  # "rate_limit" | "auth" | "model" | "unexpected" | None
    error: Optional[str] = None


def _mask(key: str) -> str:
    if not key or len(key) <= 4:
        return "****"
    return f"...{key[-4:]}"


def probe_one(
    inst: _ProbeLLM, provider_cfg: ProviderConfig, key: str, model_name: str, prompt: str
) -> ProbeResult:
    api_key = key
    key_suffix = _mask(api_key)
    messages = [{"role": "user", "content": prompt}]
    t0 = time.time()
    try:
        # _api_request lives on LLMRouter (backend-dispatch owner), not on
        # LLM_Base itself - LLM_Base only exposes chat()/process_*.
        raw, usage = inst.router._api_request(
            provider_cfg, api_key, model_name, messages, schema={}
        )
        latency_s = time.time() - t0
        tokens_out = (usage or {}).get("completion_tokens", 0)
        tok_per_s = (tokens_out / latency_s) if latency_s > 0 and tokens_out else 0.0
        return ProbeResult(
            provider_cfg.name,
            model_name,
            key_suffix,
            True,
            latency_s,
            tokens_out=tokens_out,
            tok_per_s=tok_per_s,
        )
    except _RateLimitError as e:
        return ProbeResult(
            provider_cfg.name, model_name, key_suffix, False, time.time() - t0,
            error_type="rate_limit", error=str(e),
        )
    except _AuthError as e:
        return ProbeResult(
            provider_cfg.name, model_name, key_suffix, False, time.time() - t0,
            error_type="auth", error=str(e),
        )
    except _ModelError as e:
        return ProbeResult(
            provider_cfg.name, model_name, key_suffix, False, time.time() - t0,
            error_type="model", error=str(e),
        )
    except Exception as e:
        return ProbeResult(
            provider_cfg.name, model_name, key_suffix, False, time.time() - t0,
            error_type="unexpected", error=str(e),
        )


def run(
    provider_filter: Optional[str] = None,
    prompt: str = "Reply with exactly: OK",
    timeout: int = 20,
    first_key_only: bool = False,
    delay: float = 0.0,
) -> List[ProbeResult]:
    api_cfg = APIConfig(use_api=True)  # pulls in default_provider_configs() + default provider_order
    providers = api_cfg.providers
    order = api_cfg.provider_order

    for p in providers.values():
        provider_name = p.name
        print(f"[INFO] {provider_name}: {len(p.models)} models, {len(p.api_keys)} keys configured")
        p.timeout = timeout

    inst = _ProbeLLM(api_config=api_cfg)

    results: List[ProbeResult] = []
    total_combos = 0
    for provider_name in order:
        if provider_filter and provider_name != provider_filter:
            continue

        provider_cfg = providers.get(provider_name)
        if provider_cfg is None:
            print(f"[SKIP] '{provider_name}' - not configured")
            continue
        if not provider_cfg.api_keys:
            print(f"[SKIP] '{provider_name}' - no API keys set (check its env var)")
            continue
        if not provider_cfg.models:
            print(f"[SKIP] '{provider_name}' - no models configured")
            continue

        # Default: test EVERY key for EVERY model (full sweep). Only trim to
        # the first key when explicitly asked for a quick sanity check.
        keys_to_test = provider_cfg.api_keys[:1] if first_key_only else provider_cfg.api_keys
        total_combos += len(provider_cfg.models) * len(keys_to_test)

        for model_name in provider_cfg.models:
            for key in keys_to_test:
                res = probe_one(inst, provider_cfg, key, model_name, prompt)
                results.append(res)
                if res.ok:
                    print(
                        f"[OK  ] {res.provider:12s} {res.model:48s} key={res.key_suffix:10s} "
                        f"{res.latency_s:.2f}s | {res.tokens_out} tok | {res.tok_per_s:.1f} tok/s"
                    )
                else:
                    print(
                        f"[FAIL] {res.provider:12s} {res.model:48s} key={res.key_suffix:10s} "
                        f"{res.latency_s:.2f}s [{res.error_type}] {res.error}"
                    )
                if delay:
                    time.sleep(delay)

    print(f"\n[INFO] Tested {len(results)}/{total_combos} planned (provider, model, key) combos.")
    return results


def summarize(results: List[ProbeResult]) -> bool:
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)

    if not results:
        print("Nothing was tested - check that at least one provider has keys set.")
        return True

    passed = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]
    print(f"Passed: {len(passed)}/{len(results)}")

    if passed:
        avg_latency = sum(r.latency_s for r in passed) / len(passed)
        avg_tok_s = sum(r.tok_per_s or 0 for r in passed) / len(passed)
        print(f"Avg latency: {avg_latency:.2f}s | Avg throughput: {avg_tok_s:.1f} tok/s")

        print("\nThroughput by combo (fastest first):")
        for r in sorted(passed, key=lambda r: r.tok_per_s or 0, reverse=True):
            print(
                f"  - {r.provider}/{r.model} (key {r.key_suffix}): {r.latency_s:.2f}s | "
                f"{r.tokens_out} tok | {r.tok_per_s:.1f} tok/s"
            )

    if failed:
        print("\nFailing (provider, model, key) combos:")
        for r in failed:
            print(f"  - {r.provider}/{r.model} (key {r.key_suffix})  [{r.error_type}] {r.error}")

        print("\nFailing provider/model pairs (candidates to remove or replace):")
        seen = set()
        for r in failed:
            k = (r.provider, r.model)
            if k in seen:
                continue
            seen.add(k)
            print(f"  - {r.provider}/{r.model}  [{r.error_type}] {r.error}")

    # Ready-to-paste suggestion: each provider's model list trimmed to only
    # the models where AT LEAST ONE tested key passed, in first-seen order.
    good_models = defaultdict(list)
    for r in passed:
        if r.model not in good_models[r.provider]:
            good_models[r.provider].append(r.model)

    if good_models:
        print("\nSuggested trimmed `models=[...]` lists (models with >=1 passing key):")
        for provider, models in good_models.items():
            print(f'  "{provider}": {models}')

    # Models where ALL tested keys failed (fully dead, not just one bad key).
    tested_models = defaultdict(set)
    passed_models = defaultdict(set)
    for r in results:
        tested_models[r.provider].add(r.model)
    for r in passed:
        passed_models[r.provider].add(r.model)

    fully_dead_models = []
    for provider, models in tested_models.items():
        for model in models:
            if model not in passed_models[provider]:
                fully_dead_models.append((provider, model))

    if fully_dead_models:
        print("\nModels with ZERO passing keys (fully dead - safe to drop):")
        for provider, model in fully_dead_models:
            print(f"  - {provider}/{model}")

    tested_providers = {r.provider for r in results}
    dead_providers = tested_providers - set(good_models.keys())
    if dead_providers:
        print("\nProviders with zero passing models (candidates to drop entirely):")
        for p in sorted(dead_providers):
            print(f"  - {p}")

    print("=" * 72)
    return len(failed) == 0


def main():
    parser = argparse.ArgumentParser(description="Probe all configured API providers/models/keys.")
    parser.add_argument("--provider", default=None, help="Only test this provider (e.g. groq)")
    parser.add_argument("--prompt", default="Reply with exactly: OK", help="Test prompt to send")
    parser.add_argument("--timeout", type=int, default=20, help="Per-request timeout in seconds")
    parser.add_argument(
        "--first-key-only",
        action="store_true",
        help="Only test the first key per model instead of every key (quick sanity check).",
    )
    parser.add_argument(
        "--all-keys",
        action="store_true",
        help="Deprecated / no-op: testing every key is now the default. Kept for backward compatibility.",
    )
    parser.add_argument("--delay", type=float, default=0.0, help="Seconds to pause between requests")
    parser.add_argument("--json-out", default=None, help="Also write machine-readable results to this path")
    args = parser.parse_args()

    results = run(
        provider_filter=args.provider,
        prompt=args.prompt,
        timeout=args.timeout,
        first_key_only=args.first_key_only,
        delay=args.delay,
    )
    ok = summarize(results)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        print(f"\nWrote machine-readable results to {args.json_out}")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()