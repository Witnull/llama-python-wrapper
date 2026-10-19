r"""
Test script for local llama-server with targeted model.

Supports both:
1. Decision Models (like Cloudflare Clef) via native `/v1/systemone`
2. Generative Chat Models (like Mistral, Llama, Qwen) via `/v1/chat/completions`

Targets DEFAULT_CLEF_MODEL by default with full server logging.
Accepts an optional input message as an argument (defaults to "hi").

Usage:
    # Test Clef with native decision endpoint (/v1/systemone)
    python test_server.py
    python test_server.py "Apple announced record quarterly profits of 90 billion dollars."

    # Test generative chat model (/v1/chat/completions)
    python test_server.py "Hello, who are you?" --model "E:\Godseer\models\Ministral-3-3B-Instruct-2512-Q5_K_M.gguf"

    # Force specific endpoint
    python test_server.py --endpoint systemone
    python test_server.py --endpoint chat
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Fix Windows console UnicodeEncodeError for emojis/special characters
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modules.llm.llama_server import (
    DEFAULT_SERVER_EXE,
    LlamaServerInstance,
    find_free_port,
)

DEFAULT_CLEF_MODEL = r"E:\Godseer\models\Kev-4B-Q4_K_M.gguf"
#DEFAULT_FALLBACK_MODEL = r"E:\Godseer\models\Ministral-3-3B-Instruct-2512-Q5_K_M.gguf"


def run_systemone_decision(server: LlamaServerInstance, user_input: str) -> int:
    """Tests the native /v1/systemone endpoint for Decision Models like Clef."""
    payload = {
        "state": user_input,
        "questions": {
            "sentiment": {
                "type": "choice",
                "instructions": "What is the sentiment of the statement?",
                "criteria": {
                    "positive": None,
                    "neutral": None,
                    "negative": None,
                },
            },
            "category": {
                "type": "choice",
                "instructions": "What is the primary topic or domain?",
                "criteria": {
                    "business_and_finance": None,
                    "technology": None,
                    "politics_and_policy": None,
                    "general_or_greeting": None,
                },
            },
        },
    }

    print("\n" + "=" * 80)
    print(" [SYSTEM ONE DECISION REQUEST] (Native Endpoint for Clef / Decision Models)")
    print(f" Endpoint: {server.base_url}/v1/systemone")
    print(f" Payload : {json.dumps(payload, indent=2, ensure_ascii=False)}")
    print("=" * 80 + "\n")

    t_start = time.perf_counter()
    resp = server.session.post(
        f"{server.base_url}/v1/systemone",
        json=payload,
        timeout=60.0,
    )
    t_latency = time.perf_counter() - t_start

    print("-" * 35 + " RESPONSE STATUS: " + str(resp.status_code) + " " + "-" * 35)
    print(f" Latency: {t_latency:.3f}s")
    print(f" Raw Response:\n{resp.text}\n")

    if resp.status_code == 200:
        data = resp.json()
        answers = data.get("answers", {})
        print("=" * 80)
        print(" [DECISION MODEL EVALUATIONS (Single Forward Pass)]")
        for q_name, q_val in answers.items():
            choice = q_val.get("choice")
            conf = q_val.get("confidence", 0.0)
            probs = q_val.get("probabilities", {})
            prob_str = ", ".join(f"{k}: {v*100:.1f}%" for k, v in probs.items())
            print(f" • Question : {q_name}")
            print(f"   Selected : {choice} (confidence: {conf*100:.1f}%)")
            print(f"   Breakdown: {prob_str}")
        print("=" * 80)
        usage = data.get("usage", {})
        print(f" Tokens: input={usage.get('input_tokens')}, output={usage.get('output_tokens')}")
        return 0
    else:
        print(f"[ERROR] /v1/systemone returned error code {resp.status_code}: {resp.text}")
        return 1


def run_chat_completion(
    server: LlamaServerInstance,
    user_prompt: str,
    system_prompt: str = "You are a helpful assistant.",
) -> int:
    """Tests the /v1/chat/completions endpoint for Generative Chat Models."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    payload = {
        "messages": messages,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 512,
        "stream": False,
    }

    print("\n" + "=" * 80)
    print(" [CHAT COMPLETION REQUEST] (Generative Chat Endpoint)")
    print(f" Endpoint: {server.base_url}/v1/chat/completions")
    print(f" Payload : {json.dumps(payload, indent=2, ensure_ascii=False)}")
    print("=" * 80 + "\n")

    t_start = time.perf_counter()
    resp = server.session.post(
        f"{server.base_url}/v1/chat/completions",
        json=payload,
        timeout=120.0,
    )
    t_latency = time.perf_counter() - t_start

    print("-" * 35 + " RESPONSE STATUS: " + str(resp.status_code) + " " + "-" * 35)
    print(f" Latency: {t_latency:.2f}s")
    print(f" Raw Response:\n{resp.text}\n")

    if resp.status_code == 200:
        data = resp.json()
        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        usage = data.get("usage", {})
        print("=" * 80)
        print(" [ASSISTANT RESPONSE]")
        print(content)
        print("=" * 80)
        print(
            f" Tokens: prompt={usage.get('prompt_tokens')}, "
            f"completion={usage.get('completion_tokens')}, total={usage.get('total_tokens')}"
        )
        return 0
    else:
        if resp.status_code == 500 and "logits computation" in resp.text:
            print("[DIAGNOSTIC NOTICE]")
            print(
                " Notice: This model is a Decision Model (Clef/Classifier) and does NOT generate text.\n"
                " llama.cpp requires using the `/v1/systemone` endpoint for this model architecture.\n"
                " Testing /v1/systemone now automatically...\n"
            )
            return run_systemone_decision(server, user_prompt)
        return 1


def test_server(
    model_path: str,
    user_prompt: str = "hi",
    system_prompt: str = "You are a helpful assistant.",
    endpoint: str = "auto",
    n_ctx: int = 2048,
    n_gpu_layers: int = 99,
    port: int | None = None,
    timeout_sec: float = 60.0,
):
    print("=" * 80)
    print(" [LOCAL LLAMA-SERVER TEST]")
    print(f" Target Model : {model_path}")
    print(f" Input String : {user_prompt!r}")
    print(f" Endpoint     : {endpoint}")
    print(f" Server Exe   : {DEFAULT_SERVER_EXE}")
    print("=" * 80)

    model_p = Path(model_path)
    if not model_p.exists():
        print(f"\n[ERROR] Model file not found at: {model_p}")
        return 1

    file_size_mb = model_p.stat().st_size / (1024 * 1024)
    print(f"[INFO] Model file size: {file_size_mb:.2f} MB")

    if not DEFAULT_SERVER_EXE.exists():
        print(f"\n[ERROR] llama-server executable not found at: {DEFAULT_SERVER_EXE}")
        return 1

    is_clef = "clef" in model_p.name.lower()
    server_port = port or find_free_port(start_port=8080)
    print(f"[INFO] Selected Port: {server_port}")
    print(f"[INFO] Model Category: {'Decision Model (Clef)' if is_clef else 'Generative Model'}")
    print("[INFO] Launching llama-server with verbose logging enabled...\n")
    print("-" * 35 + " SERVER LOGS START " + "-" * 35)

    server = LlamaServerInstance(
        model_path=model_p,
        exe_path=DEFAULT_SERVER_EXE,
        port=server_port,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        n_parallel=1,
        verbose=True,  # Streams full stdout/stderr from llama-server
    )

    try:
        t_start = time.perf_counter()
        server.start(timeout_sec=timeout_sec)
        t_ready = time.perf_counter()
        print("-" * 35 + " SERVER READY " + "-" * 36)
        print(f"[INFO] Server became ready in {t_ready - t_start:.2f}s on {server.base_url}")

        # Verify /health endpoint
        health_resp = server.session.get(f"{server.base_url}/health", timeout=5)
        print(f"[HEALTH] Status Code: {health_resp.status_code} | Body: {health_resp.text}")

        # Dispatch based on endpoint selection
        if endpoint == "systemone" or (endpoint == "auto" and is_clef):
            return run_systemone_decision(server, user_prompt)
        elif endpoint == "chat" or (endpoint == "auto" and not is_clef):
            return run_chat_completion(server, user_prompt, system_prompt)
        else:
            return run_chat_completion(server, user_prompt, system_prompt)

    except Exception as e:
        print(f"\n[EXCEPTION OCCURRED] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        print("\n" + "-" * 35 + " STOPPING SERVER " + "-" * 35)
        server.stop()
        print("[INFO] Server stopped and VRAM reclaimed.")
        print("-" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Test local llama-server on target model with full log output."
    )
    # Chat / evaluation message as positional argument or option
    parser.add_argument(
        "message",
        nargs="?",
        default=None,
        help="Input text for chat or decision evaluation (default: 'hi')",
    )
    parser.add_argument(
        "-p", "--prompt",
        dest="prompt_opt",
        default=None,
        help="Input text (alternative to positional argument)",
    )
    parser.add_argument(
        "-m", "--model",
        default=DEFAULT_CLEF_MODEL,
        help=f"Path to GGUF model file (default: {DEFAULT_CLEF_MODEL})",
    )
    parser.add_argument(
        "-e", "--endpoint",
        choices=["auto", "chat", "systemone"],
        default="auto",
        help="Endpoint to test: 'auto' (detects model type), 'chat' (/v1/chat/completions), or 'systemone' (/v1/systemone)",
    )
    parser.add_argument(
        "--system",
        default="You are a helpful assistant.",
        help="System prompt message (for chat completions)",
    )
    parser.add_argument(
        "--ctx",
        type=int,
        default=2048,
        help="Context window size (n_ctx, default: 2048)",
    )
    parser.add_argument(
        "--ngl",
        type=int,
        default=99,
        help="GPU layers to offload (-ngl, default: 99)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port for llama-server (default: auto find free port)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Server startup timeout in seconds (default: 60.0)",
    )

    args = parser.parse_args()

    # Determine input: CLI positional or --prompt if provided, else "hi"
    user_prompt = args.prompt_opt or args.message or "hi"

    sys.exit(
        test_server(
            model_path=args.model,
            user_prompt=user_prompt,
            system_prompt=args.system,
            endpoint=args.endpoint,
            n_ctx=args.ctx,
            n_gpu_layers=args.ngl,
            port=args.port,
            timeout_sec=args.timeout,
        )
    )


if __name__ == "__main__":
    main()
