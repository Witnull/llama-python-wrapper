"""
Main entry point for llama-python-wrapper.
Quick demonstration and environment verification.
"""

import sys
from pathlib import Path
from src import (
    Llama,
    LLM_Core,
    LLMConfig,
    GENConfig,
    get_gpu_vram_info,
)


def main():
    print("=" * 60)
    print("  llama-python-wrapper: High-Performance LLM Pipeline")
    print("=" * 60)

    # Check GPU VRAM
    total_vram, free_vram = get_gpu_vram_info()
    print(f"[Hardware] GPU VRAM: {free_vram:.0f} MB free / {total_vram:.0f} MB total")

    # Look for candidate model in common directories
    candidate_paths = [
        Path("E:/Godseer/models/Qwen3.5-2B-Q4_K_M.gguf"),
        Path("E:/Godseer/models/Ministral-3-3B-Instruct-2512-Q5_K_M.gguf"),
        Path("E:/Godseer/models/Kev-4B-Q4_K_M.gguf"),
    ]
    model_path = None
    for p in candidate_paths:
        if p.exists():
            model_path = p
            break

    if not model_path:
        print("[Info] No default GGUF model detected. Pass a model path to run inference.")
        print("Usage: python -m src.modules.llm.test_server --model <path-to-gguf>")
        return

    print(f"[Model] Using model: {model_path.name}")
    print("[Pipeline] Initializing single-session inference...")

    try:
        with Llama(model_path=str(model_path), n_ctx=2048, verbose=False) as llm:
            prompt = "What is the capital of France? Answer in one sentence."
            print(f"\nUser: {prompt}")
            response = llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=64,
            )
            msg = response["choices"][0]["message"]
            content = msg.get("content") or ""
            reasoning = msg.get("reasoning_content") or ""
            if reasoning:
                print(f"Assistant [Reasoning]: {reasoning.strip()[:100]}...")
            if content:
                print(f"Assistant: {content.strip()}")
            elif not reasoning:
                print(f"Assistant: (empty response)")
            print("\n[Pipeline] Inference completed successfully.")
    except Exception as e:
        print(f"[Error] Inference test failed: {e}")


if __name__ == "__main__":
    main()
