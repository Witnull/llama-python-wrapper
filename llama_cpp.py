"""
Drop-in compatibility module for llama_cpp.
Redirects all imports to the native llama-server background subprocess wrapper.
"""

from src.modules.llm.llama_server import (
    GGML_TYPE_F16,
    GGML_TYPE_F32,
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q4_1,
    GGML_TYPE_Q5_0,
    GGML_TYPE_Q5_1,
    GGML_TYPE_Q8_0,
    LLAMA_DEFAULT_SEED,
    Llama,
    LlamaGrammar,
    LlamaServerInstance,
    calculate_optimal_concurrency,
    get_gpu_vram_info,
    cleanup_orphan_servers,
)

__all__ = [
    "Llama",
    "LlamaGrammar",
    "LlamaServerInstance",
    "cleanup_orphan_servers",
    "GGML_TYPE_F32",
    "GGML_TYPE_F16",
    "GGML_TYPE_Q4_0",
    "GGML_TYPE_Q4_1",
    "GGML_TYPE_Q5_0",
    "GGML_TYPE_Q5_1",
    "GGML_TYPE_Q8_0",
    "LLAMA_DEFAULT_SEED",
    "calculate_optimal_concurrency",
    "get_gpu_vram_info",
]

