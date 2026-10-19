# llama-python-wrapper
This is my crude wrapper for ease of using concurrent session of llama-cpp - just for fast testing/implemeting projects. May not be good for production.

A medium-performance Python pipeline and drop-in `python-llama-cpp` replacement powered by a background-managed native `llama-server` engine.

## Key Features

- **90% Leak-Proof VRAM Lifecycle**: - yes it may leak sometimes (WIP) - Replaces in-process C-bindings with a supervised native `llama-server.exe` subprocess. GPU VRAM is completely and immediately reclaimed by the OS kernel upon exit/unload.
- **Dynamic Auto-Concurrency**: Automatically measures free VRAM via `nvidia-smi` and sizes continuous batching slots (`-np`) to safely maximize throughput without OOM errors.
- **Drop-in `python-llama-cpp` Compatibility**: Full API compatibility (`Llama`, `LlamaGrammar`, `create_chat_completion`, `create_completion`, `create_embedding`). Existing projects can swap imports without code modifications.
- **Single & Multi-Session Pipelines**:
  - `Llama`: Direct single-session or multi-turn chat and text completion.
  - `LLMPool`: Asynchronous thread-pool dispatcher running concurrent multi-session inference across parallel slots.
  - `LLM_Core` / `LLMRouter`: Unified routing with fallback, token analytics, and JSON parsing.

## Installation

```bash
git clone https://github.com/Witnull/llama-python-wrapper.git
cd llama-python-wrapper
pip install -e .
```

Dependencies:
- Python >= 3.12
- `requests`, `python-dotenv`, `psutil`, `colorama`
- Pre-compiled `llama-server.exe` binary in `exe/llama/`

## Quick Start

### 1. Single Session Inference (`Llama`)

```python
from llama_cpp import Llama

# Automatically spawns llama-server.exe on a free port
with Llama(model_path="path/to/model.gguf", n_ctx=2048) as llm:
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain quantum computing in one sentence."}
        ],
        max_tokens=64,
        temperature=0.7,
    )
    print(response["choices"][0]["message"]["content"])
```

### 2. Multi-Session Concurrent Inference (`LLMPool`)

```python
import asyncio
from src import LLMPool, LLM_Core, LLMConfig

async def run_sessions():
    # Spawns server with 4 concurrent slots (-np 4) managed by LLMPool
    pool = LLMPool(
        LLM_Core,
        n_workers=4,
        general_config=GeneralLLMConfig(model_path="path/to/model.gguf"),
        llm_config=LLMConfig(n_ctx=2048),
    )
    pool.load()

    prompts = [
        "What is machine learning?",
        "Explain gravitational waves.",
        "Summarize photosynthesis in 2 sentences.",
        "What is the theory of relativity?"
    ]

    # Run 4 sessions concurrently in parallel slots
    tasks = [pool.call("chat", prompt=p) for p in prompts]
    results = await asyncio.gather(*tasks)

    for prompt, reply in zip(prompts, results):
        print(f"Q: {prompt}\nA: {reply}\n")

    pool.unload()

asyncio.run(run_sessions())
```

## Interactive Testing

Run the included Jupyter notebook:
- `test_inference.ipynb`: Demonstrates single-session chat and concurrent multi-session inference benchmarks.

## License

[MIT](LICENSE)
