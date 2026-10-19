from dataclasses import dataclass
from datetime import datetime
import logging
import os
from src.modules.llm.llama_server import Llama
import gc
import time
import traceback
from pathlib import Path
from typing import List, Optional

from src.modules.llm.types import GeneralLLMConfig
from src.utils.logger import setup_logger


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = os.environ.get("LOG_DIR", None)
if LOG_DIR is None:
    LOG_DIR = f"./logs/EmbedLLM_{timestamp}"
    os.environ["LOG_DIR"] = LOG_DIR
os.makedirs(LOG_DIR, exist_ok=True)  # Ensure the log directory exists
logger = setup_logger(
    f"{LOG_DIR}/llm_logs.log", log_level=logging.DEBUG
)  # Adjust log level as needed


@dataclass
class EmbedConfig:
    """Configuration for the embedding model."""
    n_ctx: int = 1024
    n_threads: int = 8
    n_batch: int = 512
    verbose: bool = False
    flash_attn: bool = True  # Enforce native Flash Attention
    seed: int = 42  # default seed llama_cpp.LLAMA_DEFAULT_SEED, -1 for random
    offload_kqv: bool = True
    logits_all: bool = False
    n_ubatch: int = 512  # physical batch size, pairs with n_batch=1024
    use_mlock: bool = True  # prevents model from being swapped out of RAM


class LLMEmbed:
    """
    Standalone embedding engine using llama.cpp.
    Mirrors the lifecycle and logging style of LLM_Base.
    """

    def __init__(
        self,
        model_path: str = "E:/Godseer/models/Qwen3-Embedding-0.6B-Q8_0.gguf",
        config: Optional[EmbedConfig] = None,
        general_config: Optional["GeneralLLMConfig"] = None,  # type: ignore
    ):
        self.model_path = Path(model_path)
        self.config = config or EmbedConfig()
        self.general_config = general_config
        
        self._model: Optional[Llama] = None
        self.verbose = self.config.verbose
        
    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def load(self) -> "LLMEmbed":
        """Load the embedding model into memory. Called automatically by context manager."""
        if self._model is not None:
            return self
            
        if not self.model_path.exists():
            raise FileNotFoundError(f"[LLMEmbed] Model not found: {self.model_path}")
            
        if self.verbose:
            logger.info(f"[LLMEmbed] Loading model: {self.model_path.name}")
            t0 = time.time()
            
        # CRITICAL: embedding=True enables the embedding endpoint in llama.cpp
        self._model = Llama(
            model_path=str(self.model_path),
            embedding=True,
            n_ctx=self.config.n_ctx,
            n_threads=self.config.n_threads,
            n_batch=self.config.n_batch,
            verbose=self.verbose,
        )
        
        if self.verbose:
            logger.info(f"[LLMEmbed] Ready in {time.time() - t0:.1f}s")
            
        return self

    def unload(self):
        """Unload the model and free VRAM/RAM."""
        if self._model is not None:
            if hasattr(self._model, "close"):
                self._model.close()          # explicit synchronous teardown
            else:
                self._model._sampler = None  # drop refs that keep ctx alive on older versions
                self._model._model = None
                self._model._ctx = None
            del self._model
            self._model = None
            gc.collect()
            gc.collect()                     # second pass catches cyclic refs GC misses
            if self.verbose:
                logger.info("[LLMEmbed] Model unloaded.")

    def close(self):
        self.unload()

    def __enter__(self) -> "LLMEmbed":
        return self.load()

    def __exit__(self, *args):
        self.unload()

    def __del__(self):
        try:
            self.unload()
        except Exception:
            pass

    # ── Main Methods ───────────────────────────────────────────────────────────

    def encode(self, text: str) -> List[float]:
        """Encode a single string into an embedding vector."""
        if self._model is None:
            self.load()
            
        t0 = time.time()
        result = self._model.create_embedding(text) # type: ignore
        elapsed = time.time() - t0
        
        if self.verbose:
            usage = result.get("usage", {}) # type: ignore
            logger.info(
                f"[LLMEmbed] Encoded 1 text in {elapsed:.3f}s "
                f"(prompt tokens: {usage.get('prompt_tokens', 0)})"
            )
            
        # result["data"] is a list of dicts: [{"object": "embedding", "index": 0, "embedding": [...]}]
        return result["data"][0]["embedding"] # type: ignore

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Encode a list of strings into embedding vectors.
        Guarantees output order matches input order.
        """
        if self._model is None:
            self.load()
            
        if not texts:
            return []
            
        t0 = time.time()
        # llama_cpp.create_embedding accepts both str and List[str]
        result = self._model.create_embedding(texts) # type: ignore
        elapsed = time.time() - t0
        
        if self.verbose:
            usage = result.get("usage", {}) # type: ignore
            logger.info(
                f"[LLMEmbed] Encoded {len(texts)} texts in {elapsed:.3f}s "
                f"(prompt tokens: {usage.get('prompt_tokens', 0)})"
            )
            
        # Sort by index to guarantee order matches input (llama_cpp may return out of order)
        embeddings = [None] * len(texts)
        for item in result["data"]:
            embeddings[item["index"]] = item["embedding"] # type: ignore
            
        return embeddings # type: ignore