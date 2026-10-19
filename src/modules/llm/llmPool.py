import asyncio
import gc
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional


class LLMPool:
    """
    High-throughput async execution pool.

    With the native llama-server backend, concurrency is handled natively via
    parallel slots (-np) sharing a single set of model weights in VRAM. This replaces
    the legacy pattern of instantiating N separate model copies in VRAM while maintaining
    full backward-compatibility for callers (load, call, unload).
    """

    def __init__(
        self, llm_cls, n_workers: int = 2, verbose: bool = False, **llm_kwargs
    ):
        self.llm_cls = llm_cls
        self.n_workers = n_workers
        self.verbose = verbose
        self.llm_kwargs = llm_kwargs

        self._instance = None
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._executor: Optional[ThreadPoolExecutor] = None

    def load(self) -> "LLMPool":
        if self._instance is not None:
            return self

        if self.verbose:
            print(
                f"[LLMPool] Initializing {self.llm_cls.__name__} with {self.n_workers} concurrent slots"
            )
            t0 = time.time()

        # Configure n_parallel = self.n_workers on underlying model
        kwargs = dict(self.llm_kwargs)
        kwargs["n_parallel"] = self.n_workers

        # Instantiate single model instance with n_workers slots
        self._instance = self.llm_cls(**kwargs).load()
        self._semaphore = asyncio.Semaphore(self.n_workers)
        self._executor = ThreadPoolExecutor(max_workers=self.n_workers)

        if self.verbose:
            print(
                f"[LLMPool] Ready in {time.time() - t0:.1f}s ({self.n_workers} concurrent slots)"
            )

        return self

    def unload(self):
        if self._instance is not None:
            if hasattr(self._instance, "unload"):
                self._instance.unload()
            elif hasattr(self._instance, "close"):
                self._instance.close()
            self._instance = None
        self._semaphore = None
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        gc.collect()
        if self.verbose:
            print("[LLMPool] Instance unloaded and VRAM freed.")

    def close(self):
        self.unload()

    def __enter__(self) -> "LLMPool":
        return self.load()

    def __exit__(self, *args):
        self.unload()

    def __del__(self):
        try:
            self.unload()
        except Exception:
            pass

    async def call(self, method_name: str, *args, **kwargs):
        """
        Dispatches method calls concurrently across parallel slots, bounded by n_workers.
        """
        if self._instance is None:
            raise RuntimeError("LLMPool.load() must be called before use")

        loop = asyncio.get_running_loop()
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.n_workers)
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self.n_workers)

        async with self._semaphore:
            method = getattr(self._instance, method_name)
            return await loop.run_in_executor(
                self._executor,
                lambda: method(*args, **kwargs),
            )