"""
system1Base.py

Core System-1 decision-processing LLM layer built on top of native llama-server
`/v1/systemone` endpoint. Mirrors `LLM_Core` and `LLM_Base` architecture:

- Configuration via `LLMConfig`, `GeneralLLMConfig`, `GENConfig`
- Prompt construction and article application hooks (`_build_prompt` / `_apply_to_article`)
- Single and batch article processing
- Token usage accounting and latency logging matching `LLM_Core`
"""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
import threading
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union

from src.modules.llm.types import Article, GENConfig, GeneralLLMConfig, LLMConfig
from src.modules.llm.llama_server import (
    DEFAULT_SERVER_EXE,
    LlamaServerInstance,
    find_free_port,
)
from src.modules.llm.llmCore import logger

DEFAULT_SYSTEM1_MODEL = r"E:\Godseer\models\Kev-4B-Q4_K_M.gguf"


class System1_Base:
    """
    Base class for fast System-1 Decision Model evaluators and filters.
    Follows the configuration, lifecycle, and processing interface of `LLM_Base`.
    """

    def __init__(
        self,
        llm_config: LLMConfig = LLMConfig(),
        gen_config: GENConfig = GENConfig(),
        general_config: GeneralLLMConfig = GeneralLLMConfig(),
        schema: dict = {},
        verbose: bool = True,
        n_parallel: int = 1,
    ):
        self._infer_lock = threading.Lock()

        self.llm_config = llm_config
        self.gen_config = gen_config
        self.general_config = general_config

        # Resolve model path from general_config or fallback to DEFAULT_SYSTEM1_MODEL
        model_str = general_config.model_path if general_config.model_path else DEFAULT_SYSTEM1_MODEL
        self.model_path = Path(model_str)
        self.schema = schema
        self.verbose = verbose
        self.n_parallel = n_parallel

        # Managed background llama-server instance for native /v1/systemone
        n_ctx = llm_config.n_ctx if llm_config.n_ctx and llm_config.n_ctx > 0 else 2048
        n_gpu_layers = getattr(llm_config, "n_gpu_layers", 99)
        n_batch = max(2048, getattr(llm_config, "n_batch", 2048))
        n_ubatch = max(2048, getattr(llm_config, "n_ubatch", 2048))

        self._server = LlamaServerInstance(
            model_path=self.model_path,
            exe_path=DEFAULT_SERVER_EXE,
            port=find_free_port(),
            n_ctx=n_ctx,
            n_parallel=self.n_parallel,
            n_gpu_layers=n_gpu_layers if n_gpu_layers >= 0 else 99,
            n_batch=n_batch,
            n_ubatch=n_ubatch,
            verbose=self.verbose,
        )
        self._llm = self._server

        self._token_usage = {
            "t_prompt_tokens": 0,
            "t_completion_tokens": 0,
            "t_total_tokens": 0,
            "m_prompt_tokens": 0,
            "m_completion_tokens": 0,
            "m_total_tokens": 0,
            "entries": 0,
        }

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def load(self) -> "System1_Base":
        """Load the model into VRAM. Called automatically by the context manager."""
        if not self._server.is_running():
            self._server.start()
        return self

    def unload(self):
        """Unload the model and reclaim GPU VRAM."""
        self.analyze_token_usage()
        if hasattr(self, "_server") and self._server is not None:
            self._server.stop()

    def close(self):
        """Explicit teardown matching llama_cpp."""
        self.unload()

    def __enter__(self) -> "System1_Base":
        return self.load()

    def __exit__(self, *_):
        self.unload()

    def __del__(self):
        try:
            self.unload()
        except Exception:
            pass

    # ── Main methods ───────────────────────────────────────────────────────────

    def evaluate(
        self,
        prompt: str,
        schema_override: dict | None = None,
    ) -> Dict[str, Any]:
        """
        Executes a single forward-pass decision evaluation via ``/v1/systemone``.
        Payload: ``{state, questions}`` where each question has ``options`` (label→description).
        Response ``answers[q].selected`` is normalized to ``choice`` so downstream
        helpers (``get_choice``, ``get_probabilities``) work unchanged.
        """
        active_schema = schema_override if schema_override is not None else self.schema
        if self.verbose:
            logger.info(f"###### Prompt: \n\t{prompt}\n")
            logger.info(f"###### Active Schema: \n\t{active_schema}\n")

        if not self._server.is_running():
            self.load()

        # ── Build /v1/systemone questions payload ──────────────────────────────
        # API: each question → {type, instructions, criteria: {label: description}}
        questions: Dict[str, Any] = {}
        for q_name, q_def in active_schema.items():
            criteria = q_def.get("criteria", {})
            # Ensure None descriptions become the label string (server requires non-empty values)
            criteria_clean = {
                label: (desc if desc else label.replace("_", " "))
                for label, desc in criteria.items()
            }
            questions[q_name] = {
                "type": q_def.get("type", "choice"),
                "instructions": q_def.get("instructions", ""),
                "criteria": criteria_clean,
            }

        payload = {
            "state": prompt,
            "questions": questions,
        }

        t0 = time.time()  # begin inference
        try:
            url = f"{self._server.base_url}/v1/systemone"
            resp = self._server.session.post(url, json=payload, timeout=120.0)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error(f"[{self.__class__.__name__}] Generation failed: {e}")
            traceback.print_exc()
            return {}

        elapsed = time.time() - t0  # end inference

        usage = data.get("usage", {})
        tok_in = usage.get("input_tokens", 0)
        tok_out = usage.get("output_tokens", 0)

        if self.verbose:
            logger.info(
                f" [{elapsed:.1f}s | {tok_out} tok | local/{self.model_path.name}] "
            )
            logger.info(f"PROMPT TOK: {tok_in}")
            logger.info(f"TOTAL TOK: {tok_in + tok_out}")

        self._token_usage["t_prompt_tokens"] += tok_in
        self._token_usage["t_completion_tokens"] += tok_out
        self._token_usage["t_total_tokens"] += (tok_in + tok_out)
        self._token_usage["m_prompt_tokens"] = max(
            self._token_usage["m_prompt_tokens"], tok_in
        )
        self._token_usage["m_completion_tokens"] = max(
            self._token_usage["m_completion_tokens"], tok_out
        )
        self._token_usage["m_total_tokens"] = max(
            self._token_usage["m_total_tokens"], tok_in + tok_out
        )
        self._token_usage["entries"] += 1

        # Normalize response: {selected, probabilities} → {choice, probabilities}
        # so get_choice() / get_probabilities() work without changes.
        raw_answers = data.get("answers", {})
        answers: Dict[str, Any] = {}
        for q_name, ans in raw_answers.items():
            answers[q_name] = {
                "choice": ans.get("selected", ans.get("choice", "")),
                "confidence": ans.get("probability", ans.get("confidence", 0.0)),
                "probabilities": ans.get("probabilities", {}),
            }

        if self.verbose:
            logger.info(f" [RAW OUTPUT]\n {answers}\n")
        return answers

    def chat(self, prompt: str, sys_prompt: str = "", **kwargs) -> Dict[str, Any]:
        """Convenience alias for evaluate."""
        return self.evaluate(prompt)

    def analyze_token_usage(self):
        if not self.verbose:
            return
        logger.info("=" * 50)
        logger.info(f"[{self.__class__.__name__}] Token usage analysis:")
        logger.info(f"Total prompt tokens: {self._token_usage['t_prompt_tokens']}")
        logger.info(f"Total completion tokens: {self._token_usage['t_completion_tokens']}")
        logger.info(f"Total tokens: {self._token_usage['t_total_tokens']}")
        logger.info(
            f"Max prompt tokens in a single request: {self._token_usage['m_prompt_tokens']}"
        )
        logger.info(
            f"Max completion tokens in a single request: {self._token_usage['m_completion_tokens']}"
        )
        logger.info(
            f"Max total tokens in a single request: {self._token_usage['m_total_tokens']}"
        )
        logger.info(f"Total entries processed: {self._token_usage['entries']}")
        logger.info("=" * 50)
        logger.info(
            f"Avg. prompt tokens per entry: {self._token_usage['t_prompt_tokens'] / self._token_usage['entries'] if self._token_usage['entries'] > 0 else 0:.2f}"
        )
        logger.info(
            f"Avg. completion tokens per entry: {self._token_usage['t_completion_tokens'] / self._token_usage['entries'] if self._token_usage['entries'] > 0 else 0:.2f}"
        )
        logger.info(
            f"Avg. total tokens per entry: {self._token_usage['t_total_tokens'] / self._token_usage['entries'] if self._token_usage['entries'] > 0 else 0:.2f}"
        )
        logger.info("=" * 50)

    def process_single(self, article: Article, use_kg_rag: bool = False) -> Article:
        """Process a single article through the System-1 decision model."""
        sys_prompt, prompt = self._build_prompt(article)
        try:
            if self.verbose:
                logger.info(f"{article.title[:60] or article.url[:60]!r}")
            answers = self.evaluate(prompt)
            return self._apply_to_article(article, answers)
        except Exception as e:
            logger.info(f" [ERROR] Failed to process article '{article.title}': {e}")
            traceback.print_exc()
            return article

    def process_batch(
        self, articles: list[Article], on_progress=None, use_kg_rag: bool = False
    ) -> list[Article]:
        """Process a batch of articles."""
        if not self._server.is_running():
            self.load()

        total = len(articles)
        if self.verbose:
            logger.info(f"[{self.__class__.__name__}] Processing {total} article(s)...\n")

        t_batch = time.time()

        for i, article in enumerate(articles, 1):
            if self.verbose:
                logger.info(f" ({i}/{total})")
            try:
                articles[i - 1] = self.process_single(
                    article, use_kg_rag=use_kg_rag
                )
            except Exception as e:
                logger.info(f" [ERROR] Failed to process article '{article.title}': {e}")
                traceback.print_exc()
                continue

            if on_progress:
                on_progress(i, total, articles[i - 1])

        elapsed = time.time() - t_batch
        if self.verbose:
            avg = elapsed / total if total else 0
            logger.info(
                f"\n[{self.__class__.__name__}] Done. {total} articles in "
                f"{elapsed:.1f}s (avg {avg:.1f}s/article)"
            )
        return articles

    # ── helpers ──────────────────────────────────────────────────────────────

    def _truncate_to_budget(self, text: str) -> str:
        """Truncate text so prompt fits context window."""
        if not text:
            return ""
        budget_chars = getattr(self.general_config, "max_input_text_chars", 120_000) or 120_000
        if len(text) > budget_chars:
            logger.debug(
                f"[System1] Truncating content to {budget_chars} chars (got {len(text)} chars) to fit budget"
            )
            return text[:budget_chars] + "\n...[truncated]"
        return text

    def _build_prompt(self, article: Article) -> tuple[str, str]:
        """Build prompt for an article or document. Returns (system_prompt, user_state_prompt)."""
        sys_prompt = "You are a decision evaluator."
        user_prompt = article.original_text or article.title or ""
        return sys_prompt, user_prompt

    @staticmethod
    def _apply_to_article(article: Article, parsed: dict) -> Article:
        """Apply parsed decision output to an Article object."""
        article.data.update(parsed)
        return article

    @staticmethod
    def get_choice(answers: Dict[str, Any], question_name: str, default: str = "") -> str:
        """Extracts selected choice for a question."""
        ans = answers.get(question_name, {})
        return ans.get("choice", default)

    @staticmethod
    def get_confidence(answers: Dict[str, Any], question_name: str, default: float = 0.0) -> float:
        """Extracts confidence score (0.0 to 1.0) for a question."""
        ans = answers.get(question_name, {})
        return float(ans.get("confidence", default))

    @staticmethod
    def get_probabilities(answers: Dict[str, Any], question_name: str) -> Dict[str, float]:
        """Extracts probability distribution for a question."""
        ans = answers.get(question_name, {})
        return ans.get("probabilities", {})

    @staticmethod
    def calc_weighted_score(
        probabilities: Dict[str, float],
        weights: Dict[str, float],
        default: float = 0.0,
    ) -> float:
        """Calculates a continuous score by weighting choice probabilities."""
        if not probabilities:
            return default
        return sum(probabilities.get(k, 0.0) * w for k, w in weights.items())
