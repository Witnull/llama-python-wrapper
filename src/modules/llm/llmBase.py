"""
llmBase.py

Base processing layer built on top of LLM_Core. Provides document/article
processing hooks, batch processing, and JSON extraction.
"""

from typing import Any, Callable, Dict, List, Optional
import time
import traceback

from src.modules.llm.types import Article, GENConfig, GeneralLLMConfig, LLMConfig
from src.modules.llm.llmCore import LLM_Core, logger
from src.modules.llm.llmRouter import APIConfig


class LLM_Base(LLM_Core):
    """
    LLM_Core extension providing batch and single document/article processing workflows.
    """

    def __init__(
        self,
        llm_config: LLMConfig = LLMConfig(),
        gen_config: GENConfig = GENConfig(),
        general_config: GeneralLLMConfig = GeneralLLMConfig(),
        use_system_role: bool = True,
        schema: dict = {},
        api_config: Optional[APIConfig] = None,
        verbose: bool = True,
        n_parallel: int = 1,
    ):
        super().__init__(
            llm_config=llm_config,
            gen_config=gen_config,
            general_config=general_config,
            use_system_role=use_system_role,
            schema=schema,
            api_config=api_config,
            verbose=verbose,
            n_parallel=n_parallel,
        )
        self._llm = self.router._llm

    def _build_prompt(self, article: Article) -> tuple[str, str]:
        """Build prompt for an article or document. Returns (system_prompt, user_prompt)."""
        sys_prompt = "You are a helpful AI assistant."
        user_prompt = article.original_text or article.title or ""
        return sys_prompt, user_prompt

    @staticmethod
    def _apply_to_article(article: Article, parsed: dict) -> Article:
        """Apply parsed LLM output to an Article/Document object."""
        article.data.update(parsed)
        return article

    def process_single(self, article: Article) -> Article:
        """Process a single article or document."""
        self._llm = self.router._llm
        sys_prompt, prompt = self._build_prompt(article)
        try:
            if self.verbose:
                title_preview = article.title[:60] if article.title else (article.url[:60] if article.url else "Document")
                logger.info(f"Processing: {title_preview!r}")
            raw_chat_result = self.chat(prompt, sys_prompt)

            try:
                parsed, think = self._parse_llm_json(raw_chat_result)
                if self.verbose and think:
                    logger.info(f" [THINK]\n {think}\n")
                return self._apply_to_article(article, parsed)
            except Exception as e:
                if self.verbose:
                    logger.info(f" [WARN] Parse failed: {e}")
                return article

        except Exception:
            traceback.print_exc()
            return article

    def process_batch(
        self,
        articles: List[Article],
        on_progress: Optional[Callable[[int, int, Article], None]] = None,
    ) -> List[Article]:
        """Process a batch of articles/documents sequentially."""
        if not self.use_api and self._llm is None:
            self.load()
            self._llm = self.router._llm

        total = len(articles)
        if self.verbose:
            logger.info(f"[{self.__class__.__name__}] Processing {total} item(s)...\n")

        t_batch = time.time()

        for i, article in enumerate(articles, 1):
            if self.verbose:
                logger.info(f" ({i}/{total})")
            try:
                articles[i - 1] = self.process_single(article)
            except Exception as e:
                logger.info(f" [ERROR] Failed to process item: {e}")
                traceback.print_exc()
                continue

            if on_progress:
                on_progress(i, total, articles[i - 1])

        elapsed = time.time() - t_batch
        if self.verbose:
            avg = elapsed / total if total else 0
            logger.info(
                f"\n[{self.__class__.__name__}] Done. {total} items in "
                f"{elapsed:.1f}s (avg {avg:.1f}s/item)"
            )
        return articles