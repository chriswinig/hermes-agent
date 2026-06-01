"""Crawl4AI content extraction provider.

Local, free extraction backend for ``web_extract``. Crawl4AI renders pages with
Playwright and returns LLM-friendly markdown without requiring an external API
key. It intentionally does not implement search.
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
from typing import Any, Dict, List

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)


def _crawl4ai_importable() -> bool:
    """Return True when the optional Crawl4AI package is importable."""
    return importlib.util.find_spec("crawl4ai") is not None


def _markdown_to_text(markdown: Any) -> str:
    """Normalize Crawl4AI's StringCompatibleMarkdown / str variants."""
    if markdown is None:
        return ""
    if isinstance(markdown, str):
        return markdown
    # Newer Crawl4AI exposes a StringCompatibleMarkdown object. str(obj) gives
    # the markdown text; markdown.raw_markdown may exist on some versions.
    raw = getattr(markdown, "raw_markdown", None)
    if raw:
        return str(raw)
    return str(markdown)


def _result_title(result: Any) -> str:
    metadata = getattr(result, "metadata", None) or {}
    if isinstance(metadata, dict):
        return str(metadata.get("title") or "")
    return ""


class Crawl4AIWebSearchProvider(WebSearchProvider):
    """Local Crawl4AI markdown extraction provider."""

    @property
    def name(self) -> str:
        return "crawl4ai"

    @property
    def display_name(self) -> str:
        return "Crawl4AI"

    def is_available(self) -> bool:
        """Crawl4AI is usable when the optional package is installed."""
        return _crawl4ai_importable()

    def supports_search(self) -> bool:
        return False

    def supports_extract(self) -> bool:
        return True

    async def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        """Extract markdown content from URLs via Crawl4AI.

        Returns per-URL result dictionaries matching the Hermes web_extract
        post-processing pipeline. Lazy-installs the optional dependency if
        permitted by config.
        """
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return [{"url": u, "title": "", "content": "", "error": "Interrupted"} for u in urls]

            try:
                from tools.lazy_deps import ensure as _lazy_ensure

                _lazy_ensure("search.crawl4ai", prompt=False)
            except ImportError:
                pass
            except Exception as exc:  # noqa: BLE001 — lazy_deps gives install hint
                return [
                    {
                        "url": u,
                        "title": "",
                        "content": "",
                        "raw_content": "",
                        "error": f"Crawl4AI dependency unavailable: {exc}",
                    }
                    for u in urls
                ]

            from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig

            browser_config = BrowserConfig(
                headless=True,
                verbose=False,
                text_mode=True,
                light_mode=True,
            )
            run_config = CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                page_timeout=int(kwargs.get("page_timeout") or 60_000),
                wait_until=str(kwargs.get("wait_until") or "domcontentloaded"),
                remove_overlay_elements=True,
                remove_consent_popups=True,
                verbose=False,
            )

            results: List[Dict[str, Any]] = []
            async with AsyncWebCrawler(config=browser_config) as crawler:
                for url in urls:
                    if is_interrupted():
                        results.append({"url": url, "title": "", "content": "", "error": "Interrupted"})
                        continue
                    try:
                        crawl_result = await crawler.arun(url=url, config=run_config)
                        success = bool(getattr(crawl_result, "success", False))
                        error = str(getattr(crawl_result, "error_message", "") or "")
                        markdown = _markdown_to_text(getattr(crawl_result, "markdown", ""))
                        title = _result_title(crawl_result)
                        final_url = str(getattr(crawl_result, "redirected_url", None) or getattr(crawl_result, "url", None) or url)
                        status_code = getattr(crawl_result, "status_code", None)
                        metadata = getattr(crawl_result, "metadata", None) or {}
                        if not isinstance(metadata, dict):
                            metadata = {}
                        metadata = {**metadata, "sourceURL": final_url, "status_code": status_code, "provider": "crawl4ai"}

                        if not success and not markdown:
                            results.append(
                                {
                                    "url": url,
                                    "title": title,
                                    "content": "",
                                    "raw_content": "",
                                    "metadata": metadata,
                                    "error": error or "Crawl4AI extraction failed",
                                }
                            )
                        else:
                            results.append(
                                {
                                    "url": final_url,
                                    "title": title,
                                    "content": markdown,
                                    "raw_content": markdown,
                                    "metadata": metadata,
                                    **({"error": error} if error and not success else {}),
                                }
                            )
                    except Exception as exc:  # noqa: BLE001 — per-URL failure
                        logger.warning("Crawl4AI extract error for %s: %s", url, exc)
                        results.append(
                            {
                                "url": url,
                                "title": "",
                                "content": "",
                                "raw_content": "",
                                "error": f"Crawl4AI extract failed: {exc}",
                            }
                        )
            return results
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — whole-batch failure
            logger.warning("Crawl4AI extract batch error: %s", exc)
            return [
                {"url": u, "title": "", "content": "", "raw_content": "", "error": f"Crawl4AI extract failed: {exc}"}
                for u in urls
            ]

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Crawl4AI",
            "badge": "free/local",
            "tag": "Local Playwright-based extraction to LLM-ready markdown. No API key.",
            "env_vars": [],
        }
