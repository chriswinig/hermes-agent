"""Crawl4AI web extraction plugin — bundled, auto-loaded.

Crawl4AI is a local, free extraction backend optimized for LLM-ready
markdown. It requires the optional ``crawl4ai`` Python package and Playwright
browsers, both installed lazily when the backend is first used.
"""

from __future__ import annotations

from plugins.web.crawl4ai.provider import Crawl4AIWebSearchProvider


def register(ctx) -> None:
    """Register the Crawl4AI provider with the plugin context."""
    ctx.register_web_search_provider(Crawl4AIWebSearchProvider())
