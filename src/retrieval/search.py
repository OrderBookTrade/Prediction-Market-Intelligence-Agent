"""Tavily search wrapper — async-safe, returns SearchHit with full citation metadata."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ── Domain credibility registry ───────────────────────────────────────────────

_TRUSTED = {
    "reuters.com", "bloomberg.com", "ap.org", "apnews.com",
    "ft.com", "wsj.com", "bbc.com", "nytimes.com", "economist.com",
    "federalreserve.gov", "sec.gov", "cftc.gov", "whitehouse.gov",
    "github.com", "openai.com", "anthropic.com", "deepmind.com",
    "coinmarketcap.com", "coingecko.com", "theinformation.com",
    "nature.com", "science.org", "arxiv.org",
}

_MEDIUM = {
    "techcrunch.com", "arstechnica.com", "theverge.com", "wired.com",
    "axios.com", "politico.com", "thehill.com", "cnn.com",
    "cnbc.com", "fortune.com", "businessinsider.com", "marketwatch.com",
}


def _publisher(url: str) -> str:
    """Extract bare domain from a URL, e.g. 'https://www.reuters.com/...' → 'reuters.com'."""
    try:
        return url.split("/")[2].removeprefix("www.")
    except IndexError:
        return url


def _credibility(url: str) -> str:
    domain = _publisher(url)
    if any(t in domain for t in _TRUSTED):
        return "HIGH"
    if any(m in domain for m in _MEDIUM):
        return "MEDIUM"
    return "LOW"


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class SearchHit:
    """Normalised result from any search provider.

    `snippet` / `raw_text` are used for quote substring-validation.
    They may be identical when the provider returns a single content field.
    """

    url: str
    title: str
    publisher: str
    published_at: str | None          # ISO-8601 string or None
    snippet: str                      # ~500-char content excerpt from provider
    raw_text: str                     # same as snippet (Tavily only returns one)
    score: float = 0.0
    credibility: str = "LOW"
    query_idx: int = 0
    query_label: str = ""             # "yes_case" | "no_case" | "resolution"


# Backward-compat alias — Sprint 2 tests reference SearchResult
SearchResult = SearchHit


# ── Tavily client ─────────────────────────────────────────────────────────────

class TavilySearchClient:
    """Async wrapper around the synchronous Tavily Python SDK."""

    def __init__(self, api_key: str) -> None:
        from tavily import TavilyClient
        self._client = TavilyClient(api_key=api_key)

    async def search(
        self,
        query: str,
        *,
        max_results: int = 8,
        days_back: int = 30,
        query_idx: int = 0,
        query_label: str = "",
    ) -> tuple[list[SearchHit], int]:
        """Fire one search query.

        Returns (hits, latency_ms).
        Silently returns ([], 0) on any provider error.
        """
        loop = asyncio.get_event_loop()
        t0 = time.monotonic()

        try:
            raw = await loop.run_in_executor(
                None,
                lambda: self._client.search(
                    query,
                    max_results=max_results,
                    days=days_back,
                ),
            )
        except Exception as exc:
            logger.warning("Tavily search failed for %r: %s", query, exc)
            return [], 0

        took_ms = int((time.monotonic() - t0) * 1000)

        hits: list[SearchHit] = []
        for item in raw.get("results", []):
            url = item.get("url", "")
            content = item.get("content", "")
            hits.append(
                SearchHit(
                    url=url,
                    title=item.get("title", ""),
                    publisher=_publisher(url),
                    published_at=item.get("published_date"),
                    snippet=content,
                    raw_text=content,
                    score=float(item.get("score", 0.0)),
                    credibility=_credibility(url),
                    query_idx=query_idx,
                    query_label=query_label,
                )
            )

        return hits, took_ms
