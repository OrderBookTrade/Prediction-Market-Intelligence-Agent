"""Tavily search wrapper — async-safe, structured output."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

TRUSTED_DOMAINS = {
    "reuters.com", "bloomberg.com", "ap.org", "apnews.com",
    "ft.com", "wsj.com", "bbc.com", "nytimes.com",
    "federalreserve.gov", "sec.gov", "cftc.gov",
    "github.com", "openai.com", "anthropic.com",
    "coinmarketcap.com", "coingecko.com", "theinformation.com",
}

MEDIUM_DOMAINS = {
    "techcrunch.com", "arstechnica.com", "theverge.com",
    "wired.com", "axios.com", "politico.com", "thehill.com",
    "bloomberg.com", "cnn.com",
}


def _credibility(url: str) -> str:
    domain = url.split("/")[2].removeprefix("www.") if "://" in url else ""
    if any(t in domain for t in TRUSTED_DOMAINS):
        return "HIGH"
    if any(m in domain for m in MEDIUM_DOMAINS):
        return "MEDIUM"
    return "LOW"


@dataclass
class SearchResult:
    title: str
    url: str
    content: str
    score: float
    credibility: str
    query_idx: int = 0


class TavilySearchClient:
    """Async wrapper around the synchronous Tavily SDK."""

    def __init__(self, api_key: str) -> None:
        from tavily import TavilyClient

        self._client = TavilyClient(api_key=api_key)

    async def search(
        self,
        query: str,
        max_results: int = 5,
        query_idx: int = 0,
    ) -> tuple[list[SearchResult], int]:
        """Run a search query.

        Returns (results, took_ms).
        """
        loop = asyncio.get_event_loop()
        t0 = time.monotonic()

        try:
            raw = await loop.run_in_executor(
                None,
                lambda: self._client.search(query, max_results=max_results),
            )
        except Exception as exc:
            logger.warning("Tavily search failed for %r: %s", query, exc)
            return [], 0

        took_ms = int((time.monotonic() - t0) * 1000)
        results = []
        for item in raw.get("results", []):
            url = item.get("url", "")
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    url=url,
                    content=item.get("content", ""),
                    score=float(item.get("score", 0.0)),
                    credibility=_credibility(url),
                    query_idx=query_idx,
                )
            )
        return results, took_ms
