"""
Evidence retrieval via web search.

Uses Tavily API if available, falls back to DuckDuckGo.
Both are free/cheap and work without a proxy for most queries.
"""

import os
import httpx
from langchain_core.tools import tool


def _tavily_search(query: str, max_results: int = 5) -> list[dict]:
    """Search via Tavily API."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY not set")

    resp = httpx.post(
        "https://api.tavily.com/search",
        json={
            "api_key": api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
            "include_answer": True,
        },
        timeout=15.0,
    )
    resp.raise_for_status()
    data = resp.json()

    results = []
    if data.get("answer"):
        results.append({
            "title": "Tavily Summary",
            "url": "tavily",
            "content": data["answer"]
        })
    for r in data.get("results", []):
        results.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", "")[:500],
        })
    return results


def _ddg_search(query: str, max_results: int = 5) -> list[dict]:
    """Fallback: DuckDuckGo instant answer API (no key needed)."""
    resp = httpx.get(
        "https://api.duckduckgo.com/",
        params={
            "q": query,
            "format": "json",
            "no_html": 1,
            "skip_disambig": 1,
        },
        timeout=10.0,
    )
    resp.raise_for_status()
    data = resp.json()

    results = []
    if data.get("AbstractText"):
        results.append({
            "title": data.get("Heading", ""),
            "url": data.get("AbstractURL", ""),
            "content": data["AbstractText"][:500],
        })
    for topic in data.get("RelatedTopics", [])[:max_results - 1]:
        if isinstance(topic, dict) and topic.get("Text"):
            results.append({
                "title": topic.get("Text", "")[:100],
                "url": topic.get("FirstURL", ""),
                "content": topic.get("Text", "")[:400],
            })
    return results


def _format_results(results: list[dict]) -> str:
    if not results:
        return "No results found."
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r['title']}\n    URL: {r['url']}\n    {r['content']}")
    return "\n\n".join(lines)


@tool
def search_evidence(query: str) -> str:
    """
    Search the web for evidence related to a prediction market.
    Use specific queries like:
    - "Fed rate decision June 2026 probability"
    - "OpenAI GPT-5 release date announcement"
    - "[company/event] latest news site:reuters.com OR site:bloomberg.com"
    Returns titles, URLs, and snippets.
    """
    try:
        results = _tavily_search(query)
        return f"Search results for '{query}':\n\n" + _format_results(results)
    except Exception:
        pass

    try:
        results = _ddg_search(query)
        return f"Search results for '{query}':\n\n" + _format_results(results)
    except Exception as e:
        return f"Search failed: {e}. Try a more specific query."


@tool
def search_official_sources(query: str) -> str:
    """
    Search for official/authoritative sources only.
    Targets news sites, government sources, company blogs.
    Use for finding resolution criteria sources and official announcements.
    query example: "Federal Reserve FOMC June 2026 decision official"
    """
    # Append high-credibility domains to query
    enhanced = f"{query} site:reuters.com OR site:bloomberg.com OR site:ft.com OR site:wsj.com OR site:apnews.com OR site:bbc.com"

    try:
        results = _tavily_search(enhanced, max_results=4)
        return f"Official sources for '{query}':\n\n" + _format_results(results)
    except Exception:
        pass

    try:
        results = _ddg_search(query, max_results=4)
        return f"Sources for '{query}':\n\n" + _format_results(results)
    except Exception as e:
        return f"Search failed: {e}"
