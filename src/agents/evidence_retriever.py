"""Node 2 — evidence_retriever.

Generates search queries, fires them in parallel against Tavily,
BM25-reranks results, and streams per-query log lines to the SSE feed.
"""

from __future__ import annotations

import asyncio
import logging

from src.run_store import push_log

logger = logging.getLogger(__name__)


def _generate_queries(question: str, description: str | None = None) -> list[str]:
    """Produce 4-5 targeted search queries from the market question."""
    base = question.rstrip("?")
    queries = [
        f"{base} {2026}",
        f"{base} latest news",
        f"{base} evidence probability",
        f"{base} analysis prediction",
    ]
    # Add description-derived query if available
    if description and len(description) > 30:
        # Extract first meaningful sentence
        first = description.split(".")[0][:80]
        if first not in queries:
            queries.append(first)
    return queries[:5]


async def evidence_retriever_node(state: dict) -> dict:
    run_id: str = state["run_id"]
    snapshot: dict = state["snapshot"]

    from src.config import settings
    from src.retrieval.search import TavilySearchClient
    from src.retrieval.reranker import bm25_rerank

    question = snapshot.get("question", "")
    description = snapshot.get("description")

    queries = _generate_queries(question, description)

    await push_log(run_id, "Dispatching Tavily search queries...", "info")

    all_results = []
    search_query_log = []

    if not settings.tavily_api_key:
        await push_log(run_id, "  ⚠ TAVILY_API_KEY not set — skipping web search", "warn")
        return {"search_results": [], "search_queries": [], "sources": []}

    client = TavilySearchClient(settings.tavily_api_key)

    # Fire all queries concurrently
    async def _run_query(idx: int, q: str):
        results, took_ms = await client.search(q, max_results=4, query_idx=idx + 1)
        return idx, q, results, took_ms

    tasks = [_run_query(i, q) for i, q in enumerate(queries)]
    completed = await asyncio.gather(*tasks, return_exceptions=True)

    for item in completed:
        if isinstance(item, Exception):
            logger.warning("Search task failed: %s", item)
            continue
        idx, q, results, took_ms = item
        await push_log(run_id, f"  [{idx + 1}/{len(queries)}] \"{q[:60]}\"", "query")
        await push_log(run_id, f"          → {len(results)} results · {took_ms}ms", "dim")
        all_results.extend(results)
        search_query_log.append({
            "idx": idx + 1,
            "query": q,
            "results": len(results),
            "took_ms": took_ms,
        })

    total = len(all_results)
    await push_log(run_id, f"Retrieved {total} sources, filtering by credibility...", "info")

    # Rerank and filter
    reranked = bm25_rerank(all_results, question, top_k=12)
    medium_plus = [r for r in reranked if r.credibility in ("HIGH", "MEDIUM")]
    cited = medium_plus[:8] if medium_plus else reranked[:8]

    await push_log(run_id, f"  ✓ {len(cited)} sources passed (≥ MEDIUM credibility)", "ok")

    # Stream source previews
    await push_log(run_id, "Reading top-ranked sources...", "info")
    for r in cited[:6]:
        domain = r.url.split("/")[2].removeprefix("www.") if "://" in r.url else r.url
        await push_log(run_id, f"  {domain} — {r.title[:55]}", "dim")

    sources_out = [
        {
            "domain": r.url.split("/")[2].removeprefix("www.") if "://" in r.url else r.url,
            "title": r.title,
            "url": r.url,
            "content": r.content,
            "cred": r.credibility,
            "used": True,
            "q": r.query_idx,
        }
        for r in cited
    ]

    return {
        "search_results": [
            {"title": r.title, "url": r.url, "content": r.content, "credibility": r.credibility}
            for r in cited
        ],
        "search_queries": search_query_log,
        "sources": sources_out,
    }
