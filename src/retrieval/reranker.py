"""BM25-based reranker for retrieved search results."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def bm25_rerank(results: list, query: str, top_k: int = 8) -> list:
    """Rerank a list of SearchResult objects using BM25 against the query.

    Falls back to original order if rank-bm25 is unavailable or corpus is empty.
    """
    if not results:
        return results

    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        logger.warning("rank-bm25 not installed; skipping rerank")
        return results[:top_k]

    # Support both SearchHit.snippet (Sprint 3+) and legacy .content attribute
    corpus = [_tokenize(f"{r.title} {getattr(r, 'snippet', None) or getattr(r, 'content', '')}") for r in results]
    tokenized_query = _tokenize(query)

    try:
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(tokenized_query)
    except Exception as exc:
        logger.warning("BM25 scoring failed: %s", exc)
        return results[:top_k]

    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [results[i] for i in ranked[:top_k]]
