"""Sprint 3 — tests for evidence_retriever module.

Three required tests:
1. test_generates_distinct_queries   — 3 non-duplicate labeled (query, label) pairs
2. test_rejects_hallucinated_quotes  — _quote_valid() rejects paraphrased / invented text
3. test_dedup_by_url                 — _dedup_by_url keeps highest-confidence per URL
"""

from __future__ import annotations

import pytest

from src.agents.evidence_retriever import (
    _dedup_by_url,
    _generate_queries,
    _quote_valid,
    _extract_and_validate,
)
from src.schemas import CitedEvidence


# ── 1. Query generation ───────────────────────────────────────────────────────

def test_generates_distinct_queries():
    """_generate_queries must return exactly 3 distinct queries with valid labels."""
    question   = "Will OpenAI release GPT-5 before Q3 2026?"
    res_source = "openai.com"

    queries = _generate_queries(question, resolution_source=res_source)

    # Exactly 3 tuples
    assert len(queries) == 3, f"Expected 3 queries, got {len(queries)}"

    query_strings = [q for q, _ in queries]
    labels        = [l for _, l in queries]

    # All three query strings are distinct
    assert len(set(query_strings)) == 3, "All queries must be unique"

    # Required labels present
    assert set(labels) == {"yes_case", "no_case", "resolution"}, (
        f"Expected labels yes_case/no_case/resolution, got {labels}"
    )

    # Each query is non-empty and contains part of the question
    for q_str, label in queries:
        assert q_str.strip(), f"Query for {label!r} must not be empty"
        # The market topic should appear in the query
        assert "GPT-5" in q_str or "OpenAI" in q_str or "openai" in q_str.lower(), (
            f"Query {q_str!r} does not seem related to the market question"
        )


def test_generates_distinct_queries_no_resolution_source():
    """Works correctly when resolution_source is None."""
    queries = _generate_queries("Will it rain in Paris on 2026-06-01?")
    assert len(queries) == 3
    labels = [l for _, l in queries]
    assert "yes_case"   in labels
    assert "no_case"    in labels
    assert "resolution" in labels


# ── 2. Quote validation ───────────────────────────────────────────────────────

def test_rejects_hallucinated_quotes():
    """_quote_valid must return False for text not literally in the snippet."""
    snippet = (
        "Federal Reserve officials signalled they could raise rates by 50bp "
        "at the June meeting, according to minutes released Wednesday."
    )

    # Hallucinated / paraphrased versions that must be rejected
    hallucinated = [
        "The Fed announced a 50 basis-point rate increase",   # paraphrase
        "Central bankers confirmed a June rate hike",          # invented
        "50bp increase confirmed at June FOMC",                # not verbatim
        "",                                                     # empty
        "   ",                                                  # whitespace only
    ]
    for bad_quote in hallucinated:
        assert not _quote_valid(bad_quote, snippet), (
            f"Expected _quote_valid to reject {bad_quote!r}"
        )

    # Actual verbatim substrings must be accepted
    valid_quotes = [
        "Federal Reserve officials signalled they could raise rates by 50bp",
        "minutes released Wednesday",
        "raise rates by 50bp",
    ]
    for good_quote in valid_quotes:
        assert _quote_valid(good_quote, snippet), (
            f"Expected _quote_valid to accept {good_quote!r}"
        )


def test_quote_valid_whitespace_normalisation():
    """Whitespace normalisation lets multi-space/newline snippets match cleanly."""
    snippet = "The   court  ruled   in  favour  of  the  plaintiff."
    # Same content with normalised whitespace — should pass
    assert _quote_valid("court  ruled", snippet)
    assert _quote_valid("ruled in favour", snippet)
    # Non-matching text must fail
    assert not _quote_valid("ruled against", snippet)


# ── 3. Deduplication ─────────────────────────────────────────────────────────

def _make_ev(url: str, confidence: str, label: str = "yes_case") -> CitedEvidence:
    return CitedEvidence(
        claim=f"Some claim from {url}",
        quote="verbatim quote here",
        source_url=url,
        publisher=url.split("/")[0],
        credibility="MEDIUM",
        label=label,
        confidence=confidence,
    )


def test_dedup_by_url():
    """_dedup_by_url keeps the highest-confidence item per URL."""
    url_a = "reuters.com/article-1"
    url_b = "bloomberg.com/article-2"

    items = [
        _make_ev(url_a, "low"),      # duplicate A — low
        _make_ev(url_a, "high"),     # duplicate A — HIGH (should win)
        _make_ev(url_a, "medium"),   # duplicate A — medium
        _make_ev(url_b, "high"),     # unique B — high
        _make_ev(url_b, "medium"),   # duplicate B — medium
    ]

    deduped = _dedup_by_url(items)

    urls_out = [ev.source_url for ev in deduped]

    # Exactly one entry per URL
    assert len(deduped) == 2, f"Expected 2 unique URLs, got {len(deduped)}: {urls_out}"

    by_url = {ev.source_url: ev for ev in deduped}

    # For url_a the "high" item must survive
    assert by_url[url_a].confidence == "high", (
        f"Expected 'high' for {url_a}, got {by_url[url_a].confidence!r}"
    )
    # For url_b the "high" item must survive
    assert by_url[url_b].confidence == "high", (
        f"Expected 'high' for {url_b}, got {by_url[url_b].confidence!r}"
    )


def test_dedup_by_url_single_items():
    """Single item per URL passes through unchanged."""
    items = [
        _make_ev("techcrunch.com/a", "medium"),
        _make_ev("wired.com/b",      "low"),
    ]
    deduped = _dedup_by_url(items)
    assert len(deduped) == 2
    urls = {ev.source_url for ev in deduped}
    assert "techcrunch.com/a" in urls
    assert "wired.com/b" in urls


# ── 4. Extract-and-validate with mocked LLM ──────────────────────────────────

@pytest.mark.asyncio
async def test_extract_and_validate_rejects_bad_quote(monkeypatch):
    """When the LLM returns a hallucinated quote, _extract_and_validate returns None."""
    from src.retrieval.search import SearchHit

    hit = SearchHit(
        url="https://reuters.com/article",
        title="Reuters article",
        publisher="reuters.com",
        published_at=None,
        snippet="The Federal Reserve held rates steady at 5.25% in May 2026.",
        raw_text="The Federal Reserve held rates steady at 5.25% in May 2026.",
        score=0.9,
        credibility="HIGH",
        query_idx=1,
        query_label="yes_case",
    )

    # LLM returns a paraphrased (hallucinated) quote
    async def fake_llm(h, question, model, api_key):
        return {
            "claim":      "The Fed did not change rates.",
            "quote":      "Fed decided to keep rates unchanged",   # NOT in snippet
            "confidence": "high",
            "label":      "no_case",
        }

    import src.agents.evidence_retriever as mod
    monkeypatch.setattr(mod, "_call_extraction_llm", fake_llm)

    result = await _extract_and_validate(hit, "Will Fed cut rates?", "model", "key")
    assert result is None, "Hallucinated quote must be rejected → None"


@pytest.mark.asyncio
async def test_extract_and_validate_accepts_valid_quote(monkeypatch):
    """When the LLM returns a real verbatim quote, _extract_and_validate returns CitedEvidence."""
    from src.retrieval.search import SearchHit

    snippet = "The Federal Reserve held rates steady at 5.25% in May 2026."
    hit = SearchHit(
        url="https://reuters.com/article",
        title="Reuters article",
        publisher="reuters.com",
        published_at=None,
        snippet=snippet,
        raw_text=snippet,
        score=0.9,
        credibility="HIGH",
        query_idx=1,
        query_label="no_case",
    )

    async def fake_llm(h, question, model, api_key):
        return {
            "claim":      "The Fed held rates at 5.25%.",
            "quote":      "Federal Reserve held rates steady at 5.25%",   # IS in snippet
            "confidence": "high",
            "label":      "no_case",
        }

    import src.agents.evidence_retriever as mod
    monkeypatch.setattr(mod, "_call_extraction_llm", fake_llm)

    result = await _extract_and_validate(hit, "Will Fed cut rates?", "model", "key")
    assert result is not None
    assert isinstance(result, CitedEvidence)
    assert result.confidence == "high"
    assert result.label == "no_case"
    assert result.publisher == "reuters.com"
