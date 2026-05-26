"""Node 2 — evidence_retriever.

Generates 3 purpose-specific search queries (yes_case / no_case / resolution),
fires them concurrently via Tavily, BM25-reranks results, then calls Claude
(claude-sonnet, temperature=0) per hit to extract a verbatim-quoted claim.

The NON-NEGOTIABLE anti-hallucination rule
------------------------------------------
Every quote extracted by the LLM is validated with _quote_valid() before
a CitedEvidence object is created.  If the quote is not a literal substring
of the source snippet, the evidence item is SILENTLY DROPPED and a warning
is emitted.  Under no circumstances should a hallucinated quote survive.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from src.run_store import push_log
from src.schemas import CitedEvidence
from src.utils.secrets import safe_secret_info

logger = logging.getLogger(__name__)

# ── Extraction tool schema (no $ref — flat for Anthropic API) ─────────────────

EXTRACTION_TOOL: dict = {
    "name": "extract_evidence",
    "description": (
        "Extract one key evidence item from a search-result snippet. "
        "The 'quote' field MUST be copied verbatim from the snippet — "
        "do NOT paraphrase or invent text."
    ),
    "input_schema": {
        "type": "object",
        "required": ["claim", "quote", "confidence", "label"],
        "properties": {
            "claim": {
                "type": "string",
                "description": "Factual claim (1-2 sentences) drawn from the snippet, relevant to the market question.",
            },
            "quote": {
                "type": "string",
                "description": (
                    "Exact verbatim substring from the snippet that supports the claim. "
                    "Must appear character-for-character in the content."
                ),
            },
            "confidence": {
                "type": "string",
                "enum": ["high", "medium", "low"],
                "description": "How strongly this evidence speaks to the market outcome.",
            },
            "label": {
                "type": "string",
                "enum": ["yes_case", "no_case", "resolution"],
                "description": (
                    "yes_case  — supports YES resolution; "
                    "no_case   — supports NO resolution; "
                    "resolution — clarifies how/when the market resolves."
                ),
            },
        },
    },
}

# ── Query generation ──────────────────────────────────────────────────────────

# Tool schema for LLM-based query generation
_QUERY_GEN_TOOL: dict = {
    "name": "generate_search_queries",
    "description": "Generate three targeted web search queries to research a prediction market.",
    "input_schema": {
        "type": "object",
        "required": ["yes_case", "no_case", "resolution"],
        "properties": {
            "yes_case": {
                "type": "string",
                "description": (
                    "Search query to find recent news/evidence supporting the YES outcome. "
                    "Use specific, newsworthy keywords — NOT the market question verbatim."
                ),
            },
            "no_case": {
                "type": "string",
                "description": (
                    "Search query to find recent news/evidence supporting the NO outcome "
                    "⚠ ANTHROPIC_API_KEY invalid shape — expected sk- prefix "
                    "or showing the event is unlikely. Specific terms, not verbatim question."
                ),
            },
            "resolution": {
                "type": "string",
                "description": (
                    "Search query to find official announcements, decisions, or source "
                    "clarifications about how/when this market resolves."
                ),
            },
        },
    },
}


def _generate_queries_template(
    question: str,
    resolution_source: str | None = None,
    description: str | None = None,
) -> list[tuple[str, str]]:
    """Fallback: simple template-based queries when no API key is available."""
    base = question.rstrip("?").strip()
    res_hint = f" site:{resolution_source}" if resolution_source else ""
    return [
        (f"{base} latest news 2025 2026", "yes_case"),
        (f"{base} unlikely obstacles analysis", "no_case"),
        (f"{base} official announcement{res_hint}", "resolution"),
    ]


def _generate_queries(
    question: str,
    resolution_source: str | None = None,
    description: str | None = None,
) -> list[tuple[str, str]]:
    """Backward-compatible alias for tests and deterministic fallback use."""
    return _generate_queries_template(question, resolution_source, description)


async def _generate_queries_llm(
    question: str,
    description: str | None,
    resolution_source: str | None,
    model: str,
    api_key: str,
) -> list[tuple[str, str]]:
    """Use Claude to generate targeted, specific search queries for a market.

    Falls back to template queries on any failure.
    """
    import anthropic

    desc_line = f"\nDescription: {description[:400]}" if description else ""
    res_line  = f"\nResolution source: {resolution_source}" if resolution_source else ""

    prompt = (
        f"You are a prediction market researcher. Generate targeted web search queries.\n\n"
        f"Market question: {question}{desc_line}{res_line}\n\n"
        "Rules:\n"
        "- Use specific, newsworthy keywords (e.g. 'China PLA Taiwan strait military 2025')\n"
        "- Do NOT repeat the market question verbatim\n"
        "- Focus on recent events (past 3-6 months)\n"
        "- yes_case: evidence the event WILL happen\n"
        "- no_case: evidence the event WON'T happen or obstacles\n"
        "- resolution: official source, announcement, or deadline clarification"
    )

    loop = asyncio.get_event_loop()
    client = anthropic.Anthropic(api_key=api_key)

    try:
        response = await loop.run_in_executor(
            None,
            lambda: client.messages.create(
                model=model,
                max_tokens=300,
                temperature=0,
                tools=[_QUERY_GEN_TOOL],
                tool_choice={"type": "tool", "name": "generate_search_queries"},
                messages=[{"role": "user", "content": prompt}],
            ),
        )
        for block in response.content:
            if block.type == "tool_use" and block.name == "generate_search_queries":
                q = block.input
                return [
                    (q["yes_case"],   "yes_case"),
                    (q["no_case"],    "no_case"),
                    (q["resolution"], "resolution"),
                ]
    except Exception as exc:
        logger.warning("LLM query generation failed, using template fallback: %s", exc)

    return _generate_queries_template(question, resolution_source, description)


# ── Quote validation ──────────────────────────────────────────────────────────

def _quote_valid(quote: str, snippet: str) -> bool:
    """Return True iff `quote` is a literal (whitespace-normalised) substring of `snippet`.

    Empty or whitespace-only quotes always return False.
    """
    if not quote or not quote.strip():
        return False
    norm_quote   = " ".join(quote.lower().split())
    norm_snippet = " ".join(snippet.lower().split())
    return norm_quote in norm_snippet


# ── LLM extraction (standalone, patchable in tests) ──────────────────────────

async def _call_extraction_llm(
    hit: Any,          # SearchHit
    question: str,
    model: str,
    api_key: str,
) -> dict | None:
    """Call Claude with tool_use to extract one evidence item from a search hit.

    Returns the raw tool-input dict, or None on any failure.
    This function is a standalone coroutine so tests can patch it easily.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    prompt = (
        f"Market question: {question}\n\n"
        f"Source: {hit.title} ({hit.publisher})\n"
        f"URL: {hit.url}\n"
        f"Content:\n{hit.snippet}\n\n"
        "Extract the single most relevant piece of evidence from this snippet. "
        "The 'quote' field must be copied verbatim from the content above."
    )

    loop = asyncio.get_event_loop()
    try:
        response = await loop.run_in_executor(
            None,
            lambda: client.messages.create(
                model=model,
                max_tokens=512,
                temperature=0,
                tools=[EXTRACTION_TOOL],
                tool_choice={"type": "tool", "name": "extract_evidence"},
                messages=[{"role": "user", "content": prompt}],
            ),
        )
        for block in response.content:
            if block.type == "tool_use" and block.name == "extract_evidence":
                return block.input
    except Exception as exc:
        logger.warning("Extraction LLM failed for %s: %s", hit.url, exc)
    return None


# ── Extract + validate wrapper ────────────────────────────────────────────────

async def _extract_and_validate(
    hit: Any,          # SearchHit
    question: str,
    model: str,
    api_key: str,
) -> CitedEvidence | None:
    """Call extraction LLM then enforce quote substring rule.

    Returns CitedEvidence or None (logged warning on rejection).
    """
    data = await _call_extraction_llm(hit, question, model, api_key)
    if data is None:
        return None

    quote = data.get("quote", "")
    if not _quote_valid(quote, hit.snippet):
        logger.warning(
            "Quote validation FAILED for %s — dropping. quote=%r",
            hit.url,
            quote[:80],
        )
        return None

    return CitedEvidence(
        claim=data.get("claim", ""),
        quote=quote,
        source_url=hit.url,
        publisher=hit.publisher,
        published_at=hit.published_at,
        credibility=hit.credibility,
        label=data.get("label", hit.query_label or ""),
        confidence=data.get("confidence", "low"),
    )


# ── Deduplication ─────────────────────────────────────────────────────────────

_CONF_RANK = {"high": 3, "medium": 2, "low": 1}


def _dedup_by_url(items: list[CitedEvidence]) -> list[CitedEvidence]:
    """Keep the highest-confidence CitedEvidence per URL."""
    best: dict[str, CitedEvidence] = {}
    for ev in items:
        key = ev.source_url
        current = best.get(key)
        if current is None or _CONF_RANK.get(ev.confidence, 0) > _CONF_RANK.get(current.confidence, 0):
            best[key] = ev
    return list(best.values())


# ── No-key fallback ───────────────────────────────────────────────────────────

def _fallback_evidence(hit: Any) -> CitedEvidence:
    """Produce CitedEvidence without an LLM when ANTHROPIC_API_KEY is absent.

    Uses the title as the claim and the first 100 chars of the snippet as the
    quote — both are guaranteed to be valid substrings.
    """
    safe_quote = hit.snippet[:100] if hit.snippet else hit.title[:100]
    return CitedEvidence(
        claim=hit.title,
        quote=safe_quote,
        source_url=hit.url,
        publisher=hit.publisher,
        published_at=hit.published_at,
        credibility=hit.credibility,
        label=hit.query_label or "",
        confidence="low",
    )


# ── Main node ─────────────────────────────────────────────────────────────────

async def evidence_retriever_node(state: dict) -> dict:
    run_id: str = state["run_id"]
    snapshot: dict = state.get("snapshot") or {}

    from src.config import settings
    from src.retrieval.reranker import bm25_rerank
    from src.retrieval.search import TavilySearchClient

    question        = snapshot.get("question", "")
    description     = snapshot.get("description")
    resolution_src  = snapshot.get("resolution_source")

    anthropic_info = safe_secret_info(settings.anthropic_api_key, expected_prefix="sk-")

    # ── Generate queries: LLM if key appears usable, template fallback otherwise ──
    if anthropic_info["present"] and anthropic_info["prefix_ok"]:
        await push_log(run_id, "Generating search queries via Claude...", "info")
        await push_log(
            run_id,
            (
                "[anthropic] caller=query_planner "
                f"key_present={anthropic_info['present']} "
                f"key_prefix={anthropic_info['prefix']} "
                f"key_length={anthropic_info['length']} "
                f"fingerprint={anthropic_info['fingerprint']} "
                f"model={settings.claude_model_fast}"
            ),
            "dim",
        )
        queries = await _generate_queries_llm(
            question, description, resolution_src,
            settings.claude_model_fast, settings.anthropic_api_key,
        )
    else:
        await push_log(
            run_id,
            (
                "[anthropic] caller=query_planner skipped "
                f"key_present={anthropic_info['present']} "
                f"key_prefix={anthropic_info['prefix']} "
                f"key_length={anthropic_info['length']} "
                f"fingerprint={anthropic_info['fingerprint']}"
            ),
            "warn",
        )
        queries = _generate_queries_template(question, resolution_src, description)

    await push_log(run_id, "Dispatching 3 Tavily search queries...", "info")
    key_info = safe_secret_info(settings.tavily_api_key, expected_prefix="tvl")
    await push_log(
        run_id,
        (
            "[tavily] caller=evidence_retriever "
            f"key_present={key_info['present']} "
            f"key_prefix={key_info['prefix']} "
            f"key_length={key_info['length']} "
            f"fingerprint={key_info['fingerprint']}"
        ),
        "dim",
    )

    # ── Bail out early if no Tavily key ──────────────────────────────────────
    if not settings.tavily_api_key:
        await push_log(run_id, "  ⚠ TAVILY_API_KEY not set — skipping web search", "warn")
        return {
            "search_results":  [],
            "search_queries":  [],
            "sources":         [],
            "cited_evidence":  [],
        }

    client = TavilySearchClient(settings.tavily_api_key)

    # ── Fire queries concurrently ─────────────────────────────────────────────
    async def _run_query(idx: int, q: str, label: str):
        results, took_ms = await client.search(
            q, max_results=5, days_back=60, query_idx=idx, query_label=label
        )
        return idx, q, label, results, took_ms

    tasks = [_run_query(i + 1, q, label) for i, (q, label) in enumerate(queries)]
    completed = await asyncio.gather(*tasks, return_exceptions=True)

    all_hits: list = []
    query_log: list[dict] = []

    for item in completed:
        if isinstance(item, Exception):
            logger.warning("Search task failed: %s", item)
            continue
        idx, q, label, hits, took_ms = item
        await push_log(run_id, f"  [{idx}/{len(queries)}] [{label}] \"{q[:55]}\"", "query")
        await push_log(run_id, f"          → {len(hits)} results · {took_ms}ms", "dim")
        all_hits.extend(hits)
        query_log.append({"idx": idx, "query": q, "label": label, "results": len(hits), "took_ms": took_ms})

    await push_log(run_id, f"Search requests executed: {len(query_log)}/{len(queries)}", "info")

    await push_log(run_id, f"Retrieved {len(all_hits)} raw hits — reranking...", "info")

    # ── BM25 rerank and credibility filter ───────────────────────────────────
    reranked = bm25_rerank(all_hits, question, top_k=12)
    medium_plus = [h for h in reranked if h.credibility in ("HIGH", "MEDIUM")]
    top_hits = medium_plus[:8] if medium_plus else reranked[:8]

    await push_log(run_id, f"  ✓ {len(top_hits)} sources passed credibility filter", "ok")

    # ── Extract evidence (LLM or fallback) ───────────────────────────────────
    raw_evidence: list[CitedEvidence] = []

    if not anthropic_info["present"] or not anthropic_info["prefix_ok"]:
        await push_log(run_id, "  ⚠ Anthropic unavailable — using title-based fallback evidence", "warn")
        raw_evidence = [_fallback_evidence(h) for h in top_hits]
    else:
        await push_log(run_id, "Extracting claims from sources (claude-sonnet)...", "info")
        extraction_tasks = [
            _extract_and_validate(h, question, settings.claude_model_fast, settings.anthropic_api_key)
            for h in top_hits
        ]
        results = await asyncio.gather(*extraction_tasks, return_exceptions=True)

        for h, res in zip(top_hits, results):
            if isinstance(res, Exception):
                logger.warning("Extraction failed for %s: %s", h.url, res)
            elif res is not None:
                raw_evidence.append(res)
            else:
                logger.warning("Evidence dropped for %s (None result)", h.url)

        await push_log(run_id, f"  ✓ {len(raw_evidence)}/{len(top_hits)} items passed quote validation", "ok")

    # ── Dedup by URL ──────────────────────────────────────────────────────────
    cited: list[CitedEvidence] = _dedup_by_url(raw_evidence)

    # ── Stream previews ───────────────────────────────────────────────────────
    await push_log(run_id, "Top evidence items:", "info")
    for ev in cited[:6]:
        tone = "✓YES" if ev.label == "yes_case" else "✗NO " if ev.label == "no_case" else " RES"
        await push_log(run_id, f"  [{tone}] {ev.publisher} — {ev.claim[:55]}", "dim")

    # ── Build backward-compat `search_results` for memo_writer & risk_critic ─
    search_results_compat = [
        {
            "title":       ev.claim,
            "url":         ev.source_url,
            "content":     ev.quote,
            "credibility": ev.credibility,
        }
        for ev in cited
    ]

    sources_out = [
        {
            "domain":  ev.publisher,
            "title":   ev.claim,
            "url":     ev.source_url,
            "content": ev.quote,
            "cred":    ev.credibility,
            "label":   ev.label,
            "conf":    ev.confidence,
        }
        for ev in cited
    ]

    return {
        "search_results": search_results_compat,
        "search_queries": query_log,
        "sources":        sources_out,
        "cited_evidence": [ev.model_dump() for ev in cited],
    }
