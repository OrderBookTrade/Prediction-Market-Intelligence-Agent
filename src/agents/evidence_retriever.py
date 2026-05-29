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
import hashlib
import logging
import re
import time
from typing import Any

from src.agents.metrics import count_support, metrics_for, set_latency
from src.run_store import push_log
from src.schemas import CitedEvidence, EvidenceSide, EvidenceSupportLevel, SourceObject
from src.utils.secrets import safe_secret_info

logger = logging.getLogger(__name__)


def _source_id_for(url: str) -> str:
    """Deterministic short id for a source URL.

    CitedEvidence.source_id is required; SearchHit carries no id, so we derive
    a stable one from the URL (same URL -> same id, enabling dedup/linking).
    """
    return "src_" + hashlib.sha1((url or "").encode("utf-8")).hexdigest()[:12]


def _side_for_label(label: str) -> EvidenceSide:
    if label == "yes_case":
        return EvidenceSide.YES
    if label == "no_case":
        return EvidenceSide.NO
    return EvidenceSide.RESOLUTION


def _clean_market_topic(question: str) -> str:
    """Turn market-question phrasing into a natural search topic."""
    text = question.strip().rstrip("?")
    text = re.sub(r"^\s*will\s+", "", text, flags=re.I)
    text = re.sub(r"\b(resolve|resolves|resolved)\s+to\s+(yes|no)\b", "", text, flags=re.I)
    text = re.sub(r"\bby\s+[A-Z][a-z]+\s+\d{1,2},\s+\d{4}\b", "", text)
    text = re.sub(r"\bby\s+\d{1,2}/\d{1,2}/\d{2,4}\b", "", text)
    text = re.sub(r"\bbefore\s+[A-Z][a-z]+\s+\d{4}\b", "", text)
    text = re.sub(r"\bbefore\s+\d{4}\b", "", text, flags=re.I)
    text = re.sub(r"\bby\s+the\s+end\s+of\s+\d{4}\b", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text.replace(" x ", " ")).strip(" -:;,.")
    return text or question.strip().rstrip("?")


def _normalize_query(query: str, fallback_topic: str) -> str:
    query = re.sub(r"^\s*will\s+", "", query.strip().rstrip("?"), flags=re.I)
    query = re.sub(r"\bby\s+[A-Z][a-z]+\s+\d{1,2},\s+\d{4}\b", "", query)
    query = re.sub(r"\s+", " ", query).strip(" -:;,.")
    if len(query) < 40:
        query = f"{query or fallback_topic} latest news official sources 2026"
    if len(query) > 120:
        query = query[:120].rsplit(" ", 1)[0]
    return query


def _source_from_hit(hit: Any) -> SourceObject | None:
    url = getattr(hit, "url", "") or ""
    snippet = getattr(hit, "snippet", "") or ""
    title = getattr(hit, "title", "") or ""
    if not url or not (snippet or title):
        return None
    return SourceObject(
        source_id=_source_id_for(url),
        url=url,
        domain=getattr(hit, "publisher", "") or url,
        title=title,
        snippet=snippet,
        raw_content=getattr(hit, "raw_text", None),
        query_id=getattr(hit, "query_idx", 0) or 0,
        query_label=getattr(hit, "query_label", "") or "",
        credibility=getattr(hit, "credibility", "LOW") or "LOW",
    )


def _evidence_from_source(
    source: SourceObject,
    *,
    claim: str | None = None,
    quote: str | None = None,
    label: str | None = None,
    support_level: EvidenceSupportLevel = EvidenceSupportLevel.SNIPPET_SUPPORTED,
    confidence: str = "low",
) -> CitedEvidence | None:
    supported_text = (quote or source.snippet or source.title).strip()
    claim_text = (claim or source.title or source.snippet[:180]).strip()
    if not source.url or not source.domain or not supported_text or not claim_text:
        return None
    evidence_label = label or source.query_label
    return CitedEvidence(
        source_id=source.source_id,
        claim=claim_text,
        quote=supported_text[:500],
        source_url=source.url,
        publisher=source.domain,
        published_at=None,
        credibility=source.credibility,
        label=evidence_label,
        side=_side_for_label(evidence_label),
        support_level=support_level,
        confidence=confidence,
    )

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
    base = _clean_market_topic(question)
    res_hint = f" site:{resolution_source}" if resolution_source else ""
    return [
        (_normalize_query(f"{base} latest news evidence 2026", base), "yes_case"),
        (_normalize_query(f"{base} obstacles unlikely analysis 2026", base), "no_case"),
        (_normalize_query(f"{base} official announcement resolution source{res_hint}", base), "resolution"),
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
    provider: str = "anthropic",
) -> list[tuple[str, str]]:
    """Use Claude to generate targeted, specific search queries for a market.

    Falls back to template queries on any failure.
    """
    desc_line = f"\nDescription: {description[:400]}" if description else ""
    res_line  = f"\nResolution source: {resolution_source}" if resolution_source else ""
    clean_topic = _clean_market_topic(question)

    prompt = (
        f"You are a prediction market researcher. Generate targeted web search queries.\n\n"
        f"Market question: {question}\n"
        f"Clean topic: {clean_topic}{desc_line}{res_line}\n\n"
        "Rules:\n"
        "- Use specific, newsworthy keywords (e.g. 'China PLA Taiwan strait military 2025')\n"
        "- Do NOT repeat the market question verbatim\n"
        "- Remove 'Will', deadline phrasing, and prediction-market wording\n"
        "- Each query must be 40-120 characters and must not be truncated\n"
        "- Focus on recent events (past 3-6 months)\n"
        "- yes_case: evidence the event WILL happen\n"
        "- no_case: evidence the event WON'T happen or obstacles\n"
        "- resolution: official source, announcement, or deadline clarification"
    )

    loop = asyncio.get_event_loop()

    try:
        if provider == "deepseek":
            import openai
            import json
            client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
            tool = {
                "type": "function",
                "function": {
                    "name": _QUERY_GEN_TOOL["name"],
                    "description": _QUERY_GEN_TOOL["description"],
                    "parameters": _QUERY_GEN_TOOL["input_schema"]
                }
            }
            response = await loop.run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model=model,
                    max_tokens=300,
                    temperature=0.0,
                    tools=[tool],
                    tool_choice={"type": "function", "function": {"name": "generate_search_queries"}},
                    messages=[{"role": "user", "content": prompt}],
                ),
            )
            if response.choices[0].message.tool_calls:
                q = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
                return [
                    (_normalize_query(q.get("yes_case", ""), clean_topic), "yes_case"),
                    (_normalize_query(q.get("no_case", ""), clean_topic), "no_case"),
                    (_normalize_query(q.get("resolution", ""), clean_topic), "resolution"),
                ]
        else:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
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
                        (_normalize_query(q["yes_case"], clean_topic), "yes_case"),
                        (_normalize_query(q["no_case"], clean_topic), "no_case"),
                        (_normalize_query(q["resolution"], clean_topic), "resolution"),
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
    provider: str = "anthropic",
) -> dict | None:
    """Call Claude with tool_use to extract one evidence item from a search hit.

    Returns the raw tool-input dict, or None on any failure.
    This function is a standalone coroutine so tests can patch it easily.
    """
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
        if provider == "deepseek":
            import openai
            import json
            client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
            tool = {
                "type": "function",
                "function": {
                    "name": EXTRACTION_TOOL["name"],
                    "description": EXTRACTION_TOOL["description"],
                    "parameters": EXTRACTION_TOOL["input_schema"]
                }
            }
            response = await loop.run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model=model,
                    max_tokens=512,
                    temperature=0.0,
                    tools=[tool],
                    tool_choice={"type": "function", "function": {"name": "extract_evidence"}},
                    messages=[{"role": "user", "content": prompt}],
                ),
            )
            if response.choices[0].message.tool_calls:
                return json.loads(response.choices[0].message.tool_calls[0].function.arguments)
        else:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
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
    provider: str = "anthropic",
) -> CitedEvidence | None:
    """Call extraction LLM and return the strongest source-supported evidence.

    Exact quote matches become quote_verified evidence. If the model produces an
    unsupported quote but the search hit has a real snippet, downgrade to
    snippet_supported LOW confidence instead of dropping the source.
    """
    source = _source_from_hit(hit)
    if source is None:
        return None

    try:
        data = await _call_extraction_llm(hit, question, model, api_key, provider)
    except TypeError as exc:
        if "positional" not in str(exc):
            raise
        data = await _call_extraction_llm(hit, question, model, api_key)
    if data is None:
        return _evidence_from_source(source)

    quote = data.get("quote", "")
    label = data.get("label", hit.query_label or "")
    if _quote_valid(quote, source.snippet):
        support = (
            EvidenceSupportLevel.PRIMARY_SOURCE_VERIFIED
            if source.credibility == "HIGH"
            else EvidenceSupportLevel.QUOTE_VERIFIED
        )
        return _evidence_from_source(
            source,
            claim=data.get("claim", ""),
            quote=quote,
            label=label,
            support_level=support,
            confidence=data.get("confidence", "medium"),
        )

    raw_content = source.raw_content or ""
    if raw_content and _quote_valid(quote, raw_content):
        return _evidence_from_source(
            source,
            claim=data.get("claim", ""),
            quote=quote,
            label=label,
            support_level=EvidenceSupportLevel.RAW_CONTENT_SUPPORTED,
            confidence="low",
        )

    if source.snippet:
        logger.warning(
            "Quote validation failed for %s — downgrading to snippet_supported. quote=%r",
            hit.url,
            quote[:80],
        )
        return _evidence_from_source(source, label=label)

    return None


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
    source = _source_from_hit(hit)
    if source is None:
        raise ValueError("Cannot build fallback evidence without source support")
    evidence = _evidence_from_source(source)
    if evidence is None:
        raise ValueError("Cannot build fallback evidence without source support")
    return evidence


# ── Main node ─────────────────────────────────────────────────────────────────

async def evidence_retriever_node(state: dict) -> dict:
    run_id: str = state["run_id"]
    snapshot: dict = state.get("snapshot") or {}
    metrics = metrics_for(state)
    stage_t0 = time.monotonic()

    from src.config import settings
    from src.retrieval.reranker import bm25_rerank
    from src.retrieval.search import TavilySearchClient

    question        = snapshot.get("question", "")
    description     = snapshot.get("description")
    resolution_src  = snapshot.get("resolution_source")

    anthropic_info = safe_secret_info(settings.anthropic_api_key, expected_prefix="sk-")
    deepseek_info = safe_secret_info(settings.deepseek_api_key, expected_prefix="sk-")

    use_deepseek = deepseek_info["present"]
    use_anthropic = not use_deepseek and anthropic_info["present"] and anthropic_info["prefix_ok"]

    # ── Generate queries: LLM if key appears usable, template fallback otherwise ──
    if use_deepseek:
        await push_log(run_id, "Generating search queries via DeepSeek...", "info")
        queries = await _generate_queries_llm(
            question, description, resolution_src,
            settings.deepseek_model, settings.deepseek_api_key,
            provider="deepseek"
        )
    elif use_anthropic:
        await push_log(run_id, "Generating search queries via Claude...", "info")
        queries = await _generate_queries_llm(
            question, description, resolution_src,
            settings.claude_model_fast, settings.anthropic_api_key,
            provider="anthropic"
        )
    else:
        await push_log(run_id, "No valid LLM key found. Using template fallback queries.", "warn")
        queries = _generate_queries_template(question, resolution_src, description)

    metrics["planned_queries"] = [{"query": q, "label": label} for q, label in queries]
    set_latency(metrics, "query_planner", stage_t0)

    await push_log(run_id, f"Dispatching {len(queries)} Tavily search queries...", "info")
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
            "run_metrics": metrics,
        }

    client = TavilySearchClient(settings.tavily_api_key)
    search_t0 = time.monotonic()

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

    metrics["search_requests_sent"] = len(query_log)
    metrics["raw_hits"] = len(all_hits)
    set_latency(metrics, "search", search_t0)
    await push_log(run_id, f"Search requests executed: {len(query_log)}/{len(queries)}", "info")

    await push_log(run_id, f"Retrieved {len(all_hits)} raw hits — reranking...", "info")

    # ── BM25 rerank and credibility filter ───────────────────────────────────
    rerank_t0 = time.monotonic()
    reranked = bm25_rerank(all_hits, question, top_k=12)
    medium_plus = [h for h in reranked if h.credibility in ("HIGH", "MEDIUM")]
    top_hits = medium_plus[:8] if medium_plus else reranked[:8]
    normalized_sources = [s for h in top_hits if (s := _source_from_hit(h)) is not None]
    metrics["unique_sources"] = len({s.source_id for s in normalized_sources})
    metrics["credible_sources"] = len([s for s in normalized_sources if s.credibility in ("HIGH", "MEDIUM")])
    set_latency(metrics, "rerank", rerank_t0)

    await push_log(run_id, f"  ✓ {len(top_hits)} sources passed credibility filter", "ok")

    # ── Extract evidence (LLM or fallback) ───────────────────────────────────
    extract_t0 = time.monotonic()
    raw_evidence: list[CitedEvidence] = []

    if use_deepseek:
        await push_log(run_id, "Extracting claims from sources (deepseek)...", "info")
        extraction_tasks = [
            _extract_and_validate(h, question, settings.deepseek_model, settings.deepseek_api_key, "deepseek")
            for h in top_hits
        ]
    elif use_anthropic:
        await push_log(run_id, "Extracting claims from sources (claude-sonnet)...", "info")
        extraction_tasks = [
            _extract_and_validate(h, question, settings.claude_model_fast, settings.anthropic_api_key, "anthropic")
            for h in top_hits
        ]
    else:
        await push_log(run_id, "  ⚠ LLM unavailable — using title-based fallback evidence", "warn")
        raw_evidence = []
        for h in top_hits:
            try:
                raw_evidence.append(_fallback_evidence(h))
            except ValueError:
                logger.warning("Evidence fallback skipped unsupported hit %s", getattr(h, "url", ""))

    if use_deepseek or use_anthropic:
        results = await asyncio.gather(*extraction_tasks, return_exceptions=True)

        for h, res in zip(top_hits, results):
            if isinstance(res, Exception):
                logger.warning("Extraction failed for %s: %s", h.url, res)
            elif res is not None:
                raw_evidence.append(res)
            else:
                logger.warning("Evidence dropped for %s (None result)", h.url)

        await push_log(run_id, f"  ✓ {len(raw_evidence)}/{len(top_hits)} items passed quote validation", "ok")
    set_latency(metrics, "evidence_extraction", extract_t0)

    # ── Dedup by URL ──────────────────────────────────────────────────────────
    cited: list[CitedEvidence] = _dedup_by_url(raw_evidence)
    for ev in cited:
        count_support(metrics, ev.support_level.value if hasattr(ev.support_level, "value") else str(ev.support_level))
    metrics["claims_generated"] = len(raw_evidence)
    metrics["claims_published"] = len(cited)

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
            "source_id":   ev.source_id,
            "side":        ev.side.value if hasattr(ev.side, "value") else ev.side,
            "support_level": ev.support_level.value if hasattr(ev.support_level, "value") else ev.support_level,
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
            "source_id": ev.source_id,
            "side": ev.side.value if hasattr(ev.side, "value") else ev.side,
            "support_level": ev.support_level.value if hasattr(ev.support_level, "value") else ev.support_level,
            "conf":    ev.confidence,
        }
        for ev in cited
    ]

    return {
        "search_results": search_results_compat,
        "search_queries": query_log,
        "normalized_sources": [s.model_dump() for s in normalized_sources],
        "sources":        sources_out,
        "cited_evidence": [ev.model_dump() for ev in cited],
        "run_metrics": metrics,
    }
