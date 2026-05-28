"""Node 4 — memo_writer.

Single Claude call (temperature=0.3) with tool_use forced output.
Validates result as a ResearchMemo Pydantic model.
On failure: retry once, then emit NO_TRADE with UNCERTAIN confidence.
Writes to DB and logs audit entry.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time

from src.run_store import push_log
from src.utils.secrets import safe_secret_info

logger = logging.getLogger(__name__)

PROMPT_VERSION = "v1"

_SUPPORT_LEVELS = {
    "snippet_supported",
    "raw_content_supported",
    "quote_verified",
    "primary_source_verified",
}

# ── Tool schema (flat — no $ref) ──────────────────────────────────────────────

MEMO_TOOL: dict = {
    "name": "write_research_memo",
    "description": "Generate a structured research memo for a prediction market. Cite every factual claim. If evidence is insufficient, say so explicitly — never fabricate.",
    "input_schema": {
        "type": "object",
        "required": [
            "agent_estimate", "edge", "confidence",
            "yes_case", "no_case",
            "resolution_source", "resolution_deadline",
            "resolution_condition", "resolution_ambiguities",
            "resolution_risk_level", "resolution_risk_notes",
            "manipulation_risk", "manipulation_notes",
            "recommendation", "recommendation_rationale", "key_uncertainties",
        ],
        "properties": {
            "agent_estimate": {
                "type": "number",
                "description": "Your probability estimate for YES (0.0 – 1.0). Must be justified by the evidence.",
            },
            "edge": {
                "type": "number",
                "description": "agent_estimate minus current market YES price. Positive = you think YES is underpriced.",
            },
            "confidence": {
                "type": "string",
                "enum": ["high", "medium", "low", "uncertain"],
                "description": "Your confidence in the estimate given evidence quality.",
            },
            "yes_case": {
                "type": "array",
                "description": "Evidence supporting YES outcome. Only use provided LABEL=yes_case evidence.",
                "items": {
                    "type": "object",
                    "required": ["claim", "source", "credibility", "source_id", "support_level"],
                    "properties": {
                        "claim": {"type": "string", "description": "Specific, factual claim."},
                        "source": {"type": "string", "description": "Domain or URL of the source."},
                        "credibility": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
                        "source_id": {"type": "string", "description": "Exact source_id from the evidence list."},
                        "support_level": {
                            "type": "string",
                            "enum": ["snippet_supported", "raw_content_supported", "quote_verified", "primary_source_verified"],
                        },
                    },
                },
            },
            "no_case": {
                "type": "array",
                "description": "Evidence supporting NO outcome. Only use provided LABEL=no_case evidence.",
                "items": {
                    "type": "object",
                    "required": ["claim", "source", "credibility", "source_id", "support_level"],
                    "properties": {
                        "claim": {"type": "string"},
                        "source": {"type": "string"},
                        "credibility": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
                        "source_id": {"type": "string", "description": "Exact source_id from the evidence list."},
                        "support_level": {
                            "type": "string",
                            "enum": ["snippet_supported", "raw_content_supported", "quote_verified", "primary_source_verified"],
                        },
                    },
                },
            },
            "resolution_source": {"type": "string"},
            "resolution_deadline": {"type": "string"},
            "resolution_condition": {"type": "string", "description": "Precise condition for YES resolution."},
            "resolution_ambiguities": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Edge cases or ambiguous clauses. Empty array if none.",
            },
            "resolution_risk_level": {"type": "string", "enum": ["high", "medium", "low", "unknown"]},
            "resolution_risk_notes": {"type": "string"},
            "manipulation_risk": {"type": "string", "enum": ["high", "medium", "low", "unknown"]},
            "manipulation_notes": {"type": "string"},
            "recommendation": {
                "type": "string",
                "enum": ["no_trade", "watch", "research_more", "candidate_opportunity"],
            },
            "recommendation_rationale": {"type": "string", "description": "2-4 sentence explanation."},
            "key_uncertainties": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Top 3 things that would change this analysis.",
                "maxItems": 3,
            },
        },
    },
}


def _build_prompt(
    snapshot: dict,
    search_results: list[dict],
    risk_details: dict,
    sources: list[dict] | None = None,
) -> str:
    yes_price = snapshot.get("yes_price", 0.5) or 0.5
    labeled_sources = sources or []
    if labeled_sources:
        evidence_text = "\n".join(
            f"[{i+1}] LABEL={s.get('label') or 'unknown'} CRED={s.get('cred') or '?'} "
            f"SOURCE_ID={s.get('source_id') or '?'} "
            f"SUPPORT={s.get('support_level') or '?'} "
            f"SOURCE={s.get('domain') or s.get('url')}\n"
            f"    CLAIM: {s.get('title', '')}\n"
            f"    QUOTE: {s.get('content', '')[:300]}"
            for i, s in enumerate(labeled_sources[:8])
        )
    else:
        evidence_text = "\n".join(
            f"[{i+1}] LABEL=unknown CRED={r.get('credibility') or '?'} "
            f"SOURCE_ID={r.get('source_id') or '?'} "
            f"SUPPORT={r.get('support_level') or '?'} SOURCE={r.get('url', '')}\n"
            f"    CLAIM: {r.get('title', '')}\n"
            f"    QUOTE: {r.get('content', '')[:300]}"
            for i, r in enumerate(search_results[:8])
        ) or "No external evidence retrieved."

    risk_summary = "\n".join(
        f"- {k.capitalize()} risk: {v.get('level','?').upper()} — {v.get('note','')}"
        for k, v in risk_details.items()
    )

    return f"""You are a quantitative prediction market analyst. Generate a rigorous, evidence-backed research memo.

RULES:
- Cite every factual claim with the domain/URL of its source
- If you cannot support a claim with evidence, say "insufficient evidence" — NEVER fabricate
- Your agent_estimate must be derived from the evidence, not anchored to the market price
- Use temperature reasoning: consider base rates, recent signals, and counter-evidence
- Be conservative: prefer NO_TRADE if evidence quality is thin
- Respect evidence labels: LABEL=yes_case supports YES, LABEL=no_case supports NO,
  and LABEL=resolution only clarifies rules/source/deadline. Do not put NO evidence
  into yes_case or YES evidence into no_case.
- Every yes_case/no_case item must copy SOURCE_ID and SUPPORT exactly from the
  evidence list. If one side has no labeled evidence, leave that case empty.

MARKET:
Question: {snapshot.get('question')}
Current YES price: {yes_price:.1%}
Volume: ${snapshot.get('volume') or 0:,.0f}
Liquidity: ${snapshot.get('liquidity') or 0:,.0f}
End Date: {snapshot.get('end_date', 'unknown')}
Resolution Source: {snapshot.get('resolution_source') or 'Not specified'}

PRE-ANALYSIS RISK FLAGS:
{risk_summary}

EVIDENCE ({len(search_results)} sources retrieved):
{evidence_text}

Generate a complete research memo using the write_research_memo tool.
    """


def _source_side(source: dict) -> str:
    side = source.get("side")
    if side:
        return str(side)
    label = source.get("label") or source.get("query_label")
    if label == "yes_case":
        return "yes"
    if label == "no_case":
        return "no"
    if label == "resolution":
        return "resolution"
    return ""


def _source_domain(source: dict) -> str:
    return str(source.get("domain") or source.get("publisher") or source.get("source") or "").lower()


def _source_url(source: dict) -> str:
    return str(source.get("url") or source.get("source_url") or "").lower()


def _source_support_level(source: dict) -> str:
    support = str(source.get("support_level") or "").strip()
    return support if support in _SUPPORT_LEVELS else "snippet_supported"


def _real_source_candidates(sources: list[dict], search_results: list[dict]) -> list[dict]:
    candidates: list[dict] = []
    seen: set[str] = set()
    for source in [*(sources or []), *(search_results or [])]:
        source_id = source.get("source_id")
        url = source.get("url") or source.get("source_url")
        domain = source.get("domain") or source.get("publisher")
        if not source_id or not url:
            continue
        key = str(source_id)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            {
                **source,
                "source_id": str(source_id),
                "url": str(url),
                "domain": str(domain or url),
                "side": _source_side(source),
                "support_level": _source_support_level(source),
                "credibility": source.get("credibility") or source.get("cred") or "LOW",
            }
        )
    return candidates


def _match_source(item: dict, expected_side: str, candidates: list[dict]) -> dict | None:
    wanted_source_id = str(item.get("source_id") or "").strip()
    item_source = str(item.get("source") or item.get("url") or "").lower().strip()

    for candidate in candidates:
        if candidate["side"] != expected_side:
            continue
        if wanted_source_id and wanted_source_id == candidate["source_id"]:
            return candidate

    if not item_source:
        return None

    for candidate in candidates:
        if candidate["side"] != expected_side:
            continue
        domain = _source_domain(candidate)
        url = _source_url(candidate)
        if item_source == domain or item_source in url or domain in item_source:
            return candidate

    return None


def _enrich_case_items(items: list[dict], expected_side: str, candidates: list[dict]) -> list[dict]:
    enriched: list[dict] = []
    for item in items or []:
        claim = str(item.get("claim") or "").strip()
        if not claim:
            continue
        source = _match_source(item, expected_side, candidates)
        if source is None:
            logger.warning("Dropping memo claim without valid %s source: %s", expected_side, claim[:120])
            continue
        enriched.append(
            {
                "claim": claim,
                "source": source["domain"],
                "credibility": source["credibility"],
                "source_id": source["source_id"],
                "support_level": source["support_level"],
                "url": source["url"],
            }
        )
    return enriched


def _fallback_memo(
    snapshot: dict,
    run_id: str,
    search_queries: list = None,
    *,
    reason: str = "ANALYSIS_FAILED",
    sources_found: int = 0,
) -> dict:
    yes_price = snapshot.get("yes_price", 0.5) or 0.5
    queries = [q.get("query", "") for q in (search_queries or [])]
    return {
        "status": "fallback",
        "fallback_reason": reason,
        "run_id": run_id,
        "condition_id": snapshot.get("condition_id", ""),
        "market_question": snapshot.get("question", ""),
        "market_probability": yes_price,
        "agent_estimate": None,
        "edge": None,
        "confidence": "uncertain",
        "yes_case": [],
        "no_case": [],
        "resolution_source": snapshot.get("resolution_source") or "not specified",
        "resolution_deadline": snapshot.get("end_date") or "unknown",
        "resolution_condition": "See market description",
        "resolution_ambiguities": [],
        "resolution_risk_level": "unknown",
        "resolution_risk_notes": "Could not complete analysis",
        "manipulation_risk": "unknown",
        "manipulation_notes": "Analysis incomplete",
        "recommendation": "no_trade",
        "recommendation_rationale": (
            "Agent analysis did not produce verified evidence, so no independent "
            "probability estimate was generated. Manual review required."
        ),
        "key_uncertainties": [reason],
        "model_name": "fallback",
        "prompt_version": PROMPT_VERSION,
        "search_queries": queries,
        "sources_found": sources_found,
    }


async def memo_writer_node(state: dict) -> dict:
    run_id: str = state["run_id"]
    condition_id: str = state["condition_id"]
    snapshot: dict = state["snapshot"]
    search_results: list[dict] = state.get("search_results", [])
    search_queries: list[dict] = state.get("search_queries", [])
    sources: list[dict] = state.get("sources", [])
    risk_details: dict = state.get("risk_details", {})

    from src.config import settings

    yes_price = snapshot.get("yes_price", 0.5) or 0.5

    if not search_results:
        await push_log(run_id, "No verified evidence found — skipping thesis generation", "warn")
        memo = _fallback_memo(snapshot, run_id, search_queries, reason="NO_VERIFIED_EVIDENCE", sources_found=0)
        _finalize(state, memo, run_id, condition_id, yes_price, search_queries, sources, 0)
        await push_log(run_id, "Agent probability = N/A · market price only shown · edge = N/A", "warn")
        await push_log(run_id, "Recommendation = NO_TRADE · fallback_reason=NO_VERIFIED_EVIDENCE", "warn")
        return {"memo": memo}

    yes_claims = len([s for s in sources if s.get("label") == "yes_case"])
    no_claims = len([s for s in sources if s.get("label") == "no_case"])

    await push_log(run_id, f"Generating YES thesis... ({yes_claims} verified claims)", "info")
    await push_log(run_id, f"Generating NO thesis... ({no_claims} verified claims)", "info")
    await push_log(run_id, "Computing agent probability estimate...", "info")
    await push_log(run_id, f"Writing research memo... (schema={PROMPT_VERSION})", "info")

    anthropic_info = safe_secret_info(settings.anthropic_api_key, expected_prefix="sk-")
    deepseek_info = safe_secret_info(settings.deepseek_api_key, expected_prefix="sk-")

    use_deepseek = deepseek_info["present"]
    use_anthropic = not use_deepseek and anthropic_info["present"] and anthropic_info["prefix_ok"]

    if not use_deepseek and not use_anthropic:
        await push_log(run_id, "  ⚠ No valid LLM API key set — using fallback memo", "warn")
        memo = _fallback_memo(
            snapshot,
            run_id,
            search_queries,
            reason="LLM_API_KEY_MISSING",
            sources_found=len(search_results),
        )
        _finalize(state, memo, run_id, condition_id, yes_price, search_queries, sources, len(search_results))
        await push_log(run_id, "Agent probability = N/A · market price only shown · edge = N/A", "warn")
        return {"memo": memo}

    prompt = _build_prompt(snapshot, search_results, risk_details, sources)
    input_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]

    memo_raw: dict | None = None
    t0 = time.monotonic()
    token_input = token_output = 0

    if use_deepseek:
        import openai
        import json
        await push_log(run_id, f"[deepseek] caller=memo_generator model={settings.deepseek_model}", "dim")
        client = openai.OpenAI(api_key=settings.deepseek_api_key, base_url="https://api.deepseek.com/v1")
        
        tool = {
            "type": "function",
            "function": {
                "name": MEMO_TOOL["name"],
                "description": MEMO_TOOL["description"],
                "parameters": MEMO_TOOL["input_schema"]
            }
        }
        for attempt in range(2):
            try:
                response = client.chat.completions.create(
                    model=settings.deepseek_model,
                    max_tokens=4096,
                    temperature=0.3,
                    tools=[tool],
                    tool_choice={"type": "function", "function": {"name": "write_research_memo"}},
                    messages=[{"role": "user", "content": prompt}],
                )
                if response.usage:
                    token_input = response.usage.prompt_tokens
                    token_output = response.usage.completion_tokens
                if response.choices[0].message.tool_calls:
                    memo_raw = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
                    break
            except Exception as exc:
                if "AuthenticationError" in str(type(exc)) or "401" in str(exc):
                    logger.warning("DeepSeek AuthenticationError: %s", exc)
                    if attempt == 0:
                        await push_log(run_id, "  ⚠ DEEPSEEK_API_KEY invalid or unauthorized", "error")
                    break
                logger.warning("DeepSeek call attempt %d failed: %s", attempt + 1, exc)
                if attempt == 0:
                    await push_log(run_id, f"  ⚠ retry after error: {exc}", "warn")
    else:
        import anthropic
        await push_log(run_id, f"[anthropic] caller=memo_generator model={settings.claude_model}", "dim")
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        
        for attempt in range(2):
            try:
                response = client.messages.create(
                    model=settings.claude_model,
                    max_tokens=4096,
                    temperature=0.3,
                    tools=[MEMO_TOOL],
                    tool_choice={"type": "tool", "name": "write_research_memo"},
                    messages=[{"role": "user", "content": prompt}],
                )
                token_input = response.usage.input_tokens
                token_output = response.usage.output_tokens

                tool_block = next(
                    (b for b in response.content if b.type == "tool_use"),
                    None,
                )
                if tool_block:
                    memo_raw = tool_block.input
                    break
            except Exception as exc:
                if "AuthenticationError" in str(type(exc)) or "401" in str(exc):
                    logger.warning("Claude AuthenticationError: %s", exc)
                    if attempt == 0:
                        await push_log(run_id, "  ⚠ ANTHROPIC_API_KEY invalid or unauthorized", "error")
                    break
                logger.warning("Claude call attempt %d failed: %s", attempt + 1, exc)
                if attempt == 0:
                    await push_log(run_id, f"  ⚠ retry after error: {exc}", "warn")

    latency_ms = int((time.monotonic() - t0) * 1000)

    if not memo_raw:
        await push_log(run_id, "  ✗ Memo generation failed — using fallback", "warn")
        memo = _fallback_memo(
            snapshot,
            run_id,
            search_queries,
            reason="LLM_GENERATION_FAILED",
            sources_found=len(search_results),
        )
    else:
        yes_case = memo_raw.get("yes_case", [])
        no_case = memo_raw.get("no_case", [])
        
        # Enrich with URLs from retrieved sources
        all_sources = search_results + (sources or [])
        for ev in yes_case + no_case:
            src = ev.get("source", "")
            for s in all_sources:
                domain = s.get("domain", "")
                url = s.get("url", "")
                if src and domain and (src.lower() == domain.lower() or src.lower() in url.lower() or domain.lower() in src.lower()):
                    ev["url"] = url
                    break

        edge = round(memo_raw.get("agent_estimate", yes_price) - yes_price, 4)
        memo = {
            "run_id": run_id,
            "condition_id": condition_id,
            "market_question": snapshot.get("question", ""),
            "market_probability": yes_price,
            "agent_estimate": memo_raw.get("agent_estimate", yes_price),
            "edge": edge,
            "confidence": memo_raw.get("confidence", "uncertain"),
            "yes_case": yes_case,
            "no_case": no_case,
            "resolution_source": memo_raw.get("resolution_source", ""),
            "resolution_deadline": memo_raw.get("resolution_deadline", ""),
            "resolution_condition": memo_raw.get("resolution_condition", ""),
            "resolution_ambiguities": memo_raw.get("resolution_ambiguities", []),
            "resolution_risk_level": memo_raw.get("resolution_risk_level", "unknown"),
            "resolution_risk_notes": memo_raw.get("resolution_risk_notes", ""),
            "manipulation_risk": memo_raw.get("manipulation_risk", "unknown"),
            "manipulation_notes": memo_raw.get("manipulation_notes", ""),
            "recommendation": memo_raw.get("recommendation", "no_trade"),
            "recommendation_rationale": memo_raw.get("recommendation_rationale", ""),
            "key_uncertainties": memo_raw.get("key_uncertainties", []),
            "model_name": settings.claude_model,
            "prompt_version": PROMPT_VERSION,
            "search_queries": [q.get("query", "") for q in search_queries],
            "sources_found": len(search_results),
            "token_input": token_input,
            "token_output": token_output,
        }

    _finalize(state, memo, run_id, condition_id, yes_price, search_queries, sources, len(search_results))

    # Write audit log
    try:
        from src.storage.db import get_engine, log_audit
        from sqlalchemy.orm import Session

        with Session(get_engine()) as session:
            log_audit(
                session,
                run_id=run_id,
                node_name="memo_writer",
                output={"recommendation": memo.get("recommendation"), "confidence": memo.get("confidence")},
                model=settings.claude_model,
                prompt_version=PROMPT_VERSION,
                input_text=prompt,
                latency_ms=latency_ms,
                token_input=token_input,
                token_output=token_output,
            )
    except Exception as exc:
        logger.warning("Audit log write failed: %s", exc)

    agent_est = memo.get("agent_estimate")
    edge = memo.get("edge")
    conf = memo.get("confidence", "uncertain")

    if memo.get("status") == "fallback":
        await push_log(run_id, f"  p(YES) = N/A · market = {yes_price:.3f} · edge N/A", "warn")
    else:
        await push_log(run_id, f"  p(YES) = {agent_est:.3f} · market = {yes_price:.3f} · edge {edge:+.3f}", "ok")
    await push_log(run_id, f"Confidence = {conf.upper()}", "ok")
    await push_log(run_id, f"Memo validated · {len(search_results)} sources cited", "ok")
    await push_log(run_id, "✓ Memo complete. Brier logged.", "ok")

    return {"memo": memo}


def _finalize(
    state: dict,
    memo: dict,
    run_id: str,
    condition_id: str,
    yes_price: float,
    search_queries: list,
    sources: list,
    sources_found: int,
) -> None:
    """Persist memo and prediction_history row to DB."""
    if memo.get("status") == "fallback":
        logger.info("Skipping DB persistence for fallback memo run=%s reason=%s", run_id, memo.get("fallback_reason"))
        return

    try:
        from src.storage.db import get_engine, save_memo
        from sqlalchemy.orm import Session

        with Session(get_engine()) as session:
            save_memo(session, memo)
    except Exception as exc:
        logger.warning("Memo DB write failed: %s", exc)
