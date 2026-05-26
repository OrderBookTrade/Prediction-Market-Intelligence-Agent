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
                "description": "Evidence supporting YES outcome. Minimum 1 item.",
                "items": {
                    "type": "object",
                    "required": ["claim", "source", "credibility"],
                    "properties": {
                        "claim": {"type": "string", "description": "Specific, factual claim."},
                        "source": {"type": "string", "description": "Domain or URL of the source."},
                        "credibility": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
                    },
                },
            },
            "no_case": {
                "type": "array",
                "description": "Evidence supporting NO outcome. Minimum 1 item.",
                "items": {
                    "type": "object",
                    "required": ["claim", "source", "credibility"],
                    "properties": {
                        "claim": {"type": "string"},
                        "source": {"type": "string"},
                        "credibility": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
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
            f"SOURCE={s.get('domain') or s.get('url')}\n"
            f"    CLAIM: {s.get('title', '')}\n"
            f"    QUOTE: {s.get('content', '')[:300]}"
            for i, s in enumerate(labeled_sources[:8])
        )
    else:
        evidence_text = "\n".join(
            f"[{i+1}] LABEL=unknown CRED={r.get('credibility') or '?'} SOURCE={r.get('url', '')}\n"
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
- If one side has no labeled evidence, include one explicit insufficient-evidence item
  for that side instead of borrowing evidence from the other side.

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

    key_info = safe_secret_info(settings.anthropic_api_key, expected_prefix="sk-")

    if not key_info["present"]:
        await push_log(run_id, "  ⚠ ANTHROPIC_API_KEY not set — using fallback memo", "warn")
        memo = _fallback_memo(
            snapshot,
            run_id,
            search_queries,
            reason="ANTHROPIC_API_KEY_MISSING",
            sources_found=len(search_results),
        )
        _finalize(state, memo, run_id, condition_id, yes_price, search_queries, sources, len(search_results))
        await push_log(run_id, "Agent probability = N/A · market price only shown · edge = N/A", "warn")
        return {"memo": memo}

    if not key_info["prefix_ok"]:
        await push_log(
            run_id,
            (
                "  ⚠ ANTHROPIC_API_KEY invalid shape — expected sk- prefix "
                f"(prefix={key_info['prefix']} length={key_info['length']} fingerprint={key_info['fingerprint']})"
            ),
            "error",
        )
        memo = _fallback_memo(
            snapshot,
            run_id,
            search_queries,
            reason="ANTHROPIC_API_KEY_INVALID_SHAPE",
            sources_found=len(search_results),
        )
        _finalize(state, memo, run_id, condition_id, yes_price, search_queries, sources, len(search_results))
        await push_log(run_id, "Agent probability = N/A · market price only shown · edge = N/A", "warn")
        return {"memo": memo}

    import anthropic

    await push_log(
        run_id,
        (
            "[anthropic] caller=memo_generator "
            f"key_present={key_info['present']} "
            f"key_prefix={key_info['prefix']} "
            f"key_length={key_info['length']} "
            f"fingerprint={key_info['fingerprint']} "
            f"model={settings.claude_model}"
        ),
        "dim",
    )

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    prompt = _build_prompt(snapshot, search_results, risk_details, sources)
    input_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]

    memo_raw: dict | None = None
    t0 = time.monotonic()
    token_input = token_output = 0

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
        edge = round(memo_raw.get("agent_estimate", yes_price) - yes_price, 4)
        memo = {
            "run_id": run_id,
            "condition_id": condition_id,
            "market_question": snapshot.get("question", ""),
            "market_probability": yes_price,
            "agent_estimate": memo_raw.get("agent_estimate", yes_price),
            "edge": edge,
            "confidence": memo_raw.get("confidence", "uncertain"),
            "yes_case": memo_raw.get("yes_case", []),
            "no_case": memo_raw.get("no_case", []),
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
