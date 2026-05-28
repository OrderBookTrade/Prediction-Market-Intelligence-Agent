"""LangGraph StateGraph — 4-node pipeline with conditional routing."""

from __future__ import annotations

import logging
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from src.agents.evidence_retriever import evidence_retriever_node
from src.agents.market_analyzer import market_analyzer_node
from src.agents.metrics import default_run_metrics
from src.agents.memo_writer import memo_writer_node
from src.agents.risk_critic import risk_critic_node

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    # Inputs
    condition_id: str
    run_id: str

    # Node outputs (accumulated through the graph)
    snapshot: dict | None
    search_results: list[Any]
    search_queries: list[Any]
    sources: list[Any]
    cited_evidence: list[Any]    # list[CitedEvidence.model_dump()] from evidence_retriever
    run_metrics: dict[str, Any]
    risk_flags: list[str]
    risk_details: dict[str, Any]
    memo: dict | None
    error: str | None


def _should_write_memo(state: AgentState) -> str:
    """Conditional routing after risk_critic.

    Aborts early (skips memo_writer) when the market is untradeable:
      - Critically low liquidity (< $500) AND LOW_LIQUIDITY flag raised
      - OR market is already at extreme probability (>98% or <2%), likely resolved

    Returns "abort" → END with a NO_TRADE memo stub.
    Returns "continue" → memo_writer for full analysis.
    """
    flags = state.get("risk_flags", [])
    snapshot = state.get("snapshot") or {}
    liquidity = snapshot.get("liquidity") or 0
    yes_price = snapshot.get("yes_price")

    # Critically illiquid — price is unreliable, no point analysing
    if "LOW_LIQUIDITY" in flags and liquidity < 500:
        logger.info(
            "Aborting memo — critically low liquidity $%.0f (run=%s)",
            liquidity, state.get("run_id"),
        )
        return "abort"

    # Market at extreme: already near resolution, edge opportunity is gone
    if yes_price is not None and (yes_price > 0.98 or yes_price < 0.02):
        logger.info(
            "Aborting memo — price at extreme %.3f, market likely resolved (run=%s)",
            yes_price, state.get("run_id"),
        )
        return "abort"

    return "continue"


async def _abort_node(state: AgentState) -> dict:
    """Emit a minimal NO_TRADE memo when the graph aborts early."""
    from src.run_store import push_log

    run_id = state.get("run_id", "")
    snapshot = state.get("snapshot") or {}
    flags = state.get("risk_flags", [])

    reason = (
        "critically low liquidity" if "LOW_LIQUIDITY" in flags
        else "market price at extreme (likely resolved)"
    )
    await push_log(run_id, f"  ✗ Analysis aborted — {reason}", "warn")
    await push_log(run_id, "  → Recommendation: NO_TRADE (untradeable market)", "warn")

    yes_price = snapshot.get("yes_price", 0.5) or 0.5
    memo = {
        "run_id": run_id,
        "condition_id": state.get("condition_id", ""),
        "market_question": snapshot.get("question", ""),
        "market_probability": yes_price,
        "agent_estimate": yes_price,
        "edge": 0.0,
        "confidence": "uncertain",
        "yes_case": [],
        "no_case": [],
        "resolution_source": snapshot.get("resolution_source") or "not specified",
        "resolution_deadline": snapshot.get("end_date") or "unknown",
        "resolution_condition": "Analysis aborted — see abort reason",
        "resolution_ambiguities": [],
        "resolution_risk_level": "high",
        "resolution_risk_notes": f"Aborted: {reason}",
        "manipulation_risk": "unknown",
        "manipulation_notes": "",
        "recommendation": "no_trade",
        "recommendation_rationale": f"Market aborted early: {reason}. Manual review required.",
        "key_uncertainties": [f"Abort reason: {reason}"],
        "model_name": "aborted",
        "prompt_version": "v1",
        "sources_found": 0,
    }
    return {"memo": memo}


def _build_graph() -> Any:
    workflow = StateGraph(AgentState)

    workflow.add_node("market_analyzer", market_analyzer_node)
    workflow.add_node("evidence_retriever", evidence_retriever_node)
    workflow.add_node("risk_critic", risk_critic_node)
    workflow.add_node("memo_writer", memo_writer_node)
    workflow.add_node("abort", _abort_node)

    workflow.set_entry_point("market_analyzer")
    workflow.add_edge("market_analyzer", "evidence_retriever")
    workflow.add_edge("evidence_retriever", "risk_critic")

    # Conditional routing: untradeable markets abort before memo_writer
    workflow.add_conditional_edges(
        "risk_critic",
        _should_write_memo,
        {"continue": "memo_writer", "abort": "abort"},
    )

    workflow.add_edge("memo_writer", END)
    workflow.add_edge("abort", END)

    return workflow.compile()


# Singleton — compiled once at import time
compiled_graph = _build_graph()


async def run_analysis(condition_id: str, run_id: str) -> dict:
    """Entry point called by the API route.

    Returns the final AgentState dict.
    """
    initial: AgentState = {
        "condition_id": condition_id,
        "run_id": run_id,
        "snapshot": None,
        "search_results": [],
        "search_queries": [],
        "sources": [],
        "cited_evidence": [],
        "run_metrics": default_run_metrics(),
        "risk_flags": [],
        "risk_details": {},
        "memo": None,
        "error": None,
    }

    try:
        result = await compiled_graph.ainvoke(initial)
        return result
    except Exception as exc:
        logger.error("Graph execution failed for run %s: %s", run_id, exc, exc_info=True)
        return {**initial, "error": str(exc)}
