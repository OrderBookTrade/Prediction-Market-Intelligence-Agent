"""LangGraph StateGraph — 4-node sequential pipeline."""

from __future__ import annotations

import logging
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from src.agents.evidence_retriever import evidence_retriever_node
from src.agents.market_analyzer import market_analyzer_node
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
    risk_flags: list[str]
    risk_details: dict[str, Any]
    memo: dict | None
    error: str | None


def _build_graph() -> Any:
    workflow = StateGraph(AgentState)

    workflow.add_node("market_analyzer", market_analyzer_node)
    workflow.add_node("evidence_retriever", evidence_retriever_node)
    workflow.add_node("risk_critic", risk_critic_node)
    workflow.add_node("memo_writer", memo_writer_node)

    workflow.set_entry_point("market_analyzer")
    workflow.add_edge("market_analyzer", "evidence_retriever")
    workflow.add_edge("evidence_retriever", "risk_critic")
    workflow.add_edge("risk_critic", "memo_writer")
    workflow.add_edge("memo_writer", END)

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
