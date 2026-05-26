"""Run-level observability counters for the agent pipeline."""

from __future__ import annotations

import time
from typing import Any


def default_run_metrics() -> dict[str, Any]:
    return {
        "planned_queries": [],
        "search_requests_sent": 0,
        "raw_hits": 0,
        "unique_sources": 0,
        "credible_sources": 0,
        "snippet_supported_evidence": 0,
        "quote_verified_evidence": 0,
        "claims_generated": 0,
        "claims_published": 0,
        "fallback_reason": None,
        "latency_ms": {},
        "token_input": 0,
        "token_output": 0,
        "estimated_cost": None,
    }


def metrics_for(state: dict) -> dict[str, Any]:
    metrics = dict(default_run_metrics())
    metrics.update(state.get("run_metrics") or {})
    metrics["latency_ms"] = {
        **default_run_metrics()["latency_ms"],
        **((state.get("run_metrics") or {}).get("latency_ms") or {}),
    }
    return metrics


def set_latency(metrics: dict[str, Any], stage: str, started_at: float) -> None:
    metrics.setdefault("latency_ms", {})[stage] = int((time.monotonic() - started_at) * 1000)


def count_support(metrics: dict[str, Any], support_level: str) -> None:
    if support_level in ("quote_verified", "primary_source_verified"):
        metrics["quote_verified_evidence"] = metrics.get("quote_verified_evidence", 0) + 1
    elif support_level in ("snippet_supported", "raw_content_supported"):
        metrics["snippet_supported_evidence"] = metrics.get("snippet_supported_evidence", 0) + 1
