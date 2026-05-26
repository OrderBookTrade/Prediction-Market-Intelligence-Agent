"""Development-only debug endpoints for provider configuration checks."""

from __future__ import annotations

import time

import httpx
from fastapi import APIRouter, Query

from src.config import settings
from src.utils.secrets import safe_secret_info

router = APIRouter(prefix="/debug", tags=["debug"])


@router.get("/anthropic")
async def debug_anthropic(
    probe: bool = Query(False, description="Send a 1-token request to Anthropic"),
) -> dict:
    """Return safe Anthropic config metadata and, optionally, live auth status."""
    key = settings.anthropic_api_key
    out: dict = {
        "api_key": safe_secret_info(key, expected_prefix="sk-"),
        "model": settings.claude_model,
        "fast_model": settings.claude_model_fast,
        "probe": {"request_sent": False},
    }

    if not probe or not key:
        return out

    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": key.strip().strip('"').strip("'"),
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": settings.claude_model_fast,
                    "max_tokens": 1,
                    "messages": [{"role": "user", "content": "ping"}],
                },
            )
        out["probe"] = {
            "request_sent": True,
            "status": resp.status_code,
            "duration_ms": int((time.monotonic() - t0) * 1000),
            "ok": resp.is_success,
            "error_type": (resp.json().get("error") or {}).get("type") if not resp.is_success else None,
            "error_message": (resp.json().get("error") or {}).get("message") if not resp.is_success else None,
        }
    except Exception as exc:
        out["probe"] = {
            "request_sent": True,
            "status": None,
            "duration_ms": int((time.monotonic() - t0) * 1000),
            "ok": False,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
    return out


@router.get("/tavily")
async def debug_tavily(
    q: str = Query(..., min_length=1),
    max_results: int = Query(5, ge=1, le=10),
) -> dict:
    """Run one direct Tavily query and return provider status without secrets."""
    key = settings.tavily_api_key
    out: dict = {
        "api_key": safe_secret_info(key, expected_prefix="tvl"),
        "query": q,
        "request_sent": False,
    }
    if not key:
        return out | {"status": None, "duration_ms": 0, "result_count": 0, "results": []}

    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": key.strip().strip('"').strip("'"),
                    "query": q,
                    "max_results": max_results,
                    "search_depth": "basic",
                },
            )
        duration_ms = int((time.monotonic() - t0) * 1000)
        payload = resp.json() if resp.content else {}
        results = [
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "domain": item.get("url", "").split("/")[2].removeprefix("www.")
                if "://" in item.get("url", "")
                else item.get("url", ""),
            }
            for item in payload.get("results", [])[:max_results]
        ]
        return out | {
            "request_sent": True,
            "status": resp.status_code,
            "duration_ms": duration_ms,
            "ok": resp.is_success,
            "result_count": len(results),
            "results": results,
            "error": None if resp.is_success else payload,
        }
    except Exception as exc:
        return out | {
            "request_sent": True,
            "status": None,
            "duration_ms": int((time.monotonic() - t0) * 1000),
            "ok": False,
            "result_count": 0,
            "results": [],
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
