"""POST /api/agent/run/{condition_id}  ·  GET /api/agent/run/{run_id}/stream  ·  GET /api/agent/run/{run_id}/result"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from sse_starlette.sse import EventSourceResponse

from src.run_store import cleanup_run, get_queue, get_result, push_done, push_log, register_run, set_result

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["agent"])


async def _execute_run(condition_id: str, run_id: str) -> None:
    """Background coroutine: runs the full agent graph and handles DB lifecycle."""
    from src.agents.graph import run_analysis
    from src.storage.db import get_engine, update_run_status
    from sqlalchemy.orm import Session

    engine = get_engine()

    def _set_running():
        with Session(engine) as s:
            update_run_status(s, run_id, "running")

    await run_in_threadpool(_set_running)

    try:
        result = await run_analysis(condition_id, run_id)

        if result.get("error"):
            def _set_error():
                with Session(engine) as s:
                    update_run_status(s, run_id, "error", error_message=result["error"])
            await run_in_threadpool(_set_error)
            await push_log(run_id, f"ERROR: {result['error']}", "error")
            set_result(run_id, {"status": "error", "error": result["error"]})
        else:
            def _set_done():
                with Session(engine) as s:
                    update_run_status(s, run_id, "done")
            await run_in_threadpool(_set_done)
            set_result(run_id, {"status": "done", "memo": result.get("memo")})

    except Exception as exc:
        logger.exception("Agent run %s crashed", run_id)
        set_result(run_id, {"status": "error", "error": str(exc)})
        await push_log(run_id, f"FATAL: {exc}", "error")
    finally:
        await push_done(run_id)
        # Allow queue to drain before cleanup
        asyncio.get_event_loop().call_later(120, lambda: cleanup_run(run_id))


@router.post("/run/{condition_id}", status_code=202)
async def start_run(condition_id: str) -> dict:
    """Launch a new agent analysis. Returns run_id immediately."""
    from src.storage.db import create_agent_run, get_engine
    from sqlalchemy.orm import Session

    run_id = str(uuid.uuid4())

    # Register SSE queue before starting task so client can subscribe immediately
    register_run(run_id)

    # Persist run record
    def _create():
        with Session(get_engine()) as s:
            create_agent_run(s, run_id, condition_id)

    await run_in_threadpool(_create)

    # Fire and forget
    asyncio.create_task(_execute_run(condition_id, run_id))

    return {"run_id": run_id, "condition_id": condition_id, "status": "queued"}


@router.get("/run/{run_id}/stream")
async def stream_run(run_id: str) -> EventSourceResponse:
    """SSE stream — push log events until None sentinel or client disconnects."""
    queue = get_queue(run_id)
    if queue is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found or expired")

    async def _generate():
        while True:
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                # Keep-alive ping
                yield {"data": json.dumps({"kind": "ping", "time": datetime.now(timezone.utc).strftime("%H:%M:%S"), "text": ""})}
                continue

            if msg is None:
                # Sentinel — stream complete
                yield {"data": json.dumps({"kind": "done", "time": datetime.now(timezone.utc).strftime("%H:%M:%S"), "text": "Stream complete"})}
                break

            yield {"data": json.dumps(msg)}

    return EventSourceResponse(_generate())


@router.get("/run/{run_id}/result")
async def get_run_result(run_id: str) -> dict:
    """Poll for the final memo after the stream completes."""
    result = get_result(run_id)
    if result is None:
        # Check DB
        from src.storage.db import get_engine, get_run
        from sqlalchemy.orm import Session

        def _query():
            with Session(get_engine()) as s:
                return get_run(s, run_id)

        run = await run_in_threadpool(_query)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")
        return {"status": run.status, "run_id": run_id}

    if result.get("status") == "done" and result.get("memo"):
        # Convert memo to the MemoOut format
        from src.api.routes.memos import REC_MAP

        data = result["memo"]
        rec = data.get("recommendation", "no_trade")
        rec_meta = REC_MAP.get(rec, {"label": rec.upper(), "tone": "gray"})

        return {
            "status": "done",
            "run_id": run_id,
            "memo": {
                "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                "model": data.get("model_name", "unknown"),
                "prompt_version": data.get("prompt_version", "v1"),
                "recommendation": rec_meta,
                "market_implied": data.get("market_probability", 0.5),
                "agent_estimate": data.get("agent_estimate", 0.5),
                "edge": data.get("edge", 0.0),
                "confidence": data.get("confidence", "uncertain").upper(),
                "yes_case": data.get("yes_case", []),
                "no_case": data.get("no_case", []),
                "resolution": {
                    "official_source": data.get("resolution_source", ""),
                    "deadline": data.get("resolution_deadline", ""),
                    "condition": data.get("resolution_condition", ""),
                    "ambiguities": data.get("resolution_ambiguities", []),
                    "risk_level": data.get("resolution_risk_level", "unknown").upper(),
                },
                "risks": {
                    "manipulation": data.get("manipulation_notes", ""),
                    "uncertainties": data.get("key_uncertainties", []),
                },
                "rationale": data.get("recommendation_rationale", ""),
                "search_queries": data.get("search_queries", []),
                "sources_found": data.get("sources_found", 0),
                "token_input": data.get("token_input", 0),
                "token_output": data.get("token_output", 0),
            },
        }

    return result


@router.get("/runs")
async def list_runs(limit: int = 50, status: str | None = None) -> list[dict]:
    """List recent agent runs from DB — used by RUNS page and EVAL grading flow."""
    from src.storage.db import get_engine
    from src.storage.models import AgentRunORM
    from sqlalchemy.orm import Session

    def _query():
        with Session(get_engine()) as s:
            q = s.query(AgentRunORM).order_by(AgentRunORM.started_at.desc())
            if status:
                q = q.filter(AgentRunORM.status == status)
            rows = q.limit(limit).all()
            return [
                {
                    "run_id":       r.run_id,
                    "condition_id": r.condition_id,
                    "status":       r.status,
                    "started_at":   r.started_at.isoformat() if r.started_at else None,
                    "finished_at":  r.finished_at.isoformat() if r.finished_at else None,
                    "error_message": r.error_message,
                }
                for r in rows
            ]

    return await run_in_threadpool(_query)
