"""Shared in-process store for active agent run queues and results.

Both the agent nodes (src/agents/) and the SSE endpoint (src/api/routes/agent.py)
import from here to avoid circular dependencies.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# run_id → asyncio.Queue of SSE event dicts (None sentinel = stream closed)
_run_queues: dict[str, asyncio.Queue] = {}

# run_id → final memo dict (populated after agent completes)
_run_results: dict[str, dict] = {}


def register_run(run_id: str) -> asyncio.Queue:
    q: asyncio.Queue = asyncio.Queue()
    _run_queues[run_id] = q
    return q


def get_queue(run_id: str) -> asyncio.Queue | None:
    return _run_queues.get(run_id)


def set_result(run_id: str, result: dict) -> None:
    _run_results[run_id] = result


def get_result(run_id: str) -> dict | None:
    return _run_results.get(run_id)


def cleanup_run(run_id: str) -> None:
    _run_queues.pop(run_id, None)
    # Keep results for 10 minutes — caller's responsibility to evict if needed


async def push_log(run_id: str, text: str, kind: str = "info") -> None:
    """Push a single SSE log line to the run's queue."""
    q = _run_queues.get(run_id)
    if q is None:
        return
    await q.put({
        "time": datetime.now(timezone.utc).strftime("%H:%M:%S"),
        "text": text,
        "kind": kind,
    })


async def push_done(run_id: str) -> None:
    """Signal the SSE stream to close."""
    q = _run_queues.get(run_id)
    if q:
        await q.put(None)  # sentinel
