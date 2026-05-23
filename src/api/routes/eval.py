"""GET /api/eval/summary  ·  GET /api/eval/history"""

from __future__ import annotations

import logging

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/eval", tags=["eval"])


class EvalSummaryOut(BaseModel):
    total: int
    resolved: int
    brier_avg: float | None
    brier_label: str
    baseline_delta: float | None


class PredictionRecordOut(BaseModel):
    id: int
    market: str
    agent: float
    market_price: float
    outcome: float | None
    recommendation: str
    date: str
    brier: float | None


@router.get("/summary", response_model=EvalSummaryOut)
async def eval_summary() -> EvalSummaryOut:
    from src.storage.db import get_engine, get_eval_summary
    from sqlalchemy.orm import Session

    def _query():
        with Session(get_engine()) as s:
            return get_eval_summary(s)

    data = await run_in_threadpool(_query)
    return EvalSummaryOut(**data)


@router.get("/history", response_model=list[PredictionRecordOut])
async def prediction_history(limit: int = 50) -> list[PredictionRecordOut]:
    from src.storage.db import get_engine
    from src.storage.models import PredictionHistoryORM
    from sqlalchemy.orm import Session

    def _query():
        with Session(get_engine()) as s:
            return (
                s.query(PredictionHistoryORM)
                .order_by(PredictionHistoryORM.created_at.desc())
                .limit(limit)
                .all()
            )

    rows = await run_in_threadpool(_query)

    return [
        PredictionRecordOut(
            id=row.id,
            market=row.market_question,
            agent=row.agent_estimate,
            market_price=row.market_probability,
            outcome=float(row.outcome) if row.outcome is not None else None,
            recommendation=_short_rec(row.recommendation),
            date=row.created_at.strftime("%Y-%m-%d"),
            brier=row.brier_score,
        )
        for row in rows
    ]


def _short_rec(rec: str) -> str:
    return {
        "candidate_opportunity": "OPP",
        "watch": "WATCH",
        "research_more": "RESEARCH",
        "no_trade": "NO_TRADE",
    }.get(rec, rec.upper())
