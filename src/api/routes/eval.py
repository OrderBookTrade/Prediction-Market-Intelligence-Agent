"""GET /api/eval/summary  ·  GET /api/eval/history  ·  POST /api/eval/outcome/{condition_id}  ·  POST /api/eval/grade/{run_id}  ·  GET /api/eval/grades"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException
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


# ── LLM Judge ─────────────────────────────────────────────────────────────────

class EvalGradeOut(BaseModel):
    run_id: str
    condition_id: str
    citation_score: float
    calibration_score: float
    reasoning_score: float
    hedge_score: float
    overall: float
    weighted_overall: float
    letter_grade: str
    feedback: list[str]
    model_name: str
    created_at: str


@router.post("/grade/{run_id}", response_model=EvalGradeOut)
async def grade_memo(run_id: str) -> EvalGradeOut:
    """Run the LLM-as-judge on a completed memo and persist the grade.

    Idempotent — grading the same run_id twice replaces the previous grade.
    """
    from src.eval.judge import grade_memo as _grade
    from src.storage.db import get_engine, list_eval_grades, upsert_eval_grade
    from src.storage.models import MemoORM
    from sqlalchemy.orm import Session

    engine = get_engine()

    # Load memo from DB
    def _load():
        with Session(engine) as s:
            return s.query(MemoORM).filter_by(run_id=run_id).first()

    memo_row = await run_in_threadpool(_load)
    if memo_row is None:
        raise HTTPException(status_code=404, detail=f"No memo found for run_id={run_id!r}")

    try:
        memo_dict = json.loads(memo_row.memo_json)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Memo JSON invalid: {exc}")

    # Run judge
    grade = await _grade(
        memo_dict=memo_dict,
        run_id=run_id,
        condition_id=memo_row.condition_id,
    )

    # Persist — collect all fields inside session to avoid DetachedInstanceError
    def _save():
        with Session(engine) as s:
            row = upsert_eval_grade(s, grade)
            s.commit()
            s.refresh(row)
            return {
                "run_id": row.run_id,
                "condition_id": row.condition_id,
                "citation_score": row.citation_score,
                "calibration_score": row.calibration_score,
                "reasoning_score": row.reasoning_score,
                "hedge_score": row.hedge_score,
                "overall": row.overall,
                "weighted_overall": row.weighted_overall,
                "letter_grade": row.letter_grade,
                "feedback": json.loads(row.feedback_json),
                "model_name": row.model_name,
                "created_at": row.created_at.isoformat(),
            }

    data = await run_in_threadpool(_save)
    return EvalGradeOut(**data)


class OutcomeIn(BaseModel):
    outcome: float  # 1.0 = YES resolved, 0.0 = NO resolved


class OutcomeOut(BaseModel):
    condition_id: str
    run_id: str
    agent_estimate: float
    outcome: float
    brier_score: float
    updated: int  # number of rows updated


@router.post("/outcome/{condition_id}", response_model=OutcomeOut)
async def record_outcome(condition_id: str, body: OutcomeIn) -> OutcomeOut:
    """Record the actual market outcome (1.0=YES, 0.0=NO) for a condition_id.

    Updates ALL unresolved prediction_history rows for this condition and
    computes Brier score = (agent_estimate - outcome)² for each.
    Returns stats for the most-recent updated row.
    """
    from src.storage.db import get_engine
    from src.storage.models import PredictionHistoryORM
    from sqlalchemy.orm import Session
    from datetime import datetime, timezone

    if body.outcome not in (0.0, 1.0):
        raise HTTPException(status_code=422, detail="outcome must be 0.0 (NO) or 1.0 (YES)")

    engine = get_engine()

    def _update():
        with Session(engine) as s:
            rows = (
                s.query(PredictionHistoryORM)
                .filter(
                    PredictionHistoryORM.condition_id == condition_id,
                    PredictionHistoryORM.resolved == False,  # noqa: E712
                )
                .all()
            )
            if not rows:
                return None, 0

            now = datetime.now(timezone.utc)
            for row in rows:
                row.outcome = bool(body.outcome)
                row.brier_score = round((row.agent_estimate - body.outcome) ** 2, 6)
                row.resolved = True
                row.resolved_at = now

            s.commit()
            s.refresh(rows[-1])
            latest = rows[-1]
            return {
                "condition_id": latest.condition_id,
                "run_id": latest.run_id,
                "agent_estimate": latest.agent_estimate,
                "outcome": float(latest.outcome),
                "brier_score": latest.brier_score,
            }, len(rows)

    data, count = await run_in_threadpool(_update)
    if data is None:
        raise HTTPException(
            status_code=404,
            detail=f"No unresolved predictions found for condition_id={condition_id!r}",
        )
    return OutcomeOut(**data, updated=count)


@router.get("/grades", response_model=list[EvalGradeOut])
async def list_grades(limit: int = 50) -> list[EvalGradeOut]:
    """Return all LLM-judge grades, newest first."""
    from src.storage.db import get_engine, list_eval_grades
    from sqlalchemy.orm import Session

    engine = get_engine()

    def _load():
        with Session(engine) as s:
            rows = list_eval_grades(s, limit=limit)
            return [
                {
                    "run_id": r.run_id,
                    "condition_id": r.condition_id,
                    "citation_score": r.citation_score,
                    "calibration_score": r.calibration_score,
                    "reasoning_score": r.reasoning_score,
                    "hedge_score": r.hedge_score,
                    "overall": r.overall,
                    "weighted_overall": r.weighted_overall,
                    "letter_grade": r.letter_grade,
                    "feedback": json.loads(r.feedback_json),
                    "model_name": r.model_name,
                    "created_at": r.created_at.isoformat(),
                }
                for r in rows
            ]

    data_list = await run_in_threadpool(_load)
    return [EvalGradeOut(**d) for d in data_list]
