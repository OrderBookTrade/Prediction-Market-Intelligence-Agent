"""Database engine, session helpers, upsert and persistence logic."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from src.schemas import MarketSnapshot
from src.storage.models import (
    AgentRunORM,
    AuditLogORM,
    Base,
    MarketSnapshotHistory,
    MarketSnapshotORM,
    MemoORM,
    PredictionHistoryORM,
)

logger = logging.getLogger(__name__)

# Module-level singleton engine (lazy-initialised via get_engine())
_engine: Engine | None = None


def get_engine(database_url: str | None = None) -> Engine:
    global _engine
    if database_url is None and _engine is not None:
        return _engine

    from src.config import settings

    url = database_url or settings.database_url

    # Railway ships DATABASE_URL as postgres://, SQLAlchemy needs postgresql://
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)

    if url.startswith("sqlite:///"):
        db_path = Path(url.removeprefix("sqlite:///"))
        db_path.parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(url, pool_pre_ping=True, echo=False)

    if database_url is None:
        _engine = engine
    return engine


def init_db(engine: Engine | None = None) -> Engine:
    if engine is None:
        engine = get_engine()
    Base.metadata.create_all(engine)
    logger.info("Database tables ready (%s)", engine.url.render_as_string(hide_password=True))
    return engine


# ── Market snapshots ──────────────────────────────────────────────────────────

def upsert_snapshot(session: Session, snapshot: MarketSnapshot) -> None:
    """Insert or update the snapshot row and append a history record."""
    orm = MarketSnapshotORM(
        condition_id=snapshot.condition_id,
        question=snapshot.question,
        description=snapshot.description,
        outcomes=json.dumps(snapshot.outcomes),
        yes_price=snapshot.yes_price,
        no_price=snapshot.no_price,
        volume=snapshot.volume,
        liquidity=snapshot.liquidity,
        spread=snapshot.spread,
        end_date=snapshot.end_date,
        resolution_source=snapshot.resolution_source,
        category=snapshot.category,
        raw_rules_text=snapshot.raw_rules_text,
        updated_at=datetime.now(timezone.utc),
    )
    session.merge(orm)

    history = MarketSnapshotHistory(
        condition_id=snapshot.condition_id,
        yes_price=snapshot.yes_price,
        no_price=snapshot.no_price,
        volume=snapshot.volume,
        liquidity=snapshot.liquidity,
        fetched_at=datetime.now(timezone.utc),
    )
    session.add(history)


def get_price_history(session: Session, condition_id: str, limit: int = 20) -> list[float]:
    """Return the last N yes_price values for sparkline rendering."""
    rows = (
        session.query(MarketSnapshotHistory.yes_price)
        .filter(MarketSnapshotHistory.condition_id == condition_id)
        .filter(MarketSnapshotHistory.yes_price.isnot(None))
        .order_by(MarketSnapshotHistory.fetched_at.desc())
        .limit(limit)
        .all()
    )
    prices = [r[0] for r in reversed(rows)]
    return prices


# ── Agent runs ────────────────────────────────────────────────────────────────

def create_agent_run(session: Session, run_id: str, condition_id: str) -> AgentRunORM:
    run = AgentRunORM(
        run_id=run_id,
        condition_id=condition_id,
        status="queued",
        started_at=datetime.now(timezone.utc),
    )
    session.add(run)
    session.commit()
    return run


def update_run_status(
    session: Session,
    run_id: str,
    status: str,
    error_message: str | None = None,
) -> None:
    run = session.get(AgentRunORM, run_id)
    if run:
        run.status = status
        if error_message:
            run.error_message = error_message
        if status in ("done", "error"):
            run.finished_at = datetime.now(timezone.utc)
        session.commit()


def get_run(session: Session, run_id: str) -> AgentRunORM | None:
    return session.get(AgentRunORM, run_id)


# ── Memos ─────────────────────────────────────────────────────────────────────

def save_memo(session: Session, memo_data: dict) -> MemoORM:
    """Persist a completed ResearchMemo dict and create a prediction_history row."""
    memo_orm = MemoORM(
        run_id=memo_data["run_id"],
        condition_id=memo_data["condition_id"],
        memo_json=json.dumps(memo_data),
        recommendation=memo_data.get("recommendation", "no_trade"),
        agent_estimate=memo_data.get("agent_estimate", 0.5),
        market_probability=memo_data.get("market_probability", 0.5),
        edge=memo_data.get("edge", 0.0),
        confidence=memo_data.get("confidence", "uncertain"),
        model_name=memo_data.get("model_name", "unknown"),
        prompt_version=memo_data.get("prompt_version", "v1"),
        sources_found=memo_data.get("sources_found", 0),
        created_at=datetime.now(timezone.utc),
    )
    session.add(memo_orm)

    # Also record in prediction_history for eval tracking
    history = PredictionHistoryORM(
        run_id=memo_data["run_id"],
        condition_id=memo_data["condition_id"],
        market_question=memo_data.get("market_question", ""),
        agent_estimate=memo_data.get("agent_estimate", 0.5),
        market_probability=memo_data.get("market_probability", 0.5),
        edge=memo_data.get("edge", 0.0),
        confidence=memo_data.get("confidence", "uncertain"),
        recommendation=memo_data.get("recommendation", "no_trade"),
    )
    session.add(history)
    session.commit()
    return memo_orm


def get_latest_memo(session: Session, condition_id: str) -> MemoORM | None:
    return (
        session.query(MemoORM)
        .filter(MemoORM.condition_id == condition_id)
        .order_by(MemoORM.created_at.desc())
        .first()
    )


# ── Audit log ─────────────────────────────────────────────────────────────────

def log_audit(
    session: Session,
    run_id: str,
    node_name: str,
    output: dict | str,
    *,
    model: str | None = None,
    prompt_version: str | None = None,
    input_text: str | None = None,
    latency_ms: int | None = None,
    token_input: int | None = None,
    token_output: int | None = None,
) -> None:
    input_hash = (
        hashlib.sha256(input_text.encode()).hexdigest()[:16] if input_text else None
    )
    entry = AuditLogORM(
        run_id=run_id,
        node_name=node_name,
        model=model,
        prompt_version=prompt_version,
        input_hash=input_hash,
        output_json=json.dumps(output) if not isinstance(output, str) else output,
        latency_ms=latency_ms,
        token_input=token_input,
        token_output=token_output,
    )
    session.add(entry)
    session.commit()


# ── Eval helpers ──────────────────────────────────────────────────────────────

def get_eval_summary(session: Session) -> dict:
    all_rows = session.query(PredictionHistoryORM).all()
    resolved = [r for r in all_rows if r.resolved and r.brier_score is not None]

    brier_avg = (
        round(sum(r.brier_score for r in resolved) / len(resolved), 4)
        if resolved else None
    )

    def brier_label(score: float | None) -> str:
        if score is None:
            return "N/A"
        if score < 0.05:
            return "Excellent"
        if score < 0.15:
            return "Good"
        if score < 0.25:
            return "Fair"
        return "Poor"

    # Rough baseline: coin-flip Brier = 0.25
    baseline = 0.25
    baseline_delta = round(brier_avg - baseline, 4) if brier_avg is not None else None

    return {
        "total": len(all_rows),
        "resolved": len(resolved),
        "brier_avg": brier_avg,
        "brier_label": brier_label(brier_avg),
        "baseline_delta": baseline_delta,
    }
