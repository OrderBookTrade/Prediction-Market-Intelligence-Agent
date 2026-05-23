"""SQLAlchemy ORM models — Sprint 1 tables + Sprint 2 agent tables + Sprint 5 eval grades."""

from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


# ── Sprint 1 ──────────────────────────────────────────────────────────────────

class MarketSnapshotORM(Base):
    """Latest known state for each market. Upserted on every ingest run."""

    __tablename__ = "market_snapshots"

    condition_id: Mapped[str] = mapped_column(String, primary_key=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    outcomes: Mapped[str | None] = mapped_column(Text)  # JSON list
    yes_price: Mapped[float | None] = mapped_column(Float)
    no_price: Mapped[float | None] = mapped_column(Float)
    volume: Mapped[float | None] = mapped_column(Float)
    liquidity: Mapped[float | None] = mapped_column(Float)
    spread: Mapped[float | None] = mapped_column(Float)
    end_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    resolution_source: Mapped[str | None] = mapped_column(Text)
    category: Mapped[str | None] = mapped_column(String)
    raw_rules_text: Mapped[str | None] = mapped_column(Text)
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class MarketSnapshotHistory(Base):
    """Append-only price/volume time series. One row per ingest per market."""

    __tablename__ = "market_snapshot_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    condition_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    yes_price: Mapped[float | None] = mapped_column(Float)
    no_price: Mapped[float | None] = mapped_column(Float)
    volume: Mapped[float | None] = mapped_column(Float)
    liquidity: Mapped[float | None] = mapped_column(Float)
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )


# ── Sprint 2 ──────────────────────────────────────────────────────────────────

class AgentRunORM(Base):
    """Tracks the lifecycle of a single agent analysis run."""

    __tablename__ = "agent_runs"

    run_id: Mapped[str] = mapped_column(String, primary_key=True)
    condition_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    status: Mapped[str] = mapped_column(String, default="queued")  # queued|running|done|error
    error_message: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class MemoORM(Base):
    """Persisted research memo, one per completed agent run."""

    __tablename__ = "memos"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    condition_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    memo_json: Mapped[str] = mapped_column(Text, nullable=False)  # full ResearchMemo as JSON
    recommendation: Mapped[str] = mapped_column(String)
    agent_estimate: Mapped[float] = mapped_column(Float)
    market_probability: Mapped[float] = mapped_column(Float)
    edge: Mapped[float] = mapped_column(Float)
    confidence: Mapped[str] = mapped_column(String)
    model_name: Mapped[str] = mapped_column(String)
    prompt_version: Mapped[str] = mapped_column(String, default="v1")
    sources_found: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )


class AuditLogORM(Base):
    """Immutable audit trail for every LLM / tool call."""

    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    node_name: Mapped[str] = mapped_column(String)
    model: Mapped[str | None] = mapped_column(String)
    prompt_version: Mapped[str | None] = mapped_column(String)
    input_hash: Mapped[str | None] = mapped_column(String)  # sha256 of prompt
    output_json: Mapped[str | None] = mapped_column(Text)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    token_input: Mapped[int | None] = mapped_column(Integer)
    token_output: Mapped[int | None] = mapped_column(Integer)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )


class PredictionHistoryORM(Base):
    """One row per memo; outcome + Brier filled in after market resolves."""

    __tablename__ = "prediction_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    condition_id: Mapped[str] = mapped_column(String, nullable=False)
    market_question: Mapped[str] = mapped_column(Text)
    agent_estimate: Mapped[float] = mapped_column(Float)
    market_probability: Mapped[float] = mapped_column(Float)
    edge: Mapped[float] = mapped_column(Float)
    confidence: Mapped[str] = mapped_column(String)
    recommendation: Mapped[str] = mapped_column(String)
    resolved: Mapped[bool] = mapped_column(Boolean, default=False)
    outcome: Mapped[bool | None] = mapped_column(Boolean)
    brier_score: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


# ── Sprint 5 ──────────────────────────────────────────────────────────────────

class EvalGradeORM(Base):
    """LLM-as-judge grades for a completed ResearchMemo."""

    __tablename__ = "eval_grades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String, nullable=False, index=True, unique=True)
    condition_id: Mapped[str] = mapped_column(String, nullable=False, index=True)

    citation_score: Mapped[float] = mapped_column(Float)
    calibration_score: Mapped[float] = mapped_column(Float)
    reasoning_score: Mapped[float] = mapped_column(Float)
    hedge_score: Mapped[float] = mapped_column(Float)
    overall: Mapped[float] = mapped_column(Float)
    weighted_overall: Mapped[float] = mapped_column(Float)
    letter_grade: Mapped[str] = mapped_column(String(2))    # A / B / C / D / F

    feedback_json: Mapped[str] = mapped_column(Text)        # JSON list[str]
    model_name: Mapped[str] = mapped_column(String)
    prompt_version: Mapped[str] = mapped_column(String, default="judge-v1")

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
