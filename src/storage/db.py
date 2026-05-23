"""Database engine, session helpers, and upsert logic."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from src.schemas import MarketSnapshot
from src.storage.models import Base, MarketSnapshotHistory, MarketSnapshotORM

logger = logging.getLogger(__name__)


def get_engine(database_url: str | None = None) -> Engine:
    from src.config import settings

    url = database_url or settings.database_url
    if url.startswith("sqlite:///"):
        db_path = Path(url.removeprefix("sqlite:///"))
        db_path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(url, echo=False)


def init_db(engine: Engine | None = None) -> Engine:
    if engine is None:
        engine = get_engine()
    Base.metadata.create_all(engine)
    logger.info("Database tables ready (%s)", engine.url)
    return engine


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
