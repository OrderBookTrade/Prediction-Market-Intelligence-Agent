"""GET /api/markets  ·  GET /api/markets/{condition_id}  ·  POST /api/markets/ingest"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/markets", tags=["markets"])


# ── Response model ─────────────────────────────────────────────────────────────

class MarketResponse(BaseModel):
    condition_id: str
    question: str
    yes_price: float | None
    no_price: float | None
    volume: float | None
    liquidity: float | None
    spread: float | None
    change_24h: float | None
    end_date: str | None
    category: str | None
    platform: str = "Polymarket"
    risk_flags: list[str]
    history: list[float]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _compute_risk_flags(yes_price, liquidity, spread, end_date_str, volume) -> list[str]:
    flags: list[str] = []
    liq = liquidity or 0
    vol = volume or 0
    spd = spread or 0

    if liq < 10_000:
        flags.append("LOW_LIQUIDITY")
    if spd > 0.01:
        flags.append("WIDE_SPREAD")
    if end_date_str:
        try:
            end_dt = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
            days_left = (end_dt.replace(tzinfo=timezone.utc) - datetime.now(timezone.utc)).days
            if days_left < 30:
                flags.append("EXPIRES_SOON")
        except Exception:
            pass
    if vol > 0 and liq > 0 and vol / liq > 15:
        flags.append("HIGH_VOLUME")
    return flags


def _orm_to_response(orm, history: list[float]) -> MarketResponse:
    end_iso = orm.end_date.strftime("%Y-%m-%d") if orm.end_date else None
    flags = _compute_risk_flags(
        orm.yes_price, orm.liquidity, orm.spread, end_iso, orm.volume
    )
    change_24h = None
    if len(history) >= 2:
        change_24h = round(history[-1] - history[-2], 4)
    return MarketResponse(
        condition_id=orm.condition_id,
        question=orm.question,
        yes_price=orm.yes_price,
        no_price=orm.no_price,
        volume=orm.volume,
        liquidity=orm.liquidity,
        spread=orm.spread,
        change_24h=change_24h,
        end_date=end_iso,
        category=orm.category,
        risk_flags=flags,
        history=history,
    )


def _raw_to_response(snap, history: list[float], change_24h: float | None) -> MarketResponse:
    end_iso = snap.end_date.strftime("%Y-%m-%d") if snap.end_date else None
    flags = _compute_risk_flags(
        snap.yes_price, snap.liquidity, snap.spread, end_iso, snap.volume
    )
    return MarketResponse(
        condition_id=snap.condition_id,
        question=snap.question,
        yes_price=snap.yes_price,
        no_price=snap.no_price,
        volume=snap.volume,
        liquidity=snap.liquidity,
        spread=snap.spread,
        change_24h=change_24h,
        end_date=end_iso,
        category=snap.category,
        risk_flags=flags,
        history=history,
    )


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.get("", response_model=list[MarketResponse])
async def list_markets(
    category: str | None = Query(None),
    limit: int = Query(20, ge=1, le=100),
    sort: Literal["volume", "liquidity", "end_date"] = Query("volume"),
    platform: str = Query("polymarket"),
) -> list[MarketResponse]:
    """Live-proxy the Polymarket Gamma API, upsert results to DB, return list."""
    from src.ingestion.normalizer import normalize
    from src.ingestion.polymarket import PolymarketClient
    from src.storage.db import get_engine, get_price_history, upsert_snapshot
    from src.storage.models import MarketSnapshotORM
    from sqlalchemy.orm import Session

    engine = get_engine()

    # Fetch live from Polymarket
    try:
        async with PolymarketClient() as client:
            raws = await client.fetch_active_markets(category=category, limit=limit)
    except Exception as exc:
        logger.warning("Polymarket fetch failed: %s — falling back to DB", exc)
        raws = []

    snapshots = []
    for raw in raws:
        try:
            snap = normalize(raw)
            snapshots.append((snap, raw.one_day_price_change))
        except Exception as exc:
            logger.warning("Normalization error: %s", exc)

    # Upsert to DB
    if snapshots:
        def _upsert():
            with Session(engine) as session:
                for snap, _ in snapshots:
                    upsert_snapshot(session, snap)
                session.commit()

        await run_in_threadpool(_upsert)

    # If no live data, fall back to DB
    if not snapshots:
        def _load_from_db():
            with Session(engine) as session:
                q = session.query(MarketSnapshotORM)
                if category:
                    q = q.filter(MarketSnapshotORM.category.ilike(f"%{category}%"))
                return q.limit(limit).all()

        rows = await run_in_threadpool(_load_from_db)
        results = []
        for row in rows:
            def _hist(cid=row.condition_id):
                with Session(engine) as s:
                    return get_price_history(s, cid)
            history = await run_in_threadpool(_hist)
            results.append(_orm_to_response(row, history))
        return results

    # Sort
    if sort == "volume":
        snapshots.sort(key=lambda x: x[0].volume or 0, reverse=True)
    elif sort == "liquidity":
        snapshots.sort(key=lambda x: x[0].liquidity or 0, reverse=True)
    elif sort == "end_date":
        snapshots.sort(key=lambda x: x[0].end_date or datetime.max.replace(tzinfo=timezone.utc))

    results = []
    for snap, change_24h in snapshots:
        def _hist(cid=snap.condition_id):
            with Session(engine) as s:
                return get_price_history(s, cid)
        history = await run_in_threadpool(_hist)
        results.append(_raw_to_response(snap, history, change_24h))
    return results


@router.get("/{condition_id}", response_model=MarketResponse)
async def get_market(condition_id: str) -> MarketResponse:
    from src.ingestion.normalizer import normalize
    from src.ingestion.polymarket import PolymarketClient
    from src.storage.db import get_engine, get_price_history
    from src.storage.models import MarketSnapshotORM
    from sqlalchemy.orm import Session

    engine = get_engine()

    # Try live fetch first
    try:
        async with PolymarketClient() as client:
            raw = await client.fetch_market_detail(condition_id)
        snap = normalize(raw)

        def _hist():
            with Session(engine) as s:
                return get_price_history(s, condition_id)

        history = await run_in_threadpool(_hist)
        return _raw_to_response(snap, history, raw.one_day_price_change)
    except Exception as exc:
        logger.warning("Live fetch failed for %s: %s — trying DB", condition_id, exc)

    # Fall back to DB
    def _from_db():
        with Session(engine) as session:
            row = session.get(MarketSnapshotORM, condition_id)
            if not row:
                return None, []
            return row, get_price_history(session, condition_id)

    row, history = await run_in_threadpool(_from_db)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Market {condition_id!r} not found")
    return _orm_to_response(row, history)


# ── URL resolver ──────────────────────────────────────────────────────────────

class ResolveRequest(BaseModel):
    url: str

    @field_validator("url")
    @classmethod
    def must_be_polymarket(cls, v: str) -> str:
        if "polymarket.com" not in v:
            raise ValueError("URL must be a polymarket.com link")
        return v.strip()


class ResolveResponse(BaseModel):
    type: str                        # "yesno" | "multi"
    event_title: str | None = None   # populated for multi-outcome events
    markets: list[MarketResponse]


def _parse_polymarket_url(url: str) -> tuple[str, str]:
    """Return (kind, slug) where kind is 'event' or 'market'."""
    url = url.strip().rstrip("/")
    # strip query string
    url = url.split("?")[0]
    for marker in ("polymarket.com/event/", "polymarket.com/market/"):
        if marker in url:
            kind = "event" if "/event/" in marker else "market"
            slug = url.split(marker, 1)[1]
            return kind, slug
    raise ValueError(f"Cannot parse Polymarket URL: {url!r}")


@router.post("/resolve", response_model=ResolveResponse)
async def resolve_url(body: ResolveRequest) -> ResolveResponse:
    """Parse a Polymarket URL and return market(s) ready for analysis.

    - /event/{slug}  → multi-outcome event (type="multi")
    - /market/{slug} → single Yes/No market  (type="yesno")
    """
    from src.ingestion.normalizer import normalize
    from src.ingestion.polymarket import PolymarketClient
    from src.storage.db import get_engine, get_price_history, upsert_snapshot
    from sqlalchemy.orm import Session

    try:
        kind, slug = _parse_polymarket_url(body.url)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    engine = get_engine()

    async with PolymarketClient() as client:

        # ── Single Yes/No market ──────────────────────────────────────────
        if kind == "market":
            raw = await client.fetch_market_by_slug(slug)
            if raw is None:
                raise HTTPException(status_code=404, detail=f"Market {slug!r} not found")

            try:
                snap = normalize(raw)
            except Exception as exc:
                raise HTTPException(status_code=422, detail=f"Normalization failed: {exc}")

            def _persist_and_hist():
                with Session(engine) as s:
                    upsert_snapshot(s, snap)
                    s.commit()
                    return get_price_history(s, snap.condition_id)

            history = await run_in_threadpool(_persist_and_hist)
            market_resp = _raw_to_response(snap, history, raw.one_day_price_change)
            return ResolveResponse(type="yesno", markets=[market_resp])

        # ── Multi-outcome event ───────────────────────────────────────────
        event = await client.fetch_event_by_slug(slug)
        if event is None:
            raise HTTPException(status_code=404, detail=f"Event {slug!r} not found")

        event_title: str = event.get("title") or slug
        raw_markets: list[dict] = event.get("markets", [])

        if not raw_markets:
            raise HTTPException(status_code=404, detail="Event has no markets")

        from src.schemas import MarketRaw

        responses: list[MarketResponse] = []
        for item in raw_markets:
            try:
                raw = MarketRaw.model_validate(item)
                snap = normalize(raw)

                def _ph(cid=snap.condition_id):
                    with Session(engine) as s:
                        upsert_snapshot(s, snap)
                        s.commit()
                        return get_price_history(s, cid)

                hist = await run_in_threadpool(_ph)
                responses.append(_raw_to_response(snap, hist, raw.one_day_price_change))
            except Exception as exc:
                logger.warning("Skipping market in event: %s", exc)

        if not responses:
            raise HTTPException(status_code=422, detail="No valid markets in event")

        # Sort by YES price descending (most likely outcome first)
        responses.sort(key=lambda m: m.yes_price or 0, reverse=True)

        return ResolveResponse(
            type="multi",
            event_title=event_title,
            markets=responses,
        )


@router.post("/ingest", status_code=202)
async def trigger_ingest(
    category: str | None = Query(None),
    limit: int = Query(50),
) -> dict:
    """Background ingest — returns immediately, updates DB asynchronously."""
    import asyncio

    async def _background():
        from src.ingestion.normalizer import normalize
        from src.ingestion.polymarket import PolymarketClient
        from src.storage.db import get_engine, upsert_snapshot
        from sqlalchemy.orm import Session

        engine = get_engine()
        try:
            async with PolymarketClient() as client:
                raws = await client.fetch_active_markets(category=category, limit=limit)
            with Session(engine) as session:
                for raw in raws:
                    try:
                        upsert_snapshot(session, normalize(raw))
                    except Exception:
                        pass
                session.commit()
            logger.info("Background ingest complete: %d markets", len(raws))
        except Exception as exc:
            logger.error("Background ingest failed: %s", exc)

    asyncio.create_task(_background())
    return {"status": "accepted", "category": category, "limit": limit}
