"""Node 1 — market_analyzer.

Loads the MarketSnapshot from DB (or fetches live if missing),
validates it, and emits the first batch of SSE log lines.
"""

from __future__ import annotations

import json
import logging

from src.run_store import push_log

logger = logging.getLogger(__name__)

TRUSTED_SOURCES = {
    "reuters.com", "bloomberg.com", "ap.org", "apnews.com",
    "federalreserve.gov", "sec.gov", "openai.com", "anthropic.com",
}
AMBIGUITY_KEYWORDS = [
    "at discretion", "may", "generally", "approximately",
    "as determined by", "reportedly", "sources say",
]


async def market_analyzer_node(state: dict) -> dict:
    condition_id: str = state["condition_id"]
    run_id: str = state["run_id"]

    await push_log(run_id, f"Initializing agent · model=claude-opus-4-5 · prompt=v1", "header")
    await push_log(run_id, f"Fetching market data for {condition_id}...", "info")

    from src.storage.db import get_engine
    from src.storage.models import MarketSnapshotORM
    from src.ingestion.polymarket import PolymarketClient
    from src.ingestion.normalizer import normalize
    from sqlalchemy.orm import Session
    import time

    snapshot_dict: dict | None = None

    # Try DB first
    with Session(get_engine()) as session:
        orm = session.get(MarketSnapshotORM, condition_id)
        if orm:
            snapshot_dict = {
                "condition_id": orm.condition_id,
                "question": orm.question,
                "description": orm.description,
                "yes_price": orm.yes_price,
                "no_price": orm.no_price,
                "volume": orm.volume,
                "liquidity": orm.liquidity,
                "spread": orm.spread,
                "end_date": orm.end_date.isoformat() if orm.end_date else None,
                "resolution_source": orm.resolution_source,
                "category": orm.category,
                "raw_rules_text": orm.raw_rules_text,
            }

    # Fall back to live fetch
    if not snapshot_dict:
        await push_log(run_id, "  Market not in DB — fetching from Polymarket...", "dim")
        t0 = time.monotonic()
        async with PolymarketClient() as client:
            raw = await client.fetch_market_detail(condition_id)
        snap = normalize(raw)
        elapsed = int((time.monotonic() - t0) * 1000)
        snapshot_dict = {
            "condition_id": snap.condition_id,
            "question": snap.question,
            "description": snap.description,
            "yes_price": snap.yes_price,
            "no_price": snap.no_price,
            "volume": snap.volume,
            "liquidity": snap.liquidity,
            "spread": snap.spread,
            "end_date": snap.end_date.isoformat() if snap.end_date else None,
            "resolution_source": snap.resolution_source,
            "category": snap.category,
            "raw_rules_text": snap.raw_rules_text,
        }
        await push_log(run_id, f"  ✓ market loaded · {elapsed}ms", "ok")
    else:
        await push_log(run_id, "  ✓ market loaded from cache", "ok")

    # Log key fields
    yes_p = snapshot_dict.get("yes_price")
    await push_log(run_id, f"  YES={yes_p:.3f}  VOL=${snapshot_dict.get('volume') or 0:,.0f}  LIQ=${snapshot_dict.get('liquidity') or 0:,.0f}", "dim")

    # Resolution rules analysis
    await push_log(run_id, "Parsing resolution rules...", "info")
    rules = snapshot_dict.get("raw_rules_text") or snapshot_dict.get("resolution_source") or ""
    res_source = snapshot_dict.get("resolution_source") or ""
    if any(t in res_source.lower() for t in TRUSTED_SOURCES):
        await push_log(run_id, f"  source = {res_source[:60]} (trusted)", "dim")
    else:
        await push_log(run_id, f"  source = {res_source[:60] or 'not specified'} (unverified)", "warn")

    found_ambig = [kw for kw in AMBIGUITY_KEYWORDS if kw in rules.lower()]
    if found_ambig:
        await push_log(run_id, f"  ⚠ ambiguous language detected: {found_ambig[:2]}", "warn")

    return {"snapshot": snapshot_dict}
