"""Sprint 1 tests: normalizer, polymarket client retries, db upsert idempotency."""

from __future__ import annotations

import pytest
import httpx
from tenacity import wait_none

from src.ingestion.normalizer import normalize
from src.ingestion.polymarket import PolymarketClient
from src.schemas import MarketRaw, MarketSnapshot
from src.storage.db import get_engine, init_db, upsert_snapshot
from src.storage.models import MarketSnapshotHistory, MarketSnapshotORM
from sqlalchemy.orm import Session


# ── Normalizer ────────────────────────────────────────────────────────────────

class TestNormalizer:
    def test_normalizer_handles_missing_fields(self):
        """A MarketRaw with only an id should produce a valid snapshot with safe defaults."""
        raw = MarketRaw(id="market-001")
        snap = normalize(raw)

        assert snap.condition_id == "market-001"
        assert snap.question == ""
        assert snap.yes_price is None
        assert snap.no_price is None
        assert snap.outcomes == ["Yes", "No"]
        assert snap.spread is None

    def test_normalizer_parses_json_string_prices(self):
        """outcomePrices as a JSON-encoded string is the normal API format."""
        raw = MarketRaw.model_validate({
            "conditionId": "0xabc",
            "question": "Will X happen by 2026?",
            "outcomePrices": '["0.72", "0.28"]',
            "outcomes": '["Yes", "No"]',
        })
        snap = normalize(raw)

        assert snap.condition_id == "0xabc"
        assert snap.yes_price == pytest.approx(0.72)
        assert snap.no_price == pytest.approx(0.28)
        # spread = |1 - 0.72 - 0.28| = 0
        assert snap.spread == pytest.approx(0.0, abs=1e-6)

    def test_normalizer_handles_malformed_prices(self):
        """Non-JSON outcomePrices should not crash; prices become None."""
        raw = MarketRaw(id="bad-prices", outcome_prices="NOT_JSON")
        snap = normalize(raw)

        assert snap.yes_price is None
        assert snap.no_price is None

    def test_normalizer_falls_back_to_tokens(self):
        """When outcomePrices is absent, use tokens[].price."""
        raw = MarketRaw.model_validate({
            "id": "tok-market",
            "tokens": [
                {"outcome": "Yes", "price": 0.6},
                {"outcome": "No", "price": 0.4},
            ],
        })
        snap = normalize(raw)

        assert snap.yes_price == pytest.approx(0.6)
        assert snap.no_price == pytest.approx(0.4)

    def test_normalizer_extracts_category_from_tags(self):
        raw = MarketRaw.model_validate({
            "id": "tagged",
            "tags": [{"label": "Artificial Intelligence", "slug": "ai"}],
        })
        snap = normalize(raw)
        assert snap.category == "Artificial Intelligence"

    def test_normalizer_raises_on_no_id(self):
        """A raw object with no id and no conditionId must raise ValueError."""
        with pytest.raises(ValueError, match="no identifiable condition_id"):
            normalize(MarketRaw())


# ── Polymarket client ─────────────────────────────────────────────────────────

class TestPolymarketClient:
    def _make_client(self, handler) -> PolymarketClient:
        transport = httpx.MockTransport(handler)
        client = PolymarketClient(_retry_wait=wait_none())
        client._client = httpx.AsyncClient(transport=transport)
        return client

    async def test_polymarket_client_retries_on_429(self):
        """Client must retry up to 3 times on HTTP 429 responses."""
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return httpx.Response(429, json={"error": "rate limited"})
            return httpx.Response(200, json=[])

        client = self._make_client(handler)
        result = await client.fetch_active_markets()

        assert call_count == 3
        assert result == []

    async def test_polymarket_client_exhausted_retries_raises(self):
        """After 3 failed attempts, a RuntimeError should propagate."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, json={"error": "still rate limited"})

        client = self._make_client(handler)
        with pytest.raises((RuntimeError, httpx.HTTPStatusError)):
            await client.fetch_active_markets()

    async def test_polymarket_client_validates_market_objects(self):
        """Valid market JSON is parsed into MarketRaw instances."""
        payload = [
            {
                "id": "1",
                "conditionId": "0xabc",
                "question": "Test market?",
                "outcomePrices": '["0.6", "0.4"]',
                "outcomes": '["Yes", "No"]',
                "active": True,
                "volume": 50000.0,
                "liquidity": 12000.0,
            }
        ]

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=payload)

        client = self._make_client(handler)
        markets = await client.fetch_active_markets(limit=1)

        assert len(markets) == 1
        assert markets[0].condition_id == "0xabc"
        assert markets[0].question == "Test market?"


# ── DB upsert ─────────────────────────────────────────────────────────────────

class TestDbUpsert:
    def _engine(self, tmp_path):
        return init_db(get_engine(f"sqlite:///{tmp_path}/test.db"))

    def _make_snapshot(self, **overrides) -> MarketSnapshot:
        defaults = dict(
            condition_id="test-001",
            question="Is this idempotent?",
            outcomes=["Yes", "No"],
            yes_price=0.70,
            no_price=0.30,
            volume=10_000.0,
            liquidity=5_000.0,
            spread=0.0,
        )
        return MarketSnapshot(**(defaults | overrides))

    def test_db_upsert_idempotent(self, tmp_path):
        """Upserting the same condition_id twice updates the snapshot row but appends history."""
        engine = self._engine(tmp_path)
        snap1 = self._make_snapshot()

        with Session(engine) as session:
            upsert_snapshot(session, snap1)
            session.commit()

        snap2 = self._make_snapshot(yes_price=0.75, no_price=0.25)

        with Session(engine) as session:
            upsert_snapshot(session, snap2)
            session.commit()

        with Session(engine) as session:
            all_snapshots = session.query(MarketSnapshotORM).all()
            all_history = session.query(MarketSnapshotHistory).all()

        assert len(all_snapshots) == 1, "snapshot table must have exactly one row per market"
        assert all_snapshots[0].yes_price == pytest.approx(0.75), "snapshot must reflect latest price"
        assert len(all_history) == 2, "history table must have one row per ingest call"

    def test_db_upsert_multiple_markets(self, tmp_path):
        """Multiple distinct condition_ids each get their own snapshot row."""
        engine = self._engine(tmp_path)
        snaps = [self._make_snapshot(condition_id=f"market-{i}") for i in range(5)]

        with Session(engine) as session:
            for snap in snaps:
                upsert_snapshot(session, snap)
            session.commit()

        with Session(engine) as session:
            count = session.query(MarketSnapshotORM).count()
        assert count == 5
