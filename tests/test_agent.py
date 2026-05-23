"""Sprint 2 tests: risk critic, BM25 reranker, API endpoints, DB helpers."""

from __future__ import annotations

import json
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch


# ── Risk critic ───────────────────────────────────────────────────────────────

class TestRiskCritic:
    """Test the deterministic risk flag logic directly (no LLM, no HTTP)."""

    @pytest.fixture
    def base_state(self):
        return {
            "run_id": "test-run-001",
            "condition_id": "0xabc",
            "snapshot": {
                "condition_id": "0xabc",
                "question": "Will X happen?",
                "yes_price": 0.5,
                "no_price": 0.5,
                "volume": 100_000,
                "liquidity": 50_000,
                "spread": 0.002,
                "end_date": (datetime.now(timezone.utc) + timedelta(days=90)).isoformat(),
                "resolution_source": "reuters.com",
                "raw_rules_text": "This market resolves based on official announcement from reuters.com",
            },
            "search_results": [
                {"title": "Test", "url": "https://reuters.com/test", "credibility": "HIGH"},
                {"title": "Test2", "url": "https://bloomberg.com/t2", "credibility": "HIGH"},
            ],
        }

    async def test_low_liquidity_flag(self, base_state):
        base_state["snapshot"]["liquidity"] = 5_000
        # Patch push_log so it doesn't fail without a real queue
        with patch("src.agents.risk_critic.push_log", new=AsyncMock()):
            from src.agents.risk_critic import risk_critic_node
            result = await risk_critic_node(base_state)
        assert "LOW_LIQUIDITY" in result["risk_flags"]

    async def test_wide_spread_flag(self, base_state):
        base_state["snapshot"]["spread"] = 0.08
        with patch("src.agents.risk_critic.push_log", new=AsyncMock()):
            from src.agents.risk_critic import risk_critic_node
            result = await risk_critic_node(base_state)
        assert "WIDE_SPREAD" in result["risk_flags"]

    async def test_expires_soon_flag(self, base_state):
        base_state["snapshot"]["end_date"] = (
            datetime.now(timezone.utc) + timedelta(days=7)
        ).isoformat()
        with patch("src.agents.risk_critic.push_log", new=AsyncMock()):
            from src.agents.risk_critic import risk_critic_node
            result = await risk_critic_node(base_state)
        assert "EXPIRES_SOON" in result["risk_flags"]

    async def test_healthy_market_no_flags(self, base_state):
        with patch("src.agents.risk_critic.push_log", new=AsyncMock()):
            from src.agents.risk_critic import risk_critic_node
            result = await risk_critic_node(base_state)
        assert "LOW_LIQUIDITY" not in result["risk_flags"]
        assert "WIDE_SPREAD" not in result["risk_flags"]
        assert "EXPIRES_SOON" not in result["risk_flags"]

    async def test_risk_details_structure(self, base_state):
        with patch("src.agents.risk_critic.push_log", new=AsyncMock()):
            from src.agents.risk_critic import risk_critic_node
            result = await risk_critic_node(base_state)
        assert "liquidity" in result["risk_details"]
        assert "resolution" in result["risk_details"]
        assert "hallucination" in result["risk_details"]
        assert result["risk_details"]["liquidity"]["level"] in ("low", "medium", "high")


# ── BM25 reranker ─────────────────────────────────────────────────────────────

class TestBM25Reranker:
    def _make_results(self, n: int):
        from src.retrieval.search import SearchHit
        return [
            SearchHit(
                url=f"https://example.com/{i}",
                title=f"Title {i}",
                publisher=f"example.com",
                published_at=None,
                snippet=f"Content about prediction markets and topic {i}",
                raw_text=f"Content about prediction markets and topic {i}",
                score=1.0,
                credibility="MEDIUM",
            )
            for i in range(n)
        ]

    def test_returns_at_most_top_k(self):
        from src.retrieval.reranker import bm25_rerank
        results = self._make_results(20)
        reranked = bm25_rerank(results, "prediction markets", top_k=5)
        assert len(reranked) == 5

    def test_empty_input_returns_empty(self):
        from src.retrieval.reranker import bm25_rerank
        assert bm25_rerank([], "query", top_k=5) == []

    def test_relevant_doc_ranked_higher(self):
        """BM25 IDF needs N>2 to produce non-zero scores; pad corpus accordingly."""
        from src.retrieval.search import SearchHit
        from src.retrieval.reranker import bm25_rerank

        def _sr(title, url, content):
            return SearchHit(
                url=url,
                title=title,
                publisher=url.split("/")[2],
                published_at=None,
                snippet=content,
                raw_text=content,
                score=1.0,
                credibility="MEDIUM",
            )

        results = [
            _sr("Cats and dogs lifestyle", "https://a.com", "cats dogs pets animals"),
            _sr("GPT-5 release 2026 OpenAI", "https://b.com", "OpenAI will release GPT-5 summer 2026"),
            _sr("Football match results", "https://c.com", "soccer football premier league goals"),
            _sr("Weather forecast rain", "https://d.com", "rain cloudy sunny forecast weather"),
            _sr("Stock market update", "https://e.com", "stocks shares dow jones nasdaq market"),
        ]
        reranked = bm25_rerank(results, "OpenAI GPT-5 release 2026", top_k=5)
        assert reranked[0].url == "https://b.com"


# ── Database helpers ──────────────────────────────────────────────────────────

class TestDbHelpers:
    def test_save_and_retrieve_memo(self, tmp_path):
        from src.storage.db import get_engine, init_db, save_memo, get_latest_memo
        from sqlalchemy.orm import Session

        engine = init_db(get_engine(f"sqlite:///{tmp_path}/test.db"))
        memo_data = {
            "run_id": "run-001",
            "condition_id": "0xtest",
            "market_question": "Will X happen?",
            "market_probability": 0.5,
            "agent_estimate": 0.65,
            "edge": 0.15,
            "confidence": "medium",
            "recommendation": "watch",
            "model_name": "test-model",
            "prompt_version": "v1",
            "sources_found": 3,
            "yes_case": [],
            "no_case": [],
            "resolution_source": "",
            "resolution_deadline": "",
            "resolution_condition": "",
            "resolution_ambiguities": [],
            "resolution_risk_level": "low",
            "resolution_risk_notes": "",
            "manipulation_risk": "low",
            "manipulation_notes": "",
            "recommendation_rationale": "Test rationale",
            "key_uncertainties": [],
            "search_queries": ["q1"],
        }

        with Session(engine) as session:
            save_memo(session, memo_data)

        with Session(engine) as session:
            result = get_latest_memo(session, "0xtest")

        assert result is not None
        assert result.agent_estimate == pytest.approx(0.65)
        assert result.recommendation == "watch"

    def test_eval_summary_no_resolved(self, tmp_path):
        from src.storage.db import get_engine, init_db, get_eval_summary
        from sqlalchemy.orm import Session

        engine = init_db(get_engine(f"sqlite:///{tmp_path}/test2.db"))
        with Session(engine) as session:
            summary = get_eval_summary(session)

        assert summary["total"] == 0
        assert summary["resolved"] == 0
        assert summary["brier_avg"] is None


# ── FastAPI routes ────────────────────────────────────────────────────────────

class TestAPIRoutes:
    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/api_test.db")
        # Reload settings and engine singleton
        import importlib
        import src.storage.db as db_module
        db_module._engine = None

        from fastapi.testclient import TestClient
        from src.api.app import app
        from src.storage.db import get_engine, init_db
        init_db(get_engine())
        return TestClient(app)

    def test_health_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_eval_summary_returns_dict(self, client):
        resp = client.get("/api/eval/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert "total" in data
        assert "brier_label" in data

    def test_eval_history_returns_list(self, client):
        resp = client.get("/api/eval/history")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_memo_404_when_missing(self, client):
        resp = client.get("/api/memos/nonexistent-id")
        assert resp.status_code == 404
