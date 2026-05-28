"""GET /api/memos/{condition_id}"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/memos", tags=["memos"])


class EvidenceOut(BaseModel):
    claim: str
    source: str
    credibility: str
    url: str | None = None


class ResolutionOut(BaseModel):
    official_source: str
    deadline: str
    condition: str
    ambiguities: list[str]
    risk_level: str


class RisksOut(BaseModel):
    manipulation: str
    uncertainties: list[str]


class RecommendationOut(BaseModel):
    label: str
    tone: str


REC_MAP = {
    "candidate_opportunity": {"label": "CANDIDATE OPPORTUNITY", "tone": "yes"},
    "watch":                 {"label": "WATCH",                 "tone": "brand"},
    "research_more":         {"label": "RESEARCH MORE",         "tone": "warn"},
    "no_trade":              {"label": "NO TRADE",              "tone": "gray"},
}


class MemoOut(BaseModel):
    run_id: str | None = None          # exposed so frontend can trigger /eval/grade
    generated_at: str
    model: str
    prompt_version: str
    recommendation: RecommendationOut
    market_implied: float
    agent_estimate: float
    edge: float
    confidence: str
    yes_case: list[EvidenceOut]
    no_case: list[EvidenceOut]
    resolution: ResolutionOut
    risks: RisksOut
    rationale: str
    search_queries: list[str]
    sources_found: int
    token_input: int | None = None
    token_output: int | None = None


def _orm_to_memo_out(orm) -> MemoOut:
    data = json.loads(orm.memo_json)
    rec = data.get("recommendation", "no_trade")
    rec_meta = REC_MAP.get(rec, {"label": rec.upper(), "tone": "gray"})

    return MemoOut(
        run_id=orm.run_id,
        generated_at=orm.created_at.strftime("%Y-%m-%d %H:%M UTC"),
        model=orm.model_name,
        prompt_version=orm.prompt_version,
        recommendation=RecommendationOut(**rec_meta),
        market_implied=orm.market_probability,
        agent_estimate=orm.agent_estimate,
        edge=orm.edge,
        confidence=data.get("confidence", "uncertain").upper(),
        yes_case=[EvidenceOut(**e) for e in data.get("yes_case", [])],
        no_case=[EvidenceOut(**e) for e in data.get("no_case", [])],
        resolution=ResolutionOut(
            official_source=data.get("resolution_source", ""),
            deadline=data.get("resolution_deadline", ""),
            condition=data.get("resolution_condition", ""),
            ambiguities=data.get("resolution_ambiguities", []),
            risk_level=data.get("resolution_risk_level", "unknown").upper(),
        ),
        risks=RisksOut(
            manipulation=data.get("manipulation_notes", ""),
            uncertainties=data.get("key_uncertainties", []),
        ),
        rationale=data.get("recommendation_rationale", ""),
        search_queries=data.get("search_queries", []),
        sources_found=orm.sources_found,
        token_input=data.get("token_input"),
        token_output=data.get("token_output"),
    )


@router.get("/{condition_id}", response_model=MemoOut)
async def get_memo(condition_id: str) -> MemoOut:
    from src.storage.db import get_engine, get_latest_memo
    from sqlalchemy.orm import Session

    def _query():
        with Session(get_engine()) as session:
            return get_latest_memo(session, condition_id)

    orm = await run_in_threadpool(_query)
    if orm is None:
        raise HTTPException(status_code=404, detail=f"No memo found for {condition_id!r}")
    return _orm_to_memo_out(orm)
