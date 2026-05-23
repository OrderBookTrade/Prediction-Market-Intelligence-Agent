from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# ── Polymarket raw API response ───────────────────────────────────────────────

class MarketRaw(BaseModel):
    """Mirrors the Polymarket Gamma API market object.

    extra="allow" absorbs any fields the API adds without breaking validation.
    outcomePrices arrives as a JSON-encoded string, e.g. '["0.67","0.33"]'.
    outcomes similarly: '["Yes","No"]'.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    id: str | None = None
    condition_id: str | None = Field(None, alias="conditionId")
    question: str | None = None
    description: str | None = None
    slug: str | None = None
    resolution_source: str | None = Field(None, alias="resolutionSource")
    end_date: str | None = Field(None, alias="endDate")
    start_date: str | None = Field(None, alias="startDate")
    liquidity: float | None = None
    volume: float | None = None
    volume_24hr: float | None = Field(None, alias="volume24hr")
    active: bool | None = None
    closed: bool | None = None
    archived: bool | None = None
    outcome_prices: str | list[str] | None = Field(None, alias="outcomePrices")
    outcomes: str | list[str] | None = None
    tokens: list[dict[str, Any]] | None = None
    tags: list[dict[str, Any]] | None = None
    spread: float | None = None
    last_trade_price: float | None = Field(None, alias="lastTradePrice")
    best_ask: float | None = Field(None, alias="bestAsk")
    best_bid: float | None = Field(None, alias="bestBid")
    one_day_price_change: float | None = Field(None, alias="oneDayPriceChange")


# ── Normalized snapshot ───────────────────────────────────────────────────────

class MarketSnapshot(BaseModel):
    """Normalized, validated representation of a single market point-in-time."""

    condition_id: str
    question: str
    description: str | None = None
    outcomes: list[str] = Field(default_factory=lambda: ["Yes", "No"])
    yes_price: float | None = None
    no_price: float | None = None
    volume: float | None = None
    liquidity: float | None = None
    spread: float | None = None
    end_date: datetime | None = None
    resolution_source: str | None = None
    category: str | None = None
    raw_rules_text: str | None = None


# ── Time-series point ─────────────────────────────────────────────────────────

class PriceHistoryPoint(BaseModel):
    condition_id: str
    timestamp: datetime
    yes_price: float | None = None
    no_price: float | None = None
    volume: float | None = None
    liquidity: float | None = None


# ── Agent output schemas (used by future sprints) ─────────────────────────────

class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNCERTAIN = "uncertain"


class RiskLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class Recommendation(str, Enum):
    NO_TRADE = "no_trade"
    WATCH = "watch"
    RESEARCH_MORE = "research_more"
    CANDIDATE = "candidate_opportunity"


class EvidenceItem(BaseModel):
    claim: str
    source: str | None = None
    credibility: str | None = None


class ResolutionAnalysis(BaseModel):
    source: str
    deadline: str
    condition: str
    ambiguities: list[str] = Field(default_factory=list)
    risk_level: RiskLevel
    risk_notes: str


class MarketStateSchema(BaseModel):
    yes_price: float
    no_price: float
    volume: float
    liquidity: float
    spread: float
    price_change_24h: float | None = None
    liquidity_risk: RiskLevel
    liquidity_notes: str


class ResearchMemo(BaseModel):
    """Structured research memo — every LLM output must validate against this."""

    market_id: str
    market_question: str
    platform: str = "polymarket"

    current_probability: float
    market_state: MarketStateSchema

    agent_estimate: float = Field(ge=0, le=1)
    edge: float
    confidence: Confidence

    yes_case: list[EvidenceItem]
    no_case: list[EvidenceItem]

    resolution: ResolutionAnalysis
    manipulation_risk: RiskLevel
    manipulation_notes: str

    recommendation: Recommendation
    recommendation_rationale: str
    key_uncertainties: list[str]

    model_name: str
    prompt_version: str = "v1"
    search_queries_used: list[str] = Field(default_factory=list)
    sources_found: int = 0
