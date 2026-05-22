"""
Data schemas for the Prediction Market Intelligence Agent.

All LLM outputs are structured using these Pydantic models.
This enforces consistency and makes evals tractable.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


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
    claim: str = Field(description="The specific claim or evidence point")
    source: Optional[str] = Field(None, description="URL or source name")
    credibility: Optional[str] = Field(None, description="high/medium/low/unknown")


class ResolutionAnalysis(BaseModel):
    source: str = Field(description="Official resolution source")
    deadline: str = Field(description="Exact deadline including timezone")
    condition: str = Field(description="Precise event condition being predicted")
    ambiguities: list[str] = Field(
        default_factory=list,
        description="Potential edge cases or ambiguous clauses"
    )
    risk_level: RiskLevel = Field(description="Overall resolution risk")
    risk_notes: str = Field(description="Explanation of resolution risk")


class MarketState(BaseModel):
    yes_price: float = Field(description="Current YES price (0-1)")
    no_price: float = Field(description="Current NO price (0-1)")
    volume: float = Field(description="Total trading volume in USD")
    liquidity: float = Field(description="Available liquidity in USD")
    spread: float = Field(description="Bid-ask spread")
    price_change_24h: Optional[float] = Field(
        None, description="24h price change in percentage points"
    )
    liquidity_risk: RiskLevel = Field(description="Liquidity risk level")
    liquidity_notes: str = Field(description="Notes on liquidity")


class ResearchMemo(BaseModel):
    """
    Structured research memo output by the agent.
    Every field is required — no silent failures.
    """

    # Identity
    market_id: str
    market_question: str
    platform: str = "polymarket"

    # Market state
    current_probability: float = Field(description="Market-implied probability (YES price)")
    market_state: MarketState

    # Agent analysis
    agent_estimate: float = Field(
        description="Agent's probability estimate (0-1)",
        ge=0, le=1
    )
    edge: float = Field(
        description="Agent estimate minus market probability. Positive = agent thinks YES underpriced."
    )
    confidence: Confidence

    # Evidence
    yes_case: list[EvidenceItem] = Field(
        description="Evidence supporting YES outcome"
    )
    no_case: list[EvidenceItem] = Field(
        description="Evidence supporting NO outcome"
    )

    # Risk
    resolution: ResolutionAnalysis
    manipulation_risk: RiskLevel = Field(
        description="Risk of insider trading or market manipulation"
    )
    manipulation_notes: str

    # Output
    recommendation: Recommendation
    recommendation_rationale: str
    key_uncertainties: list[str] = Field(
        description="Top 3 things that would change this analysis"
    )

    # Metadata
    model_name: str
    prompt_version: str = "v1"
    search_queries_used: list[str] = Field(default_factory=list)
    sources_found: int = 0


class MarketSummary(BaseModel):
    """Lightweight market card for discovery dashboard."""
    market_id: str
    question: str
    category: str
    yes_price: float
    volume: float
    liquidity: float
    spread: float
    end_date: str
    price_change_24h: Optional[float] = None
    alert_flags: list[str] = Field(
        default_factory=list,
        description="e.g. ['low_liquidity', 'wide_spread', 'price_spike']"
    )
