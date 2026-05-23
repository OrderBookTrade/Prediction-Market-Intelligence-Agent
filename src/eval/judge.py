"""LLM-as-judge for evaluating ResearchMemo quality.

Scores a memo on four dimensions, each 0–10:
  citation_score   — claims backed by real, identifiable sources
  calibration_score — agent_estimate proximity to market_probability (structural)
  reasoning_score  — engagement with key uncertainties, no weasel words
  hedge_score      — appropriate "insufficient evidence" usage when warranted

Design principles:
  - `_call_judge_llm` is a standalone coroutine so tests can monkeypatch it.
  - Temperature 0 — deterministic grading, reproducible across runs.
  - Pydantic-validated output — no raw string parsing, ever.
  - calibration_score is computed structurally (no LLM needed) to avoid
    the LLM second-guessing numeric facts.
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# ── Output schema ──────────────────────────────────────────────────────────────

class MemoGrade(BaseModel):
    """LLM-as-judge output for one ResearchMemo."""

    run_id: str
    condition_id: str

    citation_score: float = Field(ge=0, le=10)
    calibration_score: float = Field(ge=0, le=10)   # structural, not LLM
    reasoning_score: float = Field(ge=0, le=10)
    hedge_score: float = Field(ge=0, le=10)
    overall: float = Field(ge=0, le=10)

    # Specific, actionable feedback items — min 1, max 5
    feedback: list[str] = Field(default_factory=list)

    model_name: str = ""
    prompt_version: str = "judge-v1"

    @field_validator("feedback")
    @classmethod
    def at_least_one_feedback(cls, v: list[str]) -> list[str]:
        if not v:
            return ["No specific feedback provided."]
        return v[:5]  # cap at 5

    @property
    def weighted_overall(self) -> float:
        """Weighted composite: calibration 30%, citation 30%, reasoning 25%, hedge 15%."""
        return round(
            self.calibration_score * 0.30
            + self.citation_score * 0.30
            + self.reasoning_score * 0.25
            + self.hedge_score * 0.15,
            2,
        )

    @property
    def letter_grade(self) -> str:
        w = self.weighted_overall
        if w >= 8.5:  return "A"
        if w >= 7.0:  return "B"
        if w >= 5.5:  return "C"
        if w >= 4.0:  return "D"
        return "F"


# ── Structural calibration score (no LLM) ─────────────────────────────────────

def _calibration_score(agent_estimate: float, market_probability: float) -> float:
    """Score 0–10 based on |agent_estimate - market_probability|.

    An estimate within 5% of market = 10.  Gap of 40%+ = 0.
    This is structural — we're NOT grading whether the agent is right,
    we're grading whether the agent's estimate is well-anchored to market data.

    A large divergence from market price needs to be *explicitly justified*
    in the memo — if it isn't, the reasoning_score will catch it.
    """
    gap = abs(agent_estimate - market_probability)
    # Linear decay from 10 → 0 over [0, 0.40]
    score = max(0.0, 10.0 * (1.0 - gap / 0.40))
    return round(score, 2)


# ── LLM call (patchable in tests) ─────────────────────────────────────────────

_JUDGE_SYSTEM = """\
You are an expert evaluator of prediction market research memos.
Your job is to score a memo on three dimensions and return a JSON object.
Be strict. Do not give high scores unless clearly deserved.
Temperature is 0 — your output must be deterministic and based only on the memo text.
"""

_JUDGE_USER_TEMPLATE = """\
MARKET QUESTION: {question}
MARKET PROBABILITY: {market_prob:.1%}
AGENT ESTIMATE: {agent_est:.1%}

--- MEMO CONTENT ---
YES CASE ({yes_n} items):
{yes_case}

NO CASE ({no_n} items):
{no_case}

RESOLUTION ANALYSIS:
  Source: {resolution_source}
  Condition: {resolution_condition}
  Ambiguities: {ambiguities}

KEY UNCERTAINTIES:
{uncertainties}

RECOMMENDATION: {recommendation} (rationale: {rationale})
---

Score this memo on THREE dimensions, each 0–10:

1. citation_score: Are claims in yes_case and no_case backed by real, identifiable sources
   (URLs, publisher names)? Deduct for "N/A", empty sources, or generic "source" strings.
   10 = all items have real URLs; 0 = no sources anywhere.

2. reasoning_score: Does the memo engage deeply with key_uncertainties?
   Does the recommendation rationale address the resolution condition specifically?
   Deduct for weasel phrases ("may", "could", "possibly" without explanation), circular
   reasoning, or uncertainties listed but never addressed.
   10 = deep, specific engagement; 0 = boilerplate/generic.

3. hedge_score: Is uncertainty expressed appropriately? The memo should NOT claim
   high confidence when evidence is thin. It SHOULD hedge when sources are weak.
   10 = honest, well-calibrated hedging; 0 = overconfident with weak evidence.

Also provide 1–5 specific, actionable feedback items (strings).
Feedback should name the SPECIFIC problem (e.g., "yes_case[0] has source='N/A'
but claims a specific poll result — add a real URL").

Return ONLY a JSON object with keys:
  citation_score, reasoning_score, hedge_score, overall, feedback

Where overall is YOUR holistic 0–10 assessment (can differ from the weighted average).
"""


async def _call_judge_llm(
    prompt: str,
    model: str,
    api_key: str,
) -> dict[str, Any]:
    """Standalone coroutine — monkeypatch this in tests."""
    import anthropic

    client = anthropic.AsyncAnthropic(api_key=api_key)
    msg = await client.messages.create(
        model=model,
        max_tokens=1024,
        temperature=0,
        system=_JUDGE_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    text = msg.content[0].text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()

    return json.loads(text)


# ── Main judge entry point ─────────────────────────────────────────────────────

async def grade_memo(
    memo_dict: dict[str, Any],
    run_id: str,
    condition_id: str,
    model: str = "claude-sonnet-4-5",
    api_key: str | None = None,
) -> MemoGrade:
    """Grade a ResearchMemo dict with an LLM judge.

    Args:
        memo_dict: The full ResearchMemo as a dict (from DB memo_json field).
        run_id: The agent run that produced this memo.
        condition_id: The market's condition_id.
        model: Claude model to use for grading.
        api_key: Anthropic API key (falls back to ANTHROPIC_API_KEY env var).

    Returns:
        MemoGrade with all scores populated.
    """
    from src.config import settings

    key = api_key or settings.anthropic_api_key

    # ── Structural score — no LLM needed ──────────────────────────────────────
    agent_est = float(memo_dict.get("agent_estimate", 0.5))
    market_prob = float(memo_dict.get("current_probability", 0.5))
    cal_score = _calibration_score(agent_est, market_prob)

    # ── Build prompt ──────────────────────────────────────────────────────────
    yes_items = memo_dict.get("yes_case", [])
    no_items = memo_dict.get("no_case", [])

    def _fmt_items(items: list[dict]) -> str:
        if not items:
            return "  (none)"
        lines = []
        for i, it in enumerate(items):
            src = it.get("source") or it.get("source_url") or "N/A"
            lines.append(f"  [{i}] claim: {it.get('claim', '')!r}  source: {src}")
        return "\n".join(lines)

    resolution = memo_dict.get("resolution", {})
    uncertainties = memo_dict.get("key_uncertainties", [])

    prompt = _JUDGE_USER_TEMPLATE.format(
        question=memo_dict.get("market_question", ""),
        market_prob=market_prob,
        agent_est=agent_est,
        yes_n=len(yes_items),
        yes_case=_fmt_items(yes_items),
        no_n=len(no_items),
        no_case=_fmt_items(no_items),
        resolution_source=resolution.get("source", "unknown"),
        resolution_condition=resolution.get("condition", "unknown"),
        ambiguities=", ".join(resolution.get("ambiguities", [])) or "none listed",
        uncertainties="\n".join(f"  - {u}" for u in uncertainties) or "  (none listed)",
        recommendation=memo_dict.get("recommendation", ""),
        rationale=memo_dict.get("recommendation_rationale", ""),
    )

    # ── LLM call ──────────────────────────────────────────────────────────────
    try:
        raw = await _call_judge_llm(prompt, model, key)
    except Exception as exc:
        logger.warning("Judge LLM call failed: %s — using fallback scores", exc)
        raw = {
            "citation_score": 5.0,
            "reasoning_score": 5.0,
            "hedge_score": 5.0,
            "overall": 5.0,
            "feedback": [f"Judge LLM unavailable: {exc}"],
        }

    return MemoGrade(
        run_id=run_id,
        condition_id=condition_id,
        citation_score=float(raw.get("citation_score", 5)),
        calibration_score=cal_score,
        reasoning_score=float(raw.get("reasoning_score", 5)),
        hedge_score=float(raw.get("hedge_score", 5)),
        overall=float(raw.get("overall", 5)),
        feedback=raw.get("feedback", []),
        model_name=model,
    )
