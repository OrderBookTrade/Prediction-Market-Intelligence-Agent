"""Tests for src/eval/judge.py — LLM-as-judge evaluation module.

All LLM calls are monkeypatched; no network traffic in CI.
"""
from __future__ import annotations

import pytest

from src.eval.judge import MemoGrade, _calibration_score, grade_memo


# ── Fixtures ──────────────────────────────────────────────────────────────────

EXCELLENT_MEMO = {
    "market_question": "Will the Fed cut rates in June 2026?",
    "current_probability": 0.72,
    "agent_estimate": 0.75,
    "yes_case": [
        {
            "claim": "Inflation fell to 2.1% in April 2026",
            "source": "https://bls.gov/cpi/2026-04",
            "credibility": "HIGH",
        },
        {
            "claim": "Fed chair signalled dovish pivot at May testimony",
            "source": "https://federalreserve.gov/newsevents/testimony/2026-05",
            "credibility": "HIGH",
        },
    ],
    "no_case": [
        {
            "claim": "Labour market still tight — unemployment at 3.8%",
            "source": "https://bls.gov/news.release/empsit.2026-05",
            "credibility": "HIGH",
        },
    ],
    "resolution": {
        "source": "federalreserve.gov",
        "condition": "Rate cut announced at June 2026 FOMC meeting",
        "ambiguities": [],
        "risk_level": "LOW",
        "risk_notes": "Clear, specific resolution source",
    },
    "key_uncertainties": [
        "May CPI print not yet released",
        "Geopolitical shocks could reverse dovish trend",
    ],
    "recommendation": "watch",
    "recommendation_rationale": (
        "Market at 72% seems slightly underpriced given fed chair testimony, "
        "but May CPI uncertainty warrants holding off on a strong call."
    ),
}

POOR_MEMO = {
    "market_question": "Will X happen?",
    "current_probability": 0.50,
    "agent_estimate": 0.90,   # huge divergence — overconfident
    "yes_case": [
        {"claim": "Things might happen", "source": None, "credibility": "LOW"},
    ],
    "no_case": [],
    "resolution": {
        "source": "unknown",
        "condition": "unclear",
        "ambiguities": ["may resolve at discretion", "no specific source"],
        "risk_level": "HIGH",
        "risk_notes": "No clear resolution path",
    },
    "key_uncertainties": [],
    "recommendation": "candidate_opportunity",
    "recommendation_rationale": "Could be good, seems likely.",
}


# ── _calibration_score ────────────────────────────────────────────────────────

def test_calibration_perfect():
    """Identical estimate and market price → score 10."""
    assert _calibration_score(0.70, 0.70) == pytest.approx(10.0)


def test_calibration_small_gap():
    """5 pp gap → score still ~8.75."""
    score = _calibration_score(0.75, 0.70)
    assert 8.0 < score <= 10.0


def test_calibration_large_gap():
    """40 pp gap → score 0."""
    assert _calibration_score(0.90, 0.50) == pytest.approx(0.0)


def test_calibration_exceeds_threshold():
    """Gap > 40 pp clamps to 0, never goes negative."""
    assert _calibration_score(1.0, 0.0) == pytest.approx(0.0)


# ── MemoGrade model ───────────────────────────────────────────────────────────

def test_grade_weighted_overall():
    """weighted_overall = 0.30*cal + 0.30*cit + 0.25*rea + 0.15*hedge."""
    g = MemoGrade(
        run_id="r1", condition_id="c1",
        citation_score=8.0, calibration_score=9.0,
        reasoning_score=7.0, hedge_score=6.0,
        overall=7.5,
        feedback=["good job"],
    )
    expected = round(0.30 * 9.0 + 0.30 * 8.0 + 0.25 * 7.0 + 0.15 * 6.0, 2)
    assert g.weighted_overall == pytest.approx(expected)


def test_grade_letter_a():
    g = MemoGrade(
        run_id="r2", condition_id="c2",
        citation_score=9.0, calibration_score=9.5,
        reasoning_score=9.0, hedge_score=8.5,
        overall=9.0, feedback=["excellent"],
    )
    assert g.letter_grade == "A"


def test_grade_letter_f():
    g = MemoGrade(
        run_id="r3", condition_id="c3",
        citation_score=1.0, calibration_score=1.0,
        reasoning_score=1.0, hedge_score=1.0,
        overall=1.0, feedback=["very poor"],
    )
    assert g.letter_grade == "F"


def test_grade_empty_feedback_gets_placeholder():
    """Validator fills in default when feedback list is empty."""
    g = MemoGrade(
        run_id="r4", condition_id="c4",
        citation_score=5, calibration_score=5,
        reasoning_score=5, hedge_score=5,
        overall=5, feedback=[],
    )
    assert len(g.feedback) == 1
    assert "No specific" in g.feedback[0]


def test_grade_feedback_capped_at_5():
    """More than 5 feedback items are silently truncated."""
    g = MemoGrade(
        run_id="r5", condition_id="c5",
        citation_score=5, calibration_score=5,
        reasoning_score=5, hedge_score=5,
        overall=5,
        feedback=["a", "b", "c", "d", "e", "f", "g"],
    )
    assert len(g.feedback) == 5


# ── grade_memo integration (monkeypatched LLM) ───────────────────────────────

@pytest.mark.asyncio
async def test_grade_excellent_memo(monkeypatch):
    """Excellent memo → all LLM scores ≥ 7, calibration high, letter A or B."""
    async def _mock_llm(prompt, model, api_key):
        return {
            "citation_score": 9.0,
            "reasoning_score": 8.5,
            "hedge_score": 8.0,
            "overall": 8.5,
            "feedback": ["Strong citation coverage", "Resolution source is authoritative"],
        }

    monkeypatch.setattr("src.eval.judge._call_judge_llm", _mock_llm)

    grade = await grade_memo(
        memo_dict=EXCELLENT_MEMO,
        run_id="test-excellent",
        condition_id="0xabc",
        api_key="dummy",
    )

    assert grade.citation_score >= 7.0
    assert grade.reasoning_score >= 7.0
    assert grade.calibration_score >= 8.0   # 3 pp gap → structural score ~9.25
    assert grade.letter_grade in ("A", "B")
    assert len(grade.feedback) >= 1


@pytest.mark.asyncio
async def test_grade_poor_memo(monkeypatch):
    """Poor memo → LLM scores low, calibration 0 (40 pp gap), letter D or F."""
    async def _mock_llm(prompt, model, api_key):
        return {
            "citation_score": 1.5,
            "reasoning_score": 2.0,
            "hedge_score": 1.0,
            "overall": 1.5,
            "feedback": [
                "No sources in yes_case",
                "no_case is empty",
                "agent_estimate 0.90 diverges massively from market 0.50",
                "recommendation_rationale is vague",
            ],
        }

    monkeypatch.setattr("src.eval.judge._call_judge_llm", _mock_llm)

    grade = await grade_memo(
        memo_dict=POOR_MEMO,
        run_id="test-poor",
        condition_id="0xdef",
        api_key="dummy",
    )

    assert grade.calibration_score == pytest.approx(0.0)  # 40 pp gap
    assert grade.citation_score <= 3.0
    assert grade.letter_grade in ("D", "F")
    assert len(grade.feedback) >= 3


@pytest.mark.asyncio
async def test_grade_llm_failure_returns_defaults(monkeypatch):
    """If LLM call raises, grade_memo returns fallback 5.0 scores and continues."""
    async def _boom(prompt, model, api_key):
        raise RuntimeError("API unavailable")

    monkeypatch.setattr("src.eval.judge._call_judge_llm", _boom)

    grade = await grade_memo(
        memo_dict=EXCELLENT_MEMO,
        run_id="test-fallback",
        condition_id="0xfff",
        api_key="dummy",
    )

    # Fallback scores are 5.0 for LLM dimensions; calibration is still structural
    assert grade.citation_score == pytest.approx(5.0)
    assert grade.reasoning_score == pytest.approx(5.0)
    assert grade.hedge_score == pytest.approx(5.0)
    assert grade.calibration_score > 0   # structural score still computed
    assert "unavailable" in grade.feedback[0].lower()
