"""
Evaluation framework for the Prediction Market Intelligence Agent.

Three eval types:
1. Rule extraction accuracy — does the agent correctly parse resolution rules?
2. Citation support — are claims backed by actual sources?
3. Calibration — on resolved markets, how accurate were probability estimates?
   (Brier score: lower = better)
"""

import json
import math
from datetime import datetime
from pathlib import Path


class BrierScore:
    """
    Measures calibration of probability estimates on resolved markets.
    Brier = (probability - outcome)^2
    Lower is better. Perfect = 0.0, Worst = 1.0, Random = 0.25
    """

    @staticmethod
    def score(predicted: float, outcome: bool) -> float:
        return (predicted - int(outcome)) ** 2

    @staticmethod
    def interpret(score: float) -> str:
        if score < 0.05:
            return "Excellent"
        elif score < 0.15:
            return "Good"
        elif score < 0.25:
            return "Fair"
        else:
            return "Poor (worse than random)"


class CitationEval:
    """
    Checks whether claims in the memo are supported by cited sources.
    Human-graded: 0=unsupported, 1=partially, 2=fully supported
    """

    @staticmethod
    def grade(claim: str, source: str, content: str) -> dict:
        """
        Return a template for human grading.
        In production this would use an LLM judge.
        """
        return {
            "claim": claim,
            "source": source,
            "content_snippet": content[:200] if content else "N/A",
            "grade": None,  # 0, 1, or 2 — filled in by human/LLM judge
            "notes": "",
        }


class EvalLogger:
    """
    Logs memo predictions for future evaluation when markets resolve.
    """

    def __init__(self, log_path: str = "eval/predictions.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(exist_ok=True)

    def log_prediction(
        self,
        market_id: str,
        market_question: str,
        agent_estimate: float,
        market_probability: float,
        confidence: str,
        recommendation: str,
        model_name: str,
        prompt_version: str,
    ):
        """Log a prediction. Call this every time a memo is generated."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "market_id": market_id,
            "market_question": market_question,
            "agent_estimate": agent_estimate,
            "market_probability": market_probability,
            "edge": agent_estimate - market_probability,
            "confidence": confidence,
            "recommendation": recommendation,
            "model_name": model_name,
            "prompt_version": prompt_version,
            "resolved": False,
            "outcome": None,       # filled in after resolution
            "brier_score": None,   # filled in after resolution
        }

        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        return entry

    def mark_resolved(self, market_id: str, outcome: bool):
        """Update a prediction with the actual outcome and compute Brier score."""
        entries = []
        with open(self.log_path) as f:
            for line in f:
                entry = json.loads(line)
                if entry["market_id"] == market_id and not entry["resolved"]:
                    entry["resolved"] = True
                    entry["outcome"] = outcome
                    entry["brier_score"] = BrierScore.score(entry["agent_estimate"], outcome)
                entries.append(entry)

        with open(self.log_path, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

    def compute_stats(self) -> dict:
        """Compute aggregate eval stats across all resolved predictions."""
        entries = []
        with open(self.log_path) as f:
            for line in f:
                entries.append(json.loads(line))

        resolved = [e for e in entries if e["resolved"]]
        if not resolved:
            return {"message": "No resolved predictions yet"}

        brier_scores = [e["brier_score"] for e in resolved if e["brier_score"] is not None]
        avg_brier = sum(brier_scores) / len(brier_scores) if brier_scores else None

        # Accuracy by confidence level
        by_confidence = {}
        for e in resolved:
            c = e.get("confidence", "unknown")
            if c not in by_confidence:
                by_confidence[c] = {"total": 0, "correct": 0, "brier_sum": 0}
            by_confidence[c]["total"] += 1
            if e["brier_score"] is not None:
                by_confidence[c]["brier_sum"] += e["brier_score"]
                # "correct" = brier < 0.25 (better than random)
                if e["brier_score"] < 0.25:
                    by_confidence[c]["correct"] += 1

        return {
            "total_predictions": len(entries),
            "resolved": len(resolved),
            "pending": len(entries) - len(resolved),
            "avg_brier_score": round(avg_brier, 4) if avg_brier else None,
            "brier_interpretation": BrierScore.interpret(avg_brier) if avg_brier else None,
            "by_confidence": by_confidence,
        }


# ── Test Cases ────────────────────────────────────────────────────────────────

RULE_EXTRACTION_TESTS = [
    {
        "id": "re_001",
        "description": "Clear rules with specific source",
        "rules_text": "This market will resolve Yes if the Federal Reserve announces a rate cut at the June 2026 FOMC meeting. Resolution source: federalreserve.gov",
        "expected": {
            "resolution_source": "federalreserve.gov",
            "condition": "rate cut announced at June 2026 FOMC",
            "ambiguity_count": 0,
        }
    },
    {
        "id": "re_002",
        "description": "Ambiguous rules",
        "rules_text": "This market resolves Yes if OpenAI generally announces GPT-5 as expected by most analysts. Resolution may be at Polymarket's discretion.",
        "expected": {
            "resolution_source": "polymarket discretion",
            "condition": "GPT-5 announcement",
            "ambiguity_count": 3,  # "generally", "as expected", "at discretion"
        }
    },
    {
        "id": "re_003",
        "description": "Missing resolution source",
        "rules_text": "Resolves Yes if Bitcoin price exceeds $100,000.",
        "expected": {
            "resolution_source": "not specified",
            "condition": "BTC > $100K",
            "ambiguity_count": 1,  # which data source?
        }
    },
]


def run_rule_extraction_eval(agent, verbose: bool = True) -> dict:
    """
    Test the agent's ability to extract resolution rules accurately.
    Uses synthetic test cases (no API call needed).
    """
    results = []
    for tc in RULE_EXTRACTION_TESTS:
        # This would call the agent with the rules text
        # For now, return the test structure
        results.append({
            "id": tc["id"],
            "description": tc["description"],
            "grade": None,  # TODO: implement LLM judge
        })

    if verbose:
        print(f"Rule extraction eval: {len(results)} test cases")
        print("(Grading requires LLM judge — run with --grade flag)")

    return {"test_cases": results}
