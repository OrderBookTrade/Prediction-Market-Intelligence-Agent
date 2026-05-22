"""
Risk analysis tools for prediction markets.

These perform deterministic checks before the LLM analysis,
reducing hallucination on factual risk signals.
"""

import httpx
from langchain_core.tools import tool

GAMMA_BASE = "https://gamma-api.polymarket.com"

# Known reliable resolution sources
TRUSTED_SOURCES = {
    "reuters.com", "bloomberg.com", "ap.org", "apnews.com",
    "ft.com", "wsj.com", "bbc.com", "polymarket.com",
    "federalreserve.gov", "sec.gov", "cftc.gov",
    "github.com", "openai.com", "anthropic.com",
    "coinmarketcap.com", "coingecko.com",
}

# Ambiguity keywords in resolution rules
AMBIGUITY_KEYWORDS = [
    "announced", "expected", "likely", "report", "sources say",
    "reportedly", "if applicable", "at discretion", "may",
    "generally", "approximately", "or equivalent",
    "as determined by", "in the opinion of",
]


@tool
def analyze_market_risk(condition_id: str) -> str:
    """
    Run deterministic risk checks on a market.
    Checks: liquidity, spread, resolution source credibility, rule ambiguity.
    Returns a structured risk report.
    Call this before generating the research memo.
    """
    try:
        resp = httpx.get(
            f"{GAMMA_BASE}/markets/{condition_id}",
            timeout=15.0
        )
        m = resp.json()
    except Exception as e:
        return f"Cannot analyze risk: {e}"

    risks = []
    warnings = []

    # 1. Liquidity check
    liquidity = float(m.get("liquidity", 0))
    volume = float(m.get("volume", 0))

    if liquidity < 1000:
        risks.append("🔴 CRITICAL: Liquidity under $1K — market price may not reflect true probability")
    elif liquidity < 10000:
        risks.append("🟡 WARNING: Low liquidity ($" + f"{liquidity:,.0f}) — large orders will move price significantly")
    else:
        warnings.append(f"✅ Liquidity adequate: ${liquidity:,.0f}")

    # 2. Volume/liquidity ratio
    if volume > 0 and liquidity > 0:
        ratio = volume / liquidity
        if ratio > 20:
            risks.append(f"🟡 High volume/liquidity ratio ({ratio:.1f}x) — book may be thin relative to interest")

    # 3. Resolution source credibility
    rules_text = m.get("rules", m.get("resolutionSource", ""))
    if not rules_text:
        risks.append("🔴 No resolution rules found — cannot verify how this market settles")
    else:
        found_trusted = any(src in rules_text.lower() for src in TRUSTED_SOURCES)
        if not found_trusted:
            risks.append("🟡 Resolution source not from known trusted list — verify manually")
        else:
            warnings.append("✅ Resolution source appears to be a trusted outlet")

        # Ambiguity check
        found_ambiguities = [kw for kw in AMBIGUITY_KEYWORDS if kw in rules_text.lower()]
        if len(found_ambiguities) >= 3:
            risks.append(f"🔴 HIGH ambiguity in rules — found keywords: {found_ambiguities}")
        elif found_ambiguities:
            risks.append(f"🟡 Some ambiguous language in rules: {found_ambiguities}")

    # 4. Price extremes
    try:
        prices = [float(p) for p in m.get("outcomePrices", "[]").strip("[]").split(",")]
        if prices:
            yes = prices[0]
            if yes < 0.02:
                risks.append(f"⚠️ YES at {yes:.1%} — near zero, check if market is essentially settled")
            elif yes > 0.98:
                risks.append(f"⚠️ YES at {yes:.1%} — near certainty, check if already resolved")
    except Exception:
        pass

    # 5. End date proximity
    end_date = m.get("endDate", "")
    if end_date:
        warnings.append(f"📅 Market ends: {end_date}")

    # Summary
    risk_count = len(risks)
    if risk_count == 0:
        overall = "LOW — no major risk flags"
    elif risk_count <= 2:
        overall = "MEDIUM — some risks, see details"
    else:
        overall = "HIGH — multiple risks detected"

    report = f"""Risk Analysis for: {m.get('question')}
Overall Risk: {overall}

Risk Flags ({risk_count}):
""" + "\n".join(f"  {r}" for r in risks) + "\n\nAdditional Context:\n" + "\n".join(f"  {w}" for w in warnings)

    return report


@tool
def extract_resolution_rules(condition_id: str) -> str:
    """
    Extract and structure the resolution rules for a market.
    Returns: resolution source, deadline, exact condition, known ambiguities.
    Call this to understand HOW the market settles before analyzing WHAT will happen.
    """
    try:
        resp = httpx.get(
            f"{GAMMA_BASE}/markets/{condition_id}",
            timeout=15.0
        )
        m = resp.json()
    except Exception as e:
        return f"Cannot fetch rules: {e}"

    question = m.get("question", "Unknown")
    rules = m.get("rules", m.get("resolutionSource", "Not specified"))
    description = m.get("description", "")[:800]
    end_date = m.get("endDate", "Not specified")

    return f"""Resolution Rules for: {question}

Question: {question}
End Date: {end_date}

Resolution Rules:
{rules}

Market Description:
{description}

---
ANALYSIS NOTES:
Before generating the research memo, consider:
1. What EXACTLY needs to happen for YES to resolve?
2. What is the OFFICIAL source that determines resolution?
3. Are there any edge cases or ambiguous phrases?
4. What timezone applies to the deadline?
5. Is there a dispute/challenge mechanism?"""
