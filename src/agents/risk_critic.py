"""Node 3 — risk_critic.

Pure deterministic analysis — no LLM calls.
Produces structured risk flags and emits SSE log lines.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from src.run_store import push_log

logger = logging.getLogger(__name__)

TRUSTED_SOURCES = {
    "reuters.com", "bloomberg.com", "ap.org", "apnews.com", "ft.com",
    "wsj.com", "bbc.com", "federalreserve.gov", "sec.gov", "cftc.gov",
    "openai.com", "anthropic.com", "coinmarketcap.com", "coingecko.com",
}

AMBIGUITY_KEYWORDS = [
    "at discretion", "may", "generally", "approximately",
    "as determined by", "reportedly", "sources say", "if applicable",
    "or equivalent", "in the opinion of",
]


def _risk_level(count: int) -> str:
    if count == 0:
        return "low"
    if count <= 2:
        return "medium"
    return "high"


async def risk_critic_node(state: dict) -> dict:
    run_id: str = state["run_id"]
    snapshot: dict = state["snapshot"]

    await push_log(run_id, "Running risk assessment...", "info")

    liquidity = snapshot.get("liquidity") or 0
    volume = snapshot.get("volume") or 0
    spread = snapshot.get("spread") or 0
    yes_price = snapshot.get("yes_price")
    res_source = (snapshot.get("resolution_source") or "").lower()
    rules = (snapshot.get("raw_rules_text") or "").lower()
    end_date_str = snapshot.get("end_date")

    risk_flags: list[str] = []
    risk_details: dict[str, dict] = {}

    # 1. Liquidity risk
    if liquidity < 1_000:
        risk_flags.append("LOW_LIQUIDITY")
        liq_level, liq_note = "high", f"CRITICAL: Liquidity ${liquidity:,.0f} — price may not reflect true probability"
    elif liquidity < 10_000:
        risk_flags.append("LOW_LIQUIDITY")
        liq_level, liq_note = "medium", f"Low liquidity ${liquidity:,.0f} — large orders cause significant slippage"
    else:
        liq_level, liq_note = "low", f"Deep book — slippage on typical orders is negligible"

    risk_details["liquidity"] = {"level": liq_level, "note": liq_note}
    flag = "✓" if liq_level == "low" else "⚠"
    await push_log(run_id, f"  Risk check: liquidity ${liquidity:,.0f} {flag}", "ok" if liq_level == "low" else "warn")

    # 2. Spread risk
    if spread > 0.05:
        risk_flags.append("WIDE_SPREAD")
        sp_note = f"Very wide spread {spread:.4f} — cost to enter/exit is high"
    elif spread > 0.01:
        risk_flags.append("WIDE_SPREAD")
        sp_note = f"Wide spread {spread:.4f} — monitor before acting"
    else:
        sp_note = f"Spread {spread:.4f} acceptable"

    await push_log(run_id, f"  Risk check: spread {spread:.4f} {'✓' if spread <= 0.01 else '⚠'}", "ok" if spread <= 0.01 else "warn")

    # 3. Resolution risk
    trusted = any(t in res_source or t in rules for t in TRUSTED_SOURCES)
    ambiguities = [kw for kw in AMBIGUITY_KEYWORDS if kw in rules]
    if not res_source:
        res_level, res_note = "medium", "No resolution source specified"
    elif not trusted:
        res_level, res_note = "medium", "Resolution source not on trusted list — verify manually"
    elif ambiguities:
        res_level, res_note = "medium", f"Ambiguous language: {ambiguities[:2]}"
    else:
        res_level, res_note = "low", "Resolution source is on the trusted list"

    await push_log(run_id, f"  Risk check: resolution risk = {res_level.upper()}", "ok" if res_level == "low" else "warn")

    risk_details["resolution"] = {"level": res_level, "note": res_note, "ambiguities": ambiguities}

    # 4. Hallucination risk (proxy: source quality)
    sources = state.get("search_results", [])
    high_cred = sum(1 for s in sources if s.get("credibility") == "HIGH")
    if len(sources) < 2:
        hall_level, hall_note = "high", "Insufficient evidence found — memo may contain unsupported claims"
    elif high_cred == 0:
        hall_level, hall_note = "medium", "No HIGH credibility sources — treat claims with caution"
    else:
        hall_level, hall_note = "low", f"Sufficient evidence found from {high_cred} primary sources"

    risk_details["hallucination"] = {"level": hall_level, "note": hall_note}

    # 5. Price extremes
    if yes_price is not None:
        if yes_price < 0.02 or yes_price > 0.98:
            risk_flags.append("RESOLVED_SOURCE")
            await push_log(run_id, f"  ⚠ price extreme ({yes_price:.1%}) — market may be near resolution", "warn")

    # 6. Expiry
    if end_date_str:
        try:
            end_dt = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
            days_left = (end_dt.replace(tzinfo=timezone.utc) - datetime.now(timezone.utc)).days
            if days_left < 14:
                risk_flags.append("EXPIRES_SOON")
                await push_log(run_id, f"  ⚠ expires in {days_left}d", "warn")
        except Exception:
            pass

    # 7. Volume/liquidity ratio
    if volume > 0 and liquidity > 0 and volume / liquidity > 15:
        risk_flags.append("HIGH_VOLUME")

    await push_log(run_id, f"Risk assessment: {len(risk_flags)} flag(s) · resolution risk={res_level.upper()}", "info")

    return {
        "risk_flags": risk_flags,
        "risk_details": risk_details,
    }
