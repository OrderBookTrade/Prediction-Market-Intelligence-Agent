"""
Polymarket data tools.

Gamma API is public — no auth required for read operations.
All market data, prices, and metadata are freely accessible.
"""

import httpx
from langchain_core.tools import tool

GAMMA_BASE = "https://gamma-api.polymarket.com"
TIMEOUT = 15.0


def _get(path: str, params: dict = None) -> dict | list:
    resp = httpx.get(f"{GAMMA_BASE}{path}", params=params, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


@tool
def get_market_by_id(condition_id: str) -> str:
    """
    Get full market data for a specific Polymarket market.
    condition_id: the market's conditionId from Polymarket.
    Returns question, prices, volume, liquidity, rules, end date.
    """
    try:
        m = _get(f"/markets/{condition_id}")
    except Exception as e:
        return f"Error fetching market: {e}"

    yes_price = "N/A"
    no_price = "N/A"
    try:
        prices = [float(p) for p in m.get("outcomePrices", "[]").strip("[]").split(",")]
        if len(prices) >= 2:
            yes_price = f"{prices[0]:.2%}"
            no_price = f"{prices[1]:.2%}"
    except Exception:
        pass

    volume = float(m.get("volume", 0))
    liquidity = float(m.get("liquidity", 0))

    # Spread from orderbook if available
    spread = "N/A"
    try:
        spread_val = float(m.get("spread", 0))
        spread = f"{spread_val:.4f}"
    except Exception:
        pass

    return f"""Market: {m.get('question')}
ID: {condition_id}
YES price: {yes_price} | NO price: {no_price}
Volume: ${volume:,.0f}
Liquidity: ${liquidity:,.0f}
Spread: {spread}
End date: {m.get('endDate', 'N/A')}
Active: {m.get('active')}
Description: {m.get('description', '')[:500]}
Rules/Resolution: {m.get('rules', m.get('resolutionSource', 'Not specified'))[:500]}
Tags: {m.get('tags', [])}"""


@tool
def scan_ai_crypto_markets(limit: int = 20) -> str:
    """
    Scan active Polymarket markets in AI and Crypto categories.
    Returns top markets sorted by volume with price movements and flags.
    Use this to discover markets worth researching.
    """
    try:
        # Try AI markets
        ai_markets = _get("/markets", {
            "limit": limit,
            "active": True,
            "order": "volume",
            "ascending": False,
        })
    except Exception as e:
        return f"Error scanning markets: {e}"

    # Filter for AI/crypto related by keyword
    keywords = [
        "ai", "artificial intelligence", "openai", "anthropic", "gpt",
        "claude", "gemini", "bitcoin", "btc", "eth", "crypto", "llm",
        "nvidia", "deepseek", "model", "chatgpt"
    ]

    filtered = []
    for m in ai_markets:
        q = m.get("question", "").lower()
        desc = m.get("description", "").lower()
        if any(kw in q or kw in desc for kw in keywords):
            filtered.append(m)

    if not filtered:
        filtered = ai_markets[:10]  # fallback to top markets

    results = []
    for m in filtered[:15]:
        flags = []
        volume = float(m.get("volume", 0))
        liquidity = float(m.get("liquidity", 0))

        if liquidity < 5000:
            flags.append("⚠️ low_liquidity")
        if volume > 0 and liquidity > 0 and volume / liquidity > 10:
            flags.append("⚠️ thin_book")
        try:
            prices = [float(p) for p in m.get("outcomePrices", "[]").strip("[]").split(",")]
            if prices and (prices[0] < 0.05 or prices[0] > 0.95):
                flags.append("🔴 extreme_price")
        except Exception:
            pass

        flag_str = " ".join(flags) if flags else "✅ normal"
        try:
            prices_raw = m.get("outcomePrices", "N/A")
        except Exception:
            prices_raw = "N/A"

        results.append(
            f"- {m.get('question')}\n"
            f"  ID: {m.get('conditionId', 'N/A')}\n"
            f"  Prices: {prices_raw} | Vol: ${volume:,.0f} | Liq: ${liquidity:,.0f}\n"
            f"  Flags: {flag_str}"
        )

    return f"AI/Crypto markets ({len(results)} found):\n\n" + "\n\n".join(results)


@tool
def get_market_price_history(condition_id: str) -> str:
    """
    Get recent price history for a market.
    Useful for understanding price trajectory and momentum.
    condition_id: market's conditionId.
    """
    try:
        history = _get(f"/prices-history", {
            "market": condition_id,
            "interval": "1d",
            "fidelity": 10,
        })
    except Exception as e:
        return f"Could not fetch price history: {e}"

    if not history or not isinstance(history, list):
        return "No price history available"

    # Show last 10 data points
    recent = history[-10:] if len(history) > 10 else history
    points = []
    for p in recent:
        t = p.get("t", "")
        price = p.get("p", "N/A")
        points.append(f"  {t}: {float(price):.2%}" if price != "N/A" else f"  {t}: N/A")

    return "Recent price history (YES):\n" + "\n".join(points)
