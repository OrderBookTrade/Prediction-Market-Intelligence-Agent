"""Transform a MarketRaw into a validated MarketSnapshot."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from src.schemas import MarketRaw, MarketSnapshot

logger = logging.getLogger(__name__)


def _to_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _parse_prices(raw: MarketRaw) -> tuple[float | None, float | None]:
    """Extract yes/no prices.

    outcomePrices is either a JSON-encoded string '["0.67","0.33"]' or
    already a list. Falls back to tokens[].price if absent.
    """
    prices = raw.outcome_prices

    if prices is not None:
        if isinstance(prices, str):
            try:
                prices = json.loads(prices)
            except json.JSONDecodeError:
                logger.warning("Could not parse outcomePrices as JSON: %r", prices)
                prices = None

        if isinstance(prices, list) and len(prices) >= 2:
            return _to_float(prices[0]), _to_float(prices[1])

    # Fall back to tokens array
    if raw.tokens:
        yes = next(
            (_to_float(t.get("price")) for t in raw.tokens if t.get("outcome") == "Yes"),
            None,
        )
        no = next(
            (_to_float(t.get("price")) for t in raw.tokens if t.get("outcome") == "No"),
            None,
        )
        return yes, no

    return None, None


def _parse_outcomes(raw: MarketRaw) -> list[str]:
    outcomes = raw.outcomes
    if outcomes is None:
        return ["Yes", "No"]
    if isinstance(outcomes, str):
        try:
            parsed = json.loads(outcomes)
            if isinstance(parsed, list):
                return [str(o) for o in parsed]
        except json.JSONDecodeError:
            pass
        return ["Yes", "No"]
    if isinstance(outcomes, list):
        return [str(o) for o in outcomes]
    return ["Yes", "No"]


def _parse_end_date(raw: MarketRaw) -> datetime | None:
    if not raw.end_date:
        return None
    try:
        return datetime.fromisoformat(raw.end_date.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        logger.warning("Could not parse endDate: %r", raw.end_date)
        return None


def _extract_category(raw: MarketRaw) -> str | None:
    if not raw.tags:
        return None
    first = raw.tags[0]
    return first.get("label") or first.get("slug")


def _compute_spread(
    raw: MarketRaw,
    yes_price: float | None,
    no_price: float | None,
) -> float | None:
    # Use API-supplied spread when available
    if raw.spread is not None:
        return raw.spread
    # Fall back: vig = how far yes+no deviate from 1
    if yes_price is not None and no_price is not None:
        return round(abs(1.0 - yes_price - no_price), 6)
    return None


def normalize(raw: MarketRaw) -> MarketSnapshot:
    """Convert a raw Gamma API market object into a validated MarketSnapshot.

    Raises ValueError if neither id nor conditionId is present.
    """
    condition_id = raw.condition_id or raw.id
    if not condition_id:
        raise ValueError(f"MarketRaw has no identifiable condition_id: {raw!r}")

    yes_price, no_price = _parse_prices(raw)

    return MarketSnapshot(
        condition_id=condition_id,
        question=raw.question or "",
        description=raw.description,
        outcomes=_parse_outcomes(raw),
        yes_price=yes_price,
        no_price=no_price,
        volume=raw.volume,
        liquidity=raw.liquidity,
        spread=_compute_spread(raw, yes_price, no_price),
        end_date=_parse_end_date(raw),
        resolution_source=raw.resolution_source,
        category=_extract_category(raw),
        raw_rules_text=raw.description,
    )
