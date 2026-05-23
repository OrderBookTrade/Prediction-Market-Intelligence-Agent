"""Async Polymarket Gamma API client with retry logic."""

from __future__ import annotations

import logging
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.schemas import MarketRaw

logger = logging.getLogger(__name__)

_DEFAULT_WAIT = wait_exponential(multiplier=1, min=1, max=10)


class PolymarketClient:
    """Async client for the Polymarket Gamma public API.

    Pass _retry_wait=wait_none() in tests to skip exponential back-off delays.
    """

    def __init__(
        self,
        base_url: str = "https://gamma-api.polymarket.com",
        timeout: float = 30.0,
        _retry_wait: Any = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout)
        self._retry_wait = _retry_wait if _retry_wait is not None else _DEFAULT_WAIT

    async def __aenter__(self) -> PolymarketClient:
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self._client.aclose()

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{self._base_url}{path}"
        last_exc: Exception | None = None

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(3),
                wait=self._retry_wait,
                retry=retry_if_exception_type(httpx.HTTPStatusError),
                reraise=True,
                before_sleep=lambda rs: logger.warning(
                    "Retry %d/3 for %s (status error)", rs.attempt_number, url
                ),
            ):
                with attempt:
                    logger.debug("GET %s params=%s", url, params)
                    response = await self._client.get(url, params=params)
                    response.raise_for_status()
                    return response.json()
        except RetryError as exc:
            raise RuntimeError(f"All retries exhausted for {url}") from exc

    async def fetch_active_markets(
        self,
        category: str | None = None,
        limit: int = 100,
    ) -> list[MarketRaw]:
        """Fetch active, non-closed markets.

        Args:
            category: Gamma API tag_slug filter (e.g. "ai", "crypto").
            limit: Maximum number of markets to return.
        """
        params: dict[str, Any] = {"active": "true", "closed": "false", "limit": limit}
        if category:
            params["tag_slug"] = category

        data = await self._get("/markets", params=params)

        # Gamma returns a bare list; guard against dict wrapping just in case
        raw_list: list[dict] = data if isinstance(data, list) else data.get("markets", [])

        markets = []
        for item in raw_list:
            try:
                markets.append(MarketRaw.model_validate(item))
            except Exception as exc:
                logger.warning("Skipping malformed market object: %s", exc)
        return markets

    async def fetch_market_detail(self, condition_id: str) -> MarketRaw:
        """Fetch a single market by its conditionId."""
        data = await self._get("/markets", params={"conditionId": condition_id})
        items = data if isinstance(data, list) else data.get("markets", [])
        if not items:
            raise ValueError(f"No market found for conditionId={condition_id!r}")
        return MarketRaw.model_validate(items[0])
