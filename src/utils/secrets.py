"""Helpers for logging secret configuration without exposing secret values."""

from __future__ import annotations

import hashlib


def safe_secret_info(value: str | None, *, expected_prefix: str | None = None) -> dict:
    """Return non-sensitive metadata for a secret value."""
    cleaned = (value or "").strip().strip('"').strip("'")
    prefix_len = len(expected_prefix or "")
    prefix = cleaned[:prefix_len] if prefix_len else cleaned[:6]
    return {
        "present": bool(cleaned),
        "prefix": prefix,
        "prefix_ok": cleaned.startswith(expected_prefix) if expected_prefix else None,
        "length": len(cleaned),
        "fingerprint": hashlib.sha256(cleaned.encode()).hexdigest()[:10] if cleaned else None,
        "has_outer_quotes": bool(value) and value.strip() != cleaned,
    }
