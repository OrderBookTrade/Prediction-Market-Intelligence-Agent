"""Tests for memo_writer source traceability helpers."""

from __future__ import annotations

from src.agents.memo_writer import _enrich_case_items, _real_source_candidates


def test_enrich_case_items_preserves_source_metadata_and_side():
    sources = [
        {
            "source_id": "src_yes",
            "domain": "reuters.com",
            "url": "https://reuters.com/yes",
            "label": "yes_case",
            "side": "yes",
            "support_level": "quote_verified",
            "cred": "HIGH",
        },
        {
            "source_id": "src_no",
            "domain": "apnews.com",
            "url": "https://apnews.com/no",
            "label": "no_case",
            "side": "no",
            "support_level": "snippet_supported",
            "cred": "MEDIUM",
        },
    ]

    candidates = _real_source_candidates(sources, [])

    yes_case = _enrich_case_items(
        [
            {"claim": "YES supported", "source": "reuters.com"},
            {"claim": "Wrong side should drop", "source": "apnews.com"},
            {"claim": "Missing source should drop", "source": "unknown.example"},
        ],
        "yes",
        candidates,
    )

    no_case = _enrich_case_items(
        [{"claim": "NO supported", "source_id": "src_no"}],
        "no",
        candidates,
    )

    assert yes_case == [
        {
            "claim": "YES supported",
            "source": "reuters.com",
            "credibility": "HIGH",
            "source_id": "src_yes",
            "support_level": "quote_verified",
            "url": "https://reuters.com/yes",
        }
    ]
    assert no_case[0]["source_id"] == "src_no"
    assert no_case[0]["support_level"] == "snippet_supported"

