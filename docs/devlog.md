# PMI Agent — Dev Log

> Rule: 5 lines minimum per session. Ship before you document, but document the same day.

---

## 2026-05-23 · Sprint 3 + URL Import feature

**What shipped:**

- **Sprint 3: Evidence Retriever rewrite** — NON-NEGOTIABLE anti-hallucination chain:
  LLM quote → `_quote_valid()` substring check → drop + warn if false, accept if true.
  No CitedEvidence is ever constructed from an unverified quote. `_call_extraction_llm`
  is a standalone coroutine so tests can monkeypatch it cleanly.

- **8 new tests** in `test_evidence_retriever.py` covering: distinct queries, label assignment,
  hallucinated-quote rejection (5 bad + 3 verbatim good), whitespace normalization, URL dedup
  by credibility, and the full `_extract_and_validate` happy/sad paths.

- **URL Import feature** (frontend + backend):
  - `POST /api/markets/resolve` — parses `polymarket.com/event/` (multi-outcome) and
    `polymarket.com/market/` (Yes/No), persists to DB, returns sorted `ResolveResponse`.
  - `PolymarketClient.fetch_market_by_slug()` + `fetch_event_by_slug()` — two new Gamma API methods.
  - `ImportModal.tsx` — dark terminal modal: URL input → FETCH → YesNoResult or MultiResult.
    MultiResult strips "Will X win..." noise → shows clean outcome names with P(YES) bars.
  - E2E verified: FIFA 2026 → 60 markets, France 17.8%, Spain 17.4%.

**Learnings:**
- BM25 corpus needed `getattr(r, 'snippet', None) or getattr(r, 'content', '')` after
  `SearchResult` → `SearchHit` rename. Always grep field access after model renames.
- `replace_all` on state variable names can mangle import lines. Surgical edits only.
- `window.history` is a reserved browser global — naming a React state variable `history`
  causes silent TS shadowing bugs. Use domain-specific names: `predHistory`, `priceHistory`.

**Metrics:** 22 tests passing · 0 TS errors · backend + frontend both hot on :8000/:3000

---

## 2026-05-24 · Sprint 4 prep

**State of codebase:**

Agents wired: `market_analyzer → evidence_retriever → risk_critic → memo_writer`  
Frontend tabs: Discovery · Watchlist · Performance · Settings  
Real data: market feed live from Gamma API · URL import works end-to-end  

**Next priorities (in order):**

1. **Sprint 5 — Evals pipeline** (the moat): Brier score tracking, calibration curve,
   LLM-as-judge for memo quality. This is what separates the project from a toy.
2. **Sprint 4 — Search UX**: Accept Polymarket URLs directly in the search bar without
   opening a modal. Low effort, high polish.
3. **Tiered models**: cheap default (haiku/sonnet) + user brings own API key in localStorage.
4. **Devlog discipline**: Keep writing here. Recruiters at Anthropic/OpenAI read the git log.

---
