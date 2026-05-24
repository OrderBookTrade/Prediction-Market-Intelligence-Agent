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

## 2026-05-24 · Sprint 5 — LLM-as-judge eval pipeline

**What shipped:**

- **`src/eval/judge.py`** — LLM-as-judge with 4 scoring dimensions:
  - `calibration_score` (structural, no LLM): `10 × (1 - |agent_est - market_prob| / 0.40)`.
    Decoupled from LLM deliberately — calibration is a math fact, not a judgment call.
  - `citation_score` (LLM): are claims backed by real URLs, not `None`/`"N/A"`?
  - `reasoning_score` (LLM): deep engagement with uncertainties, no weasel words?
  - `hedge_score` (LLM): appropriate uncertainty expression, not overconfident?
  - `weighted_overall`: `cal×0.30 + cit×0.30 + rea×0.25 + hedge×0.15`
  - `letter_grade`: A/B/C/D/F based on `weighted_overall`
  - Fallback to 5.0 scores when API key unavailable — structural calibration always works.
  - `_call_judge_llm` is a standalone coroutine — monkeypatched in all tests.

- **`EvalGradeORM`** — new table `eval_grades` with unique constraint on `run_id`.
  `upsert_eval_grade` deletes-then-inserts for idempotent re-grading.

- **`POST /api/eval/grade/{run_id}`** + **`GET /api/eval/grades`** — grade a memo by
  run_id, persist to DB, return structured JSON. Collect all fields inside session
  to avoid `DetachedInstanceError`.

- **12 new tests** in `test_eval_judge.py`: calibration math, Pydantic model invariants,
  letter grades, feedback validation, excellent/poor memo paths, LLM failure fallback.

- **Frontend** `PerformanceDashboard.tsx`: `JudgeGradesPanel` component with score bars,
  letter grade chip, top feedback item on hover. Wired to `GET /api/eval/grades`.

**Live grading results** (3 memos):
  - `0090212d`: **C** — weighted=6.05, cal=8.5 (3pp gap), LLM unavailable → fallback 5.0
  - `b6b5ec4d`: **C** — same pattern
  - `ae362f2c`: **F** — weighted=3.5, **cal=0.0** (40pp+ divergence from market price)

The F grade is the system working: that memo's agent estimate was badly miscalibrated.

**Metrics:** 45 tests passing · 0 TS errors

---

## Next sprint ideas

1. **Sprint 4 — Search UX**: Accept Polymarket URLs directly in the search bar.
2. **Tiered models**: haiku default + user brings own API key in localStorage.
3. **Auto-resolution**: cron job that checks resolved markets, fills `outcome` + Brier score.
4. **Calibration curve**: `/api/eval/calibration` endpoint with bin-averaged accuracy.

---
