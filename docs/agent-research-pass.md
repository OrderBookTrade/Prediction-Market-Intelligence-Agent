# PMI Agent Research Pass

This document explains the current backend agent flow for a single research pass:

```text
POST /api/agent/run/{condition_id}
  -> background LangGraph run
  -> GET /api/agent/run/{run_id}/stream
  -> GET /api/agent/run/{run_id}/result
```

The goal is to make the agent explainable: every memo should be traceable from market data, to search queries, to sources, to evidence claims, to the final recommendation.

---

## 1. API Lifecycle

### 1.1 Start a Run

Endpoint:

```http
POST /api/agent/run/{condition_id}
```

Current implementation:

```python
run_id = str(uuid.uuid4())
register_run(run_id)
create_agent_run(session, run_id, condition_id)
asyncio.create_task(_execute_run(condition_id, run_id))
return {"run_id": run_id, "condition_id": condition_id, "status": "queued"}
```

What happens:

1. A new `run_id` is created.
2. `register_run(run_id)` creates an in-memory queue for live logs.
3. `create_agent_run(...)` writes a run row to the database.
4. `_execute_run(...)` starts in the background.
5. The API returns immediately with `202 Accepted`.

Important point: this endpoint does not wait for research to finish. It only starts the job.

### 1.2 Stream Run Logs

Endpoint:

```http
GET /api/agent/run/{run_id}/stream
```

This is an SSE stream. Agent nodes call:

```python
await push_log(run_id, "Dispatching 3 Tavily search queries...", "info")
```

`push_log()` writes into the run queue. The stream endpoint reads from that queue and sends each message to the frontend.

Example stream events:

```text
Fetching market data...
Dispatching 3 Tavily search queries...
Retrieved 15 raw hits — reranking...
Extracting claims from sources...
Writing research memo...
```

If no event arrives for 30 seconds, the stream sends a ping. When the run completes, `push_done(run_id)` sends a sentinel and the stream closes.

### 1.3 Fetch Final Result

Endpoint:

```http
GET /api/agent/run/{run_id}/result
```

This endpoint checks the in-memory result store:

```python
result = get_result(run_id)
```

If the run is still active or the result has expired from memory, it falls back to the database run status.

If the run is done, it returns a frontend-shaped memo:

```json
{
  "status": "done",
  "run_id": "...",
  "memo": {
    "market_implied": 0.695,
    "agent_estimate": 0.42,
    "edge": -0.275,
    "yes_case": [],
    "no_case": [],
    "resolution": {},
    "sources_found": 3
  }
}
```

---

## 2. Background Research Graph

The background task calls:

```python
result = await run_analysis(condition_id, run_id)
```

`run_analysis()` executes a LangGraph pipeline:

```text
market_analyzer
  -> evidence_retriever
  -> risk_critic
  -> memo_writer
```

All nodes share one state object:

```json
{
  "condition_id": "...",
  "run_id": "...",
  "snapshot": null,
  "search_queries": [],
  "sources": [],
  "cited_evidence": [],
  "risk_details": {},
  "memo": null,
  "error": null
}
```

Each node returns partial fields, and LangGraph merges them into the state for the next node.

---

## 3. Market Analyzer

Purpose:

```text
condition_id -> market snapshot
```

The node loads market data from local DB first. If not present, it fetches from Polymarket Gamma API.

Snapshot fields:

```text
condition_id
question
description / raw rules
yes_price
no_price
volume
liquidity
spread
end_date
resolution_source
category
```

This node should not do LLM reasoning. It only creates a reliable market snapshot.

Example log:

```text
Fetching market data for 0x...
YES=0.695 VOL=$2,133,782 LIQ=$166,986
Parsing resolution rules...
source = not specified (unverified)
```

---

## 4. Query Planner

Purpose:

```text
market question -> search queries
```

The planner produces three query intents:

```text
yes_case    -> find evidence supporting YES
no_case     -> find evidence supporting NO or obstacles
resolution  -> find official rules, source, deadline, or announcement
```

Bad query:

```text
Will US and Iran sign a permanent peace deal by July 31, 2026 latest news 2025 2026
```

Better query:

```text
US Iran peace deal negotiations 2026 latest news
US Iran diplomatic agreement obstacles 2026
White House Iran peace deal official announcement
```

The key idea: search engines work better with natural event terms than with Polymarket-style questions.

The planner can use an LLM if configured. If no valid LLM key exists, it falls back to deterministic template queries.

---

## 5. Third-Party Search

Current search provider:

```text
Tavily
```

Input:

```json
{
  "query": "US Iran peace deal negotiations 2026 latest news",
  "max_results": 5,
  "days_back": 60,
  "query_label": "yes_case"
}
```

Output from Tavily is normalized into internal search hits:

```text
title
url
publisher/domain
snippet
score
published_at
query_idx
query_label
credibility
```

Important distinction:

```text
Raw hit != source-backed claim
```

At this stage, the system has only found pages. It has not yet proven any claim.

---

## 6. Source Normalization

Each raw search hit should become a normalized source object:

```json
{
  "source_id": "src_abc123",
  "url": "https://www.reuters.com/...",
  "domain": "reuters.com",
  "title": "...",
  "snippet": "...",
  "query_id": 1,
  "query_label": "yes_case",
  "credibility": "HIGH"
}
```

Why `source_id` matters:

```text
No source_id -> no claim
```

Every claim in the memo should be traceable back to a real source.

---

## 7. Aggregation and Filtering

After search, the system aggregates results:

```text
all query hits
  -> dedupe by URL
  -> BM25 rerank
  -> credibility filter
  -> top sources
```

Current credibility concept:

```text
HIGH    official sources, primary sources, major news wires
MEDIUM  reputable news/analysis sites
LOW     forums, social media, low-signal sites
```

Example run counters:

```text
planned_queries: 3
search_requests_sent: 3
raw_hits: 15
unique_sources: 12
credible_sources: 3
```

This is not memo generation yet. It only determines which sources are worth extracting evidence from.

---

## 8. Evidence Extraction

Purpose:

```text
source -> claim + quote + side
```

For each top source, the extractor asks the LLM:

```text
Given this market question and source snippet, extract the most relevant evidence.
The quote must come from the snippet.
Label it as yes_case, no_case, or resolution.
```

Expected output:

```json
{
  "claim": "US and Iran are discussing a possible agreement framework.",
  "quote": "the two sides discussed a framework...",
  "confidence": "low",
  "label": "yes_case"
}
```

Important distinction:

```text
claim = analyst-friendly summary
quote = source-supported text
label = which side this evidence supports
```

---

## 9. Source Verification and Support Levels

This is Phase 4.

The system should not treat all evidence equally. It should track support levels:

```text
quote_verified
  The quote appears verbatim in the source snippet.

raw_content_supported
  The quote appears in fetched raw article content, not just snippet.

snippet_supported
  The source snippet supports the claim, but quote is not exact.
  This can produce LOW confidence claims.

primary_source_verified
  A quote-verified claim from an official or high-authority source.
```

The rule is:

```text
No source, no claim.
```

But the system should not require perfect quote verification for every evidence item. If a source has a real URL, domain, and snippet, it can still become low-confidence `snippet_supported` evidence.

This prevents the pipeline from falling back too often when the source is useful but the LLM paraphrased the quote.

---

## 10. Risk Critic

Purpose:

```text
market snapshot + evidence quality -> deterministic risk flags
```

Checks include:

```text
liquidity risk
spread risk
resolution source risk
ambiguity risk
hallucination risk
expiry risk
price extreme risk
```

This node is deterministic. It should not use an LLM.

Example:

```text
Risk check: liquidity $166,986 OK
Risk check: spread 0.0100 OK
Risk check: resolution risk = MEDIUM
```

---

## 11. Memo Writer

Purpose:

```text
market snapshot + verified evidence + risk details -> structured memo
```

The memo writer can use an LLM to synthesize:

```text
agent_estimate
confidence
yes_case
no_case
resolution analysis
recommendation
rationale
uncertainties
```

However, the LLM should not be trusted for all fields.

Backend should compute or validate:

```text
edge_pp = (agent_estimate - market_probability) * 100
sources_found = count(real sources with URL/domain)
claim side correctness
fallback semantics
```

---

## 12. Memo Quality Gate

Before returning a memo, the backend should validate:

```text
agent_estimate is null for fallback
edge_pp is null for fallback
agent_estimate is between 0 and 1 for success
edge_pp equals (agent_estimate - market_probability) * 100
sources_found only counts real sources
yes_case only contains side=yes evidence
no_case only contains side=no evidence
each claim has source_id
each source_id maps to a real source
placeholder rows are not counted as sources
```

If validation fails, return:

```json
{
  "status": "fallback",
  "fallback_reason": "QUALITY_GATE_FAILED",
  "agent_estimate": null,
  "edge_pp": null
}
```

This is what prevents a nice-looking UI from hiding a weak memo.

---

## 13. Fallback Behavior

Fallback is not failure. It is a safe output state.

Common fallback reasons:

```text
NO_VERIFIED_EVIDENCE
LLM_API_KEY_MISSING
LLM_GENERATION_FAILED
QUALITY_GATE_FAILED
SEARCH_UNAVAILABLE
```

Fallback output should not pretend to have an estimate:

```json
{
  "status": "fallback",
  "agent_estimate": null,
  "edge_pp": null,
  "recommendation": "no_trade"
}
```

Bad fallback:

```text
p(YES)=0.500, edge=0
```

That looks like an actual agent estimate and is misleading.

---

## 14. Runs and Replay

This is Phase 5.

Each research pass should be replayable.

Run record should include:

```text
run_id
condition_id
status
started_at
finished_at
planned_queries
search_requests_sent
raw_hits
unique_sources
credible_sources
snippet_supported_evidence
quote_verified_evidence
claims_generated
claims_published
fallback_reason
latency per stage
token usage
estimated cost
final memo
logs
```

This lets the user answer:

```text
What did the agent search?
Which sources did it find?
Which sources were rejected?
Which quotes supported the claims?
Why did it fallback?
How much did it cost?
How long did each stage take?
```

That makes the agent debuggable instead of magical.

---

## 15. Current Mental Model

The core chain is:

```text
Market Question
  -> Natural Search Queries
  -> Tavily Raw Hits
  -> Normalized Sources
  -> Credible Sources
  -> Source-Supported Evidence
  -> Verified Claims
  -> Memo
  -> Quality Gate
  -> Run Replay
```

The most important product rule:

```text
No source, no claim.
```

The most important engineering rule:

```text
Do not let the LLM be the source of truth for math, citations, or quality gates.
```

