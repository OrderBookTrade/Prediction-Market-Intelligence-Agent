# Prediction Market Intelligence Agent

An AI agent that ingests real-time Polymarket data, retrieves external evidence,
analyzes resolution rules and liquidity risk, and generates structured research memos.

**This is decision support, not automated trading.**

![Agent Dashboard UI](docs/Agent-Dashboard.png)

## Architecture

```
Polymarket Gamma API
        ↓
  Market Ingestion
        ↓
  ┌─────────────────────────────────┐
  │         LangGraph Agent          │
  │  ┌──────────┐  ┌──────────────┐ │
  │  │  Market  │  │   Evidence   │ │
  │  │ Analyzer │  │  Retriever   │ │
  │  └──────────┘  └──────────────┘ │
  │  ┌──────────┐  ┌──────────────┐ │
  │  │   Risk   │  │    Memo      │ │
  │  │  Critic  │  │   Writer     │ │
  │  └──────────┘  └──────────────┘ │
  └─────────────────────────────────┘
        ↓
  Structured Research Memo (JSON)
        ↓
  Eval Dashboard (Brier score, citation accuracy)
```

## Memo Output Format

```json
{
  "market_question": "Will X happen by Y?",
  "current_probability": 0.37,
  "agent_estimate": 0.45,
  "edge": 0.08,
  "confidence": "medium",
  "yes_case": ["evidence 1", "evidence 2"],
  "no_case": ["counter 1", "counter 2"],
  "resolution_risk": "medium",
  "liquidity_risk": "low",
  "sources": [],
  "recommendation": "watch"
}
```

## Quick Start

### API Server (Local Development)

The agent runs as a FastAPI backend. To start the local server with hot-reloading:

```bash
uv sync
export ANTHROPIC_API_KEY="..."
uv run uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```
*API docs will be available at `http://localhost:8000/docs`*

### CLI Usage

```bash
# Single market memo
uv run python main.py --market <condition_id>

# Interactive mode
uv run python main.py --interactive

# Scan top AI/Crypto markets
uv run python main.py --scan
```

## Deployment

This project is configured for deployment on [Railway](https://railway.app/). 

**To deploy:**
1. Connect your GitHub repository to Railway.
2. Railway will automatically detect `railway.json` / `Dockerfile` and build the service.
3. Add your `ANTHROPIC_API_KEY` to the project variables in the Railway dashboard.

Alternatively, using the Railway CLI:
```bash
railway up
```

## Eval

```bash
python eval/run_eval.py
# → citation accuracy, rule extraction, hallucination rate
```

## Safety Boundary

- No automated trade execution
- All recommendations require human confirmation
- Sources cited for every claim
- Confidence levels explicit