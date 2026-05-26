"""FastAPI application — mounts all routers, configures CORS, initialises DB."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from fastapi.responses import RedirectResponse
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes.agent import router as agent_router
from src.api.routes.eval import router as eval_router
from src.api.routes.markets import router as markets_router
from src.api.routes.memos import router as memos_router
import time

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from src.storage.db import get_engine, init_db

    init_db(get_engine())
    logger.info("PMI Agent API ready")
    yield
    logger.info("PMI Agent API shutting down")


app = FastAPI(
    title="Prediction Market Intelligence Agent API",
    version="0.1.0",
    description="Real-time Polymarket data + LangGraph agent analysis",
    lifespan=lifespan,
)

# CORS — allow all origins in dev; restrict in prod via CORS_ORIGINS env var
from src.config import settings  # noqa: E402

origins = [o.strip() for o in settings.cors_origins.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(markets_router, prefix="/api")
app.include_router(memos_router, prefix="/api")
app.include_router(agent_router, prefix="/api")
app.include_router(eval_router, prefix="/api")


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "version": app.version,
        "service": "PMI Agent Backend API Service)",
        "serverTime": int(time.time()),
        "title": app.title,
        "description": app.description,
        "github": "https://github.com/orderbooktrade/prediction-market-intelligence-agent",
        "telegram": "https://t.me/iambaice",
    }


@app.get("/")
async def root():
    return RedirectResponse(url="/health")
