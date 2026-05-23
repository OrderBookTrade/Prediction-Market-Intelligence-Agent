"""CLI entry point.

Usage:
    uv run python main.py ingest --category ai --limit 20
    uv run python main.py retrieve --market <condition_id>
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from sqlalchemy.orm import Session

app = typer.Typer(add_completion=False)
console = Console()
logger = logging.getLogger(__name__)


@app.callback()
def _root() -> None:
    """Prediction Market Intelligence Agent."""


@app.command()
def ingest(
    category: Optional[str] = typer.Option(None, "--category", "-c", help="Gamma tag_slug filter"),
    limit: int = typer.Option(20, "--limit", "-l", help="Max markets to fetch"),
) -> None:
    """Pull markets from Polymarket, normalise, and persist to SQLite."""
    asyncio.run(_ingest(category, limit))


async def _ingest(category: str | None, limit: int) -> None:
    from src.ingestion.normalizer import normalize
    from src.ingestion.polymarket import PolymarketClient
    from src.storage.db import get_engine, init_db, upsert_snapshot

    engine = init_db(get_engine())

    async with PolymarketClient() as client:
        console.print(f"[bold cyan]Fetching {limit} markets (category={category!r})…[/bold cyan]")
        raws = await client.fetch_active_markets(category=category, limit=limit)

    snapshots = []
    errors = 0
    for raw in raws:
        try:
            snapshots.append(normalize(raw))
        except Exception as exc:
            logger.warning("Normalization error: %s", exc)
            errors += 1

    with Session(engine) as session:
        for snap in snapshots:
            upsert_snapshot(session, snap)
        session.commit()

    status = f"[green]Stored {len(snapshots)} markets"
    if errors:
        status += f" ({errors} skipped)"
    console.print(status + "[/green]\n")

    table = Table(title="Ingested Markets", show_lines=True)
    table.add_column("#", justify="right", style="dim", width=3)
    table.add_column("Question", max_width=58)
    table.add_column("YES", justify="right")
    table.add_column("Volume", justify="right")
    table.add_column("Liquidity", justify="right")
    table.add_column("End Date")

    for i, snap in enumerate(snapshots, 1):
        table.add_row(
            str(i),
            snap.question[:58],
            f"{snap.yes_price:.3f}" if snap.yes_price is not None else "—",
            f"${snap.volume:>10,.0f}" if snap.volume is not None else "—",
            f"${snap.liquidity:>10,.0f}" if snap.liquidity is not None else "—",
            snap.end_date.strftime("%Y-%m-%d") if snap.end_date else "—",
        )

    console.print(table)


# ── retrieve command ──────────────────────────────────────────────────────────

@app.command()
def retrieve(
    market: str = typer.Option(..., "--market", "-m", help="Condition ID to retrieve evidence for"),
) -> None:
    """Retrieve and display cited evidence for a market (runs evidence_retriever node)."""
    asyncio.run(_retrieve(market))


async def _retrieve(condition_id: str) -> None:
    from src.storage.db import get_engine, init_db
    from src.storage.models import MarketSnapshotORM

    engine = init_db(get_engine())

    # ── Try DB first ──────────────────────────────────────────────────────────
    snapshot_dict: dict | None = None
    with Session(engine) as session:
        orm = session.get(MarketSnapshotORM, condition_id)
        if orm:
            snapshot_dict = {
                "condition_id":     orm.condition_id,
                "question":         orm.question,
                "description":      orm.description,
                "yes_price":        orm.yes_price,
                "volume":           orm.volume,
                "liquidity":        orm.liquidity,
                "resolution_source": orm.resolution_source,
                "end_date":         orm.end_date.isoformat() if orm.end_date else None,
            }

    # ── Fall back to live fetch ───────────────────────────────────────────────
    if snapshot_dict is None:
        console.print(f"[yellow]Not in local DB — fetching live from Polymarket…[/yellow]")
        from src.ingestion.normalizer import normalize
        from src.ingestion.polymarket import PolymarketClient

        async with PolymarketClient() as client:
            raw = await client.fetch_market_detail(condition_id)

        if raw is None:
            console.print(f"[red]Market {condition_id!r} not found.[/red]")
            raise typer.Exit(1)

        snap = normalize(raw)
        with Session(engine) as session:
            from src.storage.db import upsert_snapshot
            upsert_snapshot(session, snap)
            session.commit()

        snapshot_dict = {
            "condition_id":     snap.condition_id,
            "question":         snap.question,
            "description":      snap.description,
            "yes_price":        snap.yes_price,
            "volume":           snap.volume,
            "liquidity":        snap.liquidity,
            "resolution_source": snap.resolution_source,
            "end_date":         snap.end_date.isoformat() if snap.end_date else None,
        }

    console.print(
        f"\n[bold cyan]Retrieving evidence for:[/bold cyan] "
        f"{snapshot_dict.get('question', condition_id)[:80]}"
    )
    console.print(f"[dim]condition_id = {condition_id}[/dim]\n")

    # ── Run evidence retriever node ───────────────────────────────────────────
    # Register a dummy queue so push_log calls don't crash
    from src.run_store import register_run
    register_run("cli")

    from src.agents.evidence_retriever import evidence_retriever_node

    state = {
        "condition_id": condition_id,
        "run_id":       "cli",
        "snapshot":     snapshot_dict,
    }

    result = await evidence_retriever_node(state)

    # ── Display results ───────────────────────────────────────────────────────
    cited = result.get("cited_evidence", [])
    queries = result.get("search_queries", [])

    console.print(f"[bold]Queries:[/bold]")
    for q in queries:
        label_color = "green" if q.get("label") == "yes_case" else "red" if q.get("label") == "no_case" else "yellow"
        console.print(f"  [{label_color}]{q.get('label',''):<12}[/{label_color}] \"{q.get('query','')[:70]}\"  → {q.get('results',0)} hits")

    console.print(f"\n[bold]Evidence ({len(cited)} items):[/bold]")

    if not cited:
        console.print("  [dim]No evidence found — check TAVILY_API_KEY[/dim]")
        return

    table = Table(show_lines=True)
    table.add_column("Label",     width=11, style="bold")
    table.add_column("Claim",     max_width=50)
    table.add_column("Quote",     max_width=40)
    table.add_column("Publisher", width=20)
    table.add_column("Cred",      width=6)
    table.add_column("Conf",      width=6)

    for ev in cited:
        if isinstance(ev, dict):
            label       = ev.get("label", "")
            claim       = ev.get("claim", "")
            quote       = ev.get("quote", "")
            publisher   = ev.get("publisher", "")
            credibility = ev.get("credibility", "")
            confidence  = ev.get("confidence", "")
        else:
            label       = ev.label
            claim       = ev.claim
            quote       = ev.quote
            publisher   = ev.publisher
            credibility = ev.credibility
            confidence  = ev.confidence

        tone = "green" if label == "yes_case" else "red" if label == "no_case" else "yellow"
        table.add_row(
            f"[{tone}]{label}[/{tone}]",
            claim[:50],
            quote[:40],
            publisher[:20],
            credibility,
            confidence,
        )

    console.print(table)
