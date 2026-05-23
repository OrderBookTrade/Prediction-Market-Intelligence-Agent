"""CLI entry point. Usage: uv run python main.py ingest --category ai --limit 20"""

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
