"""
src/cli.py
──────────
Main command-line interface for the NBA Props Research Tool.

Run with:
    python src/cli.py <command> [options]

Commands
--------
    db-init           Initialise database schema
    ingest            Run ingestion pipelines
    features          Rebuild player features
    train             Train / load projection models
    daily             Load props, compute projections + EV, suggest bets
    analyze           Compute EV on today's props and rank them
    backtest          Backtest projections against historical lines
    show-props        Pretty-print loaded props
    show-projections  Show model projections for today's players

Examples
--------
    python src/cli.py db-init
    python src/cli.py ingest --module games
    python src/cli.py ingest --module boxscores
    python src/cli.py ingest --module injuries
    python src/cli.py features
    python src/cli.py train --force-retrain
    python src/cli.py daily --props-file data/today_props_raw.json
    python src/cli.py analyze --date 2025-01-15
    python src/cli.py analyze --props-file data/today_props_raw.json
    python src/cli.py backtest
    python src/cli.py show-props
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import date, datetime
from pathlib import Path

import click
import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich import box

# ---------------------------------------------------------------------------
# Path setup — ensure src/ is importable when run as a script
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

load_dotenv()

_PROJECT_ROOT = _SRC_DIR.parent
_DEFAULT_PROPS_FILE = _PROJECT_ROOT / "data" / "today_props_raw.json"
_CONFIG_PATH = _PROJECT_ROOT / "config.yaml"

console = Console()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config() -> dict:
    """Load config.yaml and return as a dict."""
    if _CONFIG_PATH.exists():
        with _CONFIG_PATH.open() as f:
            return yaml.safe_load(f) or {}
    return {}


def setup_logging(level: str = "INFO") -> None:
    log_file = _PROJECT_ROOT / os.getenv("LOG_FILE", "logs/nba_props.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(log_file)),
        ],
    )


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
@click.option("--verbose", is_flag=True, default=False, help="Enable DEBUG logging.")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """NBA Player Prop Research Tool — personal EV analysis CLI."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    level = "DEBUG" if verbose else os.getenv("LOG_LEVEL", "INFO")
    setup_logging(level)
    ctx.obj["config"] = load_config()


# ---------------------------------------------------------------------------
# db-init
# ---------------------------------------------------------------------------

@cli.command("db-init")
@click.pass_context
def cmd_db_init(ctx: click.Context) -> None:
    """Initialise the DuckDB schema (safe to re-run — uses CREATE IF NOT EXISTS)."""
    from db import get_connection, init_schema
    console.print("[bold cyan]Initialising database schema...[/bold cyan]")
    con = get_connection()
    init_schema(con)
    con.close()
    console.print("[bold green]✓ Schema initialised.[/bold green]")


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------

@cli.command("ingest")
@click.option(
    "--module", "-m",
    type=click.Choice(["games", "boxscores", "injuries", "all"], case_sensitive=False),
    default="all",
    help="Which ingestion module to run (default: all).",
)
@click.pass_context
def cmd_ingest(ctx: click.Context, module: str) -> None:
    """Fetch and store NBA data (games, box scores, injuries)."""
    from db import get_connection, init_schema

    con = get_connection()
    init_schema(con)

    if module in ("games", "all"):
        console.print("[bold cyan]Ingesting games...[/bold cyan]")
        try:
            from ingest_games import ingest_teams, ingest_all_games
            ingest_teams(con=con)
            ingest_all_games(con=con)
            console.print("[green]✓ Games ingested.[/green]")
        except NotImplementedError as e:
            console.print(f"[yellow]⚠ Games ingestion not yet implemented: {e}[/yellow]")

    if module in ("boxscores", "all"):
        console.print("[bold cyan]Ingesting box scores...[/bold cyan]")
        try:
            from ingest_boxscores import ingest_all_boxscores
            ingest_all_boxscores(con=con)
            console.print("[green]✓ Box scores ingested.[/green]")
        except NotImplementedError as e:
            console.print(f"[yellow]⚠ Box score ingestion not yet implemented: {e}[/yellow]")

    if module in ("injuries", "all"):
        console.print("[bold cyan]Ingesting injuries...[/bold cyan]")
        try:
            from ingest_injuries import ingest_injuries
            ingest_injuries(con=con)
            console.print("[green]✓ Injuries ingested.[/green]")
        except NotImplementedError as e:
            console.print(f"[yellow]⚠ Injury ingestion not yet implemented: {e}[/yellow]")

    con.close()


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------

@cli.command("features")
@click.pass_context
def cmd_features(ctx: click.Context) -> None:
    """Rebuild the player_features table from box score history."""
    console.print("[bold cyan]Building player features...[/bold cyan]")
    try:
        from features import build_features
        build_features()
        console.print("[green]✓ Features built.[/green]")
    except NotImplementedError as e:
        console.print(f"[yellow]⚠ Feature build not yet implemented: {e}[/yellow]")


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

@cli.command("train")
@click.option("--force-retrain", is_flag=True, default=False,
              help="Retrain even if model .pkl files exist.")
@click.pass_context
def cmd_train(ctx: click.Context, force_retrain: bool) -> None:
    """Train (or load) projection models for P, R, A, 3PM."""
    cfg = ctx.obj["config"]
    backend = cfg.get("models", {}).get("backend", "lightgbm")
    console.print(f"[bold cyan]Training models (backend={backend})...[/bold cyan]")
    try:
        from models import train_all_models
        models = train_all_models(backend=backend, force=force_retrain)
        console.print(f"[green]✓ {len(models)} models ready.[/green]")
    except NotImplementedError as e:
        console.print(f"[yellow]⚠ Model training not yet implemented: {e}[/yellow]")


# ---------------------------------------------------------------------------
# show-props
# ---------------------------------------------------------------------------

@cli.command("show-props")
@click.option("--props-file", default=str(_DEFAULT_PROPS_FILE),
              help="Path to today's props JSON/CSV.")
@click.pass_context
def cmd_show_props(ctx: click.Context, props_file: str) -> None:
    """Pretty-print today's loaded and normalised props."""
    from props_io import load_and_prepare_props

    try:
        df = load_and_prepare_props(props_file)
    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print(
            "Save your structured props to [bold]data/today_props_raw.json[/bold] first."
        )
        return

    table = Table(title=f"Today's Props  ({len(df)} rows)", box=box.ROUNDED)
    for col in ["player_name", "market", "line", "over_odds", "under_odds", "book", "game_date"]:
        if col in df.columns:
            table.add_column(col, style="cyan" if col == "player_name" else "white")

    for _, row in df.iterrows():
        table.add_row(*[str(row.get(c, "")) for c in
                        ["player_name", "market", "line", "over_odds",
                         "under_odds", "book", "game_date"]
                        if c in df.columns])

    console.print(table)


# ---------------------------------------------------------------------------
# show-projections
# ---------------------------------------------------------------------------

@cli.command("show-projections")
@click.option("--props-file", default=str(_DEFAULT_PROPS_FILE),
              help="Path to today's props file (used to get the player list).")
@click.pass_context
def cmd_show_projections(ctx: click.Context, props_file: str) -> None:
    """Show model projections for today's players."""
    from props_io import load_and_prepare_props
    from models import load_models, predict, FEATURE_COLS
    from features import build_features_for_today
    from db import get_connection

    try:
        props_df = load_and_prepare_props(props_file)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        return

    player_ids = props_df["player_id"].dropna().unique().tolist()
    if not player_ids:
        console.print("[yellow]No player IDs resolved from props file.[/yellow]")
        return

    con = get_connection(read_only=True)
    try:
        feature_df = build_features_for_today(player_ids, date.today(), con=con)
        models = load_models()
        proj_df = predict(feature_df, models)
    except (NotImplementedError, FileNotFoundError) as e:
        console.print(f"[yellow]⚠ {e}[/yellow]")
        return
    finally:
        con.close()

    proj_cols = [c for c in proj_df.columns if c.startswith("proj_")]
    display_cols = ["player_id"] + proj_cols

    table = Table(title="Model Projections", box=box.ROUNDED)
    for col in display_cols:
        table.add_column(col, style="green" if "proj" in col else "cyan")

    for _, row in proj_df[display_cols].iterrows():
        table.add_row(*[f"{row[c]:.1f}" if isinstance(row[c], float) else str(row[c])
                        for c in display_cols])

    console.print(table)


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------

@cli.command("analyze")
@click.option("--props-file", default=str(_DEFAULT_PROPS_FILE),
              help="Path to today's props JSON/CSV.")
@click.option("--ev-threshold", type=float, default=None,
              help="Minimum EV%% to display (overrides .env).")
@click.option("--date", "game_date", default=None,
              help="Game date YYYY-MM-DD (default: today).")
@click.option("--force-retrain", is_flag=True, default=False)
@click.pass_context
def cmd_analyze(
    ctx: click.Context,
    props_file: str,
    ev_threshold: float | None,
    game_date: str | None,
    force_retrain: bool,
) -> None:
    """Run full EV analysis on today's props and print ranked output."""
    cfg = ctx.obj["config"]

    # Resolve EV threshold
    if ev_threshold is None:
        ev_threshold = float(os.getenv("EV_THRESHOLD", "0.03"))

    bankroll = float(os.getenv("BANKROLL", "1000"))
    max_kelly = float(os.getenv("MAX_KELLY_FRACTION", "0.05"))

    game_date_obj = (
        datetime.strptime(game_date, "%Y-%m-%d").date()
        if game_date else date.today()
    )

    console.rule(f"[bold]NBA Props EV Analysis — {game_date_obj}[/bold]")

    # 1. Load props
    from props_io import load_and_prepare_props
    try:
        props_df = load_and_prepare_props(props_file)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        return

    console.print(f"  Loaded [bold]{len(props_df)}[/bold] props from {props_file}")

    # 2. Get projections
    from db import get_connection
    from models import load_models, predict, train_all_models
    from features import build_features_for_today

    con = get_connection()
    player_ids = props_df["player_id"].dropna().unique().tolist()

    try:
        try:
            features_df = build_features_for_today(player_ids, game_date_obj, con=con)
        except NotImplementedError:
            console.print("[yellow]⚠ Feature builder not yet implemented; using placeholder projections.[/yellow]")
            features_df = None

        try:
            if force_retrain:
                models = train_all_models(force=True, con=con)
            else:
                models = load_models()
        except (FileNotFoundError, NotImplementedError) as e:
            console.print(f"[yellow]⚠ Models not available ({e}). Run `train` first.[/yellow]")
            return

        if features_df is not None:
            proj_df = predict(features_df, models)
        else:
            proj_df = None

    except Exception as e:
        console.print(f"[red]Error during projection: {e}[/red]")
        return
    finally:
        con.close()

    # 3. Compute EV
    from ev_calc import enrich_props_with_ev, rank_single_props, rank_combos, kelly_bet_size

    top_n_singles = cfg.get("ev", {}).get("top_singles", 5)
    top_n_combos  = cfg.get("ev", {}).get("top_combos", 3)

    try:
        ev_df = enrich_props_with_ev(
            props_df,
            proj_df,
            vig_removal=cfg.get("ev", {}).get("vig_removal", "multiplicative"),
        )
    except NotImplementedError:
        console.print("[yellow]⚠ EV calculation not yet implemented.[/yellow]")
        return

    # 4. Rank singles
    top_singles = rank_single_props(ev_df, ev_threshold=ev_threshold, top_n=top_n_singles)

    console.rule("[bold green]Top Single Props by EV[/bold green]")
    if top_singles.empty:
        console.print(f"[yellow]No props exceed EV threshold of {ev_threshold:.1%}[/yellow]")
    else:
        _print_singles_table(top_singles, bankroll, max_kelly)

    # 5. Rank combos
    top_combos = rank_combos(ev_df, n_legs=2, ev_threshold=ev_threshold, top_n=top_n_combos)

    console.rule("[bold blue]Best 2-Leg Combos[/bold blue]")
    if top_combos.empty:
        console.print("[yellow]No qualifying 2-leg combos found.[/yellow]")
    else:
        _print_combos_table(top_combos)

    # 6. Suggest top 2 singles
    if len(top_singles) >= 1:
        console.rule("[bold magenta]Suggested Bets[/bold magenta]")
        for _, row in top_singles.head(2).iterrows():
            bet_size = kelly_bet_size(row.get("kelly_best", 0), bankroll, max_kelly)
            console.print(
                f"  [bold]{row['player_name']}[/bold]  "
                f"{row['market'].upper()}  "
                f"{row['best_side'].upper()} {row['line']}  "
                f"EV: [green]{row['best_ev']:.1%}[/green]  "
                f"Suggested: [bold]${bet_size}[/bold]"
            )


def _print_singles_table(df, bankroll: float, max_kelly: float) -> None:
    from ev_calc import kelly_bet_size
    table = Table(box=box.ROUNDED)
    table.add_column("#",           style="dim",    width=4)
    table.add_column("Player",      style="bold cyan")
    table.add_column("Market",      style="white")
    table.add_column("Side",        style="white")
    table.add_column("Line",        style="white")
    table.add_column("Odds",        style="white")
    table.add_column("Projection",  style="yellow")
    table.add_column("EV%",         style="green")
    table.add_column("Bet ($)",     style="magenta")

    for _, row in df.iterrows():
        side      = row.get("best_side", "?")
        odds_col  = "over_odds" if side == "over" else "under_odds"
        bet_size  = kelly_bet_size(row.get("kelly_best", 0), bankroll, max_kelly)
        table.add_row(
            str(row.get("rank", "")),
            str(row.get("player_name", "")),
            str(row.get("market", "")),
            side.upper(),
            str(row.get("line", "")),
            str(int(row.get(odds_col, 0))),
            f"{row.get('projection', 0):.1f}",
            f"{row.get('best_ev', 0):.1%}",
            f"${bet_size:.0f}",
        )
    console.print(table)


def _print_combos_table(df) -> None:
    table = Table(box=box.ROUNDED)
    table.add_column("#",           style="dim",  width=4)
    table.add_column("Legs",        style="cyan")
    table.add_column("Combo Odds",  style="white")
    table.add_column("Combo EV%",   style="green")

    for _, row in df.iterrows():
        legs_str = " / ".join(row.get("legs", []))
        table.add_row(
            str(row.get("rank", "")),
            legs_str,
            f"{row.get('combo_decimal', 0):.2f}x",
            f"{row.get('combo_ev', 0):.1%}",
        )
    console.print(table)


# ---------------------------------------------------------------------------
# daily
# ---------------------------------------------------------------------------

@cli.command("daily")
@click.option("--props-file", required=True, type=click.Path(exists=True),
              help="Path to today's props JSON or CSV file.")
@click.option("--top-n-singles", type=int, default=10,
              help="Number of top single props to display (default: 10).")
@click.option("--top-n-combos", type=int, default=2,
              help="Number of top 2-leg combos to display (default: 2).")
@click.option("--date", "game_date", default=None,
              help="Game date YYYY-MM-DD (default: today).")
@click.pass_context
def cmd_daily(
    ctx: click.Context,
    props_file: str,
    top_n_singles: int,
    top_n_combos: int,
    game_date: str | None,
) -> None:
    """Daily workflow: load props, compute projections, rank by EV, suggest combos."""
    from datetime import datetime
    from props_io import load_and_prepare_props, preview_props
    from projections import build_projections_for_date, resolve_context_maps
    from ev_calc import (
        enrich_props_with_ev, rank_single_props, rank_combos, kelly_bet_size,
    )
    from db import get_connection

    cfg = ctx.obj["config"]
    ev_threshold = float(os.getenv("EV_THRESHOLD", "0.03"))
    bankroll     = float(os.getenv("BANKROLL", "1000"))
    max_kelly    = float(os.getenv("MAX_KELLY_FRACTION", "0.05"))

    game_date_obj = (
        datetime.strptime(game_date, "%Y-%m-%d").date()
        if game_date else date.today()
    )

    console.rule(f"[bold]Daily Props Analysis — {game_date_obj}[/bold]")

    # ── 1. Load and preview props ─────────────────────────────────────────
    console.print(f"[cyan]Loading props from {props_file}...[/cyan]")
    con = get_connection()
    try:
        props_df = load_and_prepare_props(props_file, con=con)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        con.close()
        return

    console.print(f"  Loaded [bold]{len(props_df)}[/bold] prop lines.")
    preview_props(props_df)

    if not click.confirm("Continue with EV calculation?", default=True):
        console.print("[yellow]Aborted.[/yellow]")
        con.close()
        return

    # ── 2. Build projections ──────────────────────────────────────────────
    console.print("[cyan]Building projections...[/cyan]")
    player_ids = props_df["player_id"].dropna().unique().tolist()
    if not player_ids:
        console.print("[red]No player IDs resolved — cannot compute projections.[/red]")
        console.print("Check that the players table is populated and names match.")
        con.close()
        return

    opponent_map, is_home_map = resolve_context_maps(props_df, con)

    try:
        proj_df = build_projections_for_date(
            player_ids=[int(p) for p in player_ids],
            game_date=game_date_obj,
            opponent_map=opponent_map,
            is_home_map=is_home_map,
            con=con,
        )
    except (FileNotFoundError, NotImplementedError) as e:
        console.print(f"[red]Projection error: {e}[/red]")
        con.close()
        return

    if proj_df.empty:
        console.print("[yellow]No projections could be generated. Check data.[/yellow]")
        con.close()
        return

    console.print(f"  Projections ready for [bold]{len(proj_df)}[/bold] players.")

    # ── 3. Compute EV ─────────────────────────────────────────────────────
    console.print("[cyan]Computing EV...[/cyan]")
    vig_method = cfg.get("ev", {}).get("vig_removal", "multiplicative")
    ev_df = enrich_props_with_ev(props_df, proj_df, vig_removal=vig_method)

    # ── 4. Rank singles ───────────────────────────────────────────────────
    top_singles = rank_single_props(
        ev_df, ev_threshold=ev_threshold, top_n=top_n_singles,
    )

    console.rule("[bold green]Top Single Props by EV[/bold green]")
    if top_singles.empty:
        console.print(
            f"[yellow]No props exceed EV threshold of {ev_threshold:.1%}[/yellow]"
        )
    else:
        _print_singles_table(top_singles, bankroll, max_kelly)

        # Highlight top 2 as suggested bets
        if len(top_singles) >= 1:
            console.print()
            console.rule("[bold magenta]Suggested Bets[/bold magenta]")
            for _, row in top_singles.head(2).iterrows():
                bet_size = kelly_bet_size(
                    row.get("kelly_best", 0), bankroll, max_kelly,
                )
                side = row.get("best_side", "?")
                odds_col = "over_odds" if side == "over" else "under_odds"
                console.print(
                    f"  [bold]{row['player_name']}[/bold]  "
                    f"{row['market'].upper()}  "
                    f"{side.upper()} {row['line']}  "
                    f"({int(row.get(odds_col, 0)):+d})  "
                    f"Proj: [yellow]{row.get('projection', 0):.1f}[/yellow]  "
                    f"EV: [green]{row['best_ev']:.1%}[/green]  "
                    f"Suggested: [bold]${bet_size:.0f}[/bold]"
                )

    # ── 5. Rank combos ────────────────────────────────────────────────────
    console.print()
    console.rule("[bold blue]Best 2-Leg Combos[/bold blue]")
    top_combos = rank_combos(
        ev_df, n_legs=2, ev_threshold=ev_threshold, top_n=top_n_combos,
    )
    if top_combos.empty:
        console.print("[yellow]No qualifying 2-leg combos found.[/yellow]")
    else:
        _print_combos_table(top_combos)

    # ── 6. Save report ────────────────────────────────────────────────────
    reports_dir = _PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_name = f"daily_{game_date_obj.strftime('%Y%m%d')}_props.csv"
    report_path = reports_dir / report_name
    ev_df.to_csv(report_path, index=False)
    console.print(f"\n[dim]Report saved to {report_path}[/dim]")

    con.close()


# ---------------------------------------------------------------------------
# backtest
# ---------------------------------------------------------------------------

@cli.command("backtest")
@click.option("--ev-threshold", type=float, default=0.0,
              help="Only simulate bets above this EV (default: all bets).")
@click.pass_context
def cmd_backtest(ctx: click.Context, ev_threshold: float) -> None:
    """Backtest model projections against historical prop lines."""
    try:
        from backtest import run_backtest, print_backtest_report
        bt_df = run_backtest(ev_threshold=ev_threshold)
        print_backtest_report(bt_df)
    except NotImplementedError as e:
        console.print(f"[yellow]⚠ Backtest not yet fully implemented: {e}[/yellow]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli(obj={})
