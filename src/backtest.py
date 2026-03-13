"""
src/backtest.py
───────────────
Backtest the projection models against historical prop lines.

Two modes
---------
1. SYNTHETIC backtest (default, no historical lines needed)
   ─────────────────────────────────────────────────────────
   Uses model projections on the validation season's box scores.
   For each player-game, it simulates a prop line by adding ±k * std
   to the projection, where k is drawn from {-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5}.
   This tests whether the EV pipeline works end-to-end and gives a rough
   calibration check without requiring real line data.

   The synthetic lines are stored in historical_props so that the real
   backtest (mode 2) can also read them if you don't have real lines yet.

2. REAL backtest (activated with --mode real)
   ──────────────────────────────────────────
   Reads rows from historical_props that have both a line and actual_value.
   Computes EV for each bet using model projections, grades the bet
   (win/loss/push), and reports P&L and ROI.

   To populate historical_props with real lines: save a CSV with the
   schema to data/historical_props.csv and run:
       python src/backtest.py --import-csv data/historical_props.csv

Summary output
--------------
   • Per-market RMSE / MAE of projections vs actuals.
   • Win rate, P&L, and ROI for simulated bets at various EV thresholds.
   • Results CSV saved to data/backtest_results.csv.

Run directly:
    python src/backtest.py --mode synthetic
    python src/backtest.py --mode real --ev-threshold 0.03
    python src/backtest.py --import-csv data/historical_props.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from db import get_connection, init_schema, upsert_dataframe, query_df
from ev_calc import (
    american_to_decimal,
    compute_ev,
    estimate_win_prob_normal,
    kelly_fraction,
    remove_vig_multiplicative,
)

load_dotenv()
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_RESULTS_PATH = _PROJECT_ROOT / "data" / "backtest_results.csv"

# Synthetic line parameters
_SYNTHETIC_ODDS_OVER  = -110
_SYNTHETIC_ODDS_UNDER = -110
_LINE_OFFSET_STDS     = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]

# Mapping: target name → player_game_stats column for actual value
_TARGET_TO_STAT: dict[str, str] = {
    "points":   "points",
    "rebounds": "rebounds",
    "assists":  "assists",
    "threepm":  "fg3m",
}


# ---------------------------------------------------------------------------
# Import historical props from CSV
# ---------------------------------------------------------------------------

def import_props_csv(path: Path, con) -> None:
    """Import historical prop lines from a user-supplied CSV.

    Expected columns (extra columns are ignored):
      player_name, game_date, book, market, side, line, odds_american,
      actual_value (optional)

    player_id is resolved from the players table by name.

    Parameters
    ----------
    path : Path
    con : duckdb connection
    """
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    # Resolve player IDs
    from ingest_injuries import _load_player_name_map, _resolve_ids
    name_map = _load_player_name_map(con)
    df = _resolve_ids(df, name_map)

    df["game_date"] = pd.to_datetime(df.get("game_date", pd.NaT), errors="coerce").dt.date
    df["odds_american"] = pd.to_numeric(df.get("odds_american", -110), errors="coerce").astype(int)
    df["line"]          = pd.to_numeric(df.get("line", 0), errors="coerce")
    df["odds_decimal"]  = df["odds_american"].apply(american_to_decimal)

    if "actual_value" in df.columns:
        df["actual_value"] = pd.to_numeric(df["actual_value"], errors="coerce")

    valid = df.dropna(subset=["player_id", "game_date", "market", "side", "line"])
    logger.info("Importing %d historical prop rows from %s", len(valid), path)
    upsert_dataframe(valid, "historical_props",
                     ["player_id", "game_date", "market", "side", "book"],
                     con=con)


# ---------------------------------------------------------------------------
# Synthetic backtest
# ---------------------------------------------------------------------------

def _load_projections_for_backtest(validation_season: str, con) -> pd.DataFrame:
    """Load projections for the validation season from player_features + actuals."""
    sql = f"""
        SELECT
            pf.player_id,
            pf.game_id,
            pf.game_date,
            pf.pts_std_L10,
            pf.reb_std_L10,
            pf.ast_std_L10,
            pf.fg3m_std_L10,
            pgs.points,
            pgs.rebounds,
            pgs.assists,
            pgs.fg3m
        FROM player_features pf
        JOIN player_game_stats pgs
          ON pf.player_id = pgs.player_id
         AND pf.game_id   = pgs.game_id
        JOIN games g ON pf.game_id = g.game_id
        WHERE g.season = '{validation_season}'
          AND pgs.did_not_play = FALSE
          AND pf.pts_avg_L5 IS NOT NULL
    """
    return con.execute(sql).df()


def run_synthetic_backtest(models: dict, con) -> pd.DataFrame:
    """Run the synthetic backtest pipeline.

    1. Find the most recent season in player_features (used as val season).
    2. Load feature rows for that season.
    3. Generate projections.
    4. For each target × line_offset combination, simulate a prop line and
       compute EV, grade, and P&L.

    Parameters
    ----------
    models : dict[str, Pipeline]
    con : duckdb connection

    Returns
    -------
    pd.DataFrame  per-bet results
    """
    from models import predict, FEATURE_COLS

    # Find the most recent season
    seasons_df = con.execute(
        "SELECT DISTINCT season FROM games ORDER BY season DESC LIMIT 2"
    ).df()
    if seasons_df.empty:
        raise RuntimeError("No seasons in games table — run ingest_games first.")

    val_season = seasons_df["season"].iloc[0]
    logger.info("Synthetic backtest using validation season: %s", val_season)

    raw = _load_projections_for_backtest(val_season, con)
    if raw.empty:
        raise RuntimeError(
            f"No feature rows for season {val_season}. "
            "Run features.py first."
        )

    logger.info("  %d player-game rows for validation.", len(raw))

    # Build feature matrix
    feat_sql = f"""
        SELECT pf.*
        FROM player_features pf
        JOIN games g ON pf.game_id = g.game_id
        WHERE g.season = '{val_season}'
          AND pf.pts_avg_L5 IS NOT NULL
    """
    features_df = con.execute(feat_sql).df()
    proj_df = predict(features_df, models)

    # Merge projections with actuals
    proj_cols = [c for c in proj_df.columns if c.startswith("proj_")] + ["player_id", "game_id"]
    merged = raw.merge(
        proj_df[proj_cols],
        on=["player_id", "game_id"],
        how="inner",
    )

    logger.info("  Generating synthetic lines for %d player-games ...", len(merged))

    rows = []
    rng = np.random.default_rng(seed=42)

    for _, row in merged.iterrows():
        for target, actual_col in _TARGET_TO_STAT.items():
            proj_col   = f"proj_{target}"
            std_col_map = {
                "points":   "pts_std_L10",
                "rebounds": "reb_std_L10",
                "assists":  "ast_std_L10",
                "threepm":  "fg3m_std_L10",
            }
            std_col = std_col_map[target]

            projection  = row.get(proj_col)
            actual      = row.get(actual_col)
            std_val     = row.get(std_col)

            if pd.isna(projection) or pd.isna(actual):
                continue
            if pd.isna(std_val) or std_val <= 0:
                std_val = max(projection * 0.3, 1.0)  # fallback: 30% of projection

            # Simulate one prop line per row (pick a random offset)
            k      = rng.choice(_LINE_OFFSET_STDS)
            line   = round(projection + k * std_val * 0.5 + 0.5, 1)  # half-unit increments
            line   = max(0.5, line)

            for side in ("over", "under"):
                odds = _SYNTHETIC_ODDS_OVER if side == "over" else _SYNTHETIC_ODDS_UNDER
                decimal_odds = american_to_decimal(odds)

                p_win = estimate_win_prob_normal(projection, line, std_val, side)
                ev    = compute_ev(p_win, decimal_odds)
                kelly = kelly_fraction(p_win, decimal_odds)

                # Grade
                if actual == line:
                    result = "push"
                elif side == "over" and actual > line:
                    result = "win"
                elif side == "under" and actual < line:
                    result = "win"
                else:
                    result = "loss"

                pnl = (decimal_odds - 1) if result == "win" else (0 if result == "push" else -1)

                rows.append({
                    "player_id":    int(row["player_id"]),
                    "game_id":      row["game_id"],
                    "game_date":    row["game_date"],
                    "target":       target,
                    "side":         side,
                    "line":         round(line, 1),
                    "odds_american":odds,
                    "projection":   round(float(projection), 2),
                    "actual":       float(actual),
                    "std":          round(float(std_val), 2),
                    "p_win":        round(p_win, 4),
                    "ev":           round(ev, 4),
                    "kelly":        round(kelly, 4),
                    "result":       result,
                    "pnl":          round(pnl, 4),
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Real backtest (reads historical_props)
# ---------------------------------------------------------------------------

def run_real_backtest(models: dict, ev_threshold: float = 0.0, con=None) -> pd.DataFrame:
    """Run backtest using real historical prop lines from historical_props.

    Parameters
    ----------
    models : dict[str, Pipeline]
    ev_threshold : float    Only include bets where our EV >= threshold.
    con : duckdb connection

    Returns
    -------
    pd.DataFrame  per-bet results
    """
    from models import predict, FEATURE_COLS

    graded = con.execute("""
        SELECT hp.*, p.full_name
        FROM historical_props hp
        LEFT JOIN players p ON hp.player_id = p.player_id
        WHERE hp.actual_value IS NOT NULL
          AND hp.line IS NOT NULL
          AND hp.odds_american IS NOT NULL
        ORDER BY hp.game_date
    """).df()

    if graded.empty:
        logger.warning("No graded historical props found. Import them with --import-csv.")
        return pd.DataFrame()

    logger.info("Running real backtest on %d graded props.", len(graded))

    # Load features for matching player-game rows
    player_ids = graded["player_id"].dropna().unique().tolist()
    game_ids   = graded["game_id"].dropna().unique().tolist()

    if not player_ids or not game_ids:
        logger.warning("No player or game IDs in graded props.")
        return pd.DataFrame()

    pid_str = ", ".join(str(p) for p in player_ids)
    gid_str = ", ".join(f"'{g}'" for g in game_ids)

    feat_sql = f"""
        SELECT * FROM player_features
        WHERE player_id IN ({pid_str})
          AND game_id IN ({gid_str})
    """
    features_df = con.execute(feat_sql).df()
    if features_df.empty:
        logger.warning("No feature rows found for graded props. Run features.py.")
        return pd.DataFrame()

    proj_df = predict(features_df, models)

    # Merge projections into graded
    target_to_proj = {
        "points":   "proj_points",
        "rebounds": "proj_rebounds",
        "assists":  "proj_assists",
        "threepm":  "proj_threepm",
        "points_rebounds":          "proj_points_rebounds",
        "points_assists":           "proj_points_assists",
        "rebounds_assists":         "proj_rebounds_assists",
        "points_rebounds_assists":  "proj_points_rebounds_assists",
    }
    std_map = {
        "points":   "pts_std_L10",
        "rebounds": "reb_std_L10",
        "assists":  "ast_std_L10",
        "threepm":  "fg3m_std_L10",
    }

    proj_keep = (
        ["player_id", "game_id"]
        + [c for c in proj_df.columns if c.startswith("proj_")]
        + [c for c in ["pts_std_L10","reb_std_L10","ast_std_L10","fg3m_std_L10"]
           if c in proj_df.columns]
    )
    merged = graded.merge(proj_df[proj_keep], on=["player_id","game_id"], how="left")

    rows = []
    for _, row in merged.iterrows():
        market    = str(row.get("market",""))
        proj_col  = target_to_proj.get(market)
        std_col   = std_map.get(market, "pts_std_L10")

        if not proj_col or proj_col not in row or pd.isna(row.get(proj_col)):
            continue

        projection = float(row[proj_col])
        line       = float(row["line"])
        side       = str(row["side"]).lower()
        odds       = int(row["odds_american"])
        actual     = float(row["actual_value"])
        std_val    = float(row.get(std_col, max(projection * 0.3, 1.0)))
        if std_val <= 0:
            std_val = max(projection * 0.3, 1.0)

        decimal_odds = american_to_decimal(odds)
        p_win = estimate_win_prob_normal(projection, line, std_val, side)
        ev    = compute_ev(p_win, decimal_odds)
        kelly = kelly_fraction(p_win, decimal_odds)

        if ev < ev_threshold:
            continue

        # Grade
        if actual == line:
            result = "push"
        elif side == "over" and actual > line:
            result = "win"
        elif side == "under" and actual < line:
            result = "win"
        else:
            result = "loss"

        pnl = (decimal_odds - 1) if result == "win" else (0 if result == "push" else -1)

        rows.append({
            "player_name":  row.get("full_name",""),
            "game_date":    row["game_date"],
            "market":       market,
            "side":         side,
            "line":         line,
            "odds_american":odds,
            "projection":   round(projection, 2),
            "actual":       actual,
            "p_win":        round(p_win, 4),
            "ev":           round(ev, 4),
            "kelly":        round(kelly, 4),
            "result":       result,
            "pnl":          round(pnl, 4),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def summarise(bt_df: pd.DataFrame, label: str = "Backtest") -> dict:
    """Compute and log summary statistics for a backtest results DataFrame."""
    if bt_df.empty:
        logger.warning("Empty backtest DataFrame — nothing to summarise.")
        return {}

    non_push = bt_df[bt_df["result"] != "push"]
    wins     = (non_push["result"] == "win").sum()
    losses   = (non_push["result"] == "loss").sum()
    pushes   = (bt_df["result"] == "push").sum()
    win_rate = wins / max(wins + losses, 1)
    total_pnl = bt_df["pnl"].sum()
    roi       = total_pnl / max(len(non_push), 1)

    # Projection accuracy
    if "actual" in bt_df.columns and "projection" in bt_df.columns:
        valid = bt_df[bt_df["actual"].notna() & bt_df["projection"].notna()]
        if not valid.empty:
            rmse = float(np.sqrt(((valid["actual"] - valid["projection"]) ** 2).mean()))
            mae  = float((valid["actual"] - valid["projection"]).abs().mean())
        else:
            rmse = mae = float("nan")
    else:
        rmse = mae = float("nan")

    summary = {
        "label":      label,
        "total_bets": len(bt_df),
        "wins":       wins,
        "losses":     losses,
        "pushes":     pushes,
        "win_rate":   round(win_rate, 4),
        "total_pnl":  round(total_pnl, 4),
        "roi":        round(roi, 4),
        "proj_rmse":  round(rmse, 4),
        "proj_mae":   round(mae, 4),
    }

    logger.info("─" * 60)
    logger.info("%-20s %s", "Label:",       label)
    logger.info("%-20s %d", "Total bets:",  summary["total_bets"])
    logger.info("%-20s %d / %d / %d (W/L/P)", "Record:", wins, losses, pushes)
    logger.info("%-20s %.1f%%", "Win rate:",  win_rate * 100)
    logger.info("%-20s %+.2f units", "Total P&L:", total_pnl)
    logger.info("%-20s %+.2f%%", "ROI:",       roi * 100)
    if not np.isnan(rmse):
        logger.info("%-20s %.3f", "Proj RMSE:", rmse)
        logger.info("%-20s %.3f", "Proj MAE:",  mae)
    logger.info("─" * 60)

    # Per-market breakdown
    if "target" in bt_df.columns:
        grp_col = "target"
    elif "market" in bt_df.columns:
        grp_col = "market"
    else:
        grp_col = None

    if grp_col:
        logger.info("Per-market breakdown:")
        for mkt, g in bt_df.groupby(grp_col):
            g_non_push = g[g["result"] != "push"]
            mkt_wins   = (g_non_push["result"] == "win").sum()
            mkt_total  = len(g_non_push)
            mkt_wr     = mkt_wins / max(mkt_total, 1)
            mkt_pnl    = g["pnl"].sum()
            logger.info("  %-28s  W/L=%d/%d  WR=%.1f%%  P&L=%+.2f",
                        mkt, mkt_wins, mkt_total - mkt_wins, mkt_wr * 100, mkt_pnl)

    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
    )
    parser = argparse.ArgumentParser(description="NBA props backtest.")
    parser.add_argument(
        "--mode", choices=["synthetic", "real"], default="synthetic",
        help="Backtest mode (default: synthetic).",
    )
    parser.add_argument(
        "--ev-threshold", type=float, default=0.0,
        help="Only include bets at or above this EV (default: 0 = all bets).",
    )
    parser.add_argument(
        "--import-csv", metavar="PATH", default=None,
        help="Import historical prop lines from a CSV file into historical_props.",
    )
    args = parser.parse_args(argv)

    con = get_connection()
    init_schema(con)

    if args.import_csv:
        import_props_csv(Path(args.import_csv), con)
        logger.info("Import complete.")

    from models import load_models, train_all_models
    try:
        models = load_models()
    except FileNotFoundError:
        logger.warning("No saved models found — training now...")
        models = train_all_models(con=con)

    if args.mode == "synthetic":
        bt_df = run_synthetic_backtest(models, con)
        label = "Synthetic backtest"
    else:
        bt_df = run_real_backtest(models, ev_threshold=args.ev_threshold, con=con)
        label = f"Real backtest (EV≥{args.ev_threshold:.1%})"

    if not bt_df.empty:
        summarise(bt_df, label=label)

        # Filter to positive-EV bets for a focused summary
        pos_ev_df = bt_df[bt_df["ev"] > 0]
        if not pos_ev_df.empty and args.mode == "synthetic":
            summarise(pos_ev_df, label="Synthetic — positive EV bets only")

        bt_df.to_csv(_RESULTS_PATH, index=False)
        logger.info("Results saved to %s", _RESULTS_PATH)
    else:
        logger.warning("No backtest results generated.")

    con.close()


if __name__ == "__main__":
    main()
