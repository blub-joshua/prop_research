"""
src/review.py
─────────────
Review yesterday's predictions against actual box-score results.

Daily tracking workflow
───────────────────────
1. Each time you run `daily`, the tool saves a prediction log to
   ``data/predictions/YYYY-MM-DD_predictions.json``.
2. After ingesting the next day's box scores, run `review` to compare
   predictions vs actuals.
3. Results are appended to ``data/tracking/performance_log.csv`` — a
   cumulative record of every graded pick.

CLI usage:
    python src/cli.py review                          # reviews yesterday
    python src/cli.py review --date 2025-03-12        # reviews a specific date
    python src/cli.py review --all                    # reviews all un-graded days
    python src/cli.py performance                     # shows cumulative stats
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PREDICTIONS_DIR = _PROJECT_ROOT / "data" / "predictions"
_TRACKING_DIR    = _PROJECT_ROOT / "data" / "tracking"
_PERF_LOG        = _TRACKING_DIR / "performance_log.csv"

_PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
_TRACKING_DIR.mkdir(parents=True, exist_ok=True)

# Mapping from market key to the stats column(s) in player_game_stats
_MARKET_TO_STATS: dict[str, list[str]] = {
    "points":                  ["points"],
    "rebounds":                ["rebounds"],
    "assists":                 ["assists"],
    "threepm":                 ["fg3m"],
    "points_rebounds":         ["points", "rebounds"],
    "points_assists":          ["points", "assists"],
    "rebounds_assists":        ["rebounds", "assists"],
    "points_rebounds_assists": ["points", "rebounds", "assists"],
}


# ---------------------------------------------------------------------------
# Save predictions (called by `daily` command)
# ---------------------------------------------------------------------------

def save_daily_predictions(
    ev_df: pd.DataFrame,
    game_date: date,
    top_n: int = 20,
) -> Path:
    """Save today's top predictions for later review.

    Saves the top N picks (by EV) to a JSON file in data/predictions/.

    Parameters
    ----------
    ev_df : pd.DataFrame
        Output of enrich_props_with_ev().
    game_date : date
        The date these predictions are for.
    top_n : int
        Save the top N picks for tracking.

    Returns
    -------
    Path
        Path to the saved predictions file.
    """
    # Filter to picks with positive EV and valid projections
    valid = ev_df[
        (ev_df["best_ev"] > 0) &
        (ev_df["projection"].notna())
    ].copy()

    if valid.empty:
        logger.warning("No positive-EV picks to save.")
        return None

    # Sort by EV and take top N
    valid = valid.sort_values("best_ev", ascending=False).head(top_n)

    records = []
    for _, row in valid.iterrows():
        records.append({
            "player_name":  str(row.get("player_name", "")),
            "player_id":    int(row["player_id"]) if pd.notna(row.get("player_id")) else None,
            "market":       str(row.get("market", "")),
            "line":         float(row.get("line", 0)),
            "best_side":    str(row.get("best_side", "")),
            "over_odds":    int(row.get("over_odds", -110)),
            "under_odds":   int(row.get("under_odds", -110)),
            "projection":   round(float(row.get("projection", 0)), 2),
            "model_p_over": round(float(row.get("model_p_over", 0)), 4),
            "model_p_under":round(float(row.get("model_p_under", 0)), 4),
            "best_ev":      round(float(row.get("best_ev", 0)), 4),
            "kelly_best":   round(float(row.get("kelly_best", 0)), 4),
            "std":          round(float(row.get("std", 0)), 2),
            "prob_method":  str(row.get("prob_method", "normal")),
            "game_date":    game_date.isoformat(),
            "graded":       False,
        })

    pred_path = _PREDICTIONS_DIR / f"{game_date.isoformat()}_predictions.json"
    with pred_path.open("w") as f:
        json.dump(records, f, indent=2)

    logger.info("Saved %d predictions to %s", len(records), pred_path)
    return pred_path


# ---------------------------------------------------------------------------
# Grade predictions against actuals
# ---------------------------------------------------------------------------

def grade_predictions(
    prediction_date: date,
    con,
) -> pd.DataFrame | None:
    """Grade saved predictions against actual box-score results.

    Parameters
    ----------
    prediction_date : date
        The date to review predictions for.
    con : duckdb connection

    Returns
    -------
    pd.DataFrame or None
        Graded results with columns:
        player_name, market, line, side, projection, actual,
        result (win/loss/push), ev_at_pick, pnl
    """
    pred_path = _PREDICTIONS_DIR / f"{prediction_date.isoformat()}_predictions.json"
    if not pred_path.exists():
        logger.warning("No predictions file found for %s", prediction_date)
        return None

    with pred_path.open() as f:
        predictions = json.load(f)

    if not predictions:
        logger.warning("Empty predictions file for %s", prediction_date)
        return None

    # Load actual stats for the prediction date
    date_str = prediction_date.isoformat()
    actuals_sql = f"""
        SELECT
            pgs.player_id,
            p.full_name,
            pgs.points,
            pgs.rebounds,
            pgs.assists,
            pgs.fg3m,
            pgs.minutes,
            pgs.did_not_play
        FROM player_game_stats pgs
        JOIN players p ON pgs.player_id = p.player_id
        WHERE pgs.game_date = '{date_str}'
          AND pgs.did_not_play = FALSE
    """
    actuals_df = con.execute(actuals_sql).df()

    if actuals_df.empty:
        logger.warning(
            "No box-score data found for %s. "
            "Run `python src/cli.py ingest -m boxscores` first.",
            prediction_date,
        )
        return None

    # Build lookup: player_id -> stats dict
    actuals_by_player: dict[int, dict] = {}
    for _, row in actuals_df.iterrows():
        pid = int(row["player_id"])
        actuals_by_player[pid] = {
            "points":   float(row.get("points", 0) or 0),
            "rebounds":  float(row.get("rebounds", 0) or 0),
            "assists":   float(row.get("assists", 0) or 0),
            "fg3m":      float(row.get("fg3m", 0) or 0),
            "minutes":   float(row.get("minutes", 0) or 0),
        }

    # Grade each prediction
    graded_rows = []
    for pred in predictions:
        pid = pred.get("player_id")
        if pid is None or pid not in actuals_by_player:
            logger.debug("  Player %s (id=%s) not found in actuals — skipping.",
                        pred.get("player_name"), pid)
            continue

        actual_stats = actuals_by_player[pid]
        market = pred["market"]
        line = pred["line"]
        side = pred["best_side"]

        # Compute actual value for this market
        stat_cols = _MARKET_TO_STATS.get(market, [])
        if not stat_cols:
            logger.debug("  Unknown market '%s' — skipping.", market)
            continue

        actual_value = sum(actual_stats.get(s, 0) for s in stat_cols)

        # Grade
        if actual_value == line:
            result = "push"
            pnl = 0.0
        elif side == "over" and actual_value > line:
            result = "win"
        elif side == "under" and actual_value < line:
            result = "win"
        else:
            result = "loss"

        # Compute P&L
        odds_col = "over_odds" if side == "over" else "under_odds"
        odds = pred.get(odds_col, -110)
        from ev_calc import american_to_decimal
        dec_odds = american_to_decimal(odds)

        if result == "win":
            pnl = round(dec_odds - 1, 4)
        elif result == "push":
            pnl = 0.0
        else:
            pnl = -1.0

        graded_rows.append({
            "game_date":    prediction_date.isoformat(),
            "player_name":  pred["player_name"],
            "player_id":    pid,
            "market":       market,
            "line":         line,
            "side":         side,
            "projection":   pred["projection"],
            "actual":       actual_value,
            "diff":         round(actual_value - pred["projection"], 2),
            "result":       result,
            "odds":         odds,
            "ev_at_pick":   pred["best_ev"],
            "model_prob":   pred.get(f"model_p_{side}", 0),
            "prob_method":  pred.get("prob_method", "normal"),
            "pnl":          pnl,
        })

    if not graded_rows:
        logger.warning("No predictions could be graded for %s", prediction_date)
        return None

    graded_df = pd.DataFrame(graded_rows)

    # Update the predictions file to mark as graded
    for pred in predictions:
        pred["graded"] = True
    with pred_path.open("w") as f:
        json.dump(predictions, f, indent=2)

    # Append to cumulative performance log
    _append_to_performance_log(graded_df)

    return graded_df


def _append_to_performance_log(graded_df: pd.DataFrame) -> None:
    """Append graded results to the cumulative performance log CSV."""
    if _PERF_LOG.exists():
        existing = pd.read_csv(_PERF_LOG)
        # Avoid duplicates: remove any existing rows for this date + player + market
        dates_to_add = graded_df["game_date"].unique()
        existing = existing[~existing["game_date"].isin(dates_to_add)]
        combined = pd.concat([existing, graded_df], ignore_index=True)
    else:
        combined = graded_df

    combined.to_csv(_PERF_LOG, index=False)
    logger.info("Performance log updated: %s (%d total rows)", _PERF_LOG, len(combined))


# ---------------------------------------------------------------------------
# Find un-graded prediction dates
# ---------------------------------------------------------------------------

def find_ungraded_dates() -> list[date]:
    """Find dates with saved predictions that haven't been graded yet."""
    ungraded = []
    for pred_file in sorted(_PREDICTIONS_DIR.glob("*_predictions.json")):
        with pred_file.open() as f:
            preds = json.load(f)
        if preds and not all(p.get("graded", False) for p in preds):
            date_str = pred_file.stem.replace("_predictions", "")
            try:
                ungraded.append(date.fromisoformat(date_str))
            except ValueError:
                continue
    return ungraded


# ---------------------------------------------------------------------------
# Performance summary
# ---------------------------------------------------------------------------

def compute_performance_summary(
    days: int | None = None,
) -> dict:
    """Compute cumulative performance statistics from the log.

    Parameters
    ----------
    days : int, optional
        Only look at the last N days. None = all time.

    Returns
    -------
    dict
        Summary statistics.
    """
    if not _PERF_LOG.exists():
        return {"error": "No performance data yet. Run `review` after grading some predictions."}

    df = pd.read_csv(_PERF_LOG)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date

    if days:
        cutoff = date.today() - timedelta(days=days)
        df = df[df["game_date"] >= cutoff]

    if df.empty:
        return {"error": "No data in the selected time range."}

    non_push = df[df["result"] != "push"]
    wins   = (non_push["result"] == "win").sum()
    losses = (non_push["result"] == "loss").sum()
    pushes = (df["result"] == "push").sum()
    total_bets = len(non_push)
    win_rate = wins / max(total_bets, 1)
    total_pnl = df["pnl"].sum()
    roi = total_pnl / max(total_bets, 1)

    # Projection accuracy
    valid = df[df["actual"].notna() & df["projection"].notna()]
    if not valid.empty:
        mae = float((valid["actual"] - valid["projection"]).abs().mean())
        rmse = float(np.sqrt(((valid["actual"] - valid["projection"]) ** 2).mean()))
    else:
        mae = rmse = float("nan")

    # Calibration check: group by predicted probability buckets
    calibration = []
    if "model_prob" in df.columns and not df["model_prob"].isna().all():
        df["prob_bucket"] = pd.cut(df["model_prob"], bins=[0, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0])
        for bucket, grp in df.groupby("prob_bucket", observed=True):
            grp_np = grp[grp["result"] != "push"]
            if len(grp_np) < 5:
                continue
            actual_wr = (grp_np["result"] == "win").mean()
            avg_prob = grp_np["model_prob"].mean()
            calibration.append({
                "prob_range":     str(bucket),
                "n_bets":         len(grp_np),
                "predicted_wr":   round(avg_prob, 3),
                "actual_wr":      round(actual_wr, 3),
                "gap":            round(actual_wr - avg_prob, 3),
            })

    # Per-market breakdown
    per_market = {}
    for market, grp in non_push.groupby("market"):
        mkt_wins = (grp["result"] == "win").sum()
        mkt_total = len(grp)
        mkt_pnl = grp["pnl"].sum()
        per_market[market] = {
            "wins": int(mkt_wins),
            "total": int(mkt_total),
            "win_rate": round(mkt_wins / max(mkt_total, 1), 3),
            "pnl": round(float(mkt_pnl), 2),
        }

    # Per-day breakdown
    daily_results = []
    for gd, grp in df.groupby("game_date"):
        grp_np = grp[grp["result"] != "push"]
        daily_results.append({
            "date": str(gd),
            "bets": len(grp_np),
            "wins": int((grp_np["result"] == "win").sum()),
            "losses": int((grp_np["result"] == "loss").sum()),
            "pnl": round(float(grp["pnl"].sum()), 2),
        })

    unique_days = df["game_date"].nunique()

    return {
        "total_days":     int(unique_days),
        "total_bets":     int(total_bets),
        "wins":           int(wins),
        "losses":         int(losses),
        "pushes":         int(pushes),
        "win_rate":       round(win_rate, 4),
        "total_pnl":      round(float(total_pnl), 2),
        "roi":            round(roi, 4),
        "proj_mae":       round(mae, 2) if not np.isnan(mae) else None,
        "proj_rmse":      round(rmse, 2) if not np.isnan(rmse) else None,
        "per_market":     per_market,
        "daily_results":  daily_results,
        "calibration":    calibration,
    }
