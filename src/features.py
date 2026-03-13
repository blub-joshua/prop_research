"""
src/features.py
───────────────
Build player-level features for prop projection models.

Feature groups
--------------
1. Rolling averages  (L5, L10 games) for PTS, REB, AST, FG3M, MIN
2. Rolling std-devs  (L5, L10) for the same stats
3. Season-to-date averages (expanding mean, no leakage)
4. Home / away splits (season-to-date expanding mean)
5. Opponent defensive context:
      - Opponent's allowed PTS/REB/AST/FG3M per game in last 10 games
        to the same player position.  If position data is unavailable,
        falls back to team-level totals.
6. Schedule context: days_rest, is_back_to_back, is_home, games_played_season
7. Pace: game-level pace from advanced box scores (avg of both teams' pace)
8. Injury status flag (0=Active/None, 1=Probable, 2=Questionable,
                        3=Doubtful, 4=Out)

All rolling/expanding calculations are computed as LAGGED (shift-1) so
that features for game N use only data from games 1..N-1. This makes
the same feature pipeline safe for both training (no future leakage)
and live inference (features for today's game).

Writes results to the player_features table (one row per player per game).
Also writes a player_features_today_*.csv under data/ for inspection.

Run directly:
    python src/features.py [--seasons 2022-23 2023-24 2024-25]

Or via CLI:
    python src/cli.py features
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from db import get_connection, init_schema, upsert_dataframe, query_df

load_dotenv()
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Rolling window sizes (number of games)
WINDOWS = [5, 10]

# Stat columns we compute rolling features for
STAT_COLS = ["points", "rebounds", "assists", "fg3m", "minutes"]

# Injury severity map (higher = worse)
_INJURY_SEVERITY = {
    "Out":          4,
    "Doubtful":     3,
    "Questionable": 2,
    "GTD":          2,
    "Probable":     1,
    "Active":       0,
}


# ---------------------------------------------------------------------------
# 1. Load raw data
# ---------------------------------------------------------------------------

def _load_stats(seasons: list[str] | None, con) -> pd.DataFrame:
    """Load player_game_stats joined with games for the requested seasons.

    Returns a DataFrame sorted by (player_id, game_date).
    Excludes DNP rows from rolling calculations but keeps them in the table
    so we can compute days_rest correctly.
    """
    season_filter = ""
    if seasons:
        placeholders = ", ".join(f"'{s}'" for s in seasons)
        season_filter = f"AND g.season IN ({placeholders})"

    sql = f"""
        SELECT
            pgs.player_id,
            pgs.game_id,
            pgs.team_id,
            pgs.game_date,
            pgs.did_not_play,
            pgs.is_starter,
            pgs.points,
            pgs.rebounds,
            pgs.assists,
            pgs.fg3m,
            pgs.minutes,
            pgs.usage_pct,
            pgs.pace,
            g.season,
            g.home_team_id,
            g.away_team_id,
            CASE WHEN pgs.team_id = g.home_team_id THEN 1 ELSE 0 END AS is_home,
            CASE WHEN pgs.team_id = g.home_team_id
                 THEN g.away_team_id ELSE g.home_team_id END AS opponent_id
        FROM player_game_stats pgs
        JOIN games g ON pgs.game_id = g.game_id
        WHERE 1=1 {season_filter}
        ORDER BY pgs.player_id, pgs.game_date, pgs.game_id
    """
    df = con.execute(sql).df()
    logger.info("Loaded %d player-game rows.", len(df))

    # Coerce types
    for col in STAT_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["game_date"]    = pd.to_datetime(df["game_date"])
    df["did_not_play"] = df["did_not_play"].fillna(False).astype(bool)
    df["is_home"]      = df["is_home"].fillna(0).astype(int)
    return df


# ---------------------------------------------------------------------------
# 2. Rolling & season features
# ---------------------------------------------------------------------------

def _short(col: str) -> str:
    """Short prefix for a stat column name."""
    return {
        "points":   "pts",
        "rebounds": "reb",
        "assists":  "ast",
        "fg3m":     "fg3m",
        "minutes":  "min",
    }.get(col, col[:4])


def compute_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all rolling / expanding features per player.

    Avoids groupby().apply() entirely — uses explicit per-player loops so
    that player_id is never moved into the index, which happens with apply()
    in some pandas versions.

    All features are shift(1) so game[i] only sees games[0..i-1].
    DNP rows do not contribute to rolling windows.
    """
    logger.info("Computing rolling and season averages...")
    df = df.sort_values(["player_id", "game_date", "game_id"]).reset_index(drop=True)

    # Pre-allocate output columns as NaN
    new_cols: dict[str, list] = {}
    for col in STAT_COLS:
        short = _short(col)
        for w in WINDOWS:
            new_cols[f"{short}_avg_L{w}"] = [np.nan] * len(df)
            new_cols[f"{short}_std_L{w}"] = [np.nan] * len(df)
        new_cols[f"{short}_avg_season"] = [np.nan] * len(df)

    # Process one player at a time — no groupby().apply(), no index magic
    for pid, grp_idx in df.groupby("player_id", sort=False).groups.items():
        grp = df.loc[grp_idx].sort_values(["game_date", "game_id"])
        played_idx = grp.index[~grp["did_not_play"].fillna(False)]
        played = grp.loc[played_idx]

        for col in STAT_COLS:
            short = _short(col)
            series = played[col].astype(float)
            shifted = series.shift(1)

            for w in WINDOWS:
                min_p = max(1, w // 2)
                avg_vals = shifted.rolling(w, min_periods=min_p).mean()
                std_vals = shifted.rolling(w, min_periods=min_p).std().fillna(0)
                for i, idx in enumerate(played_idx):
                    new_cols[f"{short}_avg_L{w}"][idx] = avg_vals.iloc[i]
                    new_cols[f"{short}_std_L{w}"][idx] = std_vals.iloc[i]

            exp_mean = shifted.expanding().mean()
            for i, idx in enumerate(played_idx):
                new_cols[f"{short}_avg_season"][idx] = exp_mean.iloc[i]

    for col_name, values in new_cols.items():
        df[col_name] = values

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Home / away splits
# ---------------------------------------------------------------------------

def compute_home_away_splits(df: pd.DataFrame) -> pd.DataFrame:
    """Add season-to-date H/A average columns (pts, reb, ast, fg3m)."""
    logger.info("Computing home/away splits...")
    df = df.reset_index(drop=True)   # ensure player_id is a column, not index level
    df = df.sort_values(["player_id", "season", "game_date", "game_id"])

    for split_val, suffix in [(1, "home"), (0, "away")]:
        mask = df["is_home"] == split_val
        for col in ["points", "rebounds", "assists", "fg3m"]:
            short = _short(col)
            col_name = f"{short}_avg_{suffix}"
            # Within each (player, season) group, compute expanding mean
            # on the subset, then re-align to the full index
            df[col_name] = np.nan
            for (pid, szn), grp in df.groupby(["player_id", "season"]):
                sub = grp[mask[grp.index]]
                if sub.empty:
                    continue
                played_sub = sub[~sub["did_not_play"]][col]
                exp_mean = played_sub.shift(1).expanding().mean()
                # Forward-fill from last available value into non-split games
                # (so every row has an estimate)
                df.loc[exp_mean.index, col_name] = exp_mean.values

    # Forward-fill NaN cells within (player, season) groups
    for col_name in [f"{_short(c)}_avg_{s}"
                     for c in ["points","rebounds","assists","fg3m"]
                     for s in ["home","away"]]:
        df[col_name] = (
            df.groupby(["player_id","season"])[col_name]
            .transform(lambda x: x.ffill())
        )

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. Rest / schedule context
# ---------------------------------------------------------------------------

def compute_rest_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add days_rest, is_back_to_back, and games_played_season."""
    logger.info("Computing rest features...")
    df = df.sort_values(["player_id", "game_date", "game_id"])

    prev_date = df.groupby("player_id")["game_date"].shift(1)
    df["days_rest"] = (df["game_date"] - prev_date).dt.days - 1
    df["is_back_to_back"] = (df["days_rest"] == 0).astype(int)

    df["games_played_season"] = (
        df[~df["did_not_play"]]
        .groupby(["player_id", "season"])
        .cumcount()
    )
    df["games_played_season"] = df["games_played_season"].fillna(0).astype(int)

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5. Opponent defensive context
# ---------------------------------------------------------------------------

def compute_opponent_defense(df: pd.DataFrame, con) -> pd.DataFrame:
    """Add opponent allowed-stat rolling averages.

    For each game, we compute the opponent team's last-10-game rolling
    averages of stats ALLOWED (i.e., scored by opposing players).

    Implementation strategy:
    1. Aggregate player_game_stats at the team level — sum stats per game
       for all players on the opposing team.
    2. For each team × game, compute the rolling L10 average of stats they
       allowed.
    3. Join back to df on (opponent_id, game_date).
    """
    logger.info("Computing opponent defense features...")

    # Total stats scored AGAINST each team in each game
    opp_sql = """
        SELECT
            g.game_id,
            g.game_date,
            CASE WHEN pgs.team_id = g.home_team_id
                 THEN g.away_team_id ELSE g.home_team_id END AS defending_team_id,
            SUM(COALESCE(pgs.points,   0)) AS allowed_pts,
            SUM(COALESCE(pgs.rebounds, 0)) AS allowed_reb,
            SUM(COALESCE(pgs.assists,  0)) AS allowed_ast,
            SUM(COALESCE(pgs.fg3m,     0)) AS allowed_fg3m
        FROM player_game_stats pgs
        JOIN games g ON pgs.game_id = g.game_id
        WHERE pgs.did_not_play = FALSE
        GROUP BY g.game_id, g.game_date, defending_team_id
        ORDER BY defending_team_id, g.game_date
    """
    opp_df = con.execute(opp_sql).df()
    opp_df["game_date"] = pd.to_datetime(opp_df["game_date"])

    # Rolling L10 averages per team
    opp_df = opp_df.sort_values(["defending_team_id", "game_date"])
    for stat in ["allowed_pts", "allowed_reb", "allowed_ast", "allowed_fg3m"]:
        roll_col = f"opp_{stat}_L10"
        opp_df[roll_col] = (
            opp_df.groupby("defending_team_id")[stat]
            .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
        )

    # Keep only the columns we need for the join
    opp_join = opp_df[[
        "game_id", "defending_team_id",
        "opp_allowed_pts_L10", "opp_allowed_reb_L10",
        "opp_allowed_ast_L10", "opp_allowed_fg3m_L10",
    ]].rename(columns={"defending_team_id": "opponent_id"})

    df = df.merge(opp_join, on=["game_id", "opponent_id"], how="left")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 6. Injury feature
# ---------------------------------------------------------------------------

def attach_injury_feature(df: pd.DataFrame, con) -> pd.DataFrame:
    """Merge the most recent injury severity for each player as of each game date.

    Uses a left merge_asof (sorted by date) so that features use only
    reports published before the game.
    """
    logger.info("Attaching injury features...")

    inj_sql = """
        SELECT player_id, report_date, status
        FROM injuries
        ORDER BY player_id, report_date
    """
    inj_df = con.execute(inj_sql).df()
    if inj_df.empty:
        df["injury_severity"] = 0
        return df

    inj_df["report_date"] = pd.to_datetime(inj_df["report_date"])
    inj_df["injury_severity"] = inj_df["status"].map(
        lambda s: _INJURY_SEVERITY.get(str(s), 0)
    )

    # Keep only the most severe status per (player, date)
    inj_df = (
        inj_df.sort_values(["player_id","report_date","injury_severity"])
        .drop_duplicates(subset=["player_id","report_date"], keep="last")
    )

    df_sorted = df.sort_values("game_date")
    inj_sorted = inj_df.sort_values("report_date")

    merged = pd.merge_asof(
        df_sorted.reset_index(drop=True),
        inj_sorted[["player_id","report_date","injury_severity","status"]].rename(
            columns={"report_date":"game_date", "status":"injury_status"}
        ),
        on="game_date",
        by="player_id",
        direction="backward",
    )
    merged["injury_severity"] = merged["injury_severity"].fillna(0).astype(int)
    return merged.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 7. Pace feature
# ---------------------------------------------------------------------------

def compute_pace_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Add a game-level pace estimate.

    Uses the 'pace' column from advanced box scores (if available).
    Falls back to a rolling team-pace estimate computed from box scores.
    """
    if "pace" in df.columns and df["pace"].notna().sum() > 0:
        # Player-level pace from advanced — use per-player rolling mean
        df["pace_L5"] = (
            df.groupby("player_id")["pace"]
            .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        )
    else:
        df["pace_L5"] = np.nan  # Will be imputed with league mean later

    return df


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def build_features(seasons: list[str] | None = None, con=None) -> pd.DataFrame:
    """Build the full player_features table.

    Parameters
    ----------
    seasons : list[str], optional
        Restrict to these seasons (e.g. ["2022-23","2023-24"]).
        None = all seasons in the database.
    con : duckdb connection, optional

    Returns
    -------
    pd.DataFrame  feature rows written to player_features
    """
    _close = con is None
    if con is None:
        con = get_connection()

    try:
        df = _load_stats(seasons, con)
        if df.empty:
            logger.warning("No stats rows found — run ingest_boxscores first.")
            return pd.DataFrame()

        df = compute_rolling_features(df)
        df = compute_home_away_splits(df)
        df = compute_rest_features(df)
        df = compute_opponent_defense(df, con)
        df = compute_pace_feature(df)
        df = attach_injury_feature(df, con)

        # ── Final column selection & rename to match player_features schema ──
        # Map computed columns to schema names
        rename_map = {
            "pts_avg_L5":               "pts_avg_L5",
            "pts_avg_L10":              "pts_avg_L10",
            "pts_std_L5":               "pts_std_L5",
            "pts_std_L10":              "pts_std_L10",
            "pts_avg_season":           "pts_avg_season",
            "reb_avg_L5":               "reb_avg_L5",
            "reb_avg_L10":              "reb_avg_L10",
            "reb_std_L5":               "reb_std_L5",
            "reb_std_L10":              "reb_std_L10",
            "reb_avg_season":           "reb_avg_season",
            "ast_avg_L5":               "ast_avg_L5",
            "ast_avg_L10":              "ast_avg_L10",
            "ast_std_L5":               "ast_std_L5",
            "ast_std_L10":              "ast_std_L10",
            "ast_avg_season":           "ast_avg_season",
            "fg3m_avg_L5":              "fg3m_avg_L5",
            "fg3m_avg_L10":             "fg3m_avg_L10",
            "fg3m_std_L5":              "fg3m_std_L5",
            "fg3m_std_L10":             "fg3m_std_L10",
            "fg3m_avg_season":          "fg3m_avg_season",
            "min_avg_L5":               "min_avg_L5",
            "min_avg_L10":              "min_avg_L10",
            "min_avg_season":           "min_avg_season",
            "pts_avg_home":             "pts_avg_home",
            "pts_avg_away":             "pts_avg_away",
            "reb_avg_home":             "reb_avg_home",
            "reb_avg_away":             "reb_avg_away",
            "ast_avg_home":             "ast_avg_home",
            "ast_avg_away":             "ast_avg_away",
            "opp_allowed_pts_L10":      "opp_pts_allowed_avg",
            "opp_allowed_reb_L10":      "opp_reb_allowed_avg",
            "opp_allowed_ast_L10":      "opp_ast_allowed_avg",
            "opp_allowed_fg3m_L10":     "opp_fg3m_allowed_avg",
            "days_rest":                "days_rest",
            "is_back_to_back":          "is_back_to_back",
            "is_home":                  "is_home",
            "games_played_season":      "games_played_season",
            "injury_severity":          "injury_severity",
            "pace_L5":                  "pace_L5",
        }

        # Drop rows with no usable features (e.g. first game of career)
        # Keep all rows but allow NaN — models handle imputation.

        # Build the output DataFrame
        keep_cols = ["player_id", "game_id", "game_date", "team_id", "opponent_id",
                     "season", "is_home"] + list(rename_map.keys())
        out = df[[c for c in keep_cols if c in df.columns]].copy()
        out = out.rename(columns=rename_map)

        logger.info("Writing %d feature rows to player_features ...", len(out))
        upsert_dataframe(out, "player_features", ["player_id", "game_id"], con=con)

        # Save a CSV snapshot for quick inspection
        snap_path = _PROJECT_ROOT / "data" / "player_features_snapshot.csv"
        out.tail(5000).to_csv(snap_path, index=False)
        logger.info("Snapshot saved to %s", snap_path)

        return out
    finally:
        if _close:
            con.close()


# ---------------------------------------------------------------------------
# Live-inference entry (used by models.py at prediction time)
# ---------------------------------------------------------------------------

def build_features_for_today(
    player_ids: list[int],
    game_date,
    opponent_map: dict[int, int] | None = None,
    is_home_map: dict[int, int] | None = None,
    con=None,
) -> pd.DataFrame:
    """Build feature rows for a specific set of players for today's games.

    Uses only historical data (up to and including yesterday's games).

    Parameters
    ----------
    player_ids : list[int]
        Players appearing in today's props.
    game_date : date or str
        Today's date (used as the prediction horizon).
    opponent_map : dict[int, int], optional
        {player_id: opponent_team_id}  — filled from today's props if available.
    is_home_map : dict[int, int], optional
        {player_id: 1 or 0}
    con : duckdb connection, optional

    Returns
    -------
    pd.DataFrame   one row per player, with all feature columns
    """
    import datetime as dt
    _close = con is None
    if con is None:
        con = get_connection(read_only=True)

    if isinstance(game_date, str):
        game_date = pd.Timestamp(game_date)
    else:
        game_date = pd.Timestamp(game_date)

    try:
        if not player_ids:
            return pd.DataFrame()

        pid_filter = ", ".join(str(p) for p in player_ids)
        sql = f"""
            SELECT pgs.player_id, pgs.game_id, pgs.team_id, pgs.game_date,
                   pgs.did_not_play, pgs.is_starter,
                   pgs.points, pgs.rebounds, pgs.assists, pgs.fg3m, pgs.minutes,
                   pgs.usage_pct, pgs.pace, g.season,
                   g.home_team_id, g.away_team_id,
                   CASE WHEN pgs.team_id = g.home_team_id THEN 1 ELSE 0 END AS is_home,
                   CASE WHEN pgs.team_id = g.home_team_id
                        THEN g.away_team_id ELSE g.home_team_id END AS opponent_id
            FROM player_game_stats pgs
            JOIN games g ON pgs.game_id = g.game_id
            WHERE pgs.player_id IN ({pid_filter})
              AND pgs.game_date < '{game_date.date()}'
            ORDER BY pgs.player_id, pgs.game_date
        """
        df = con.execute(sql).df()
        if df.empty:
            logger.warning("No historical data for requested players.")
            return pd.DataFrame()

        for col in STAT_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["game_date"] = pd.to_datetime(df["game_date"])
        df["did_not_play"] = df["did_not_play"].fillna(False).astype(bool)

        # Compute features
        df = compute_rolling_features(df)
        df = compute_rest_features(df)
        df = compute_pace_feature(df)
        df = attach_injury_feature(df, con)

        # Take the last row per player (most recent game = today's features)
        latest = df.sort_values("game_date").groupby("player_id").last().reset_index()

        # Override today's context if caller provided it
        if opponent_map:
            latest["opponent_id"] = latest["player_id"].map(opponent_map).fillna(latest["opponent_id"])
        if is_home_map:
            latest["is_home"] = latest["player_id"].map(is_home_map).fillna(latest["is_home"])

        # Add opponent defense features using the updated opponent_id
        latest = compute_opponent_defense_for_players(latest, con)

        # Set game_date to today for downstream labeling
        latest["game_date"] = game_date

        return latest
    finally:
        if _close:
            con.close()


def compute_opponent_defense_for_players(df: pd.DataFrame, con) -> pd.DataFrame:
    """Lightweight opponent defense lookup for a small inference-time DataFrame."""
    if df.empty or "opponent_id" not in df.columns:
        return df

    # Load pre-aggregated opponent defense from the full historical table
    opp_sql = """
        SELECT
            CASE WHEN pgs.team_id = g.home_team_id
                 THEN g.away_team_id ELSE g.home_team_id END AS defending_team_id,
            g.game_date,
            SUM(COALESCE(pgs.points,   0)) AS allowed_pts,
            SUM(COALESCE(pgs.rebounds, 0)) AS allowed_reb,
            SUM(COALESCE(pgs.assists,  0)) AS allowed_ast,
            SUM(COALESCE(pgs.fg3m,     0)) AS allowed_fg3m
        FROM player_game_stats pgs
        JOIN games g ON pgs.game_id = g.game_id
        WHERE pgs.did_not_play = FALSE
        GROUP BY defending_team_id, g.game_date
        ORDER BY defending_team_id, g.game_date
    """
    opp = con.execute(opp_sql).df()
    opp["game_date"] = pd.to_datetime(opp["game_date"])
    opp = opp.sort_values(["defending_team_id", "game_date"])

    for stat in ["allowed_pts", "allowed_reb", "allowed_ast", "allowed_fg3m"]:
        opp[f"opp_{stat}_L10"] = (
            opp.groupby("defending_team_id")[stat]
            .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
        )

    # Get the most recent value per team
    latest_opp = (
        opp.sort_values("game_date")
        .groupby("defending_team_id")
        .last()
        .reset_index()
        [["defending_team_id",
          "opp_allowed_pts_L10","opp_allowed_reb_L10",
          "opp_allowed_ast_L10","opp_allowed_fg3m_L10"]]
        .rename(columns={"defending_team_id": "opponent_id",
                         "opp_allowed_pts_L10":  "opp_pts_allowed_avg",
                         "opp_allowed_reb_L10":  "opp_reb_allowed_avg",
                         "opp_allowed_ast_L10":  "opp_ast_allowed_avg",
                         "opp_allowed_fg3m_L10": "opp_fg3m_allowed_avg"})
    )
    df = df.merge(latest_opp, on="opponent_id", how="left")
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
    )
    parser = argparse.ArgumentParser(description="Build player features for modeling.")
    parser.add_argument("--seasons", nargs="+", default=None,
                        help="Seasons to build features for (default: all).")
    args = parser.parse_args(argv)

    con = get_connection()
    init_schema(con)
    df = build_features(seasons=args.seasons, con=con)
    logger.info("Feature build complete.  Shape: %s", df.shape)
    con.close()


if __name__ == "__main__":
    main()
