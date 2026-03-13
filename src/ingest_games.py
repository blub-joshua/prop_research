"""
src/ingest_games.py
───────────────────
Ingest NBA schedules, final scores, and team metadata.

Data source
-----------
nba_api  (https://github.com/swar/nba_api)
  • nba_api.stats.static.teams        → static team list
  • nba_api.stats.endpoints.leaguegamelog.LeagueGameLog
      season="2024-25", season_type_all_star="Regular Season"|"Playoffs"
      Returns one row per (team, game) — we de-duplicate to one row per game.

Endpoint notes
--------------
LeagueGameLog columns (relevant subset):
  SEASON_ID, GAME_ID, GAME_DATE, TEAM_ID, TEAM_ABBREVIATION,
  MATCHUP, WL, PTS, (opponent PTS is not directly here — we derive it
  by joining the same game's two team rows)

Run directly:
    python src/ingest_games.py [--seasons 2021-22 2022-23 2023-24]

Or via CLI:
    python src/cli.py ingest --module games
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Allow running as `python src/ingest_games.py` from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from db import get_connection, init_schema, upsert_dataframe

load_dotenv()
logger = logging.getLogger(__name__)

_API_SLEEP = float(os.getenv("NBA_API_SLEEP", "0.7"))  # seconds between API calls

_DEFAULT_SEASONS: list[str] = [
    s.strip()
    for s in os.getenv("SEASONS", "2020-21,2021-22,2022-23,2023-24,2024-25").split(",")
]


# ---------------------------------------------------------------------------
# Teams
# ---------------------------------------------------------------------------

def ingest_teams(con) -> pd.DataFrame:
    """Fetch all 30 NBA franchises from nba_api static data and upsert into teams.

    The static teams list never changes mid-season so this only needs to run once.

    Returns
    -------
    pd.DataFrame  (team_id, abbreviation, full_name, city, conference, division)
    """
    from nba_api.stats.static import teams as nba_static_teams

    raw = nba_static_teams.get_teams()  # list of dicts
    df = pd.DataFrame(raw).rename(columns={
        "id":           "team_id",
        "full_name":    "full_name",
        "abbreviation": "abbreviation",
        "nickname":     "nickname",
        "city":         "city",
        "state":        "state",
        "year_founded": "year_founded",
    })

    # Schema columns only
    df = df[["team_id", "abbreviation", "full_name", "city"]].copy()
    df["conference"] = None   # Not in static data; populated by a later call if needed
    df["division"]   = None

    upsert_dataframe(df, "teams", ["team_id"], con=con)
    logger.info("Upserted %d teams.", len(df))
    return df


# ---------------------------------------------------------------------------
# Games
# ---------------------------------------------------------------------------

def _fetch_game_log(season: str, season_type: str) -> pd.DataFrame:
    """Fetch a single LeagueGameLog from the NBA API.

    Parameters
    ----------
    season : str        e.g. "2024-25"
    season_type : str   "Regular Season" | "Playoffs"

    Returns
    -------
    pd.DataFrame  raw API DataFrame (one row per team per game)
    """
    from nba_api.stats.endpoints import leaguegamelog

    log = leaguegamelog.LeagueGameLog(
        season=season,
        season_type_all_star=season_type,
        league_id="00",
        timeout=60,
    )
    df = log.get_data_frames()[0]
    logger.debug("  Raw rows from API: %d", len(df))
    return df


def _normalize_game_log(df: pd.DataFrame, season: str, season_type: str) -> pd.DataFrame:
    """Convert a raw LeagueGameLog DataFrame to the games table schema.

    The API returns two rows per game (one per team).  We pivot to one row
    per game by identifying home vs away from the MATCHUP string:
      "LAL vs. GSW"  → LAL is home
      "LAL @ GSW"    → LAL is away  (GSW is home)

    Parameters
    ----------
    df : pd.DataFrame   raw from _fetch_game_log()
    season : str
    season_type : str

    Returns
    -------
    pd.DataFrame  one row per game_id
    """
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    # Parse game date
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date

    # Identify home team: MATCHUP with "vs." means home; "@" means away
    df["is_home"] = df["matchup"].str.contains(r"\bvs\.", regex=True)

    home_rows = df[df["is_home"]].copy()
    away_rows = df[~df["is_home"]].copy()

    # Merge on game_id to get both teams in one row
    merged = home_rows.merge(
        away_rows[["game_id", "team_id", "pts"]],
        on="game_id",
        suffixes=("_home", "_away"),
    )

    result = pd.DataFrame({
        "game_id":       merged["game_id"],
        "game_date":     merged["game_date"],
        "season":        season,
        "season_type":   season_type,
        "home_team_id":  merged["team_id_home"],
        "away_team_id":  merged["team_id_away"],
        "home_score":    pd.to_numeric(merged["pts_home"], errors="coerce").astype("Int16"),
        "away_score":    pd.to_numeric(merged["pts_away"], errors="coerce").astype("Int16"),
        "status":        "Final",
    })

    # Remove any rows where we couldn't identify both teams
    result = result.dropna(subset=["home_team_id", "away_team_id"])
    result["home_team_id"] = result["home_team_id"].astype(int)
    result["away_team_id"] = result["away_team_id"].astype(int)

    return result.drop_duplicates(subset=["game_id"]).reset_index(drop=True)


def ingest_games_for_season(season: str, con, include_playoffs: bool = True) -> int:
    """Fetch and upsert all games for one NBA season.

    Parameters
    ----------
    season : str            e.g. "2024-25"
    con                     DuckDB connection
    include_playoffs : bool Include playoff games (default True)

    Returns
    -------
    int   number of game rows upserted
    """
    total = 0
    season_types = ["Regular Season"]
    if include_playoffs:
        season_types.append("Playoffs")

    for stype in season_types:
        logger.info("  Fetching %s — %s ...", season, stype)
        try:
            raw = _fetch_game_log(season, stype)
            time.sleep(_API_SLEEP)
            if raw.empty:
                logger.info("    No data returned.")
                continue
            games_df = _normalize_game_log(raw, season, stype)
            logger.info("    Normalised %d games.", len(games_df))
            upsert_dataframe(games_df, "games", ["game_id"], con=con)
            total += len(games_df)
        except Exception as exc:
            logger.warning("    Error for %s %s: %s", season, stype, exc)
            time.sleep(2)  # back off a bit on error

    return total


def ingest_all_games(seasons: list[str] | None = None, con=None) -> None:
    """Ingest games for all configured seasons.

    Parameters
    ----------
    seasons : list[str], optional   defaults to SEASONS env var
    con : duckdb connection, optional
    """
    seasons = seasons or _DEFAULT_SEASONS
    _close = con is None
    if con is None:
        con = get_connection()
    try:
        for season in seasons:
            logger.info("Ingesting games: season %s", season)
            n = ingest_games_for_season(season, con)
            logger.info("  Total rows upserted for %s: %d", season, n)
    finally:
        if _close:
            con.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Ingest NBA game schedules and scores.")
    parser.add_argument(
        "--seasons", nargs="+", default=_DEFAULT_SEASONS,
        help="Seasons to ingest, e.g. 2022-23 2023-24 2024-25",
    )
    args = parser.parse_args(argv)

    con = get_connection()
    init_schema(con)

    logger.info("Ingesting teams...")
    ingest_teams(con)

    logger.info("Ingesting games for seasons: %s", args.seasons)
    ingest_all_games(seasons=args.seasons, con=con)

    from db import table_counts
    counts = table_counts(con)
    logger.info("Table counts: %s", counts)
    con.close()
    logger.info("ingest_games done.")


if __name__ == "__main__":
    main()
