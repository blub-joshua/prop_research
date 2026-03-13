"""
src/ingest_boxscores.py
───────────────────────
Ingest per-player box score statistics for every completed game.

Data source
-----------
nba_api  (https://github.com/swar/nba_api)
  • BoxScoreTraditionalV3 — per-player counting stats
  • BoxScoreAdvancedV3   — per-player advanced metrics (merged in if available)

V3 quirks vs V2
---------------
  • All column names are camelCase instead of ALL_CAPS.
  • `minutes` is a decimal float string (e.g. "34.566667"), NOT "MM:SS".
  • Player name is two fields: firstName + familyName.
  • Starter flag is `position` (non-empty = starter), not START_POSITION.
  • Five extra kwargs are required or the API returns nothing:
        end_period=1, end_range=0, range_type=0, start_period=1, start_range=0

Schema contract
---------------
  • Do NOT supply `id` — DuckDB auto-assigns it from seq_pgs_id.
  • Composite columns (pra, points_rebounds, …) are plain SMALLINT columns
    computed here (GENERATED ALWAYS STORED is not supported in older DuckDB).
  • team_id FK is nullable — old games occasionally have unmapped team IDs.

Rate-limit / resilience strategy
----------------------------------
  • Each game's API calls are wrapped in a broad try/except.
  • ConnectionResetError, TimeoutError, HTTPError, and any other exception
    are caught, logged as WARNING, and that game is skipped.
  • Configurable sleep between calls via NBA_API_SLEEP env var (default 0.7 s).
  • On a per-game exception the script sleeps an extra back-off period before
    continuing to avoid hammering a rate-limited endpoint.

Run directly:
    python src/ingest_boxscores.py [--limit N] [--seasons 2023-24 2024-25]

Or via CLI:
    python src/cli.py ingest --module boxscores
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from db import get_connection, init_schema, upsert_dataframe

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_API_SLEEP   = float(os.getenv("NBA_API_SLEEP", "0.7"))
_ERROR_SLEEP = float(os.getenv("NBA_API_ERROR_SLEEP", "3.0"))
_TIMEOUT     = int(os.getenv("NBA_API_TIMEOUT", "60"))

# Required by every V3 boxscore endpoint — omitting these causes KeyError
_V3_EXTRA = dict(
    end_period=1,
    end_range=0,
    range_type=0,
    start_period=1,
    start_range=0,
)


# ---------------------------------------------------------------------------
# Low-level fetchers  (each returns a raw DataFrame or raises)
# ---------------------------------------------------------------------------

def _fetch_traditional(game_id: str) -> pd.DataFrame:
    """Call BoxScoreTraditionalV3 and return the PlayerStats DataFrame.

    Raises any exception — callers are responsible for catching.
    """
    from nba_api.stats.endpoints.boxscoretraditionalv3 import BoxScoreTraditionalV3
    bs = BoxScoreTraditionalV3(game_id=game_id, timeout=_TIMEOUT, **_V3_EXTRA)
    frames = bs.get_data_frames()
    if not frames or frames[0].empty:
        raise ValueError(f"BoxScoreTraditionalV3 returned no data for {game_id}")
    return frames[0]


def _fetch_advanced(game_id: str) -> Optional[pd.DataFrame]:
    """Call BoxScoreAdvancedV3 and return the PlayerStats DataFrame.

    Returns None on any failure (advanced stats are optional — we log a
    warning and continue without them rather than failing the whole game).
    """
    try:
        from nba_api.stats.endpoints.boxscoreadvancedv3 import BoxScoreAdvancedV3
        bs = BoxScoreAdvancedV3(game_id=game_id, timeout=_TIMEOUT, **_V3_EXTRA)
        frames = bs.get_data_frames()
        if frames and not frames[0].empty:
            return frames[0]
        logger.warning("    BoxScoreAdvancedV3 returned empty for %s — skipping adv stats", game_id)
        return None
    except Exception as exc:
        logger.warning("    BoxScoreAdvancedV3 failed for %s (%s) — skipping adv stats", game_id, exc)
        return None


# ---------------------------------------------------------------------------
# Column rename maps  (V3 camelCase → our snake_case DB names)
# ---------------------------------------------------------------------------

_TRAD_RENAME = {
    "personId":                 "player_id",
    "gameId":                   "game_id",
    "teamId":                   "team_id",
    # firstName + familyName combined into full_name separately
    "position":                 "start_position",   # non-empty → starter
    "comment":                  "comment",
    "minutes":                  "min_raw",           # decimal float string
    "fieldGoalsMade":           "fgm",
    "fieldGoalsAttempted":      "fga",
    "fieldGoalsPercentage":     "fg_pct",
    "threePointersMade":        "fg3m",
    "threePointersAttempted":   "fg3a",
    "threePointersPercentage":  "fg3_pct",
    "freeThrowsMade":           "ftm",
    "freeThrowsAttempted":      "fta",
    "freeThrowsPercentage":     "ft_pct",
    "reboundsOffensive":        "off_reb",
    "reboundsDefensive":        "def_reb",
    "reboundsTotal":            "rebounds",
    "assists":                  "assists",
    "steals":                   "steals",
    "blocks":                   "blocks",
    "turnovers":                "turnovers",
    "foulsPersonal":            "fouls",
    "points":                   "points",
    "plusMinusPoints":          "plus_minus",
}

_ADV_RENAME = {
    "personId":              "player_id",
    "gameId":                "game_id",
    "usagePercentage":       "usage_pct",
    "offensiveRating":       "off_rating",
    "defensiveRating":       "def_rating",
    "pace":                  "pace",
}


# ---------------------------------------------------------------------------
# Minutes parsing
# ---------------------------------------------------------------------------

def _parse_minutes(val) -> Optional[float]:
    """Parse the API minutes field to a decimal float.

    V3: decimal string e.g. "34.566667"
    V2 legacy fallback: "MM:SS" e.g. "34:22"
    Also handles float/int values passed directly.
    """
    if val is None:
        return None
    if isinstance(val, (int, float)):
        f = float(val)
        return None if np.isnan(f) else f
    s = str(val).strip()
    if not s or s.upper() in ("DNP", "DND", "NWT", "INACTIVE", ""):
        return None
    try:
        if ":" in s:
            m, sec = s.split(":", 1)
            return float(m) + float(sec) / 60.0
        return float(s)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def _normalize_traditional(raw: pd.DataFrame, game_date) -> pd.DataFrame:
    """Rename columns, derive helper fields, cast types."""

    # Combine first + last name before any renaming
    fn = raw.get("firstName", pd.Series([""] * len(raw), index=raw.index))
    ln = raw.get("familyName", pd.Series([""] * len(raw), index=raw.index))
    raw = raw.copy()
    raw["full_name"] = (fn.fillna("").str.strip() + " " + ln.fillna("").str.strip()).str.strip()

    df = raw.rename(columns=_TRAD_RENAME)

    # Keep only columns from the rename map + full_name
    wanted = list(dict.fromkeys([v for v in _TRAD_RENAME.values()] + ["full_name"]))
    df = df[[c for c in wanted if c in df.columns]].copy()

    # Derived fields
    df["minutes"]      = df["min_raw"].apply(_parse_minutes)
    df["game_date"]    = game_date

    # did_not_play: no minutes, or explicit DNP/DND comment
    def _dnp(row):
        comment = str(row.get("comment", "")).upper()
        if any(k in comment for k in ("DNP", "DND", "NWT", "INACTIVE", "NOT WITH TEAM")):
            return True
        mins = row.get("minutes")
        return mins is None or (isinstance(mins, float) and np.isnan(mins))

    df["did_not_play"] = df.apply(_dnp, axis=1)

    # is_starter: V3 position field non-empty means the player started
    sp = df.get("start_position", pd.Series([""] * len(df), index=df.index))
    df["is_starter"] = sp.apply(
        lambda x: bool(x) and str(x).strip().lower() not in ("", "nan", "none")
    )

    # Compute combo props here (plain columns, not GENERATED)
    def _si(col):
        return pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int) if col in df.columns else 0

    pts = _si("points")
    reb = _si("rebounds")
    ast = _si("assists")
    df["points_rebounds"]  = (pts + reb).astype("Int16")
    df["points_assists"]   = (pts + ast).astype("Int16")
    df["rebounds_assists"] = (reb + ast).astype("Int16")
    df["pra"]              = (pts + reb + ast).astype("Int16")

    # Integer stats
    int_cols = [
        "points", "rebounds", "assists", "steals", "blocks", "turnovers",
        "fouls", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "plus_minus",
        "off_reb", "def_reb",
    ]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int16")

    # Float stats
    for col in ("fg_pct", "fg3_pct", "ft_pct", "minutes"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _normalize_advanced(raw: pd.DataFrame) -> pd.DataFrame:
    """Rename V3 advanced columns to DB names and cast to float."""
    df = raw.rename(columns=_ADV_RENAME)
    keep = [v for v in _ADV_RENAME.values() if v in df.columns]
    df = df[keep].copy()
    for col in ("usage_pct", "off_rating", "def_rating", "pace"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Player stub upsert
# ---------------------------------------------------------------------------

def _upsert_player_stubs(df: pd.DataFrame, con) -> None:
    """Ensure every player in this box score has a row in players."""
    if "player_id" not in df.columns or "full_name" not in df.columns:
        return
    stubs = (
        df[["player_id", "full_name"]]
        .drop_duplicates("player_id")
        .pipe(lambda d: d[d["player_id"].notna()])
        .copy()
    )
    stubs["player_id"] = stubs["player_id"].astype(int)
    upsert_dataframe(stubs, "players", ["player_id"], con=con)


# ---------------------------------------------------------------------------
# Per-game ingestion
# ---------------------------------------------------------------------------

def ingest_boxscore_for_game(game_id: str, game_date, con) -> int:
    """Fetch and upsert box scores for one game.

    Returns the number of player rows written (0 on complete failure).
    Raises ValueError / API exceptions — caller decides whether to skip.
    """
    # --- Traditional (mandatory) ---
    logger.debug("    Fetching traditional V3 for %s …", game_id)
    trad_raw = _fetch_traditional(game_id)
    time.sleep(_API_SLEEP)

    # --- Advanced (optional) ---
    logger.debug("    Fetching advanced V3 for %s …", game_id)
    adv_raw = _fetch_advanced(game_id)
    time.sleep(_API_SLEEP)

    # --- Normalise ---
    trad = _normalize_traditional(trad_raw, game_date)
    if trad.empty:
        raise ValueError(f"Traditional normalisation produced empty DataFrame for {game_id}")

    # --- Merge advanced ---
    if adv_raw is not None:
        adv = _normalize_advanced(adv_raw)
        # adv may not have all players; left-join so trad rows are never dropped
        df = trad.merge(adv, on=["player_id", "game_id"], how="left")
    else:
        df = trad.copy()

    # --- Final cleanup ---
    df = df[df["player_id"].notna()].copy()
    if df.empty:
        raise ValueError(f"No valid player rows after normalisation for {game_id}")
    df["player_id"] = df["player_id"].astype(int)
    df["game_id"]   = df["game_id"].astype(str)

    # team_id: cast to int where present, leave NaN as None
    if "team_id" in df.columns:
        df["team_id"] = pd.to_numeric(df["team_id"], errors="coerce").astype("Int64")

    # --- Upsert ---
    _upsert_player_stubs(df, con)
    # Do NOT include 'id' — sequence handles it
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    upsert_dataframe(df, "player_game_stats", ["player_id", "game_id"], con=con)
    return len(df)


# ---------------------------------------------------------------------------
# Missing-game resolver
# ---------------------------------------------------------------------------

def get_missing_game_ids(con, seasons: list[str] | None = None) -> list[tuple[str, object]]:
    """Return (game_id, game_date) tuples for Final games with no box score rows.

    Parameters
    ----------
    seasons : list[str], optional
        If provided, restrict to these season strings (e.g. ["2023-24"]).
    """
    season_filter = ""
    params: list = []
    if seasons:
        placeholders = ", ".join("?" * len(seasons))
        season_filter = f"AND g.season IN ({placeholders})"
        params = list(seasons)

    sql = f"""
        SELECT g.game_id, g.game_date
        FROM games g
        LEFT JOIN (
            SELECT DISTINCT game_id FROM player_game_stats
        ) p ON g.game_id = p.game_id
        WHERE p.game_id IS NULL
          AND g.status = 'Final'
          {season_filter}
        ORDER BY g.game_date ASC
    """
    df = con.execute(sql, params).df()
    return list(zip(df["game_id"], df["game_date"]))


# ---------------------------------------------------------------------------
# Batch ingestion
# ---------------------------------------------------------------------------

def ingest_all_boxscores(
    limit: int | None = None,
    seasons: list[str] | None = None,
    con=None,
) -> None:
    """Ingest box scores for all un-ingested Final games.

    Parameters
    ----------
    limit : int, optional
        Process at most this many games per run.
    seasons : list[str], optional
        Restrict to these season strings.
    con : duckdb connection, optional
    """
    _close = con is None
    if con is None:
        con = get_connection()

    try:
        missing = get_missing_game_ids(con, seasons=seasons)
        total_found = len(missing)
        logger.info("Games without box scores: %d", total_found)

        if limit:
            missing = missing[:limit]
            logger.info("Capped to first %d games (--limit %d).", len(missing), limit)

        attempted = len(missing)
        succeeded = 0
        skipped   = 0

        for i, (game_id, game_date) in enumerate(missing, 1):
            logger.info("[%d/%d] %s  (%s)", i, attempted, game_id, game_date)
            try:
                n = ingest_boxscore_for_game(game_id, game_date, con)
                logger.info("  ✓ %d player rows", n)
                succeeded += 1

            except (ConnectionResetError, TimeoutError) as exc:
                logger.warning("  ✗ Network error for %s: %s — skipping", game_id, exc)
                skipped += 1
                time.sleep(_ERROR_SLEEP)

            except Exception as exc:
                # Catch everything else: KeyError on resultSet, ValueError from
                # empty frames, JSON decode errors, HTTP 4xx/5xx, etc.
                logger.warning("  ✗ Failed for %s: %s — skipping", game_id, type(exc).__name__, exc)
                skipped += 1
                time.sleep(_ERROR_SLEEP)

        logger.info(
            "─── Summary ───  attempted=%d  succeeded=%d  skipped=%d  "
            "(total outstanding before run: %d)",
            attempted, succeeded, skipped, total_found,
        )

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
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="Ingest NBA player box scores into player_game_stats.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # First 50 games (quick smoke test)
  python src/ingest_boxscores.py --limit 50

  # Full back-fill for one season
  python src/ingest_boxscores.py --seasons 2023-24

  # Chunked back-fill (run multiple times)
  python src/ingest_boxscores.py --limit 500

Environment variables
---------------------
  NBA_API_SLEEP       seconds between successful API calls  (default 0.7)
  NBA_API_ERROR_SLEEP seconds to wait after a failed game   (default 3.0)
  NBA_API_TIMEOUT     HTTP timeout per request in seconds   (default 60)
""",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max games to process this run.",
    )
    parser.add_argument(
        "--seasons", nargs="+", default=None,
        metavar="SEASON",
        help="One or more season strings to restrict to, e.g. 2023-24 2024-25.",
    )
    args = parser.parse_args(argv)

    con = get_connection()
    init_schema(con)
    ingest_all_boxscores(limit=args.limit, seasons=args.seasons, con=con)
    con.close()


if __name__ == "__main__":
    main()
