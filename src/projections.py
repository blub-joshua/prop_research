"""
src/projections.py
──────────────────
Helper module that loads trained models and computes a projections DataFrame
for a set of players on a given date.

Used by the ``daily`` and ``analyze`` CLI commands to bridge the gap between
the feature pipeline and the EV calculator.

The main entry point is ``build_projections_for_date()``.
"""

from __future__ import annotations

import logging
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from db import get_connection
from features import build_features_for_today
from models import load_models, predict, FEATURE_COLS

load_dotenv()
logger = logging.getLogger(__name__)


def build_projections_for_date(
    player_ids: list[int],
    game_date: date | str,
    opponent_map: dict[int, int] | None = None,
    is_home_map: dict[int, int] | None = None,
    con=None,
) -> pd.DataFrame:
    """Build a projections DataFrame for the given players and date.

    This is the single entry point that the daily / analyze commands call.
    It orchestrates:
      1. Loading (or training) projection models.
      2. Building live feature vectors for each player as of ``game_date``.
      3. Running inference to get projected stats.

    Parameters
    ----------
    player_ids : list[int]
        NBA API player IDs appearing in today's props.
    game_date : date or str
        The target game date (e.g. today).
    opponent_map : dict, optional
        {player_id: opponent_team_id}.
    is_home_map : dict, optional
        {player_id: 1 or 0}.
    con : duckdb connection, optional

    Returns
    -------
    pd.DataFrame
        One row per player with columns:
        player_id, proj_points, proj_rebounds, proj_assists, proj_threepm,
        proj_points_rebounds, proj_points_assists, proj_rebounds_assists,
        proj_points_rebounds_assists, pts_std_L10, reb_std_L10, ast_std_L10,
        fg3m_std_L10, and all feature columns.
    """
    _close = con is None
    if con is None:
        con = get_connection(read_only=True)

    try:
        if isinstance(game_date, str):
            game_date = pd.Timestamp(game_date).date()

        if not player_ids:
            logger.warning("No player IDs supplied — returning empty projections.")
            return pd.DataFrame()

        # 1. Load trained models
        models = load_models()
        logger.info("Loaded %d projection models.", len(models))

        # 2. Build feature vectors for today
        features_df = build_features_for_today(
            player_ids=player_ids,
            game_date=game_date,
            opponent_map=opponent_map,
            is_home_map=is_home_map,
            con=con,
        )

        if features_df.empty:
            logger.warning(
                "No features could be built for %d players on %s.",
                len(player_ids), game_date,
            )
            return pd.DataFrame()

        logger.info(
            "Built features for %d players (requested %d).",
            len(features_df), len(player_ids),
        )

        # 3. Predict
        proj_df = predict(features_df, models)

        # Ensure std columns are present for EV calc
        for std_col in ["pts_std_L10", "reb_std_L10", "ast_std_L10", "fg3m_std_L10"]:
            if std_col not in proj_df.columns:
                proj_df[std_col] = np.nan

        return proj_df

    finally:
        if _close:
            con.close()


def resolve_context_maps(
    props_df: pd.DataFrame,
    con,
) -> tuple[dict[int, int], dict[int, int]]:
    """Derive opponent_map and is_home_map from the props DataFrame.

    Uses the ``team`` and ``opponent`` columns (abbreviation strings) in the
    props and resolves them to team IDs via the teams table.

    Parameters
    ----------
    props_df : pd.DataFrame
        Must have player_id, and optionally team, opponent columns.
    con : duckdb connection

    Returns
    -------
    (opponent_map, is_home_map) : tuple[dict, dict]
    """
    opponent_map: dict[int, int] = {}
    is_home_map: dict[int, int] = {}

    if "team" not in props_df.columns or "opponent" not in props_df.columns:
        return opponent_map, is_home_map

    # Load team abbreviation -> team_id
    teams = con.execute(
        "SELECT team_id, abbreviation FROM teams"
    ).df()
    if teams.empty:
        return opponent_map, is_home_map

    abbr_to_id = dict(zip(
        teams["abbreviation"].str.upper(),
        teams["team_id"].astype(int),
    ))

    for _, row in props_df.iterrows():
        pid = row.get("player_id")
        if pd.isna(pid):
            continue
        pid = int(pid)

        opp_abbr = str(row.get("opponent", "")).upper().strip()
        if opp_abbr in abbr_to_id:
            opponent_map[pid] = abbr_to_id[opp_abbr]

        # Heuristic: if the team column matches the home_team for today's
        # schedule, mark as home.  Without schedule data for today, we
        # default to 0 (away) since is_home has a small effect.
        is_home_map.setdefault(pid, 0)

    return opponent_map, is_home_map
