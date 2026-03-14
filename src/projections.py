"""
src/projections.py
──────────────────
Helper module that loads trained models and computes a projections DataFrame
for a set of players on a given date.

v2: now uses the full model stack (mean + quantile + minutes).
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

    Returns a DataFrame with:
      - proj_minutes, proj_points, proj_rebounds, proj_assists, proj_threepm
      - composite proj_* columns
      - quantile predictions: {target}_q{level}
      - IQR-based std: {target}_iqr_std
      - std columns: pts_std_L10, reb_std_L10, ast_std_L10, fg3m_std_L10
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

        # 1. Load trained models (full stack)
        models = load_models()
        logger.info("Loaded models: %s", [k for k in models.keys() if not k.endswith("_quantiles") and not k.endswith("_calibrator")])

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

        # 3. Predict (mean + quantile + minutes)
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
    """Derive opponent_map and is_home_map from the props DataFrame."""
    opponent_map: dict[int, int] = {}
    is_home_map: dict[int, int] = {}

    if "team" not in props_df.columns or "opponent" not in props_df.columns:
        return opponent_map, is_home_map

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

        is_home_map.setdefault(pid, 0)

    return opponent_map, is_home_map
