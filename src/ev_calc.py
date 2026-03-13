"""
src/ev_calc.py
──────────────
Expected Value (EV) calculation, Kelly sizing, and combo ranking for NBA
player props.

Theory
------
For a binary bet (Over/Under):
  1. Convert American odds to implied probability:
       implied_prob = 100 / (|odds| + 100)        if odds < 0
       implied_prob = odds / (odds + 100)          if odds >= 0 (positive)

  2. Remove the vig to get the "true" no-vig probability:
       Method: multiplicative (default)
       no_vig_over  = implied_over  / (implied_over + implied_under)
       no_vig_under = implied_under / (implied_over + implied_under)

  3. Estimate our modelled win probability for the bet.
       For an Over:   p_win ≈ P(actual > line)
       Approximation: use a normal CDF with mean = projection, std = historical_std.
       For an Under:  p_win = 1 - p_over

  4. EV% = (p_win * decimal_odds - 1)
       where decimal_odds = (100 / |odds| + 1) for negatives,
                          = (odds / 100 + 1)   for positives.

  5. Kelly fraction = (p_win * decimal_odds - 1) / (decimal_odds - 1)
       Recommended: use fractional Kelly (e.g. 0.25 × Kelly).

2-Leg Combo EV (independence assumption)
-----------------------------------------
  EV_combo = EV_leg1 * EV_leg2   (decimal odds multiply, assuming independence)
  In practice: decimal_combo = decimal_1 × decimal_2
               p_win_combo    = p_win_1 × p_win_2
               EV_combo       = p_win_combo × decimal_combo - 1
"""

from __future__ import annotations

import itertools
import logging
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Odds conversion utilities
# ---------------------------------------------------------------------------

def american_to_decimal(odds: float) -> float:
    """Convert American odds to decimal odds.

    Parameters
    ----------
    odds : float
        American odds (e.g. -110, +150).

    Returns
    -------
    float
        Decimal odds (e.g. 1.9091, 2.5).
    """
    if odds < 0:
        return 1 + (100 / abs(odds))
    else:
        return 1 + (odds / 100)


def decimal_to_american(decimal_odds: float) -> float:
    """Convert decimal odds to American odds.

    Parameters
    ----------
    decimal_odds : float
        e.g. 1.9091

    Returns
    -------
    float
        American odds (e.g. -110).
    """
    if decimal_odds >= 2.0:
        return (decimal_odds - 1) * 100
    else:
        return -100 / (decimal_odds - 1)


def american_to_implied_prob(odds: float) -> float:
    """Convert American odds to raw implied probability (includes vig).

    Parameters
    ----------
    odds : float

    Returns
    -------
    float
        Implied probability in [0, 1].
    """
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    else:
        return 100 / (odds + 100)


# ---------------------------------------------------------------------------
# Vig removal
# ---------------------------------------------------------------------------

def remove_vig_multiplicative(
    over_odds: float, under_odds: float
) -> tuple[float, float]:
    """Remove vig using the multiplicative method.

    Parameters
    ----------
    over_odds : float
        American odds for the Over.
    under_odds : float
        American odds for the Under.

    Returns
    -------
    (no_vig_over_prob, no_vig_under_prob) : tuple[float, float]
        Fair probabilities that sum to 1.0.
    """
    p_over  = american_to_implied_prob(over_odds)
    p_under = american_to_implied_prob(under_odds)
    total   = p_over + p_under
    return p_over / total, p_under / total


def remove_vig_additive(
    over_odds: float, under_odds: float
) -> tuple[float, float]:
    """Remove vig using the additive (equal-share) method.

    Each side's probability is reduced by half the overround.
    """
    p_over  = american_to_implied_prob(over_odds)
    p_under = american_to_implied_prob(under_odds)
    overround = (p_over + p_under) - 1.0
    return p_over - overround / 2, p_under - overround / 2


# ---------------------------------------------------------------------------
# Model win probability estimation
# ---------------------------------------------------------------------------

def estimate_win_prob_normal(
    projection: float,
    line: float,
    std: float,
    side: str = "over",
) -> float:
    """Estimate P(actual > line) or P(actual < line) using a normal approximation.

    Parameters
    ----------
    projection : float
        Model's point estimate (mean of the distribution).
    line : float
        The prop line.
    std : float
        Estimated standard deviation of the player's performance.
        Tip: use rolling L10 std from features.
    side : str
        ``"over"`` or ``"under"``.

    Returns
    -------
    float
        Estimated win probability in (0, 1).
    """
    if std <= 0:
        std = max(std, 1e-6)
    # P(X > line) where X ~ N(projection, std)
    p_over = 1 - norm.cdf(line, loc=projection, scale=std)
    if side == "over":
        return float(p_over)
    else:
        return float(1 - p_over)


# ---------------------------------------------------------------------------
# EV computation
# ---------------------------------------------------------------------------

def compute_ev(
    p_win: float,
    decimal_odds: float,
) -> float:
    """Compute expected value as a fraction of stake.

    Parameters
    ----------
    p_win : float
        Estimated probability of winning.
    decimal_odds : float
        Decimal (European) odds for the bet.

    Returns
    -------
    float
        EV fraction, e.g. 0.05 means +5% EV.
        Negative values indicate a losing bet.
    """
    return p_win * decimal_odds - 1.0


def kelly_fraction(p_win: float, decimal_odds: float) -> float:
    """Compute the full Kelly criterion fraction.

    Parameters
    ----------
    p_win : float
    decimal_odds : float

    Returns
    -------
    float
        Kelly fraction (proportion of bankroll to wager).
        Clamped to [0, 1].  Apply your own fractional multiplier.
    """
    b = decimal_odds - 1.0  # net odds
    if b <= 0:
        return 0.0
    q = 1 - p_win
    kelly = (b * p_win - q) / b
    return float(np.clip(kelly, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Per-prop EV enrichment
# ---------------------------------------------------------------------------

def enrich_props_with_ev(
    props_df: pd.DataFrame,
    projections_df: pd.DataFrame,
    std_col_map: dict[str, str] | None = None,
    vig_removal: str = "multiplicative",
) -> pd.DataFrame:
    """Merge projections into props and compute EV for each row.

    Parameters
    ----------
    props_df : pd.DataFrame
        Normalised props with columns: player_id, market, line, over_odds,
        under_odds.  Each row represents one prop line (Over and Under
        on separate rows, OR combined on one row with both odds).
    projections_df : pd.DataFrame
        Must contain: player_id, proj_{market} columns, and std columns
        for each market (e.g. pts_std_L10).
    std_col_map : dict, optional
        Mapping of market → std column name in projections_df.
        Defaults to: {"points": "pts_std_L10", ...}
    vig_removal : str
        ``"multiplicative"`` or ``"additive"``.

    Returns
    -------
    pd.DataFrame
        props_df with added columns:
        projection, model_p_over, model_p_under,
        no_vig_p_over, no_vig_p_under,
        ev_over, ev_under,
        kelly_over, kelly_under,
        best_side, best_ev.
    """
    _DEFAULT_STD_MAP = {
        "points":                  "pts_std_L10",
        "rebounds":                "reb_std_L10",
        "assists":                 "ast_std_L10",
        "threepm":                 "fg3m_std_L10",
        # Composite stds — approximate with Pythagorean combination
        "points_rebounds":         None,
        "points_assists":          None,
        "rebounds_assists":        None,
        "points_rebounds_assists": None,
    }
    std_col_map = std_col_map or _DEFAULT_STD_MAP

    # TODO: full implementation
    #
    # High-level steps:
    #   1. Merge projections_df into props_df on player_id.
    #   2. For each row:
    #      a. Pull projection = projections_df[f"proj_{market}"]
    #      b. Pull std from std_col_map[market] (or approximate for composites)
    #      c. Compute model_p_over = estimate_win_prob_normal(proj, line, std, "over")
    #      d. Remove vig: no_vig_p_over, no_vig_p_under = remove_vig_*(over_odds, under_odds)
    #         (The no_vig probs are the book's estimate; model prob is ours.)
    #      e. Compute decimal_over = american_to_decimal(over_odds)
    #      f. ev_over = compute_ev(model_p_over, decimal_over)
    #      g. kelly_over = kelly_fraction(model_p_over, decimal_over)
    #      h. Repeat for under.
    #      i. best_side = "over" if ev_over > ev_under else "under"
    #         best_ev   = max(ev_over, ev_under)
    #   3. Return enriched DataFrame.
    raise NotImplementedError("enrich_props_with_ev() not yet implemented")


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def rank_single_props(
    ev_df: pd.DataFrame,
    ev_threshold: float = 0.0,
    top_n: int = 10,
) -> pd.DataFrame:
    """Rank individual props by best_ev descending.

    Parameters
    ----------
    ev_df : pd.DataFrame
        Output of enrich_props_with_ev().
    ev_threshold : float
        Minimum EV% to include (e.g. 0.03 for 3%).
    top_n : int
        Return at most this many rows.

    Returns
    -------
    pd.DataFrame
        Filtered and sorted DataFrame.
    """
    df = ev_df[ev_df["best_ev"] >= ev_threshold].copy()
    df = df.sort_values("best_ev", ascending=False).head(top_n)
    df["rank"] = range(1, len(df) + 1)
    return df.reset_index(drop=True)


def rank_combos(
    ev_df: pd.DataFrame,
    n_legs: int = 2,
    ev_threshold: float = 0.0,
    top_n: int = 5,
) -> pd.DataFrame:
    """Find the best N-leg parlays by combined EV (assuming independence).

    Only positive-EV single legs are considered as combo candidates to avoid
    dragging EV down.

    Parameters
    ----------
    ev_df : pd.DataFrame
        Output of enrich_props_with_ev() with best_ev and best_side columns.
    n_legs : int
        Number of legs per combo (currently 2 supported).
    ev_threshold : float
        Minimum EV of each individual leg to qualify.
    top_n : int
        Return the top N combos.

    Returns
    -------
    pd.DataFrame
        Columns: leg_1_desc, leg_2_desc, combo_decimal_odds, combo_ev, combo_kelly.
    """
    # Filter to positive-EV legs
    candidates = ev_df[ev_df["best_ev"] >= ev_threshold].copy()

    if len(candidates) < n_legs:
        logger.warning(
            "Only %d qualifying legs for a %d-leg combo; returning empty.",
            len(candidates), n_legs,
        )
        return pd.DataFrame()

    combo_rows = []
    for combo in itertools.combinations(candidates.itertuples(index=False), n_legs):
        # Compute joint probability and decimal odds
        p_win_joint   = 1.0
        decimal_joint = 1.0

        for leg in combo:
            side = leg.best_side
            if side == "over":
                p_win_joint   *= american_to_implied_prob(leg.over_odds)   # placeholder — use model_p_over
                decimal_joint *= american_to_decimal(leg.over_odds)
            else:
                p_win_joint   *= american_to_implied_prob(leg.under_odds)
                decimal_joint *= american_to_decimal(leg.under_odds)
            # TODO: replace american_to_implied_prob above with model_p_{side}
            #       once enrich_props_with_ev() is implemented.

        ev_combo    = compute_ev(p_win_joint, decimal_joint)
        kelly_combo = kelly_fraction(p_win_joint, decimal_joint)

        combo_rows.append({
            "legs":             [f"{leg.player_name} {leg.market} {leg.best_side} {leg.line}" for leg in combo],
            "combo_decimal":    round(decimal_joint, 3),
            "combo_ev":         round(ev_combo, 4),
            "combo_kelly":      round(kelly_combo, 4),
        })

    df_combos = pd.DataFrame(combo_rows)
    df_combos = df_combos.sort_values("combo_ev", ascending=False).head(top_n)
    df_combos["rank"] = range(1, len(df_combos) + 1)
    return df_combos.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Kelly bankroll sizing display
# ---------------------------------------------------------------------------

def kelly_bet_size(
    kelly_fraction_val: float,
    bankroll: float,
    max_fraction: float = 0.05,
    kelly_multiplier: float = 0.25,
) -> float:
    """Convert a Kelly fraction to a dollar bet size.

    Parameters
    ----------
    kelly_fraction_val : float
        Full Kelly fraction from kelly_fraction().
    bankroll : float
        Total bankroll in dollars.
    max_fraction : float
        Hard cap on fraction of bankroll per bet (default 5%).
    kelly_multiplier : float
        Fractional Kelly multiplier (default 0.25 = quarter Kelly).

    Returns
    -------
    float
        Suggested bet size in dollars.
    """
    adjusted = kelly_fraction_val * kelly_multiplier
    capped    = min(adjusted, max_fraction)
    return round(bankroll * capped, 2)
