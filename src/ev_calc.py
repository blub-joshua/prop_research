"""
src/ev_calc.py
──────────────
Expected Value (EV) calculation, Kelly sizing, and combo ranking for NBA
player props.

v2 improvements
───────────────
  • Quantile-based win probability estimation (replaces crude Normal CDF).
  • Isotonic calibration applied when a calibrator is available.
  • IQR-derived std replaces rolling L10 std for uncertainty estimation.
  • Composite market uncertainty uses Pythagorean combination of
    per-stat IQR-derived stds.
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
    """Convert American odds to decimal odds."""
    if odds < 0:
        return 1 + (100 / abs(odds))
    else:
        return 1 + (odds / 100)


def decimal_to_american(decimal_odds: float) -> float:
    """Convert decimal odds to American odds."""
    if decimal_odds >= 2.0:
        return (decimal_odds - 1) * 100
    else:
        return -100 / (decimal_odds - 1)


def american_to_implied_prob(odds: float) -> float:
    """Convert American odds to raw implied probability (includes vig)."""
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
    """Remove vig using the multiplicative method."""
    p_over  = american_to_implied_prob(over_odds)
    p_under = american_to_implied_prob(under_odds)
    total   = p_over + p_under
    return p_over / total, p_under / total


def remove_vig_additive(
    over_odds: float, under_odds: float
) -> tuple[float, float]:
    """Remove vig using the additive (equal-share) method."""
    p_over  = american_to_implied_prob(over_odds)
    p_under = american_to_implied_prob(under_odds)
    overround = (p_over + p_under) - 1.0
    return p_over - overround / 2, p_under - overround / 2


# ---------------------------------------------------------------------------
# Win probability estimation
# ---------------------------------------------------------------------------

def estimate_win_prob_normal(
    projection: float,
    line: float,
    std: float,
    side: str = "over",
) -> float:
    """Estimate P(actual > line) or P(actual < line) using Normal CDF.

    Fallback method when quantile models are not available.
    """
    if std <= 0:
        std = max(std, 1e-6)
    p_over = 1 - norm.cdf(line, loc=projection, scale=std)
    if side == "over":
        return float(p_over)
    else:
        return float(1 - p_over)


def estimate_win_prob_quantile(
    line: float,
    quantile_preds: dict[float, float],
    mean_pred: float,
    est_std: float,
    side: str = "over",
) -> float:
    """Estimate P(over/under) using quantile regression predictions.

    Uses linear interpolation between quantile predictions to estimate
    the CDF at the line, giving a more accurate probability than the
    Normal assumption.

    Parameters
    ----------
    line : float
        The prop line.
    quantile_preds : dict[float, float]
        {alpha: predicted_value} from quantile models.
    mean_pred : float
        Mean model prediction (used as fallback).
    est_std : float
        Estimated std from IQR or rolling.
    side : str
        "over" or "under".

    Returns
    -------
    float
        Estimated probability in (0.01, 0.99).
    """
    if not quantile_preds:
        # Fallback to normal
        return estimate_win_prob_normal(mean_pred, line, est_std, side)

    sorted_alphas = sorted(quantile_preds.keys())
    sorted_vals = [quantile_preds[a] for a in sorted_alphas]

    # Enforce monotonicity
    for i in range(1, len(sorted_vals)):
        if sorted_vals[i] < sorted_vals[i - 1]:
            sorted_vals[i] = sorted_vals[i - 1]

    # Estimate CDF at line via interpolation
    if line <= sorted_vals[0]:
        # Line is below the lowest quantile — high P(over)
        # Use exponential extrapolation from the lowest quantile
        if est_std > 0:
            z = (sorted_vals[0] - line) / est_std
            cdf_at_line = sorted_alphas[0] * np.exp(-z)
        else:
            cdf_at_line = sorted_alphas[0] * 0.5
        p_over = float(np.clip(1.0 - cdf_at_line, 0.01, 0.99))

    elif line >= sorted_vals[-1]:
        # Line is above the highest quantile — low P(over)
        if est_std > 0:
            z = (line - sorted_vals[-1]) / est_std
            cdf_at_line = sorted_alphas[-1] + (1 - sorted_alphas[-1]) * (1 - np.exp(-z))
        else:
            cdf_at_line = sorted_alphas[-1] + (1 - sorted_alphas[-1]) * 0.5
        p_over = float(np.clip(1.0 - cdf_at_line, 0.01, 0.99))

    else:
        # Interpolate between quantiles
        for i in range(len(sorted_vals) - 1):
            if sorted_vals[i] <= line <= sorted_vals[i + 1]:
                span = sorted_vals[i + 1] - sorted_vals[i]
                if span <= 0:
                    frac = 0.5
                else:
                    frac = (line - sorted_vals[i]) / span
                cdf_at_line = sorted_alphas[i] + frac * (sorted_alphas[i + 1] - sorted_alphas[i])
                p_over = float(np.clip(1.0 - cdf_at_line, 0.01, 0.99))
                break
        else:
            p_over = estimate_win_prob_normal(mean_pred, line, est_std, "over")

    if side == "over":
        return p_over
    else:
        return 1.0 - p_over


# ---------------------------------------------------------------------------
# EV computation
# ---------------------------------------------------------------------------

def compute_ev(
    p_win: float,
    decimal_odds: float,
) -> float:
    """Compute expected value as a fraction of stake."""
    return p_win * decimal_odds - 1.0


def kelly_fraction(p_win: float, decimal_odds: float) -> float:
    """Compute the full Kelly criterion fraction."""
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    q = 1 - p_win
    kelly = (b * p_win - q) / b
    return float(np.clip(kelly, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Per-prop EV enrichment (v2 — quantile + calibration)
# ---------------------------------------------------------------------------

def enrich_props_with_ev(
    props_df: pd.DataFrame,
    projections_df: pd.DataFrame,
    models: dict | None = None,
    std_col_map: dict[str, str] | None = None,
    vig_removal: str = "multiplicative",
) -> pd.DataFrame:
    """Merge projections into props and compute EV for each row.

    v2 improvements:
      - Uses quantile predictions for probability estimation when available.
      - Applies isotonic calibration when calibrator is loaded.
      - Uses IQR-derived std instead of rolling L10 std.
      - Falls back to Normal CDF when quantile models are not available.

    Parameters
    ----------
    props_df : pd.DataFrame
        Normalised props.
    projections_df : pd.DataFrame
        Must contain: player_id, proj_{market} columns, and optionally
        quantile prediction columns ({target}_q{level}) and
        {target}_iqr_std columns.
    models : dict, optional
        Full models dict (used for calibrators). If None, no calibration.
    std_col_map : dict, optional
        Mapping of market -> std column name.
    vig_removal : str
        "multiplicative" or "additive".

    Returns
    -------
    pd.DataFrame
        props_df with EV columns added.
    """
    _DEFAULT_STD_MAP: dict[str, str | None] = {
        "points":                  "points_iqr_std",
        "rebounds":                "rebounds_iqr_std",
        "assists":                 "assists_iqr_std",
        "threepm":                 "threepm_iqr_std",
        "points_rebounds":         None,
        "points_assists":          None,
        "rebounds_assists":        None,
        "points_rebounds_assists": None,
    }
    # Fallback to L10 std if IQR std not available
    _FALLBACK_STD_MAP: dict[str, str] = {
        "points":   "pts_std_L10",
        "rebounds":  "reb_std_L10",
        "assists":   "ast_std_L10",
        "threepm":   "fg3m_std_L10",
    }
    std_col_map = std_col_map or _DEFAULT_STD_MAP

    _PROJ_COL: dict[str, str] = {
        "points":                  "proj_points",
        "rebounds":                "proj_rebounds",
        "assists":                 "proj_assists",
        "threepm":                 "proj_threepm",
        "points_rebounds":         "proj_points_rebounds",
        "points_assists":          "proj_points_assists",
        "rebounds_assists":        "proj_rebounds_assists",
        "points_rebounds_assists": "proj_points_rebounds_assists",
    }

    # Quantile column prefix map
    _QUANTILE_PREFIX: dict[str, str] = {
        "points":   "points",
        "rebounds":  "rebounds",
        "assists":   "assists",
        "threepm":   "threepm",
    }

    _COMPOSITE_STD_PARTS: dict[str, list[str]] = {
        "points_rebounds":         ["points_iqr_std", "rebounds_iqr_std"],
        "points_assists":          ["points_iqr_std", "assists_iqr_std"],
        "rebounds_assists":        ["rebounds_iqr_std", "assists_iqr_std"],
        "points_rebounds_assists": ["points_iqr_std", "rebounds_iqr_std", "assists_iqr_std"],
    }
    # Fallback composite stds (L10 rolling)
    _COMPOSITE_STD_FALLBACK: dict[str, list[str]] = {
        "points_rebounds":         ["pts_std_L10", "reb_std_L10"],
        "points_assists":          ["pts_std_L10", "ast_std_L10"],
        "rebounds_assists":        ["reb_std_L10", "ast_std_L10"],
        "points_rebounds_assists": ["pts_std_L10", "reb_std_L10", "ast_std_L10"],
    }

    _remove_vig = (
        remove_vig_multiplicative if vig_removal == "multiplicative"
        else remove_vig_additive
    )

    _DEFAULT_STD_FALLBACK: dict[str, float] = {
        "points": 6.0, "rebounds": 2.5, "assists": 2.0, "threepm": 1.2,
    }

    # ── 1. Merge projections ─────────────────────────────────────────────
    df = props_df.copy()

    if projections_df is not None and not projections_df.empty:
        proj_cols_to_keep = ["player_id"]
        proj_cols_to_keep += [c for c in projections_df.columns if c.startswith("proj_")]
        proj_cols_to_keep += [c for c in projections_df.columns if c.endswith("_std_L10")]
        proj_cols_to_keep += [c for c in projections_df.columns if c.endswith("_iqr_std")]
        proj_cols_to_keep += [c for c in projections_df.columns if "_q" in c and c[0] != "_"]
        proj_cols_to_keep = list(dict.fromkeys(proj_cols_to_keep))
        available = [c for c in proj_cols_to_keep if c in projections_df.columns]
        df = df.merge(projections_df[available], on="player_id", how="left")
    else:
        logger.warning("No projections supplied — EV columns will be NaN.")

    # ── 2. Per-row EV computation ─────────────────────────────────────────
    out_projection   = []
    out_std          = []
    out_mp_over      = []
    out_mp_under     = []
    out_nv_over      = []
    out_nv_under     = []
    out_ev_over      = []
    out_ev_under     = []
    out_kelly_over   = []
    out_kelly_under  = []
    out_best_side    = []
    out_best_ev      = []
    out_kelly_best   = []
    out_method       = []  # track which method was used

    for _, row in df.iterrows():
        market = str(row.get("market", ""))
        line   = float(row.get("line", 0))
        over_odds  = float(row.get("over_odds", -110))
        under_odds = float(row.get("under_odds", -110))

        # --- projection ---
        proj_col = _PROJ_COL.get(market)
        projection = float(row.get(proj_col, np.nan)) if proj_col and proj_col in df.columns else np.nan

        # --- std ---
        std_col = std_col_map.get(market)
        std_val = np.nan

        if std_col and std_col in df.columns:
            std_val = float(row.get(std_col, np.nan))

        # Try fallback L10 std
        if (pd.isna(std_val) or std_val <= 0) and market in _FALLBACK_STD_MAP:
            fallback_col = _FALLBACK_STD_MAP[market]
            if fallback_col in df.columns:
                std_val = float(row.get(fallback_col, np.nan))

        # Composite market std
        if (pd.isna(std_val) or std_val <= 0) and market in _COMPOSITE_STD_PARTS:
            parts = []
            # Try IQR stds first
            for sc in _COMPOSITE_STD_PARTS[market]:
                v = float(row.get(sc, np.nan)) if sc in df.columns else np.nan
                if pd.isna(v) or v <= 0:
                    # Try L10 fallback
                    fallback_parts = _COMPOSITE_STD_FALLBACK.get(market, [])
                    idx = _COMPOSITE_STD_PARTS[market].index(sc)
                    if idx < len(fallback_parts) and fallback_parts[idx] in df.columns:
                        v = float(row.get(fallback_parts[idx], np.nan))
                if pd.isna(v) or v <= 0:
                    base = sc.replace("_iqr_std", "")
                    v = _DEFAULT_STD_FALLBACK.get(base, 3.0)
                parts.append(v)
            std_val = float(np.sqrt(sum(s ** 2 for s in parts)))

        # Final fallback
        if pd.isna(std_val) or std_val <= 0:
            base_market = market.split("_")[0] if "_" not in market else market
            std_val = _DEFAULT_STD_FALLBACK.get(base_market,
                max(projection * 0.3, 1.0) if not pd.isna(projection) else 5.0)

        out_projection.append(projection)
        out_std.append(std_val)

        if pd.isna(projection):
            out_mp_over.append(np.nan); out_mp_under.append(np.nan)
            out_nv_over.append(np.nan); out_nv_under.append(np.nan)
            out_ev_over.append(np.nan); out_ev_under.append(np.nan)
            out_kelly_over.append(np.nan); out_kelly_under.append(np.nan)
            out_best_side.append(None); out_best_ev.append(np.nan)
            out_kelly_best.append(np.nan); out_method.append(None)
            continue

        # --- Win probability estimation ---
        # Try quantile-based estimation first
        base_market = market if market in _QUANTILE_PREFIX else None
        quantile_preds = {}
        method = "normal"

        if base_market:
            prefix = _QUANTILE_PREFIX[base_market]
            for alpha in [0.10, 0.25, 0.50, 0.75, 0.90]:
                qcol = f"{prefix}_q{int(alpha*100)}"
                if qcol in df.columns:
                    qval = float(row.get(qcol, np.nan))
                    if not pd.isna(qval):
                        quantile_preds[alpha] = qval

        if len(quantile_preds) >= 3:
            mp_over = estimate_win_prob_quantile(
                line, quantile_preds, projection, std_val, "over"
            )
            method = "quantile"
        else:
            mp_over = estimate_win_prob_normal(projection, line, std_val, "over")

        # --- Apply calibration if available ---
        if models and base_market:
            cal_key = f"{base_market}_calibrator"
            if cal_key in models:
                calibrator = models[cal_key]
                try:
                    mp_over = float(calibrator.predict([mp_over])[0])
                    mp_over = float(np.clip(mp_over, 0.01, 0.99))
                    method += "+calibrated"
                except Exception:
                    pass

        mp_under = 1.0 - mp_over

        # No-vig probabilities
        nv_over, nv_under = _remove_vig(over_odds, under_odds)

        # Decimal odds
        dec_over  = american_to_decimal(over_odds)
        dec_under = american_to_decimal(under_odds)

        # EV
        ev_over  = compute_ev(mp_over, dec_over)
        ev_under = compute_ev(mp_under, dec_under)

        # Kelly
        k_over  = kelly_fraction(mp_over, dec_over)
        k_under = kelly_fraction(mp_under, dec_under)

        # Best side
        if ev_over >= ev_under:
            best_side, best_ev, k_best = "over", ev_over, k_over
        else:
            best_side, best_ev, k_best = "under", ev_under, k_under

        out_mp_over.append(mp_over);   out_mp_under.append(mp_under)
        out_nv_over.append(nv_over);   out_nv_under.append(nv_under)
        out_ev_over.append(ev_over);   out_ev_under.append(ev_under)
        out_kelly_over.append(k_over); out_kelly_under.append(k_under)
        out_best_side.append(best_side)
        out_best_ev.append(best_ev)
        out_kelly_best.append(k_best)
        out_method.append(method)

    # ── 3. Attach computed columns ────────────────────────────────────────
    df["projection"]     = out_projection
    df["std"]            = out_std
    df["model_p_over"]   = out_mp_over
    df["model_p_under"]  = out_mp_under
    df["no_vig_p_over"]  = out_nv_over
    df["no_vig_p_under"] = out_nv_under
    df["ev_over"]        = out_ev_over
    df["ev_under"]       = out_ev_under
    df["kelly_over"]     = out_kelly_over
    df["kelly_under"]    = out_kelly_under
    df["best_side"]      = out_best_side
    df["best_ev"]        = out_best_ev
    df["kelly_best"]     = out_kelly_best
    df["prob_method"]    = out_method

    logger.info("EV enrichment complete: %d rows.", len(df))
    return df


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def rank_single_props(
    ev_df: pd.DataFrame,
    ev_threshold: float = 0.0,
    top_n: int = 10,
) -> pd.DataFrame:
    """Rank individual props by best_ev descending."""
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
    """Find the best N-leg parlays by combined EV (assuming independence)."""
    candidates = ev_df[ev_df["best_ev"] >= ev_threshold].copy()

    if len(candidates) < n_legs:
        logger.warning(
            "Only %d qualifying legs for a %d-leg combo; returning empty.",
            len(candidates), n_legs,
        )
        return pd.DataFrame()

    combo_rows = []
    for combo in itertools.combinations(candidates.itertuples(index=False), n_legs):
        p_win_joint   = 1.0
        decimal_joint = 1.0

        for leg in combo:
            side = leg.best_side
            if side == "over":
                p_leg = getattr(leg, "model_p_over", None)
                if p_leg is None or pd.isna(p_leg):
                    p_leg = american_to_implied_prob(leg.over_odds)
                decimal_joint *= american_to_decimal(leg.over_odds)
            else:
                p_leg = getattr(leg, "model_p_under", None)
                if p_leg is None or pd.isna(p_leg):
                    p_leg = american_to_implied_prob(leg.under_odds)
                decimal_joint *= american_to_decimal(leg.under_odds)
            p_win_joint *= p_leg

        ev_combo    = compute_ev(p_win_joint, decimal_joint)
        kelly_combo = kelly_fraction(p_win_joint, decimal_joint)

        combo_rows.append({
            "legs":             [f"{leg.player_name} {leg.market} {leg.best_side} {leg.line}" for leg in combo],
            "p_combo":          round(p_win_joint, 6),
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
    """Convert a Kelly fraction to a dollar bet size."""
    adjusted = kelly_fraction_val * kelly_multiplier
    capped    = min(adjusted, max_fraction)
    return round(bankroll * capped, 2)
