"""
src/models.py
─────────────
Train, evaluate, save, and load projection models for PTS, REB, AST, FG3M,
and MINUTES.

Model design
------------
  • One MEAN model per stat target (point estimate).
  • One set of QUANTILE models per stat target (10th, 25th, 50th, 75th, 90th
    percentiles) for distributional uncertainty estimation.
  • A dedicated MINUTES model provides predicted minutes as a feature for
    stat models.
  • Algorithm: LightGBM (preferred) with sklearn GBM fallback.
  • Train / validation split: train on all seasons EXCEPT the most recent;
    validate on the most recent.
  • Calibration: after quantile models are trained, isotonic regression maps
    raw model P(over) to calibrated probabilities using validation-set
    outcomes.

Run directly:
    python src/models.py [--force-retrain] [--backend lightgbm|gbm|ridge]

Or via CLI:
    python src/cli.py train [--force-retrain]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from db import get_connection, init_schema, query_df

load_dotenv()
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MODELS_DIR   = _PROJECT_ROOT / os.getenv("MODELS_DIR", "models")
_EVAL_DIR     = _MODELS_DIR / "eval"
_MODELS_DIR.mkdir(parents=True, exist_ok=True)
_EVAL_DIR.mkdir(parents=True, exist_ok=True)
_FORCE_RETRAIN = os.getenv("FORCE_RETRAIN", "false").lower() == "true"

# ---------------------------------------------------------------------------
# Targets and features
# ---------------------------------------------------------------------------

TARGET_COLUMNS: dict[str, str] = {
    "points":   "points",
    "rebounds": "rebounds",
    "assists":  "assists",
    "threepm":  "fg3m",
}

# Minutes is trained separately so it can feed into stat models
MINUTES_TARGET = "minutes"

COMPOSITE_TARGETS: dict[str, list[str]] = {
    "points_rebounds":           ["points", "rebounds"],
    "points_assists":            ["points", "assists"],
    "rebounds_assists":          ["rebounds", "assists"],
    "points_rebounds_assists":   ["points", "rebounds", "assists"],
}

# Feature columns expected from player_features (must be numeric)
FEATURE_COLS: list[str] = [
    # Rolling averages
    "pts_avg_L5",   "pts_avg_L10",   "pts_avg_season",
    "reb_avg_L5",   "reb_avg_L10",   "reb_avg_season",
    "ast_avg_L5",   "ast_avg_L10",   "ast_avg_season",
    "fg3m_avg_L5",  "fg3m_avg_L10",  "fg3m_avg_season",
    "min_avg_L5",   "min_avg_L10",   "min_avg_season",
    # Rolling std-devs
    "pts_std_L5",   "pts_std_L10",
    "reb_std_L5",   "reb_std_L10",
    "ast_std_L5",   "ast_std_L10",
    "fg3m_std_L5",  "fg3m_std_L10",
    # Home/away
    "pts_avg_home", "pts_avg_away",
    "reb_avg_home", "reb_avg_away",
    "ast_avg_home", "ast_avg_away",
    # Opponent defense
    "opp_pts_allowed_avg", "opp_reb_allowed_avg",
    "opp_ast_allowed_avg", "opp_fg3m_allowed_avg",
    # Context
    "is_home", "days_rest", "is_back_to_back",
    "games_played_season", "injury_severity",
    "pace_L5",
]

# Extended features: adds projected minutes as input for stat models
FEATURE_COLS_WITH_MINUTES: list[str] = FEATURE_COLS + ["proj_minutes"]

# Quantile levels for distributional modeling
QUANTILE_LEVELS: list[float] = [0.10, 0.25, 0.50, 0.75, 0.90]


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

def _build_pipeline(backend: str, **hparams) -> Pipeline:
    """Build a sklearn Pipeline with imputation, scaling, and a regressor."""
    if backend == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
            defaults = dict(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
            defaults.update(hparams)
            regressor = LGBMRegressor(**defaults)
            logger.info("  Using LightGBM backend.")
        except ImportError:
            logger.warning("LightGBM not installed; falling back to GBM.")
            backend = "gbm"

    if backend == "gbm":
        defaults = dict(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            min_samples_leaf=20,
            subsample=0.8,
            random_state=42,
        )
        defaults.update(hparams)
        regressor = GradientBoostingRegressor(**defaults)
        logger.info("  Using GradientBoostingRegressor backend.")

    if backend == "ridge":
        defaults = dict(alpha=1.0)
        defaults.update(hparams)
        regressor = Ridge(**defaults)
        logger.info("  Using Ridge backend.")

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   regressor),
    ])


def _build_quantile_pipeline(backend: str, alpha: float, **hparams) -> Pipeline:
    """Build a pipeline for quantile regression at a given alpha level."""
    if backend == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
            defaults = dict(
                objective="quantile",
                alpha=alpha,
                n_estimators=400,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
            defaults.update(hparams)
            regressor = LGBMRegressor(**defaults)
        except ImportError:
            logger.warning("LightGBM not installed; quantile regression requires it.")
            return None
    elif backend == "gbm":
        defaults = dict(
            loss="quantile",
            alpha=alpha,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            min_samples_leaf=20,
            subsample=0.8,
            random_state=42,
        )
        defaults.update(hparams)
        regressor = GradientBoostingRegressor(**defaults)
    else:
        logger.warning("Quantile regression not supported for backend '%s'.", backend)
        return None

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   regressor),
    ])


# ---------------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------------

def load_training_data(target_stat: str, con, feature_cols=None) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Join player_features with player_game_stats to get (X, y, season)."""
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    sql = f"""
        SELECT
            pf.*,
            pgs.{target_stat}        AS target,
            pgs.did_not_play
        FROM player_features pf
        JOIN player_game_stats pgs
          ON pf.player_id = pgs.player_id
         AND pf.game_id   = pgs.game_id
        WHERE pgs.did_not_play = FALSE
          AND pgs.{target_stat} IS NOT NULL
          AND pf.pts_avg_L5 IS NOT NULL
    """
    df = con.execute(sql).df()
    logger.info("  Training rows loaded: %d", len(df))

    if df.empty:
        raise ValueError(
            f"No training rows for target '{target_stat}'. "
            "Run features.py first."
        )

    present_features = [c for c in feature_cols if c in df.columns]
    X = df[present_features].copy()
    for col in feature_cols:
        if col not in X.columns:
            X[col] = np.nan
    X = X[feature_cols]

    y       = pd.to_numeric(df["target"],  errors="coerce")
    season  = df["season"].fillna("unknown")

    valid = y.notna()
    return X[valid].reset_index(drop=True), y[valid].reset_index(drop=True), season[valid].reset_index(drop=True)


def _train_val_split(X, y, season):
    """Split into train and validation by holding out the most recent season."""
    seasons_sorted = sorted(season.unique())
    if len(seasons_sorted) < 2:
        logger.warning("  Only one season in data — using 80/20 random split.")
        n = int(len(X) * 0.8)
        return X.iloc[:n], X.iloc[n:], y.iloc[:n], y.iloc[n:]

    val_season = seasons_sorted[-1]
    train_mask = season != val_season
    val_mask   = season == val_season
    logger.info("  Train seasons: %s | Val season: %s",
                [s for s in seasons_sorted if s != val_season], val_season)
    logger.info("  Train rows: %d  |  Val rows: %d",
                train_mask.sum(), val_mask.sum())
    return (X[train_mask].reset_index(drop=True),
            X[val_mask].reset_index(drop=True),
            y[train_mask].reset_index(drop=True),
            y[val_mask].reset_index(drop=True))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _evaluate(pipeline, X_train, y_train, X_val, y_val, target: str) -> dict:
    """Compute RMSE and MAE on train and validation sets; save residual plot."""
    pred_train = pipeline.predict(X_train)
    pred_val   = pipeline.predict(X_val)

    metrics = {
        "target":   target,
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, pred_train))),
        "train_mae":  float(mean_absolute_error(y_train, pred_train)),
        "val_rmse":   float(np.sqrt(mean_squared_error(y_val, pred_val))),
        "val_mae":    float(mean_absolute_error(y_val, pred_val)),
        "n_train":    len(X_train),
        "n_val":      len(X_val),
    }
    logger.info(
        "  [%s]  Train RMSE=%.3f MAE=%.3f  |  Val RMSE=%.3f MAE=%.3f",
        target, metrics["train_rmse"], metrics["train_mae"],
        metrics["val_rmse"],  metrics["val_mae"],
    )

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        residuals = y_val.values - pred_val
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].hist(residuals, bins=40, edgecolor="k")
        axes[0].axvline(0, color="red", linestyle="--")
        axes[0].set_title(f"{target} — Residuals (val)")
        axes[0].set_xlabel("Actual − Predicted")

        axes[1].scatter(pred_val, y_val.values, alpha=0.3, s=8)
        lims = [min(pred_val.min(), y_val.min()), max(pred_val.max(), y_val.max())]
        axes[1].plot(lims, lims, "r--")
        axes[1].set_xlabel("Predicted")
        axes[1].set_ylabel("Actual")
        axes[1].set_title(f"{target} — Predicted vs Actual (val)")

        fig.suptitle(
            f"{target}  |  Val RMSE={metrics['val_rmse']:.2f}  MAE={metrics['val_mae']:.2f}",
            fontsize=12,
        )
        fig.tight_layout()
        plot_path = _EVAL_DIR / f"{target}_eval.png"
        fig.savefig(plot_path, dpi=100)
        plt.close(fig)
        logger.info("  Evaluation plot saved: %s", plot_path)
    except Exception as exc:
        logger.warning("  Could not save eval plot: %s", exc)

    return metrics


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def _log_feature_importance(pipeline: Pipeline, target: str, feature_names: list[str]) -> None:
    """Log top feature importances."""
    try:
        model = pipeline.named_steps["model"]
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_)
        else:
            return

        fi = pd.Series(importances, index=feature_names).sort_values(ascending=False)
        logger.info("  Top 10 features for %s:", target)
        for feat, imp in fi.head(10).items():
            logger.info("    %-35s %.4f", feat, imp)

        fi_path = _EVAL_DIR / f"{target}_feature_importance.csv"
        fi.to_csv(fi_path, header=["importance"])
    except Exception as exc:
        logger.debug("  Could not extract feature importance: %s", exc)


# ---------------------------------------------------------------------------
# Train single model
# ---------------------------------------------------------------------------

def train_model(
    target: str,
    backend: str = "gbm",
    con=None,
    feature_cols=None,
    **hparams,
) -> tuple[Pipeline, dict]:
    """Train a projection model for a single stat target."""
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    logger.info("Training model: target=%s  backend=%s", target, backend)
    _close = con is None
    if con is None:
        con = get_connection()

    try:
        stat_col = TARGET_COLUMNS.get(target, target)
        X, y, season = load_training_data(stat_col, con, feature_cols=feature_cols)
        X_train, X_val, y_train, y_val = _train_val_split(X, y, season)

        pipeline = _build_pipeline(backend, **hparams)
        pipeline.fit(X_train, y_train)

        metrics = _evaluate(pipeline, X_train, y_train, X_val, y_val, target)
        _log_feature_importance(pipeline, target, feature_cols)

        return pipeline, metrics
    finally:
        if _close:
            con.close()


# ---------------------------------------------------------------------------
# Train quantile models for a single target
# ---------------------------------------------------------------------------

def train_quantile_models(
    target: str,
    backend: str = "gbm",
    con=None,
    feature_cols=None,
    **hparams,
) -> dict[float, Pipeline]:
    """Train quantile regression models at QUANTILE_LEVELS for a target.

    Returns dict: {quantile_alpha: fitted_pipeline}
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    logger.info("Training quantile models: target=%s  backend=%s", target, backend)
    _close = con is None
    if con is None:
        con = get_connection()

    try:
        stat_col = TARGET_COLUMNS.get(target, target)
        X, y, season = load_training_data(stat_col, con, feature_cols=feature_cols)
        X_train, X_val, y_train, y_val = _train_val_split(X, y, season)

        quantile_models = {}
        for alpha in QUANTILE_LEVELS:
            pipeline = _build_quantile_pipeline(backend, alpha, **hparams)
            if pipeline is None:
                logger.warning("  Skipping quantile %.2f — unsupported backend.", alpha)
                continue
            pipeline.fit(X_train, y_train)
            pred_val = pipeline.predict(X_val)
            coverage = float((y_val <= pred_val).mean())
            logger.info("  Quantile %.2f  |  Val coverage=%.3f (expected ~%.2f)",
                        alpha, coverage, alpha)
            quantile_models[alpha] = pipeline

        return quantile_models
    finally:
        if _close:
            con.close()


# ---------------------------------------------------------------------------
# Calibration (isotonic regression on validation set)
# ---------------------------------------------------------------------------

def train_calibrator(
    target: str,
    mean_model: Pipeline,
    quantile_models: dict[float, Pipeline],
    backend: str = "gbm",
    con=None,
    feature_cols=None,
    minutes_model: Pipeline | None = None,
) -> IsotonicRegression | None:
    """Train an isotonic calibrator for P(over line) using validation data.

    The calibrator maps raw model-estimated P(over) to calibrated P(over)
    using historical outcomes from the validation season.

    We simulate prop lines at the player's projection ± offsets, compute
    raw P(over) from the quantile models, then compare to actual outcomes.

    Returns an IsotonicRegression object or None if insufficient data.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    logger.info("Training calibrator for target=%s", target)
    _close = con is None
    if con is None:
        con = get_connection()

    try:
        stat_col = TARGET_COLUMNS.get(target, target)
        X, y, season = load_training_data(stat_col, con, feature_cols=feature_cols)
        _, X_val, _, y_val = _train_val_split(X, y, season)

        if len(X_val) < 100:
            logger.warning("  Not enough validation data for calibration (%d rows).", len(X_val))
            return None

        # Mean model expects proj_minutes — add it if we have a minutes model
        X_val_mean = X_val.copy()
        if minutes_model is not None:
            min_pred = minutes_model.predict(X_val)
            X_val_mean["proj_minutes"] = np.clip(min_pred, 0, 48)

        # Get mean predictions and quantile predictions on val set
        mean_preds = mean_model.predict(X_val_mean)

        # Compute model-estimated P(over line) for synthetic lines
        raw_probs = []
        actual_outcomes = []
        rng = np.random.default_rng(seed=42)

        for i in range(len(X_val)):
            pred_mean = mean_preds[i]
            actual = y_val.iloc[i]

            # Get quantile predictions for uncertainty estimation
            q_preds = {}
            for alpha, qmodel in quantile_models.items():
                q_preds[alpha] = qmodel.predict(X_val.iloc[[i]])[0]

            # Estimate std from quantiles (IQR-based)
            if 0.75 in q_preds and 0.25 in q_preds:
                iqr = q_preds[0.75] - q_preds[0.25]
                est_std = max(iqr / 1.35, 0.5)  # IQR / 1.35 ≈ std for normal
            else:
                est_std = max(pred_mean * 0.25, 1.0)

            # Simulate a prop line near the prediction
            offset = rng.choice([-1.0, -0.5, 0, 0.5, 1.0])
            line = max(0.5, round(pred_mean + offset * est_std * 0.5, 1))

            # Compute raw P(over) using quantile-based distribution
            p_over = _estimate_p_over_from_quantiles(line, q_preds, pred_mean, est_std)
            raw_probs.append(p_over)
            actual_outcomes.append(1 if actual > line else 0)

        raw_probs = np.array(raw_probs)
        actual_outcomes = np.array(actual_outcomes)

        # Fit isotonic regression: raw_prob -> calibrated_prob
        calibrator = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
        calibrator.fit(raw_probs, actual_outcomes)

        # Log calibration improvement
        from sklearn.metrics import brier_score_loss
        brier_raw = brier_score_loss(actual_outcomes, raw_probs)
        calibrated_probs = calibrator.predict(raw_probs)
        brier_cal = brier_score_loss(actual_outcomes, calibrated_probs)
        logger.info("  Calibration: Brier score %.4f -> %.4f (%.1f%% improvement)",
                    brier_raw, brier_cal,
                    (1 - brier_cal / max(brier_raw, 1e-9)) * 100)

        return calibrator

    finally:
        if _close:
            con.close()


def _estimate_p_over_from_quantiles(
    line: float,
    q_preds: dict[float, float],
    mean_pred: float,
    est_std: float,
) -> float:
    """Estimate P(actual > line) using quantile predictions.

    Uses linear interpolation between quantile predictions to estimate
    the CDF at the line value, then returns 1 - CDF(line).
    """
    # Sort quantile predictions and enforce monotonicity
    sorted_alphas = sorted(q_preds.keys())
    sorted_vals = [q_preds[a] for a in sorted_alphas]

    # Enforce monotonicity (fix crossing quantiles)
    for i in range(1, len(sorted_vals)):
        if sorted_vals[i] < sorted_vals[i - 1]:
            sorted_vals[i] = sorted_vals[i - 1]

    # If line is below the lowest quantile prediction, P(over) is high
    if line <= sorted_vals[0]:
        return min(0.99, 1.0 - sorted_alphas[0] * (line / max(sorted_vals[0], 0.1)))

    # If line is above the highest quantile prediction, P(over) is low
    if line >= sorted_vals[-1]:
        return max(0.01, (1.0 - sorted_alphas[-1]) * (sorted_vals[-1] / max(line, 0.1)))

    # Interpolate between quantiles
    for i in range(len(sorted_vals) - 1):
        if sorted_vals[i] <= line <= sorted_vals[i + 1]:
            # Linear interpolation of the CDF between these two quantiles
            span = sorted_vals[i + 1] - sorted_vals[i]
            if span <= 0:
                frac = 0.5
            else:
                frac = (line - sorted_vals[i]) / span
            cdf_at_line = sorted_alphas[i] + frac * (sorted_alphas[i + 1] - sorted_alphas[i])
            return float(np.clip(1.0 - cdf_at_line, 0.01, 0.99))

    # Fallback: use mean and std with normal CDF
    from scipy.stats import norm
    return float(np.clip(1.0 - norm.cdf(line, loc=mean_pred, scale=est_std), 0.01, 0.99))


# ---------------------------------------------------------------------------
# Train / load ALL models (mean + quantile + minutes + calibration)
# ---------------------------------------------------------------------------

def train_all_models(
    backend: str | None = None,
    force: bool | None = None,
    con=None,
    **hparams,
) -> dict[str, Pipeline]:
    """Train or load all models: mean, quantile, minutes, and calibrators.

    Saves:
      models/{target}_model.pkl          — mean regression model
      models/{target}_q{int(alpha*100)}_model.pkl  — quantile models
      models/minutes_model.pkl           — minutes prediction model
      models/{target}_calibrator.pkl     — isotonic calibrator
    """
    if backend is None:
        try:
            import yaml
            cfg_path = _PROJECT_ROOT / "config.yaml"
            if cfg_path.exists():
                with cfg_path.open() as f:
                    cfg = yaml.safe_load(f)
                backend = cfg.get("models", {}).get("backend", "gbm")
            else:
                backend = "gbm"
        except Exception:
            backend = "gbm"

    force = force if force is not None else _FORCE_RETRAIN
    models: dict[str, Pipeline] = {}
    all_metrics: list[dict] = []

    _close = con is None
    if con is None:
        con = get_connection()

    try:
        # ── 1. Train MINUTES model first ─────────────────────────────────
        min_pkl = _MODELS_DIR / "minutes_model.pkl"
        if min_pkl.exists() and not force:
            logger.info("Loading cached minutes model: %s", min_pkl)
            models["minutes"] = joblib.load(min_pkl)
        else:
            logger.info("Training MINUTES model...")
            min_pipeline, min_metrics = train_model(
                "minutes", backend=backend, con=con,
                feature_cols=FEATURE_COLS, **hparams,
            )
            joblib.dump(min_pipeline, min_pkl)
            logger.info("  Saved minutes model: %s", min_pkl)
            models["minutes"] = min_pipeline
            all_metrics.append(min_metrics)

        # ── 2. Train MEAN stat models (with proj_minutes as feature) ─────
        # To use proj_minutes as a feature, we need to pre-compute it on
        # training data. We do this by predicting minutes on the full feature
        # set and adding it as a column.
        for target in TARGET_COLUMNS:
            pkl_path = _MODELS_DIR / f"{target}_model.pkl"
            if pkl_path.exists() and not force:
                logger.info("Loading cached model: %s", pkl_path)
                models[target] = joblib.load(pkl_path)
            else:
                # For stat models, add proj_minutes as an extra feature
                pipeline, metrics = _train_with_minutes(
                    target, backend, models["minutes"], con, **hparams,
                )
                joblib.dump(pipeline, pkl_path)
                logger.info("  Saved: %s", pkl_path)
                models[target] = pipeline
                all_metrics.append(metrics)

        # ── 3. Train QUANTILE models ─────────────────────────────────────
        for target in TARGET_COLUMNS:
            q_models_for_target = {}
            all_cached = True
            for alpha in QUANTILE_LEVELS:
                q_pkl = _MODELS_DIR / f"{target}_q{int(alpha*100)}_model.pkl"
                if q_pkl.exists() and not force:
                    q_models_for_target[alpha] = joblib.load(q_pkl)
                else:
                    all_cached = False

            if all_cached:
                logger.info("Loading cached quantile models for %s", target)
                models[f"{target}_quantiles"] = q_models_for_target
            else:
                q_models_for_target = train_quantile_models(
                    target, backend=backend, con=con,
                    feature_cols=FEATURE_COLS, **hparams,
                )
                for alpha, qpipe in q_models_for_target.items():
                    q_pkl = _MODELS_DIR / f"{target}_q{int(alpha*100)}_model.pkl"
                    joblib.dump(qpipe, q_pkl)
                    logger.info("  Saved quantile model: %s", q_pkl)
                models[f"{target}_quantiles"] = q_models_for_target

        # ── 4. Train CALIBRATORS ─────────────────────────────────────────
        for target in TARGET_COLUMNS:
            cal_pkl = _MODELS_DIR / f"{target}_calibrator.pkl"
            if cal_pkl.exists() and not force:
                logger.info("Loading cached calibrator: %s", cal_pkl)
                models[f"{target}_calibrator"] = joblib.load(cal_pkl)
            else:
                q_key = f"{target}_quantiles"
                if q_key in models and models[q_key]:
                    calibrator = train_calibrator(
                        target,
                        mean_model=models[target],
                        quantile_models=models[q_key],
                        backend=backend,
                        con=con,
                        feature_cols=FEATURE_COLS,
                        minutes_model=models.get("minutes"),
                    )
                    if calibrator is not None:
                        joblib.dump(calibrator, cal_pkl)
                        logger.info("  Saved calibrator: %s", cal_pkl)
                        models[f"{target}_calibrator"] = calibrator
                    else:
                        logger.warning("  Calibrator training failed for %s", target)
                else:
                    logger.warning("  No quantile models for %s — skipping calibration.", target)

        # Save metrics summary
        if all_metrics:
            metrics_df = pd.DataFrame(all_metrics)
            metrics_path = _EVAL_DIR / "training_metrics.csv"
            metrics_df.to_csv(metrics_path, index=False)
            logger.info("Training metrics saved to %s", metrics_path)

        return models

    finally:
        if _close:
            con.close()


def _train_with_minutes(
    target: str,
    backend: str,
    minutes_model: Pipeline,
    con,
    **hparams,
) -> tuple[Pipeline, dict]:
    """Train a stat model with proj_minutes as an additional feature.

    1. Load training data with base FEATURE_COLS.
    2. Predict minutes using the minutes model → add as proj_minutes column.
    3. Train the stat model using FEATURE_COLS_WITH_MINUTES.
    """
    logger.info("Training model with minutes feature: target=%s", target)
    stat_col = TARGET_COLUMNS[target]

    # Load raw data
    sql = f"""
        SELECT
            pf.*,
            pgs.{stat_col}   AS target,
            pgs.minutes      AS actual_minutes,
            pgs.did_not_play
        FROM player_features pf
        JOIN player_game_stats pgs
          ON pf.player_id = pgs.player_id
         AND pf.game_id   = pgs.game_id
        WHERE pgs.did_not_play = FALSE
          AND pgs.{stat_col} IS NOT NULL
          AND pf.pts_avg_L5 IS NOT NULL
    """
    df = con.execute(sql).df()
    if df.empty:
        raise ValueError(f"No training rows for target '{stat_col}'.")

    # Build base feature matrix
    X_base = pd.DataFrame(index=df.index)
    for col in FEATURE_COLS:
        X_base[col] = pd.to_numeric(df.get(col, np.nan), errors="coerce")

    # Predict minutes and add as feature
    min_pred = minutes_model.predict(X_base)
    X_full = X_base.copy()
    X_full["proj_minutes"] = np.clip(min_pred, 0, 48)

    y = pd.to_numeric(df["target"], errors="coerce")
    season = df["season"].fillna("unknown")
    valid = y.notna()
    X_full = X_full[valid].reset_index(drop=True)
    y = y[valid].reset_index(drop=True)
    season = season[valid].reset_index(drop=True)

    # Split
    X_train, X_val, y_train, y_val = _train_val_split(X_full, y, season)

    # Train
    pipeline = _build_pipeline(backend, **hparams)
    pipeline.fit(X_train, y_train)

    metrics = _evaluate(pipeline, X_train, y_train, X_val, y_val, target)
    _log_feature_importance(pipeline, target, FEATURE_COLS_WITH_MINUTES)

    return pipeline, metrics


# ---------------------------------------------------------------------------
# Load saved models
# ---------------------------------------------------------------------------

def load_models() -> dict[str, Pipeline]:
    """Load all saved model files (mean + quantile + minutes + calibrators).

    Returns dict with keys like:
      "points", "rebounds", "assists", "threepm" — mean models
      "minutes" — minutes model
      "points_quantiles" — dict of {alpha: pipeline}
      "points_calibrator" — IsotonicRegression
    """
    models: dict = {}

    # Minutes model
    min_pkl = _MODELS_DIR / "minutes_model.pkl"
    if min_pkl.exists():
        models["minutes"] = joblib.load(min_pkl)
    else:
        logger.warning("Minutes model not found; stat models may be less accurate.")

    # Mean models (required)
    for target in TARGET_COLUMNS:
        pkl_path = _MODELS_DIR / f"{target}_model.pkl"
        if not pkl_path.exists():
            raise FileNotFoundError(
                f"Model not found: {pkl_path}\n"
                "Run `python src/models.py` or `python src/cli.py train` first."
            )
        models[target] = joblib.load(pkl_path)

    # Quantile models (optional)
    for target in TARGET_COLUMNS:
        q_models = {}
        for alpha in QUANTILE_LEVELS:
            q_pkl = _MODELS_DIR / f"{target}_q{int(alpha*100)}_model.pkl"
            if q_pkl.exists():
                q_models[alpha] = joblib.load(q_pkl)
        if q_models:
            models[f"{target}_quantiles"] = q_models

    # Calibrators (optional)
    for target in TARGET_COLUMNS:
        cal_pkl = _MODELS_DIR / f"{target}_calibrator.pkl"
        if cal_pkl.exists():
            models[f"{target}_calibrator"] = joblib.load(cal_pkl)

    logger.debug("Loaded models: %s", list(models.keys()))
    return models


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict(
    features_df: pd.DataFrame,
    models: dict,
) -> pd.DataFrame:
    """Run all models on a feature DataFrame.

    Adds columns:
      proj_minutes, proj_points, proj_rebounds, proj_assists, proj_threepm,
      composite proj_* columns,
      {target}_q{level} quantile predictions,
      {target}_iqr for IQR-based std estimate.
    """
    df = features_df.copy()

    # Build base X matrix
    X_base = pd.DataFrame(index=df.index)
    for col in FEATURE_COLS:
        X_base[col] = pd.to_numeric(df.get(col, np.nan), errors="coerce")

    # ── 1. Predict minutes first ─────────────────────────────────────────
    if "minutes" in models:
        min_preds = models["minutes"].predict(X_base)
        df["proj_minutes"] = np.clip(min_preds, 0, 48)
    else:
        df["proj_minutes"] = df.get("min_avg_L10", np.nan)

    # Build extended X with proj_minutes
    X_ext = X_base.copy()
    X_ext["proj_minutes"] = df["proj_minutes"]

    # ── 2. Mean predictions ──────────────────────────────────────────────
    for target in TARGET_COLUMNS:
        if target not in models:
            continue
        pipeline = models[target]
        col = f"proj_{target}"
        # Determine which feature set the model expects
        n_features = _get_model_n_features(pipeline)
        if n_features == len(FEATURE_COLS_WITH_MINUTES):
            preds = pipeline.predict(X_ext)
        else:
            preds = pipeline.predict(X_base)
        df[col] = np.clip(preds, 0, None)

    # ── 3. Composite projections ─────────────────────────────────────────
    for composite, base_targets in COMPOSITE_TARGETS.items():
        base_cols = [f"proj_{t}" for t in base_targets]
        if all(c in df.columns for c in base_cols):
            df[f"proj_{composite}"] = df[base_cols].sum(axis=1)

    # ── 4. Quantile predictions ──────────────────────────────────────────
    for target in TARGET_COLUMNS:
        q_key = f"{target}_quantiles"
        if q_key not in models:
            continue
        q_models = models[q_key]
        q_preds = {}
        for alpha, qpipe in sorted(q_models.items()):
            col_name = f"{target}_q{int(alpha*100)}"
            preds = qpipe.predict(X_base)
            q_preds[alpha] = preds
            df[col_name] = np.clip(preds, 0, None)

        # Fix quantile crossing (ensure monotonicity)
        alpha_sorted = sorted(q_preds.keys())
        for row_i in range(len(df)):
            prev_val = -np.inf
            for alpha in alpha_sorted:
                col_name = f"{target}_q{int(alpha*100)}"
                if df.iloc[row_i][col_name] < prev_val:
                    df.iloc[row_i, df.columns.get_loc(col_name)] = prev_val
                prev_val = df.iloc[row_i][col_name]

        # IQR-based std estimate
        q25_col = f"{target}_q25"
        q75_col = f"{target}_q75"
        if q25_col in df.columns and q75_col in df.columns:
            iqr = df[q75_col] - df[q25_col]
            df[f"{target}_iqr_std"] = (iqr / 1.35).clip(lower=0.5)

    return df


def _get_model_n_features(pipeline: Pipeline) -> int:
    """Determine how many features a trained pipeline expects."""
    try:
        model = pipeline.named_steps["model"]
        if hasattr(model, "n_features_in_"):
            return model.n_features_in_
        if hasattr(model, "n_features_"):
            return model.n_features_
    except Exception:
        pass
    # Fallback: check imputer
    try:
        imputer = pipeline.named_steps["imputer"]
        if hasattr(imputer, "n_features_in_"):
            return imputer.n_features_in_
    except Exception:
        pass
    return len(FEATURE_COLS)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
    )
    parser = argparse.ArgumentParser(description="Train NBA prop projection models.")
    parser.add_argument(
        "--backend", choices=["gbm", "lightgbm", "ridge"],
        default=None,
        help="Model backend (default: from config.yaml or 'gbm').",
    )
    parser.add_argument(
        "--force-retrain", action="store_true", default=False,
        help="Retrain even if .pkl files exist.",
    )
    args = parser.parse_args(argv)

    con = get_connection()
    init_schema(con)

    models = train_all_models(
        backend=args.backend,
        force=args.force_retrain,
        con=con,
    )
    logger.info("All models ready: %s", list(models.keys()))
    con.close()


if __name__ == "__main__":
    main()
