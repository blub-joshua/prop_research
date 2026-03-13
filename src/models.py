"""
src/models.py
─────────────
Train, evaluate, save, and load projection models for PTS, REB, AST, FG3M.

Model design
------------
  • One model per stat target.
  • Algorithm: GradientBoostingRegressor (sklearn, always available) with an
    optional upgrade to LightGBM when installed.  Ridge regression is kept
    as a fast baseline for smoke-testing.
  • Train / validation split: train on all seasons EXCEPT the most recent
    season present in the feature table; validate on the most recent season.
  • Preprocessing: median imputation + StandardScaler inside a Pipeline.
  • Metrics logged: RMSE and MAE for train and validation sets.
    Residual histogram saved as PNG to models/eval/.
  • Models saved as models/{target}_model.pkl via joblib.

Composite predictions (PRA, P+R, etc.) are sums of base projections — no
separate model is needed.

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


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

def _build_pipeline(backend: str, **hparams) -> Pipeline:
    """Build a sklearn Pipeline with imputation, scaling, and a regressor.

    Parameters
    ----------
    backend : str   "gbm" | "lightgbm" | "ridge"
    **hparams       Hyperparameter overrides passed to the regressor.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
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


# ---------------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------------

def load_training_data(target_stat: str, con) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Join player_features with player_game_stats to get (X, y, season).

    Parameters
    ----------
    target_stat : str
        Column name in player_game_stats (e.g. "points", "fg3m").
    con : duckdb connection

    Returns
    -------
    X : pd.DataFrame    feature matrix
    y : pd.Series       target values
    season : pd.Series  season label per row (for train/val split)
    """
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
          AND pf.pts_avg_L5 IS NOT NULL   -- require at least some history
    """
    df = con.execute(sql).df()
    logger.info("  Training rows loaded: %d", len(df))

    if df.empty:
        raise ValueError(
            f"No training rows for target '{target_stat}'. "
            "Run features.py first."
        )

    # Keep only numeric feature columns that are in FEATURE_COLS
    present_features = [c for c in FEATURE_COLS if c in df.columns]
    missing = set(FEATURE_COLS) - set(present_features)
    if missing:
        logger.debug("  Feature columns absent (will be imputed): %s", missing)

    X = df[present_features].copy()
    # Add any missing columns as NaN (imputer will fill them)
    for col in FEATURE_COLS:
        if col not in X.columns:
            X[col] = np.nan
    X = X[FEATURE_COLS]

    y       = pd.to_numeric(df["target"],  errors="coerce")
    season  = df["season"].fillna("unknown")

    # Drop rows where target is NaN
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

    # Save residual histogram
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

def _log_feature_importance(pipeline: Pipeline, target: str) -> None:
    """Log top feature importances (if the regressor supports it)."""
    try:
        model = pipeline.named_steps["model"]
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_)
        else:
            return

        fi = pd.Series(importances, index=FEATURE_COLS).sort_values(ascending=False)
        logger.info("  Top 10 features for %s:", target)
        for feat, imp in fi.head(10).items():
            logger.info("    %-35s %.4f", feat, imp)

        # Save full importance CSV
        fi_path = _EVAL_DIR / f"{target}_feature_importance.csv"
        fi.to_csv(fi_path, header=["importance"])
        logger.info("  Feature importances saved: %s", fi_path)
    except Exception as exc:
        logger.debug("  Could not extract feature importance: %s", exc)


# ---------------------------------------------------------------------------
# Train / load single model
# ---------------------------------------------------------------------------

def train_model(
    target: str,
    backend: str = "gbm",
    con=None,
    **hparams,
) -> tuple[Pipeline, dict]:
    """Train a projection model for a single stat target.

    Parameters
    ----------
    target : str        key in TARGET_COLUMNS
    backend : str       "gbm" | "lightgbm" | "ridge"
    con                 DuckDB connection
    **hparams           Hyperparameter overrides

    Returns
    -------
    (pipeline, metrics_dict)
    """
    logger.info("Training model: target=%s  backend=%s", target, backend)
    _close = con is None
    if con is None:
        con = get_connection()

    try:
        stat_col = TARGET_COLUMNS[target]
        X, y, season = load_training_data(stat_col, con)
        X_train, X_val, y_train, y_val = _train_val_split(X, y, season)

        pipeline = _build_pipeline(backend, **hparams)
        pipeline.fit(X_train, y_train)

        metrics = _evaluate(pipeline, X_train, y_train, X_val, y_val, target)
        _log_feature_importance(pipeline, target)

        return pipeline, metrics
    finally:
        if _close:
            con.close()


# ---------------------------------------------------------------------------
# Train / load ALL models
# ---------------------------------------------------------------------------

def train_all_models(
    backend: str | None = None,
    force: bool | None = None,
    con=None,
    **hparams,
) -> dict[str, Pipeline]:
    """Train or load all four base models.

    Parameters
    ----------
    backend : str, optional     Defaults to config.yaml → models.backend → "gbm"
    force : bool, optional      Defaults to FORCE_RETRAIN env var
    con                         DuckDB connection, optional
    **hparams                   Hyperparameter overrides

    Returns
    -------
    dict[str, Pipeline]   {target_name: fitted_pipeline}
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

    for target in TARGET_COLUMNS:
        pkl_path = _MODELS_DIR / f"{target}_model.pkl"
        if pkl_path.exists() and not force:
            logger.info("Loading cached model: %s", pkl_path)
            models[target] = joblib.load(pkl_path)
        else:
            pipeline, metrics = train_model(target, backend=backend, con=con, **hparams)
            joblib.dump(pipeline, pkl_path)
            logger.info("  Saved: %s", pkl_path)
            models[target] = pipeline
            all_metrics.append(metrics)

    # Save metrics summary
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        metrics_path = _EVAL_DIR / "training_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        logger.info("Training metrics saved to %s", metrics_path)

    return models


# ---------------------------------------------------------------------------
# Load saved models
# ---------------------------------------------------------------------------

def load_models() -> dict[str, Pipeline]:
    """Load all saved .pkl model files.

    Raises FileNotFoundError if any model is missing.

    Returns
    -------
    dict[str, Pipeline]
    """
    models: dict[str, Pipeline] = {}
    for target in TARGET_COLUMNS:
        pkl_path = _MODELS_DIR / f"{target}_model.pkl"
        if not pkl_path.exists():
            raise FileNotFoundError(
                f"Model not found: {pkl_path}\n"
                "Run `python src/models.py` or `python src/cli.py train` first."
            )
        models[target] = joblib.load(pkl_path)
        logger.debug("Loaded model: %s", pkl_path)
    return models


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict(
    features_df: pd.DataFrame,
    models: dict[str, Pipeline],
) -> pd.DataFrame:
    """Run all base models on a feature DataFrame.

    Parameters
    ----------
    features_df : pd.DataFrame
        Must have columns matching FEATURE_COLS (NaN OK — imputed internally).
    models : dict[str, Pipeline]

    Returns
    -------
    pd.DataFrame
        Input df + projection columns: proj_points, proj_rebounds,
        proj_assists, proj_threepm, and composite proj_* columns.
    """
    df = features_df.copy()

    # Build X matrix
    X = pd.DataFrame(index=df.index)
    for col in FEATURE_COLS:
        X[col] = pd.to_numeric(df.get(col, np.nan), errors="coerce")

    for target, pipeline in models.items():
        col = f"proj_{target}"
        preds = pipeline.predict(X)
        df[col] = np.clip(preds, 0, None)   # clip negative projections to 0

    # Composite projections
    for composite, base_targets in COMPOSITE_TARGETS.items():
        base_cols = [f"proj_{t}" for t in base_targets]
        if all(c in df.columns for c in base_cols):
            df[f"proj_{composite}"] = df[base_cols].sum(axis=1)

    return df


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
