"""
src/props_io.py
───────────────
Read, validate, and normalise today's prop lines from JSON or CSV.

The expected input is the structured output you get from pasting the OCR
text into an LLM.  The canonical JSON schema is:

    [
      {
        "player_name":  "LeBron James",
        "team":         "LAL",
        "opponent":     "GSW",
        "market":       "points",         ← internal key from config.yaml
        "line":         25.5,
        "over_odds":    -115,
        "under_odds":   -105,
        "book":         "DraftKings",
        "game_date":    "2025-01-15"      ← YYYY-MM-DD
      },
      ...
    ]

CSV equivalent has the same column headers.

The module resolves player names to ``player_id`` via the ``players`` table,
and normalises market strings using the ``markets.aliases`` mapping in
``config.yaml``.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "config.yaml"

# Required columns that must be present after normalisation
_REQUIRED_COLS = [
    "player_name", "market", "line", "over_odds", "under_odds",
]

# Optional columns with their default values
_OPTIONAL_COLS: dict[str, object] = {
    "team":       None,
    "opponent":   None,
    "book":       "Unknown",
    "game_date":  None,   # filled with today if absent
    "player_id":  None,   # resolved later
}


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_market_aliases() -> dict[str, str]:
    """Load the markets.aliases mapping from config.yaml.

    Returns
    -------
    dict[str, str]
        Mapping of raw sportsbook string → internal market key.
    """
    if not _CONFIG_PATH.exists():
        logger.warning("config.yaml not found; no market alias resolution.")
        return {}
    with _CONFIG_PATH.open() as f:
        cfg = yaml.safe_load(f)
    return cfg.get("markets", {}).get("aliases", {})


def _load_supported_markets() -> list[str]:
    """Return list of supported internal market keys from config.yaml."""
    if not _CONFIG_PATH.exists():
        return []
    with _CONFIG_PATH.open() as f:
        cfg = yaml.safe_load(f)
    return cfg.get("markets", {}).get("supported", [])


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_props_json(path: str | Path) -> pd.DataFrame:
    """Load props from a JSON file.

    Parameters
    ----------
    path : str or Path
        Path to the JSON file (list of prop objects).

    Returns
    -------
    pd.DataFrame
        Raw props DataFrame before normalisation.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Props file not found: {path}")
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected a JSON array at the top level.")
    df = pd.DataFrame(data)
    logger.info("Loaded %d props from %s", len(df), path)
    return df


def load_props_csv(path: str | Path) -> pd.DataFrame:
    """Load props from a CSV file.

    Parameters
    ----------
    path : str or Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Raw props DataFrame before normalisation.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Props file not found: {path}")
    df = pd.read_csv(path)
    logger.info("Loaded %d props from %s", len(df), path)
    return df


def load_props(path: str | Path) -> pd.DataFrame:
    """Auto-detect JSON or CSV and load props.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    pd.DataFrame
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        return load_props_json(path)
    elif suffix in (".csv", ".tsv"):
        return load_props_csv(path)
    else:
        # Try JSON first, then CSV
        try:
            return load_props_json(path)
        except Exception:
            return load_props_csv(path)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalise_market(raw: str, aliases: dict[str, str]) -> str:
    """Map a raw market string to an internal market key.

    Parameters
    ----------
    raw : str
        Market string as it appears in the JSON/CSV.
    aliases : dict[str, str]
        Alias mapping from config.yaml.

    Returns
    -------
    str
        Internal market key (e.g. ``"points"``) or the original string
        lowercased if no alias found.
    """
    # Direct match first
    if raw in aliases:
        return aliases[raw]
    # Case-insensitive fallback
    lower = raw.lower().strip()
    for k, v in aliases.items():
        if k.lower() == lower:
            return v
    # Return lowercased original if no alias
    return lower


def normalise_props(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalise a raw props DataFrame.

    Steps:
    1. Strip whitespace from string columns.
    2. Map market strings to internal keys.
    3. Fill missing optional columns with defaults.
    4. Cast numeric columns.
    5. Fill missing game_date with today.
    6. Drop rows with missing required fields or unsupported markets.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Normalised props DataFrame.
    """
    aliases = _load_market_aliases()
    supported = set(_load_supported_markets())

    df = df.copy()

    # Strip whitespace
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].str.strip()

    # Normalise market names
    if "market" in df.columns:
        df["market"] = df["market"].apply(lambda m: normalise_market(str(m), aliases))

    # Fill optional columns
    for col, default in _OPTIONAL_COLS.items():
        if col not in df.columns:
            df[col] = default

    # Fill game_date with today
    df["game_date"] = pd.to_datetime(
        df["game_date"].fillna(str(date.today()))
    ).dt.date

    # Cast numeric columns
    for col in ("line", "over_odds", "under_odds"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows missing required fields
    before = len(df)
    df = df.dropna(subset=_REQUIRED_COLS)
    if len(df) < before:
        logger.warning("Dropped %d rows with missing required fields.", before - len(df))

    # Filter to supported markets only
    if supported:
        before = len(df)
        df = df[df["market"].isin(supported)]
        if len(df) < before:
            logger.warning(
                "Dropped %d rows with unsupported market types.", before - len(df)
            )

    logger.info("Normalised props: %d rows remain.", len(df))
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Player ID resolution
# ---------------------------------------------------------------------------

def resolve_player_ids_for_props(df: pd.DataFrame, con) -> pd.DataFrame:
    """Match player_name in props to player_id in the players table.

    Parameters
    ----------
    df : pd.DataFrame
        Normalised props DataFrame.
    con : duckdb.DuckDBPyConnection

    Returns
    -------
    pd.DataFrame
        Input df with player_id column populated where a match is found.
    """
    # TODO:
    #   1. Load distinct (player_id, full_name) from players.
    #   2. Normalise both sides (lowercase, remove punctuation).
    #   3. Exact match first; fuzzy match (difflib or rapidfuzz) as fallback.
    #   4. Merge player_id back into df.
    #   5. Log unmatched names.
    raise NotImplementedError("resolve_player_ids_for_props() not yet implemented")


# ---------------------------------------------------------------------------
# Full load pipeline
# ---------------------------------------------------------------------------

def load_and_prepare_props(path: str | Path, con=None) -> pd.DataFrame:
    """Load, normalise, and resolve player IDs for today's props.

    Parameters
    ----------
    path : str or Path
        Path to today_props_raw.json or .csv.
    con : duckdb.DuckDBPyConnection, optional
        If provided, player names are resolved to IDs.

    Returns
    -------
    pd.DataFrame
        Ready-to-use props DataFrame with all columns normalised.
    """
    df = load_props(path)
    df = normalise_props(df)

    if con is not None:
        try:
            df = resolve_player_ids_for_props(df, con)
        except NotImplementedError:
            logger.warning("Player ID resolution not yet implemented; skipping.")

    return df


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def save_props_json(df: pd.DataFrame, path: str | Path) -> None:
    """Save a normalised props DataFrame back to JSON.

    Parameters
    ----------
    df : pd.DataFrame
    path : str or Path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = df.to_dict(orient="records")
    # Convert date objects to strings for JSON serialisation
    for rec in records:
        if isinstance(rec.get("game_date"), date):
            rec["game_date"] = rec["game_date"].isoformat()
    with path.open("w") as f:
        json.dump(records, f, indent=2)
    logger.info("Saved %d props to %s", len(df), path)
