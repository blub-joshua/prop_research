"""
src/ingest_injuries.py
──────────────────────
Ingest NBA player injury / availability data.

Approach (minimal & extensible)
---------------------------------
Full historical injury data is not freely available in machine-readable form
from a single reliable public API.  This module therefore takes a three-tier
approach:

Tier 1 — Manual CSV  (primary for back-testing)
    Place a CSV file at  data/injuries_manual.csv  with columns:
        player_id, player_name, report_date, status, injury_type, notes
    Run:  python src/ingest_injuries.py --source csv

Tier 2 — Rotowire lineup page  (today's injury report for live use)
    Parses the public Rotowire NBA lineups/injuries page with BeautifulSoup.
    Run:  python src/ingest_injuries.py --source rotowire

Tier 3 — ESPN unofficial endpoint  (fallback if Rotowire changes)
    Uses ESPN's public (undocumented) injuries JSON API.
    Run:  python src/ingest_injuries.py --source espn

Status vocabulary (NBA convention)
------------------------------------
  "Out"          → will not play
  "Doubtful"     → very unlikely (~25 % chance of playing)
  "Questionable" → 50/50
  "Probable"     → expected to play (~75 %)
  "Active"       → confirmed in line-up
  "GTD"          → Game-Time Decision (alias for Questionable)

Run directly:
    python src/ingest_injuries.py --source csv
    python src/ingest_injuries.py --source rotowire
    python src/ingest_injuries.py --source espn

Or via CLI:
    python src/cli.py ingest --module injuries
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from db import get_connection, init_schema, upsert_dataframe, query_df

load_dotenv()
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MANUAL_CSV    = _PROJECT_ROOT / "data" / "injuries_manual.csv"
_MANUAL_CSV_TEMPLATE = _PROJECT_ROOT / "data" / "injuries_manual_template.csv"

_ROTOWIRE_URL = "https://www.rotowire.com/basketball/nba-lineups.php"
_ESPN_URL     = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"

_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


# ---------------------------------------------------------------------------
# Player name → ID resolution
# ---------------------------------------------------------------------------

def _load_player_name_map(con) -> dict[str, int]:
    """Return a dict mapping normalised player name → player_id."""
    df = query_df("SELECT player_id, full_name FROM players", con=con)
    if df.empty:
        return {}
    mapping = {}
    for _, row in df.iterrows():
        key = _normalise_name(row["full_name"])
        mapping[key] = int(row["player_id"])
    return mapping


def _normalise_name(name: str) -> str:
    """Lowercase, strip punctuation, collapse spaces."""
    import re
    name = str(name).lower().strip()
    name = re.sub(r"[^a-z\s]", "", name)
    name = re.sub(r"\s+", " ", name)
    return name.strip()


def _resolve_ids(df: pd.DataFrame, name_map: dict[str, int]) -> pd.DataFrame:
    """Add player_id column by matching player_name to the players table."""
    if "player_id" not in df.columns:
        df["player_id"] = None

    mask = df["player_id"].isna()
    df.loc[mask, "player_id"] = df.loc[mask, "player_name"].apply(
        lambda n: name_map.get(_normalise_name(n))
    )

    unresolved = df["player_id"].isna().sum()
    if unresolved:
        logger.warning("%d player names could not be resolved to player_id.", unresolved)

    return df


# ---------------------------------------------------------------------------
# Tier 1 — Manual CSV
# ---------------------------------------------------------------------------

def _write_template_csv() -> None:
    """Write an empty template CSV so the user knows the expected schema."""
    _MANUAL_CSV_TEMPLATE.parent.mkdir(parents=True, exist_ok=True)
    if not _MANUAL_CSV_TEMPLATE.exists():
        with _MANUAL_CSV_TEMPLATE.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "player_id", "player_name", "report_date",
                "status", "injury_type", "notes",
            ])
            # Example row
            writer.writerow([
                "203076", "Anthony Davis", "2024-01-10",
                "Out", "Left foot - stress fracture", "",
            ])
        logger.info("Wrote template CSV: %s", _MANUAL_CSV_TEMPLATE)


def load_from_csv(path: Path = _MANUAL_CSV, con=None) -> pd.DataFrame:
    """Load and ingest injuries from a manually maintained CSV file.

    Expected columns: player_id (optional), player_name, report_date,
                      status, injury_type, notes

    Parameters
    ----------
    path : Path     CSV file path (default data/injuries_manual.csv)
    con             DuckDB connection

    Returns
    -------
    pd.DataFrame    normalised injury rows
    """
    if not path.exists():
        logger.warning("No manual CSV found at %s — writing template and skipping.", path)
        _write_template_csv()
        return pd.DataFrame()

    df = pd.read_csv(path, dtype=str).fillna("")
    df.columns = [c.strip().lower() for c in df.columns]

    required = ["player_name", "report_date", "status"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"injuries_manual.csv is missing required column: {col!r}")

    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce").dt.date
    df = df.dropna(subset=["report_date"])
    df["source"] = "manual_csv"

    # Resolve player IDs if not already present
    if con is not None:
        name_map = _load_player_name_map(con)
        df = _resolve_ids(df, name_map)

    if "player_id" in df.columns:
        df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce")

    logger.info("Loaded %d injury rows from CSV.", len(df))
    return df


# ---------------------------------------------------------------------------
# Tier 2 — Rotowire
# ---------------------------------------------------------------------------

def load_from_rotowire() -> pd.DataFrame:
    """Scrape today's injury/status data from the Rotowire lineup page.

    Rotowire's NBA lineups page lists player status badges (Out, GTD, Q, P)
    next to player names.  We parse these with BeautifulSoup.

    Returns
    -------
    pd.DataFrame  with columns: player_name, status, injury_type,
                                report_date, notes, source
    """
    from bs4 import BeautifulSoup

    logger.info("Fetching Rotowire injuries from %s ...", _ROTOWIRE_URL)
    resp = requests.get(_ROTOWIRE_URL, headers=_REQUEST_HEADERS, timeout=20)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml")
    today = date.today()
    rows = []

    # Rotowire wraps each player in a div with class "lineup__player"
    # The structure can change; we look for player-status elements broadly.
    for player_div in soup.select(".lineup__player"):
        name_el   = player_div.select_one(".lineup__player-name, a.player-name, span.player-name")
        status_el = player_div.select_one(".lineup__injury-tag, .player-status, span[class*='status']")
        if not name_el:
            continue
        name   = name_el.get_text(strip=True)
        status = status_el.get_text(strip=True) if status_el else "Active"
        # Normalise status aliases
        status = _normalise_status(status)
        rows.append({
            "player_name": name,
            "status":      status,
            "injury_type": "",
            "notes":       "",
            "report_date": today,
            "source":      "rotowire",
        })

    # Fallback: also parse the dedicated injuries section
    for inj_row in soup.select(".injury-report__row, tr.injury-row"):
        cells = inj_row.select("td")
        if len(cells) >= 3:
            name   = cells[0].get_text(strip=True)
            status = _normalise_status(cells[1].get_text(strip=True))
            injury = cells[2].get_text(strip=True) if len(cells) > 2 else ""
            rows.append({
                "player_name": name,
                "status":      status,
                "injury_type": injury,
                "notes":       "",
                "report_date": today,
                "source":      "rotowire",
            })

    df = pd.DataFrame(rows).drop_duplicates(subset=["player_name"])
    logger.info("Parsed %d Rotowire player rows.", len(df))
    return df


# ---------------------------------------------------------------------------
# Tier 3 — ESPN
# ---------------------------------------------------------------------------

def load_from_espn() -> pd.DataFrame:
    """Fetch today's NBA injuries from ESPN's public (unofficial) API.

    Endpoint: https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries
    No authentication required.  Returns a JSON list of injuries with
    athlete name, status, and description.

    Returns
    -------
    pd.DataFrame
    """
    logger.info("Fetching ESPN injuries from %s ...", _ESPN_URL)
    resp = requests.get(_ESPN_URL, headers=_REQUEST_HEADERS, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    today = date.today()
    rows = []

    # ESPN injury JSON structure (may change without notice):
    # { "items": [
    #     { "athlete": {"displayName": "...", "id": "..."},
    #       "status": {"type": {"description": "Questionable"}},
    #       "details": [{"detail": "Knee"}] },
    #     ...
    #   ]
    # }
    for item in data.get("items", []):
        athlete = item.get("athlete", {})
        name    = athlete.get("displayName", "")
        espn_id = athlete.get("id", "")

        status_obj = item.get("status", {}).get("type", {})
        status = _normalise_status(status_obj.get("description", "Active"))

        details = item.get("details", [])
        injury  = details[0].get("detail", "") if details else ""

        rows.append({
            "player_name": name,
            "status":      status,
            "injury_type": injury,
            "notes":       "",
            "report_date": today,
            "source":      "espn",
        })

    df = pd.DataFrame(rows)
    logger.info("Parsed %d ESPN injury rows.", len(df))
    return df


def _normalise_status(raw: str) -> str:
    """Map common sportsbook/site status strings to our vocabulary."""
    mapping = {
        "out":          "Out",
        "doubtful":     "Doubtful",
        "questionable": "Questionable",
        "q":            "Questionable",
        "gtd":          "Questionable",
        "probable":     "Probable",
        "p":            "Probable",
        "active":       "Active",
        "healthy":      "Active",
        "":             "Active",
    }
    return mapping.get(raw.strip().lower(), raw.strip().title())


# ---------------------------------------------------------------------------
# Ingest orchestrator
# ---------------------------------------------------------------------------

def ingest_injuries(source: str = "csv", con=None) -> None:
    """Fetch and store today's (or historical) injury data.

    Parameters
    ----------
    source : str    "csv" | "rotowire" | "espn" | "auto"
                    "auto" tries rotowire → espn → csv in order.
    con : duckdb connection, optional
    """
    _close = con is None
    if con is None:
        con = get_connection()

    try:
        if source == "csv":
            df = load_from_csv(con=con)
        elif source == "rotowire":
            df = load_from_rotowire()
        elif source == "espn":
            df = load_from_espn()
        elif source == "auto":
            for src_fn in (load_from_rotowire, load_from_espn):
                try:
                    df = src_fn()
                    if not df.empty:
                        break
                except Exception as exc:
                    logger.warning("Source %s failed: %s", src_fn.__name__, exc)
            else:
                df = load_from_csv(con=con)
        else:
            raise ValueError(f"Unknown injury source: {source!r}")

        if df.empty:
            logger.info("No injury rows to ingest.")
            return

        # Resolve player IDs from the players table
        name_map = _load_player_name_map(con)
        df = _resolve_ids(df, name_map)

        # Drop rows without player_id (can't FK reference)
        before = len(df)
        df = df[df["player_id"].notna()].copy()
        df["player_id"] = df["player_id"].astype(int)
        if len(df) < before:
            logger.warning("Dropped %d rows with unresolved player IDs.", before - len(df))

        upsert_dataframe(df, "injuries", ["player_id", "report_date", "status"], con=con)
        logger.info("Upserted %d injury rows.", len(df))
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
    parser = argparse.ArgumentParser(description="Ingest NBA injury reports.")
    parser.add_argument(
        "--source", choices=["csv", "rotowire", "espn", "auto"],
        default="auto",
        help="Data source (default: auto = rotowire → espn → csv).",
    )
    args = parser.parse_args(argv)

    con = get_connection()
    init_schema(con)
    ingest_injuries(source=args.source, con=con)
    con.close()
    logger.info("ingest_injuries done.")


if __name__ == "__main__":
    main()
