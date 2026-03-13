"""
src/db.py
─────────
DuckDB connection helper, schema initialiser, and utility functions.

All other modules import get_connection() from here.  The database file path
is resolved from the DB_PATH env var (relative to the project root) or
defaults to data/nba_props.duckdb.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import duckdb
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DB_PATH: Path = _PROJECT_ROOT / os.getenv("DB_PATH", "data/nba_props.duckdb")
_SCHEMA_PATH: Path = Path(__file__).resolve().parent / "schema.sql"


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def get_connection(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Return an open DuckDB connection to the project database file.

    The connection uses DuckDB's default in-process mode — one writer at a
    time, unlimited concurrent readers with read_only=True.

    Parameters
    ----------
    read_only : bool
        Open in read-only mode (safe for multiple concurrent readers).

    Returns
    -------
    duckdb.DuckDBPyConnection
    """
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.debug("DuckDB open: %s  read_only=%s", _DB_PATH, read_only)
    con = duckdb.connect(str(_DB_PATH), read_only=read_only)
    # Improve performance for analytical queries
    con.execute("PRAGMA threads=4")
    return con


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def init_schema(con: duckdb.DuckDBPyConnection | None = None) -> None:
    """Execute schema.sql against the database.

    Idempotent — uses CREATE TABLE IF NOT EXISTS throughout.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection, optional
        Reuse an existing connection; otherwise one is opened and closed.
    """
    _close = con is None
    if con is None:
        con = get_connection()
    logger.info("Applying schema from %s", _SCHEMA_PATH)
    sql = _SCHEMA_PATH.read_text(encoding="utf-8")
    # DuckDB executes the whole script in one call
    con.execute(sql)
    logger.info("Schema applied.")
    if _close:
        con.close()


# ---------------------------------------------------------------------------
# DataFrame upsert
# ---------------------------------------------------------------------------

def upsert_dataframe(
    df: pd.DataFrame,
    table: str,
    conflict_columns: list[str],
    con: duckdb.DuckDBPyConnection | None = None,
) -> int:
    """Insert a DataFrame, ignoring rows that conflict on conflict_columns.

    Uses DuckDB's INSERT OR IGNORE semantics via a temporary view.

    Parameters
    ----------
    df : pd.DataFrame
    table : str
    conflict_columns : list[str]
        Columns forming the UNIQUE constraint.
    con : duckdb.DuckDBPyConnection, optional

    Returns
    -------
    int
        Approximate number of rows inserted (0 on all-duplicate batches).
    """
    if df.empty:
        return 0

    _close = con is None
    if con is None:
        con = get_connection()

    try:
        # Only keep columns that exist in the target table
        existing_cols = _get_table_columns(table, con)
        df = df[[c for c in df.columns if c in existing_cols]]

        con.register("_upsert_src", df)
        cols = ", ".join(f'"{c}"' for c in df.columns)
        con.execute(f"INSERT OR IGNORE INTO {table} ({cols}) SELECT {cols} FROM _upsert_src")
        return len(df)
    finally:
        try:
            con.unregister("_upsert_src")
        except Exception:
            pass
        if _close:
            con.close()


def _get_table_columns(table: str, con: duckdb.DuckDBPyConnection) -> list[str]:
    """Return the column names of an existing table."""
    rows = con.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = ? ORDER BY ordinal_position",
        [table],
    ).fetchall()
    return [r[0] for r in rows]


# ---------------------------------------------------------------------------
# Inspection helpers
# ---------------------------------------------------------------------------

def table_counts(con: duckdb.DuckDBPyConnection | None = None) -> dict[str, int]:
    """Return row counts for all user tables.

    Returns
    -------
    dict[str, int]
        e.g. {"teams": 30, "games": 12000, ...}
    """
    _close = con is None
    if con is None:
        con = get_connection(read_only=True)
    try:
        tables = con.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main' AND table_type = 'BASE TABLE'"
        ).fetchall()
        counts = {}
        for (tbl,) in tables:
            n = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
            counts[tbl] = n
        return counts
    finally:
        if _close:
            con.close()


def table_exists(table: str, con: duckdb.DuckDBPyConnection | None = None) -> bool:
    """Return True if the given table exists in the database."""
    _close = con is None
    if con is None:
        con = get_connection(read_only=True)
    try:
        rows = con.execute(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_name = ? AND table_schema = 'main'",
            [table],
        ).fetchone()
        return rows[0] > 0
    finally:
        if _close:
            con.close()


def query_df(sql: str, params: list | None = None,
             con: duckdb.DuckDBPyConnection | None = None) -> pd.DataFrame:
    """Execute a SELECT and return a pandas DataFrame.

    Parameters
    ----------
    sql : str
    params : list, optional
    con : duckdb.DuckDBPyConnection, optional

    Returns
    -------
    pd.DataFrame
    """
    _close = con is None
    if con is None:
        con = get_connection(read_only=True)
    try:
        if params:
            return con.execute(sql, params).df()
        return con.execute(sql).df()
    finally:
        if _close:
            con.close()
