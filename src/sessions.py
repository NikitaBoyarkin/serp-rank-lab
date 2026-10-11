"""Sessionisation for the SERP Rank Lab event stream.

Sessionisation runs in DuckDB over ``sql/02_sessions.sql`` -- no Python loop.
Assignment rule (fixed by the assignment spec):

* a new session starts after **30 minutes of inactivity**, AND
* a session is broken whenever the calendar **date changes**, even if the gap
  is under 30 minutes.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

SQL_DIR = Path(__file__).resolve().parent.parent / "sql"
SESSIONS_SQL = SQL_DIR / "02_sessions.sql"

SESSION_COLUMNS = ["user_id", "ab_group", "start_ts", "end_ts", "pdate"]


def load_sql(name: str) -> str:
    """Read a SQL file from the project's ``sql/`` directory."""
    path = SQL_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing SQL file: {path}")
    return path.read_text(encoding="utf-8")


def connect_with_events(events_df: pd.DataFrame) -> duckdb.DuckDBPyConnection:
    """Open an in-memory DuckDB connection with ``events_df`` registered as ``events``."""
    con = duckdb.connect()
    con.register("events", events_df)
    return con


def build_sessions(events_df: pd.DataFrame) -> pd.DataFrame:
    """Register ``events_df`` and execute ``sql/02_sessions.sql``.

    Returns one row per session: ``user_id, ab_group, start_ts, end_ts, pdate``.
    ``pdate`` is returned as a Python ``date`` (DuckDB hands it back as a
    timestamp, so it is normalised here to match ``load_events``).
    """
    con = connect_with_events(events_df)
    try:
        sessions = con.execute(load_sql("02_sessions.sql")).df()
    finally:
        con.close()

    sessions = sessions.loc[:, SESSION_COLUMNS]
    sessions["pdate"] = pd.to_datetime(sessions["pdate"]).dt.date
    return sessions
