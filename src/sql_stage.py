"""SQL stage for SERP Rank Lab: materialise the business SQL views.

    python run.py --stage sql

Runs ``sql/03_funnel.sql`` and ``sql/04_revenue_per_search.sql`` against
``data/processed/serp_offers.parquet`` (registered as ``serp_offers``) and
writes every view the two files define back to ``data/processed/<view>.parquet``.
``serp_offers`` carries both ``partner`` (name) and ``partner_type`` (CPA/CPC),
so the SQL's ``partner`` reference resolves without a rename.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SQL_DIR = PROJECT_ROOT / "sql"
SERP_OFFERS_PARQUET = PROCESSED_DIR / "serp_offers.parquet"

# SQL files in execution order, and the views they define.
_SQL_FILES = ["03_funnel.sql", "04_revenue_per_search.sql"]
_VIEWS = [
    "funnel_by_position",
    "partner_quality",
    "search_funnel",
    "revenue_per_search",
    "revenue_by_position",
]


def build(serp_offers: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Execute the SQL layer over ``serp_offers``; return one frame per view."""
    con = duckdb.connect()
    try:
        con.register("serp_offers", serp_offers)
        for name in _SQL_FILES:
            con.execute((SQL_DIR / name).read_text(encoding="utf-8"))
        return {view: con.execute(f"SELECT * FROM {view}").df() for view in _VIEWS}
    finally:
        con.close()


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(prog="python -m src.sql_stage").parse_args(argv)

    if not SERP_OFFERS_PARQUET.exists():
        raise FileNotFoundError(f"Missing {SERP_OFFERS_PARQUET}. Run `run.py --stage serp` first.")

    print("=== sql ===")
    frames = build(pd.read_parquet(SERP_OFFERS_PARQUET))
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for view, frame in frames.items():
        frame.to_parquet(PROCESSED_DIR / f"{view}.parquet", index=False)
        print(f"{view:22s} rows: {len(frame):>8,}  -> {view}.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
