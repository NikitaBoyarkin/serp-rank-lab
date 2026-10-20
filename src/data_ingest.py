"""Ingest the real Aviasales test-task datasets into the SERP Rank Lab warehouse layer.

Three source files are read (read-only, they live outside this project):

1. ``data.csv``               -- 135,508 flight bookings, one JSON blob of legs each.
2. ``booking_analytics_test_df.csv`` -- 8,966 paid service orders.
3. ``task_1_events.csv``      -- 44,568 raw product events for the A/B assignment.

The booking feed is *denormalised on purpose*: a booking may contain several
flight legs (connections), so it is exploded to one row per leg while every
booking-level field stays attached to each leg. That leg-level grain is what
the SERP ranking simulator in Phase 2 consumes.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# The three source feeds are third-party interview datasets and are **not** in this
# repository (see README "Data"). Resolve their directory at call time from
# ``AVIASALES_SOURCE_DIR`` (or ``--source-dir``), falling back to ``data/raw/`` --
# the directory must contain the three subdirectories named in ``SOURCE_SUBDIRS``.
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "data" / "raw"

SOURCE_SUBDIRS = {
    "bookings": "Aviastats Аналитик",
    "orders": "Booking — Продуктовый аналитик",
    "events": "Продуктовый аналитик-2",
}
SOURCE_FILES = {
    "bookings": "data.csv",
    "orders": "booking_analytics_test_df.csv",
    "events": "task_1_events.csv",
}


def source_dir() -> Path:
    """Directory holding the source feeds: ``AVIASALES_SOURCE_DIR`` or ``data/raw``."""
    override = os.environ.get("AVIASALES_SOURCE_DIR")
    return Path(override).expanduser() if override else DEFAULT_SOURCE_DIR


def source_path(kind: str) -> Path:
    """Resolved path of one source feed (``bookings`` | ``orders`` | ``events``)."""
    return source_dir() / SOURCE_SUBDIRS[kind] / SOURCE_FILES[kind]


LEG_COLUMNS = [
    "booking_id",
    "booking_month",
    "passengers",
    "price",
    "leg_index",
    "origin",
    "destination",
    "airline",
    "baggage",
    "is_connection",
    "has_baggage",
]

BAGGAGE_TOKEN = "With baggage"


def _require(path: Path) -> Path:
    """Fail loudly when a required source file is absent."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required source dataset not found: {path}\n"
            f"Expected the Aviasales test-task feeds under: {source_dir()}\n"
            f"Set AVIASALES_SOURCE_DIR or pass --source-dir (see README 'Data')."
        )
    return path


def _resolve(kind: str, path: Path | str | None) -> Path:
    """Explicit path wins; otherwise resolve ``kind`` against the source directory."""
    return _require(Path(path) if path is not None else source_path(kind))


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #


def load_bookings(path: Path | str | None = None) -> pd.DataFrame:
    """Read the bookings feed and explode ``flights_info`` to one row per leg.

    Returns a leg-level frame carrying all booking-level fields denormalised,
    plus ``is_connection`` (leg_index > 0) and ``has_baggage``.

    Data quality: the leg JSON is *ragged*. A subset of real legs carries no
    ``origin``, no ``destination``, no ``airline`` or no ``baggage`` key at all.
    Those fields are therefore cast to nullable ``string`` (``pd.NA``) rather than
    dropped or imputed, so downstream counts stay honest. ``baggage`` alone falls
    back to ``"No baggage"`` because ``has_baggage`` is a boolean flag that must
    not be nullable.
    """
    path = _resolve("bookings", path)
    bookings = pd.read_csv(path)

    # flights_info arrives as a JSON string holding a list of legs.
    bookings["flights_info"] = bookings["flights_info"].map(json.loads)
    bookings["leg_index"] = bookings["flights_info"].map(lambda legs: list(range(len(legs))))

    exploded = bookings.explode(["flights_info", "leg_index"], ignore_index=True)
    if exploded.empty:
        raise ValueError(f"Bookings feed produced no legs: {path}")

    legs = pd.json_normalize(exploded["flights_info"].tolist()).reset_index(drop=True)
    out = pd.concat(
        [
            exploded.drop(columns=["flights_info"]).reset_index(drop=True),
            legs,
        ],
        axis=1,
    )

    for column in ("booking_month", "passengers", "price", "leg_index"):
        out[column] = pd.to_numeric(out[column], errors="raise").astype("int64")

    # Ragged JSON: keep absent origin/destination/airline as pd.NA, never impute.
    for column in ("origin", "destination", "airline"):
        out[column] = out[column].astype("string")

    out["baggage"] = out["baggage"].fillna("No baggage").astype("string")
    out["is_connection"] = out["leg_index"] > 0
    # .eq() on a nullable string dtype yields a nullable boolean; force plain bool.
    out["has_baggage"] = out["baggage"].eq(BAGGAGE_TOKEN).fillna(False).astype(bool)

    return out.loc[:, LEG_COLUMNS]


def load_orders(path: Path | str | None = None) -> pd.DataFrame:
    """Read the paid-service orders feed, coercing ``order_date`` to datetime."""
    path = _resolve("orders", path)
    orders = pd.read_csv(path)
    orders["order_date"] = pd.to_datetime(orders["order_date"], errors="raise")
    return orders


def load_events(path: Path | str | None = None) -> pd.DataFrame:
    """Read the raw product events feed, parsing ``ts`` and ``pdate``."""
    path = _resolve("events", path)
    events = pd.read_csv(path)
    events["ts"] = pd.to_datetime(events["ts"], errors="raise")
    events["pdate"] = pd.to_datetime(events["pdate"], errors="raise").dt.date
    return events


# --------------------------------------------------------------------------- #
# Writers
# --------------------------------------------------------------------------- #


def write_parquet(df: pd.DataFrame, name: str) -> Path:
    """Write ``df`` to ``data/processed/<name>.parquet`` and return the path."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    target = PROCESSED_DIR / f"{name}.parquet"
    df.to_parquet(target, index=False)
    return target


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

_STAGES = ("bookings", "orders", "events", "all")

_LOADERS: dict[str, tuple[Any, str, str]] = {
    "bookings": (load_bookings, "bookings", "bookings_legs"),
    "orders": (load_orders, "orders", "orders"),
    "events": (load_events, "events", "events"),
}


def run_stage(stage: str) -> dict[str, int]:
    """Materialise one stage; returns ``{parquet_name: row_count}``."""
    loader, kind, name = _LOADERS[stage]
    print(f"=== ingest: {stage} ===")
    print(f"source: {source_path(kind)}")
    frame = loader()
    target = write_parquet(frame, name)
    print(f"rows:   {len(frame):,}")
    print(f"cols:   {len(frame.columns)}")
    print(f"written: {target}")
    if stage == "bookings":
        print_ragged_note(frame)
    return {name: len(frame)}


def print_ragged_note(legs: pd.DataFrame) -> None:
    """Report legs whose JSON omitted a key -- the feed really is ragged."""
    for column in ("origin", "destination", "airline"):
        missing = int(legs[column].isna().sum())
        print(f"  legs without {column:<12}: {missing:,}")
    print(f"  connecting legs (leg_index > 0): {int(legs['is_connection'].sum()):,}")
    print(f"  legs with baggage:              {int(legs['has_baggage'].sum()):,}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.data_ingest",
        description="Ingest the Aviasales test-task datasets into data/processed/*.parquet",
    )
    parser.add_argument(
        "--stage",
        choices=_STAGES,
        default="all",
        help="which dataset to materialise (default: all)",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="directory holding the source feeds (overrides AVIASALES_SOURCE_DIR)",
    )
    args = parser.parse_args(argv)

    if args.source_dir is not None:
        os.environ["AVIASALES_SOURCE_DIR"] = str(args.source_dir)

    stages = list(_LOADERS) if args.stage == "all" else [args.stage]

    counts: dict[str, int] = {}
    for stage in stages:
        counts.update(run_stage(stage))
        print()

    print("=== ingest summary ===")
    for name, rows in counts.items():
        print(f"{name:<16} {rows:>10,} rows")

    if "bookings_legs" in counts and "bookings" in stages:
        legs = counts["bookings_legs"]
        print(f"\nbooking legs expanded: {legs:,} leg rows from 135,508 bookings")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
