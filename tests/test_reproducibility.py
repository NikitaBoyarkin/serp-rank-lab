"""Reproducibility contract: a clean clone must build itself from a seed.

Three properties, one per test group:

1. ``run.py --stage all`` completes in a copy of the repository whose ``data/``
   directory is empty -- no third-party feeds, no prebuilt parquet.
2. The raw generator is deterministic: same seed -> byte-identical CSVs.
3. The generated raw files satisfy the schema ``src.data_ingest`` expects, so
   the loader is never weakened to accommodate the fixture.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.data_ingest import LEG_COLUMNS, load_bookings, load_events, load_orders
from src.synthetic_raw import SyntheticConfig, generate_raw, main, target_paths

REPO_ROOT = Path(__file__).resolve().parents[1]

# Every parquet the pipeline is contracted to produce.
EXPECTED_PARQUET = (
    "bookings_legs",
    "orders",
    "events",
    "sessions",
    "serp_offers",
    "funnel_by_position",
    "partner_quality",
    "search_funnel",
    "revenue_per_search",
    "revenue_by_position",
)

_IGNORE = shutil.ignore_patterns(
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "data",
    "reports",
    "notebooks",
    "*.parquet",
)


def _copy_repository(destination: Path) -> Path:
    """Copy the working tree without ignored artifacts, then create empty data dirs."""
    shutil.copytree(REPO_ROOT, destination, ignore=_IGNORE)
    (destination / "data" / "raw").mkdir(parents=True)
    (destination / "data" / "processed").mkdir(parents=True)
    return destination


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pipeline_runs_from_empty_data_dir(tmp_path):
    """The acceptance test for REQ-P01: clone -> uv sync -> run.py --stage all."""
    repo = _copy_repository(tmp_path / "repo")

    env = dict(os.environ)
    env.pop("AVIASALES_SOURCE_DIR", None)  # force the seeded path, not local feeds
    env["MPLBACKEND"] = "Agg"

    result = subprocess.run(
        [sys.executable, "run.py", "--stage", "all"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, (
        f"pipeline failed from an empty data dir\n"
        f"--- stdout ---\n{result.stdout[-4000:]}\n"
        f"--- stderr ---\n{result.stderr[-4000:]}"
    )

    # The raw stage must have materialised the feeds from the seed.
    raw = target_paths(repo / "data" / "raw")
    for feed, path in raw.items():
        assert path.exists(), f"raw stage did not write the {feed} feed: {path}"
        assert path.stat().st_size > 0

    processed = repo / "data" / "processed"
    for name in EXPECTED_PARQUET:
        path = processed / f"{name}.parquet"
        assert path.exists(), f"pipeline did not write {path.name}"
        assert len(pd.read_parquet(path)) > 0, f"{path.name} is empty"


@pytest.fixture(scope="module")
def generated_raw(tmp_path_factory):
    """One seeded generation shared by the determinism and schema tests."""
    out_dir = tmp_path_factory.mktemp("raw_a")
    counts = generate_raw(out_dir, SyntheticConfig(seed=7))
    return out_dir, counts


def test_synthetic_raw_is_deterministic(tmp_path, generated_raw):
    """Same seed -> identical bytes; a different seed -> different bytes."""
    first_dir, counts = generated_raw
    second_dir = tmp_path / "raw_b"
    second_counts = generate_raw(second_dir, SyntheticConfig(seed=7))
    assert counts == second_counts

    for feed in counts:
        assert _digest(target_paths(first_dir)[feed]) == _digest(target_paths(second_dir)[feed]), (
            f"{feed} feed is not reproducible from the seed"
        )

    third_dir = tmp_path / "raw_c"
    generate_raw(third_dir, SyntheticConfig(seed=8))
    assert _digest(target_paths(first_dir)["events"]) != _digest(
        target_paths(third_dir)["events"]
    ), "the seed does not actually drive the generator"


def test_raw_stage_skips_when_feeds_already_exist(generated_raw):
    """`run.py --stage all` must not clobber feeds that are already there."""
    out_dir, counts = generated_raw
    before = {feed: _digest(path) for feed, path in target_paths(out_dir).items()}

    assert main(["--out-dir", str(out_dir)]) == 0

    after = {feed: _digest(path) for feed, path in target_paths(out_dir).items()}
    assert after == before
    assert counts  # the fixture really did generate something


def test_raw_generator_matches_loader_schema(generated_raw):
    """The unmodified loaders accept the fixture and expose the contract columns."""
    out_dir, counts = generated_raw
    paths = target_paths(out_dir)

    legs = load_bookings(paths["bookings"])
    assert legs.columns.tolist() == LEG_COLUMNS
    assert counts["bookings"] == 135_508
    assert legs["booking_id"].nunique() == 135_508
    assert set(legs["booking_month"].unique()) <= set(range(1, 9))
    assert legs["origin"].notna().all() and legs["destination"].notna().all()
    assert legs["has_baggage"].dtype == bool
    assert legs["is_connection"].dtype == bool
    assert 0.005 < legs["is_connection"].mean() < 0.02

    routes = legs.groupby(["origin", "destination"]).size()
    assert len(routes) == 20, "the SERP simulator anchors on exactly 20 OD pairs"

    orders = load_orders(paths["orders"])
    assert len(orders) == counts["orders"] == 8_966
    assert orders["order_id"].nunique() == 8_803  # 163 multi-service duplicates
    assert pd.api.types.is_datetime64_any_dtype(orders["order_date"])
    assert {"refund", "changes", "notification", "support"} <= set(
        orders["service"].dropna().unique()
    )
    assert orders["is_return"].between(0, 1).all()
    assert 0.20 < orders["is_return"].mean() < 0.30

    events = load_events(paths["events"])
    assert len(events) == counts["events"] == 44_568
    assert events["user_id"].nunique() == 5_000
    assert pd.api.types.is_datetime64_any_dtype(events["ts"])
    assert events.groupby("ab_group")["user_id"].nunique().to_dict() == {
        "A": 2_500,
        "B": 2_500,
    }
    # One user belongs to one arm.
    assert events.groupby("user_id")["ab_group"].nunique().eq(1).all()
