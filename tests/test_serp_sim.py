"""Contract tests for the SERP simulator's grain and determinism.

The grain contract: a SERP is one user's search, so every offer that shares a
``search_id`` must also share the same ``user_id``. This was violated once (the
user was drawn per offer), which silently broke every per-user metric downstream
-- the A/B readout, CUPED and the ratio test all assume user coherence.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.serp_sim import PARTNERS, SimConfig, load_routes, load_user_ids, simulate_serps

_ARTIFACT = Path(__file__).resolve().parents[1] / "data" / "processed" / "bookings_legs.parquet"
pytestmark = pytest.mark.skipif(
    not _ARTIFACT.exists(),
    reason=(
        "needs data/processed/bookings_legs.parquet, which is built from the gitignored "
        "raw Aviasales feeds and therefore unavailable in CI"
    ),
)

SMALL = SimConfig(n_searches=300)

# Captured on the default config *before* ``simulate_serps`` gained its optional
# ``partners`` argument. These numbers must not move: if they do, the default path
# stopped being the default path and every Phase 2/3 result silently shifts.
GOLDEN = {
    "rows": 2847,
    "cols": 19,
    "revenue": 324092.788229,
    "clicks": 450,
    "bookings": 93,
    "sha256": "89a57d39069ec474d7acef46fb04a8a09b21b8f6209aa12873baf88f818e111f",
}


@pytest.fixture(scope="module")
def offers():
    return simulate_serps(load_routes(), load_user_ids(), SMALL)


def test_one_user_per_search(offers):
    """No search_id may span more than one user."""
    leaky = offers.groupby("search_id")["user_id"].nunique().gt(1).sum()
    assert leaky == 0


def test_every_user_is_a_real_user_id(offers):
    real = set(load_user_ids())
    assert set(offers["user_id"]).issubset(real)


def test_positions_are_dense_and_unique_within_a_search(offers):
    per_search = offers.groupby("search_id")["position"]
    assert (per_search.min() == 1).all()
    assert per_search.apply(lambda s: s.is_unique).all()


def test_simulation_is_deterministic():
    routes, users = load_routes(), load_user_ids()
    a = simulate_serps(routes, users, SMALL)
    b = simulate_serps(routes, users, SMALL)
    assert a.equals(b)


def test_different_seeds_differ():
    routes, users = load_routes(), load_user_ids()
    a = simulate_serps(routes, users, SMALL)
    b = simulate_serps(routes, users, SimConfig(n_searches=300, seed=7))
    assert not a.equals(b)


def test_revenue_is_non_negative(offers):
    assert np.all(offers["commission"].to_numpy() >= 0.0)


def test_default_run_is_bit_identical_to_the_captured_snapshot(offers):
    """A5: injecting partners must not have moved the default path at all."""
    assert len(offers) == GOLDEN["rows"]
    assert offers.shape[1] == GOLDEN["cols"]
    assert float(offers["commission"].sum()) == pytest.approx(GOLDEN["revenue"], abs=1e-6)
    assert int(offers["clicked"].sum()) == GOLDEN["clicks"]
    assert int(offers["booked"].sum()) == GOLDEN["bookings"]
    digest = hashlib.sha256(offers.to_csv(index=False).encode()).hexdigest()
    assert digest == GOLDEN["sha256"]


def test_passing_the_module_partner_table_explicitly_is_a_no_op(offers):
    """``partners=None`` means ``PARTNERS``, so an explicit pass must be identical."""
    explicit = simulate_serps(load_routes(), load_user_ids(), SMALL, partners=PARTNERS)
    pd.testing.assert_frame_equal(offers, explicit)


def test_an_alternative_partner_table_changes_the_economics(offers):
    """Mutation control for the snapshot above: the knob is live, not decorative."""
    dearer = [dict(p, price_index=float(p["price_index"]) * 2.0) for p in PARTNERS]
    other = simulate_serps(load_routes(), load_user_ids(), SMALL, partners=dearer)
    assert not offers.equals(other)
