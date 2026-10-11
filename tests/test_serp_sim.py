"""Contract tests for the SERP simulator's grain and determinism.

The grain contract: a SERP is one user's search, so every offer that shares a
``search_id`` must also share the same ``user_id``. This was violated once (the
user was drawn per offer), which silently broke every per-user metric downstream
-- the A/B readout, CUPED and the ratio test all assume user coherence.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.serp_sim import SimConfig, load_routes, load_user_ids, simulate_serps

SMALL = SimConfig(n_searches=300)


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
