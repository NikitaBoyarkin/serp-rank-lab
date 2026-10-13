"""Tests for re-anchoring the simulator on measurable anchors (Phase 4).

``docs/adr/0001`` keeps ``serp_offers`` (modelled) and the order/leg feeds
(measured) apart. This module reads anchors out of the measured islands only and
writes them into the simulator's knobs only. These tests pin three properties:

* the measured anchor really is the planted dispersion (not a proxy);
* the anchored config moves *only* the knobs the anchors identify;
* the sensitivity sweep answers the "is this just your parameters?" objection
  with a verdict rather than a number (**A3**).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.calibration import (
    anchor_config,
    anchor_partners,
    anchors_from_islands,
    partner_index_sigma,
    within_route_price_sigma,
)
from src.ranking_eval import sensitivity_sweep
from src.serp_sim import PARTNERS, SimConfig, load_routes, load_user_ids, simulate_serps

_ARTIFACT = Path(__file__).resolve().parents[1] / "data" / "processed" / "bookings_legs.parquet"
_NEEDS_DATA_REASON = (
    "needs data/processed/bookings_legs.parquet, which is built from the gitignored "
    "raw Aviasales feeds and therefore unavailable in CI"
)
NEEDS_DATA = pytest.mark.skipif(not _ARTIFACT.exists(), reason=_NEEDS_DATA_REASON)

ROUTE_SIGMAS = {("SVO", "LED"): 0.20, ("SVO", "AER"): 0.35}
MONTH_COUNTS = {1: 30, 2: 90, 3: 60}
LEGS_PER_ROUTE = sum(MONTH_COUNTS.values())


@pytest.fixture(scope="module")
def legs():
    """Two routes with planted log-price dispersion and an uneven month mix."""
    rng = np.random.default_rng(0)
    rows = []
    for (origin, destination), sigma in ROUTE_SIGMAS.items():
        for month, count in MONTH_COUNTS.items():
            for _ in range(count):
                rows.append(
                    {
                        "origin": origin,
                        "destination": destination,
                        "price": float(np.exp(6.0 + sigma * rng.normal())),
                        "booking_month": month,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def orders():
    return pd.DataFrame(
        {
            "order_price": [10_000.0, 20_000.0, 30_000.0, 40_000.0],
            "service_price": [0.0, 2_000.0, 1_000.0, 0.0],
            "service_profit": [0.0, 600.0, 250.0, 0.0],
        }
    )


# --------------------------------------------------------------------------- #
# The anchors are measurements
# --------------------------------------------------------------------------- #


def test_within_route_sigma_recovers_the_planted_dispersion(legs):
    expected = float(np.mean(list(ROUTE_SIGMAS.values())))
    assert within_route_price_sigma(legs, min_legs=30) == pytest.approx(expected, abs=0.03)


def test_within_route_sigma_drops_routes_below_the_leg_threshold(legs):
    tiny = pd.DataFrame(
        {
            "origin": ["SVO"] * 5,
            "destination": ["HKT"] * 5,
            "price": np.linspace(1e4, 9e4, 5),
            "booking_month": [1] * 5,
        }
    )
    with_tiny = pd.concat([legs, tiny], ignore_index=True)
    expected = float(np.mean(list(ROUTE_SIGMAS.values())))
    assert within_route_price_sigma(with_tiny, min_legs=30) == pytest.approx(expected, abs=0.03)

    # Raise the threshold above every route's size and nothing is measurable.
    with pytest.raises(ValueError):
        within_route_price_sigma(with_tiny, min_legs=1_000)


def test_anchors_report_levels_and_unit_economics(orders, legs):
    anchors = anchors_from_islands(orders, legs)
    assert anchors.n_orders == 4
    assert anchors.n_legs == 2 * LEGS_PER_ROUTE
    assert anchors.order_price_median == pytest.approx(25_000.0)
    assert anchors.order_price_q10 == pytest.approx(13_000.0)
    assert anchors.service_attach_rate == pytest.approx(0.5)
    assert anchors.service_margin_median == pytest.approx(0.275)
    assert sum(anchors.month_share.values()) == pytest.approx(1.0)


def test_anchors_refuse_an_orders_frame_without_usable_prices(legs):
    with pytest.raises(ValueError):
        anchors_from_islands(pd.DataFrame({"order_price": [0.0, -1.0]}), legs)


# --------------------------------------------------------------------------- #
# The anchored config moves exactly the identified knobs
# --------------------------------------------------------------------------- #


def test_anchor_config_solves_price_sigma_from_the_measured_spread(orders, legs):
    anchors = anchors_from_islands(orders, legs)
    cfg = anchor_config(anchors)
    index_sigma = partner_index_sigma()
    assert float(np.hypot(cfg.price_sigma, index_sigma)) == pytest.approx(
        anchors.within_route_log_price_sigma, rel=1e-9
    )
    assert cfg.price_sigma < anchors.within_route_log_price_sigma


def test_anchor_config_moves_only_price_sigma_and_the_calendar(orders, legs):
    base = SimConfig()
    cfg = anchor_config(anchors_from_islands(orders, legs))
    changed = {f for f in base.__dataclass_fields__ if getattr(base, f) != getattr(cfg, f)}
    assert changed == {"price_sigma", "months", "month_weights"}


def test_anchor_config_normalises_the_measured_month_weights(orders, legs):
    cfg = anchor_config(anchors_from_islands(orders, legs))
    assert sum(cfg.month_weights) == pytest.approx(1.0)
    total = float(sum(MONTH_COUNTS.values()))
    expected = {m: c / total for m, c in MONTH_COUNTS.items()}
    assert dict(zip(cfg.months, cfg.month_weights, strict=True)) == pytest.approx(expected)


def test_anchor_partners_recentres_without_reordering():
    anchored = anchor_partners()
    index = np.array([float(p["price_index"]) for p in anchored])
    assert float(np.median(index)) == pytest.approx(1.0)
    assert [p["partner"] for p in anchored] == [p["partner"] for p in PARTNERS]
    assert np.array_equal(
        np.argsort(index), np.argsort([float(p["price_index"]) for p in PARTNERS])
    )


# --------------------------------------------------------------------------- #
# A3 -- the sweep returns a verdict, not just numbers
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def sweep_offers():
    return simulate_serps(load_routes(), load_user_ids(), SimConfig(n_searches=300, seed=3))


@NEEDS_DATA
def test_a3_eta_axis_covers_the_grid_and_states_a_verdict(sweep_offers):
    sweep = sensitivity_sweep(sweep_offers, k=10, eta_grid=(0.6, 1.0, 1.4), n_searches=200)
    assert [row["eta"] for row in sweep["eta"]] == [0.6, 1.0, 1.4]
    assert sweep["n_searches"] == 200
    assert sweep["verdict"]
    assert sweep["verdict"].startswith(("does not vanish", "holds only", "vanishes"))


@NEEDS_DATA
def test_a3_every_eta_row_is_a_boolean_verdict_with_both_deltas(sweep_offers):
    sweep = sensitivity_sweep(sweep_offers, k=10, eta_grid=(0.6, 1.4), n_searches=200)
    for row in sweep["eta"]:
        assert isinstance(row["divergence_holds"], bool)
        assert row["divergence_holds"] == (row["ndcg_delta"] > 0 and row["revenue_delta"] < 0)
        assert np.isfinite(row["ndcg_delta"]) and np.isfinite(row["revenue_delta"])


@NEEDS_DATA
def test_a3_tilt_axis_is_skipped_without_the_route_feeds(sweep_offers):
    sweep = sensitivity_sweep(sweep_offers, k=10, eta_grid=(1.0,), n_searches=100)
    assert sweep["money_tilt"] == []


@NEEDS_DATA
def test_a3_tilt_axis_regenerates_serps_and_the_tilt_moves_the_logged_order(sweep_offers):
    """Mutation control: a tilt that changed nothing would make the axis vacuous."""
    sweep = sensitivity_sweep(
        sweep_offers,
        k=10,
        eta_grid=(1.0,),
        tilt_grid=(0.0, 1.0),
        n_searches=150,
        routes=load_routes(),
        user_ids=load_user_ids(),
    )
    rows = sweep["money_tilt"]
    assert [row["money_tilt"] for row in rows] == [0.0, 1.0]
    for row in rows:
        assert np.isfinite(row["logged_ndcg_at_k"])
        assert np.isfinite(row["logged_revenue_per_search"])
    assert rows[0]["logged_ndcg_at_k"] != rows[1]["logged_ndcg_at_k"]


@NEEDS_DATA
def test_a3_sampling_whole_searches_is_deterministic(sweep_offers):
    a = sensitivity_sweep(sweep_offers, eta_grid=(1.0,), n_searches=100, seed=5)
    b = sensitivity_sweep(sweep_offers, eta_grid=(1.0,), n_searches=100, seed=5)
    assert a == b
