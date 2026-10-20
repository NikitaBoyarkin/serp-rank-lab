"""Tests for the SERP ranking evaluation (Phase 2).

Two of these are load-bearing and must be *real numeric assertions*:

* ``test_ipw_recovers_true_ctr_*`` / ``test_uniform_propensity_fails_*`` -- the
  PBM recovery test and its mutation control. Together they encode that the
  inverse-propensity correction recovers the position-free click rate **and**
  that substituting a uniform propensity (the classic bug) does not.
* ``test_ndcg_vs_revenue_divergence_*`` -- the project's headline claim, on a
  hand-built case whose revenue and NDCG deltas are computed by hand below.

The rest pin NDCG/MRR formulas, the interleaving decision, simulator
determinism, and a SQL <-> pandas revenue cross-check.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from src.policies import context_for, train_test_split_searches
from src.ranking_eval import (
    REPORT_JSON,
    breakeven_rub_per_forgone_click,
    compare_rankings,
    estimate_propensity_em,
    headline_report,
    interleaving_experiment,
    mrr,
    naive_ctr,
    ndcg_at_k,
    policy_scorecard,
    team_draft_interleaving,
    unbiased_ctr,
)
from src.serp_sim import CONTRACT_COLUMNS, SimConfig, simulate_serps

# --------------------------------------------------------------------------- #
# PBM recovery + mutation control
# --------------------------------------------------------------------------- #


def _pbm_sample(n: int = 60_000, k_max: int = 10, eta: float = 1.0, seed: int = 123):
    """Impressions whose click probability follows the PBM exactly.

    ``P(click) = (1/k)**eta * relevance``, with a known intrinsic relevance per
    impression. The position-free click rate (the quantity IPW must recover) is
    ``mean(relevance)``.
    """
    rng = np.random.default_rng(seed)
    positions = rng.integers(1, k_max + 1, n)
    relevance = rng.uniform(0.2, 0.9, n)
    p_click = (1.0 / positions) ** eta * relevance
    clicked = rng.random(n) < p_click
    return clicked, positions, relevance, eta


def test_ipw_recovers_true_ctr_while_naive_is_biased_low():
    clicked, positions, relevance, eta = _pbm_sample()
    true_ctr = float(relevance.mean())

    naive = naive_ctr(clicked)
    corrected = unbiased_ctr(clicked, positions, eta=eta)

    # The naive observed CTR is materially biased downward ...
    assert naive < true_ctr - 0.10
    # ... and the inverse-propensity correction recovers the true rate.
    assert abs(corrected - true_ctr) < 0.02


def test_uniform_propensity_does_not_recover_true_ctr():
    """Mutation control: the correction is only correct with the real curve."""
    clicked, positions, relevance, eta = _pbm_sample()
    true_ctr = float(relevance.mean())

    uniform = unbiased_ctr(clicked, positions, propensity=np.ones(len(clicked)))

    # A uniform propensity collapses the estimator back onto the naive CTR ...
    assert abs(uniform - naive_ctr(clicked)) < 1e-9
    # ... so it does NOT recover the true rate: the propensity curve matters.
    assert abs(uniform - true_ctr) > 0.10


def test_em_recovers_a_known_examination_curve():
    """The log-only estimator recovers a known curve when groups repeat."""
    rng = np.random.default_rng(2024)
    n, k_max, eta, n_groups = 60_000, 8, 1.0, 8
    positions = rng.integers(1, k_max + 1, n)
    groups = rng.integers(0, n_groups, n)
    relevance = rng.uniform(0.15, 0.95, n_groups)[groups]
    clicked = rng.random(n) < (1.0 / positions) ** eta * relevance

    estimated = estimate_propensity_em(clicked, positions, groups)

    assert estimated[1] == pytest.approx(1.0)
    for k in range(1, k_max + 1):
        assert estimated[k] == pytest.approx((1.0 / k) ** eta, abs=0.05)


# --------------------------------------------------------------------------- #
# Offline metrics -- hand-computed
# --------------------------------------------------------------------------- #


def test_ndcg_matches_hand_computed_value():
    # NDCG@3 of [1.0, 0.0, 0.5]:
    #   DCG  = 1/1 + 0/log2(3) + 0.5/log2(4) = 1.25
    #   IDCG = 1/1 + 0.5/log2(3)             = 1.3154648768
    assert ndcg_at_k([1.0, 0.0, 0.5], 3) == pytest.approx(1.25 / 1.3154648768, abs=1e-9)
    # Perfectly sorted is the ideal ordering.
    assert ndcg_at_k([1.0, 0.5, 0.0], 3) == pytest.approx(1.0)
    # NDCG@1 of an unsorted list is the ratio of its first gain to the max.
    assert ndcg_at_k([0.5, 1.0], 1) == pytest.approx(0.5)


def test_ndcg_is_lower_when_the_best_item_is_pushed_down():
    assert ndcg_at_k([0.9, 0.3], 2) > ndcg_at_k([0.3, 0.9], 2)


def test_mrr_matches_hand_computed_values():
    assert mrr([0.1, 0.9, 0.2], threshold=0.5) == pytest.approx(0.5)  # rank 2
    assert mrr([0.9], threshold=0.5) == pytest.approx(1.0)  # rank 1
    assert mrr([0.1, 0.2], threshold=0.5) == pytest.approx(0.0)  # none relevant


def test_mrr_is_part_of_every_policy_row():
    """MRR is computed, not promised (REQ-T04 #1).

    On the hand-built SERP the relevance policy puts the only relevant offer
    (0.9) first -> MRR 1.0; the revenue policy puts it second -> MRR 0.5.
    """
    table = compare_rankings(_two_offer_search(), k=2).set_index("policy")
    assert "mrr" in table.columns
    assert float(table.loc["relevance", "mrr"]) == pytest.approx(1.0)
    assert float(table.loc["revenue", "mrr"]) == pytest.approx(0.5)


@pytest.mark.skipif(not REPORT_JSON.exists(), reason="needs reports/ranking_eval.json")
def test_the_committed_report_carries_mrr_per_policy():
    payload = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    assert "mrr_by_policy" in payload
    assert {"relevance", "price_asc"} <= set(payload["mrr_by_policy"])


# --------------------------------------------------------------------------- #
# Team-draft interleaving
# --------------------------------------------------------------------------- #


def test_team_draft_interleaving_is_a_balanced_permutation():
    rank_a = ["x", "y", "z", "w"]
    rank_b = ["z", "w", "x", "y"]
    rng = np.random.default_rng(0)

    interleaved = team_draft_interleaving(rank_a, rank_b, rng)
    items = [item for item, _ in interleaved]
    teams = [team for _, team in interleaved]

    assert sorted(items) == ["w", "x", "y", "z"]  # every offer exactly once
    assert set(teams) <= {"A", "B"}
    assert abs(teams.count("A") - teams.count("B")) <= 1  # equal exposure

    # Identical seeds reproduce the identical interleaving.
    again = team_draft_interleaving(rank_a, rank_b, np.random.default_rng(0))
    assert again == interleaved


def test_interleaving_prefers_the_higher_relevance_ranker():
    relevance = {"x": 0.95, "y": 0.90, "z": 0.15, "w": 0.10}
    rank_good = ["x", "y", "z", "w"]  # relevance-sorted
    rank_bad = ["z", "w", "x", "y"]  # relevance-ignoring

    result = interleaving_experiment(rank_good, rank_bad, relevance, eta=1.0, n_users=4000, seed=7)

    assert result["decision"] == "A"
    assert result["win_rate_a"] > result["win_rate_b"]


# --------------------------------------------------------------------------- #
# Headline: NDCG rises while revenue falls
# --------------------------------------------------------------------------- #


def _two_offer_search() -> pd.DataFrame:
    """One SERP where the relevance-best offer is the *less* valuable one.

    TripMasters (CPA rate 0.13) quotes a high price; OTA Global (CPA rate 0.05)
    is cheaper and far more relevant. Expected revenue given a click is
    ``rate * price * redirect * book``: TripMasters 19.9797, OTA Global 1.22496.
    Desktop eta = 1.0, so examination is [1.0, 0.5]. By hand:

    * relevance order [OTA, Trip]: NDCG@2 = 1.0 (ideal);
      revenue = 0.9*1.22496 + 0.5*0.3*19.9797 = 4.099419
    * revenue order   [Trip, OTA]: DCG = 0.3 + 0.9/log2(3), IDCG = 0.9 + 0.3/log2(3),
      NDCG@2 = 0.796708; revenue = 0.3*19.9797 + 0.5*0.9*1.22496 = 6.545142
    """
    return pd.DataFrame(
        [
            {
                "search_id": "S1",
                "position": 1,
                "device": "desktop",
                "partner": "TripMasters",
                "price": 1090.0,
                "relevance": 0.30,
            },
            {
                "search_id": "S1",
                "position": 2,
                "device": "desktop",
                "partner": "OTA Global",
                "price": 96.0,
                "relevance": 0.90,
            },
        ]
    )


def test_ndcg_vs_revenue_divergence_on_a_hand_built_search():
    summary = compare_rankings(_two_offer_search(), k=2)
    delta = summary[summary["policy"] == "delta_relevance_minus_revenue"].iloc[0]

    # Moving from the revenue policy to the relevance policy: NDCG up, revenue down.
    assert delta["ndcg_at_k"] > 0
    assert delta["revenue_per_search"] < 0
    assert delta["ndcg_at_k"] == pytest.approx(0.203292, abs=1e-5)
    assert delta["revenue_per_search"] == pytest.approx(-2.445723, abs=1e-4)


def test_relevance_policy_has_the_highest_ndcg_by_construction():
    summary = compare_rankings(_two_offer_search(), k=2)
    by_policy = summary.set_index("policy")
    assert by_policy.loc["relevance", "ndcg_at_k"] == pytest.approx(1.0)
    assert by_policy.loc["revenue", "ndcg_at_k"] < by_policy.loc["relevance", "ndcg_at_k"]
    assert (
        by_policy.loc["revenue", "revenue_per_search"]
        > by_policy.loc["relevance", "revenue_per_search"]
    )


# --------------------------------------------------------------------------- #
# Breakeven: the price of a forgone click
# --------------------------------------------------------------------------- #


def test_breakeven_is_the_signed_ratio_of_the_two_deltas():
    """``breakeven = dRevenue_per_search / dClicks_per_search``, sign included."""
    assert breakeven_rub_per_forgone_click(10.0, -5.0) == pytest.approx(-2.0)
    assert breakeven_rub_per_forgone_click(-10.0, 5.0) == pytest.approx(-2.0)
    assert breakeven_rub_per_forgone_click(10.0, 5.0) == pytest.approx(2.0)
    assert breakeven_rub_per_forgone_click(-10.0, -5.0) == pytest.approx(2.0)
    assert breakeven_rub_per_forgone_click(0.0, -5.0) == pytest.approx(0.0)


def test_breakeven_sign_agrees_with_the_signs_of_both_deltas():
    for delta_rev, delta_clicks in ((10.0, -5.0), (-10.0, 5.0), (10.0, 5.0), (-10.0, -5.0)):
        value = breakeven_rub_per_forgone_click(delta_rev, delta_clicks)
        assert value is not None
        assert (value > 0) == (np.sign(delta_rev) == np.sign(delta_clicks))


def test_breakeven_is_undefined_when_no_click_is_forgone():
    """No denominator, no number: returning a big figure would be an invention."""
    assert breakeven_rub_per_forgone_click(10.0, 0.0) is None


def test_headline_breakeven_is_recomputable_from_the_report_itself(tiny_routes):
    """The JSON must carry enough to check the rate, not just state it."""
    offers = simulate_serps(tiny_routes, ["u1", "u2", "u3"], SimConfig(seed=11, n_searches=120))
    train, test = train_test_split_searches(offers, test_frac=0.5, seed=0)
    ctx = context_for(test, train=train)

    payload = headline_report(test, ctx, k=10, commit="0" * 12)
    hl = payload["headline"]

    assert hl["clicks_per_search_delta_abs"] != 0
    assert hl["breakeven_rub_per_forgone_click"] == pytest.approx(
        hl["revenue_per_search_delta_abs"] / hl["clicks_per_search_delta_abs"]
    )
    assert np.sign(hl["breakeven_rub_per_forgone_click"]) == np.sign(
        hl["revenue_per_search_delta_abs"]
    ) * np.sign(hl["clicks_per_search_delta_abs"])
    assert hl["information_set"] == "logged"
    assert payload["generated_from"]["searches"] == test["search_id"].nunique()


def test_conversion_metric_is_reported_for_every_policy(tiny_routes):
    """``conversion`` is bookings per click, so it is a rate in (0, 1]."""
    offers = simulate_serps(tiny_routes, ["u1", "u2"], SimConfig(seed=12, n_searches=80))
    ctx = context_for(offers, train=offers)
    card = policy_scorecard(offers, ctx, k=10)
    assert "conversion" in card.columns
    assert ((card["conversion"] > 0) & (card["conversion"] <= 1)).all()


# --------------------------------------------------------------------------- #
# SQL <-> pandas cross-check
# --------------------------------------------------------------------------- #


def _sql(name: str) -> str:
    return (Path(__file__).resolve().parent.parent / "sql" / name).read_text(encoding="utf-8")


def test_revenue_per_search_sql_matches_pandas():
    frame = pd.DataFrame(
        [
            # search A: a CPA booking (commission 500) and a CPC click (18)
            {
                "search_id": "A",
                "user_id": "u1",
                "position": 1,
                "partner": "AviaCenter",
                "price": 5555.0,
                "commission": 500.0,
                "clicked": True,
                "redirected": True,
                "booked": True,
            },
            {
                "search_id": "A",
                "user_id": "u1",
                "position": 2,
                "partner": "MetaFly",
                "price": 5600.0,
                "commission": 38.0,
                "clicked": True,
                "redirected": False,
                "booked": False,
            },
            {
                "search_id": "A",
                "user_id": "u1",
                "position": 3,
                "partner": "GoFare",
                "price": 5700.0,
                "commission": 0.0,
                "clicked": False,
                "redirected": False,
                "booked": False,
            },
            # search B: one CPC click only
            {
                "search_id": "B",
                "user_id": "u2",
                "position": 1,
                "partner": "CloudTrip",
                "price": 5200.0,
                "commission": 22.0,
                "clicked": True,
                "redirected": False,
                "booked": False,
            },
            {
                "search_id": "B",
                "user_id": "u2",
                "position": 2,
                "partner": "SkyTour",
                "price": 5300.0,
                "commission": 0.0,
                "clicked": False,
                "redirected": False,
                "booked": False,
            },
        ]
    )
    con = duckdb.connect()
    try:
        con.register("serp_offers", frame)
        con.execute(_sql("04_revenue_per_search.sql"))
        per_search = con.execute(
            "SELECT search_id, revenue, offers, clicks, bookings FROM revenue_per_search "
            "ORDER BY search_id"
        ).df()
        by_position = con.execute(
            "SELECT position, revenue FROM revenue_by_position ORDER BY position"
        ).df()
    finally:
        con.close()

    expected_revenue = frame.groupby("search_id")["commission"].sum().sort_index()
    assert per_search["search_id"].tolist() == ["A", "B"]
    assert per_search["revenue"].tolist() == pytest.approx(expected_revenue.tolist())
    assert per_search.set_index("search_id").loc["A", "clicks"] == 2
    assert per_search.set_index("search_id").loc["A", "bookings"] == 1
    # Per-position revenue aggregates the same way.
    expected_by_position = frame.groupby("position")["commission"].sum().sort_index()
    assert by_position["revenue"].tolist() == pytest.approx(expected_by_position.tolist())


# --------------------------------------------------------------------------- #
# Simulator contract
# --------------------------------------------------------------------------- #


@pytest.fixture
def tiny_routes() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"origin": "SVO", "destination": "LED", "avg_booking_price": 6006.76},
            {"origin": "VKO", "destination": "LED", "avg_booking_price": 5299.04},
        ]
    )


def test_simulator_is_deterministic_for_a_fixed_seed(tiny_routes):
    users = ["u1", "u2", "u3"]
    config = SimConfig(seed=5, n_searches=40)
    first = simulate_serps(tiny_routes, users, config)
    second = simulate_serps(tiny_routes, users, config)
    assert first.equals(second)


def test_simulator_keeps_the_funnel_contract_columns(tiny_routes):
    offers = simulate_serps(tiny_routes, ["u1", "u2"], SimConfig(seed=1, n_searches=30))
    assert offers.columns[: len(CONTRACT_COLUMNS)].tolist() == CONTRACT_COLUMNS
    assert "relevance" in offers.columns
    # Bookings are only possible after click and redirect.
    assert (offers["booked"] <= offers["redirected"]).all()
    assert (offers["redirected"] <= offers["clicked"]).all()


def test_positions_are_contiguous_per_search(tiny_routes):
    offers = simulate_serps(tiny_routes, ["u1", "u2"], SimConfig(seed=3, n_searches=25))
    for _, frame in offers.groupby("search_id"):
        assert frame["position"].tolist() == list(range(1, len(frame) + 1))


def test_partners_repeat_so_partner_quality_is_meaningful(tiny_routes):
    offers = simulate_serps(tiny_routes, ["u1", "u2"], SimConfig(seed=9, n_searches=200))
    distinct_partners = offers.groupby("search_id")["partner"].nunique()
    offers_per_search = offers.groupby("search_id").size()
    # Every SERP carries more offers than distinct partners (partners repeat).
    assert (distinct_partners < offers_per_search).all()
