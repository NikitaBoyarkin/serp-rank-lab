"""Tests for the ranking-policy registry (Phase 4, increment 1).

The project's headline is that offline relevance (NDCG) and revenue per search
diverge. Phase 2 showed it with two policies that *both* read the simulator's
ground-truth ``relevance`` column, so a skeptic can dismiss the result as "you
compared the ideal order with the greedy one". These tests pin the two things
that make the claim defensible:

* **A1** -- at least two policies that cannot see ``relevance`` produce an order
  different from both references and score below the oracle's 1.0.
* **A2** -- the learned ranker is fitted and scored on disjoint searches, so its
  apparent quality is not memorisation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.policies import (
    POLICIES,
    REALISTIC_POLICIES,
    PolicyContext,
    context_for,
    train_test_split_searches,
)
from src.ranking_eval import compare_rankings
from src.serp_sim import SimConfig, load_routes, load_user_ids, simulate_serps

_ARTIFACT = Path(__file__).resolve().parents[1] / "data" / "processed" / "bookings_legs.parquet"
_NEEDS_DATA_REASON = (
    "needs data/processed/bookings_legs.parquet, which is built from the gitignored "
    "raw Aviasales feeds and therefore unavailable in CI"
)
NEEDS_DATA = pytest.mark.skipif(not _ARTIFACT.exists(), reason=_NEEDS_DATA_REASON)

SIM = SimConfig(n_searches=400, seed=7)
ALL_POLICIES = ("relevance", "revenue", *REALISTIC_POLICIES)


@pytest.fixture(scope="module")
def offers():
    return simulate_serps(load_routes(), load_user_ids(), SIM)


@pytest.fixture(scope="module")
def split(offers):
    return train_test_split_searches(offers, test_frac=0.5, seed=0)


@pytest.fixture(scope="module")
def ctx(split):
    train, _ = split
    return context_for(train, train=train)


def _first_frame(offers):
    group = offers[offers["search_id"] == offers["search_id"].iloc[0]]
    return group.sort_values("position").reset_index(drop=True)


def _orders(offers, ctx, names):
    """Per-search display order for each policy, in a stable search order."""
    out = {name: [] for name in names}
    for _, group in offers.groupby("search_id", sort=True):
        frame = group.sort_values("position").reset_index(drop=True)
        for name in names:
            out[name].append(np.asarray(POLICIES[name](frame, ctx), dtype=int))
    return out


# --------------------------------------------------------------------------- #
# A1 -- realistic policies, and they really cannot see relevance
# --------------------------------------------------------------------------- #


@NEEDS_DATA
def test_a1_at_least_two_realistic_policies_differ_from_both_references(split, ctx):
    """Each realistic policy must disagree with the oracle *and* with the greedy
    optimum on a strict majority of searches -- that is what "not an oracle" means."""
    _, test = split
    orders = _orders(test, ctx, ALL_POLICIES)
    n = len(orders["relevance"])

    def disagreement(name: str, reference: str) -> int:
        return sum(not np.array_equal(orders[name][i], orders[reference][i]) for i in range(n))

    distinct = [
        name
        for name in REALISTIC_POLICIES
        if disagreement(name, "relevance") > n / 2 and disagreement(name, "revenue") > n / 2
    ]
    assert len(distinct) >= 2, f"only {distinct} differ from both references on a majority"


@NEEDS_DATA
def test_a1_realistic_policies_score_below_the_oracle(split, ctx):
    _, test = split
    summary = compare_rankings(test, k=10, policies=ALL_POLICIES, ctx=ctx).set_index("policy")

    oracle = float(summary.loc["relevance", "ndcg_at_k"])
    assert oracle == pytest.approx(1.0)
    for name in REALISTIC_POLICIES:
        assert float(summary.loc[name, "ndcg_at_k"]) < oracle, name


@NEEDS_DATA
def test_a1_scrambling_relevance_does_not_move_a_realistic_policy(offers, ctx):
    """Order under scrambled ``relevance`` must be identical -- i.e. unread."""
    frame = _first_frame(offers)
    rng = np.random.default_rng(0)
    scrambled = frame.copy()
    scrambled["relevance"] = rng.permutation(scrambled["relevance"].to_numpy())

    for name in REALISTIC_POLICIES:
        assert np.array_equal(POLICIES[name](frame, ctx), POLICIES[name](scrambled, ctx)), (
            f"{name} reads the relevance column"
        )


@NEEDS_DATA
def test_learned_policies_refuse_to_run_without_coefficients(offers):
    frame = _first_frame(offers)
    bare = PolicyContext()
    for name in ("click_trained", "propensity_logistic"):
        with pytest.raises(ValueError):
            POLICIES[name](frame, bare)


@NEEDS_DATA
def test_dropping_a_policy_from_the_table_does_not_renumber_the_others(offers, ctx):
    """The registry is keyed by name; row position carries no meaning."""
    two = compare_rankings(offers, k=10, policies=("relevance", "revenue"), ctx=ctx)
    five = compare_rankings(offers, k=10, policies=ALL_POLICIES, ctx=ctx)
    assert list(two["policy"]) == ["relevance", "revenue", "delta_relevance_minus_revenue"]
    assert len(five) == len(ALL_POLICIES) + 1
    for _, row in two.iterrows():
        match = five[five["policy"] == row["policy"]]
        assert len(match) == 1
        assert float(match.iloc[0]["ndcg_at_k"]) == pytest.approx(row["ndcg_at_k"])


def test_a_two_offer_reference_table_matches_the_hand_computed_divergence():
    """The defaults still reproduce the Phase 2 numbers on a hand-built search."""
    frame = pd.DataFrame(
        {
            "search_id": ["s1"] * 3,
            "position": [1, 2, 3],
            "device": ["desktop"] * 3,
            "partner": ["SkyTour", "TripMasters", "CloudTrip"],
            "price": [12_000.0, 15_000.0, 9_000.0],
            "relevance": [0.9, 0.4, 0.6],
        }
    )
    summary = compare_rankings(frame, k=10).set_index("policy")
    assert float(summary.loc["relevance", "ndcg_at_k"]) == pytest.approx(1.0)
    assert float(summary.loc["delta_relevance_minus_revenue", "ndcg_at_k"]) > 0
    assert float(summary.loc["delta_relevance_minus_revenue", "revenue_per_search"]) < 0


# --------------------------------------------------------------------------- #
# A2 -- no leakage between the fit and the score
# --------------------------------------------------------------------------- #


@NEEDS_DATA
def test_a2_train_and_test_searches_are_disjoint(offers, split):
    train, test = split
    assert set(train["search_id"]).isdisjoint(set(test["search_id"]))
    assert len(train) + len(test) == len(offers)
    assert train["search_id"].nunique() > 0
    assert test["search_id"].nunique() > 0


@NEEDS_DATA
def test_a2_the_split_is_deterministic(offers):
    a_train, a_test = train_test_split_searches(offers, test_frac=0.5, seed=0)
    b_train, b_test = train_test_split_searches(offers, test_frac=0.5, seed=0)
    assert set(a_train["search_id"]) == set(b_train["search_id"])
    assert set(a_test["search_id"]) == set(b_test["search_id"])


@NEEDS_DATA
def test_a2_the_fit_uses_the_train_half_and_not_the_whole_frame(offers, split):
    """Mutation control: if ``context_for`` ignored its ``train`` argument and
    fitted on everything, the two halves would give the same coefficients."""
    train, test = split
    on_train = np.asarray(context_for(train, train=train).params["click_trained"])
    on_test = np.asarray(context_for(test, train=test).params["click_trained"])
    assert on_train.shape == (3,)
    assert not np.allclose(on_train, on_test)


@NEEDS_DATA
def test_a2_a_single_search_cannot_be_split(offers):
    single = offers[offers["search_id"] == offers["search_id"].iloc[0]]
    with pytest.raises(ValueError):
        train_test_split_searches(single)


# --------------------------------------------------------------------------- #
# The learned pair is a controlled pair: one fitter, one weight apart
# --------------------------------------------------------------------------- #


@NEEDS_DATA
def test_both_learned_models_fit_finitely(ctx):
    for name in ("click_trained", "propensity_logistic"):
        beta = np.asarray(ctx.params[name], dtype=float)
        assert beta.shape == (3,)
        assert np.all(np.isfinite(beta))


@NEEDS_DATA
def test_ipw_weighting_actually_moves_the_price_coefficient(ctx):
    """If the two fits were identical the 'controlled pair' would be decoration."""
    raw = np.asarray(ctx.params["click_trained"], dtype=float)
    ipw = np.asarray(ctx.params["propensity_logistic"], dtype=float)
    assert not np.allclose(raw, ipw)
