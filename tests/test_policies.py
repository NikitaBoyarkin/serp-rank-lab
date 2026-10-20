"""Tests for the ranking-policy registry (Phase 4, increment 1).

The project's headline is that offline relevance (NDCG) and revenue per search
diverge. Phase 2 showed it with two policies that *both* read the simulator's
ground-truth ``relevance`` column, so a skeptic can dismiss the result as "you
compared the ideal order with the greedy one". These tests pin the things that
make the claim defensible:

* **A1** -- at least two policies that cannot see ``relevance`` produce an order
  different from both references and score below the oracle's 1.0.
* **A2** -- the learned ranker is fitted and scored on disjoint searches, so its
  apparent quality is not memorisation.
* **A3** -- the *feasible pair* (``ctr_objective`` / ``revenue_objective``) is one
  model with two objectives: same ``X``, same ``beta``, same slot mechanism; only
  the utility multiplier differs. Neither reads ground truth, and the guard that
  says so is itself tested by mutation (substitute true revenue -> the guard must
  go red).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.policies as policies_module
from src.policies import (
    FEASIBLE_POLICIES,
    INFORMATION_SET,
    ORACLE_POLICIES,
    POLICIES,
    REALISTIC_POLICIES,
    PolicyContext,
    click_features,
    context_for,
    fitted_click_probability,
    rev_engaged,
    train_test_split_searches,
)
from src.ranking_eval import compare_rankings, headline_report
from src.serp_sim import SimConfig, examination_prob, load_routes, load_user_ids, simulate_serps

_ARTIFACT = Path(__file__).resolve().parents[1] / "data" / "processed" / "bookings_legs.parquet"
_NEEDS_DATA_REASON = (
    "needs data/processed/bookings_legs.parquet; run `python run.py --stage all` "
    "first (CI does exactly that, so these tests run there)"
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


# --------------------------------------------------------------------------- #
# A3 -- the feasible pair: one model, two objectives
# --------------------------------------------------------------------------- #

GROUND_TRUTH_COLUMNS = ("relevance", "expected_revenue", "click_prob")


def _bare(frame: pd.DataFrame) -> pd.DataFrame:
    """The frame a serving-time ranker actually has: no simulator ground truth."""
    return frame.drop(columns=[c for c in GROUND_TRUTH_COLUMNS if c in frame.columns])


def _greedy_from_utility(frame: pd.DataFrame, utility: np.ndarray, eta: float) -> np.ndarray:
    """Independent re-implementation of the greedy slot mechanism (k = 1..n).

    Written out here on purpose: if the policy ever stops using "pick
    ``argmax examine_prob(k) * utility`` for slot k", this stops matching.
    """
    remaining = list(range(len(frame)))
    order: list[int] = []
    for k in range(1, len(frame) + 1):
        e = examination_prob(k, eta)
        scores = [(e * float(utility[i]), -i) for i in remaining]
        best = max(scores)[1]
        chosen = -best
        remaining.remove(chosen)
        order.append(chosen)
    return np.array(order, dtype=int)


@NEEDS_DATA
def test_feasible_policies_share_information_set(offers, ctx):
    """Same ``X``, same ``beta``, same slot mechanism -- only the multiplier differs.

    This is the test that makes "the gap is the objective, nothing else"
    checkable: both utilities are rebuilt from *one* :func:`fitted_click_probability`
    call, the revenue arm is that same vector times :func:`rev_engaged`, and both
    orders are reproduced by the same greedy assignment.
    """
    frame = _first_frame(offers)
    eta = ctx.eta(str(frame["device"].iloc[0]))

    p_hat = fitted_click_probability(frame, ctx)
    # ... and that one vector is the IPW fit applied to the serving features.
    beta = np.asarray(ctx.params[policies_module.FITTED_MODEL], dtype=float)
    np.testing.assert_allclose(
        1.0 / (1.0 + np.exp(-(click_features(frame) @ beta))), p_hat, rtol=0, atol=0
    )

    money = rev_engaged(frame, ctx)
    # The revenue utility is the CTR utility times the contract money: no other term.
    np.testing.assert_allclose(p_hat * money / p_hat, money)

    assert np.array_equal(
        POLICIES["ctr_objective"](frame, ctx), _greedy_from_utility(frame, p_hat, eta)
    )
    assert np.array_equal(
        POLICIES["revenue_objective"](frame, ctx), _greedy_from_utility(frame, p_hat * money, eta)
    )


@NEEDS_DATA
def test_the_money_multiplier_actually_changes_the_order(split, ctx):
    """Mutation control for the symmetry claim: the multiplier is not decoration.

    Utility orders differ on a strict majority of searches, so the two policies
    are genuinely two objectives -- not the same ranking under two names.
    """
    _, test = split
    orders = _orders(test, ctx, FEASIBLE_POLICIES)
    n = len(orders[FEASIBLE_POLICIES[0]])
    differing = sum(
        not np.array_equal(orders[FEASIBLE_POLICIES[0]][i], orders[FEASIBLE_POLICIES[1]][i])
        for i in range(n)
    )
    assert differing > n / 2, "the money multiplier changes nothing; the pair is decorative"


@NEEDS_DATA
def test_headline_policies_do_not_read_ground_truth(offers, ctx):
    """Both feasible policies rank a frame that has no truth columns at all.

    Mutation control lives next door: a ``rev_engaged`` that read
    ``expected_revenue`` cannot even run on this frame.
    """
    frame = _first_frame(offers)
    bare = _bare(frame)
    assert not set(GROUND_TRUTH_COLUMNS) & set(bare.columns)

    for name in FEASIBLE_POLICIES:
        order = np.asarray(POLICIES[name](bare, ctx), dtype=int)
        assert sorted(order.tolist()) == list(range(len(bare))), name
        # ... and it is the same order as on the full frame.
        assert np.array_equal(order, np.asarray(POLICIES[name](frame, ctx), dtype=int)), name


@NEEDS_DATA
def test_scrambling_ground_truth_does_not_move_a_feasible_policy(offers, ctx):
    """Even when the truth columns are present, shuffling them must not matter."""
    frame = _first_frame(offers)
    rng = np.random.default_rng(0)
    scrambled = frame.copy()
    for column in GROUND_TRUTH_COLUMNS:
        scrambled[column] = rng.permutation(scrambled[column].to_numpy())

    for name in FEASIBLE_POLICIES:
        assert np.array_equal(POLICIES[name](frame, ctx), POLICIES[name](scrambled, ctx)), (
            f"{name} reads a ground-truth column"
        )


@NEEDS_DATA
def test_mutation_control_true_revenue_utility_trips_the_guard(offers, ctx, monkeypatch):
    """Substitute ``rev_engaged`` with the simulator's true revenue -> the guard goes red.

    Two ways to detect the substitution, both asserted here:

    1. the bare frame has no ``expected_revenue``, so the leaky policy raises;
    2. on a full frame whose truth is scrambled, the leaky policy *changes order*
       -- which is exactly what ``test_scrambling_ground_truth_...`` would catch.

    Without this the "does not read ground truth" test would only prove that the
    columns are absent, not that reading them is impossible.
    """
    frame = _first_frame(offers)
    baseline = np.asarray(policies_module.revenue_objective(frame, ctx), dtype=int)

    def leaky_rev_engaged(frame_, ctx_):  # the bug we are guarding against
        return frame_["expected_revenue"].to_numpy(dtype=float)

    with monkeypatch.context() as patch:
        patch.setattr(policies_module, "rev_engaged", leaky_rev_engaged)
        with pytest.raises(KeyError):
            policies_module.revenue_objective(_bare(frame), ctx)

        scrambled = frame.copy()
        rng = np.random.default_rng(1)
        scrambled["expected_revenue"] = rng.permutation(scrambled["expected_revenue"].to_numpy())
        assert not np.array_equal(policies_module.revenue_objective(scrambled, ctx), baseline), (
            "a true-revenue utility produced the same order: the guard has no teeth"
        )

    # Unpatched again: identical order, i.e. the healthy policy is deterministic.
    assert np.array_equal(np.asarray(policies_module.revenue_objective(frame, ctx)), baseline)


@NEEDS_DATA
def test_headline_policies_refuse_to_run_without_the_ipw_fit(offers):
    """Both arms need the same fitted model; missing coefficients raise."""
    frame = _first_frame(offers)
    for name in FEASIBLE_POLICIES:
        with pytest.raises(ValueError):
            POLICIES[name](frame, PolicyContext())


def test_oracles_never_enter_headline_block():
    """``headline_report`` refuses an oracle and labels the ceiling ``modeled``."""
    frame = pd.DataFrame(
        [
            {
                "search_id": "s1",
                "position": 1,
                "device": "desktop",
                "partner": "TripMasters",
                "partner_type": "CPA",
                "price": 10_000.0,
                "relevance": 0.30,
                "expected_revenue": 1.0,
                "click_prob": 0.3,
            },
            {
                "search_id": "s1",
                "position": 2,
                "device": "desktop",
                "partner": "CloudTrip",
                "partner_type": "CPC",
                "price": 9_500.0,
                "relevance": 0.70,
                "expected_revenue": 2.0,
                "click_prob": 0.7,
            },
            {
                "search_id": "s1",
                "position": 3,
                "device": "desktop",
                "partner": "OTA Global",
                "partner_type": "CPA",
                "price": 9_000.0,
                "relevance": 0.90,
                "expected_revenue": 3.0,
                "click_prob": 0.9,
            },
        ]
    )
    ctx = PolicyContext(params={"propensity_logistic": np.array([0.5, -1.0, 0.0])})

    payload = headline_report(frame, ctx, k=3, searches=1, commit="deadbeef")

    assert set(payload["policies"]) == {
        "ctr_objective",
        "revenue_objective",
        "production",
        "price_asc",
    }
    assert not set(ORACLE_POLICIES) & set(payload["policies"])
    assert all(row["information_set"] == "logged" for row in payload["policies"].values())
    assert payload["headline"]["information_set"] == "logged"
    # The oracle pair is present, but only as the labelled ceiling.
    assert payload["ceiling_oracle"]["label"] == "modeled"
    assert "serp_sim.py:57-61" in payload["ceiling_oracle"]["note"]
    for oracle in ORACLE_POLICIES:
        assert oracle not in payload["headline"]
        assert oracle not in payload["failed_hypothesis"]


def test_information_set_marks_exactly_the_oracles():
    """The registry's labels are the guard the headline relies on."""
    oracles = {name for name, kind in INFORMATION_SET.items() if kind == "oracle"}
    assert oracles == set(ORACLE_POLICIES)
    for name in FEASIBLE_POLICIES:
        assert INFORMATION_SET[name] == "logged"
    # Every registered policy is classified and none is named after a truth column.
    assert set(INFORMATION_SET) == set(POLICIES)
    assert not {"expected_revenue", "click_prob"} & set(POLICIES)
