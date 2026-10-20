"""Tests for the A/B readout (Phase 3, PRD REQ-D01..D04).

The readout used to be an "A/B test" without a treatment: two independent
Bernoulli subsets of the same logged rows. These tests pin the properties that
make it an experiment instead:

* the arms are **disjoint on the randomisation unit** (``user_id``), and a user's
  whole history lands in one arm;
* the two arms carry **different outcomes**, produced by re-ranking and re-drawing
  the click model -- same offers, different slots;
* ``outcomes_under_policy(..., "logged")`` **reproduces the logged click rate**,
  which is what licenses the generator for the counterfactual orders;
* **SRM reddens on a deliberately broken split** (mutation control), so a green
  SRM means something;
* the **power block** carries the MDE, the n/arm and the scenario label;
* heterogeneity is **CTR only**, with BH over the two device comparisons and an
  explicit ``underpowered_on_revenue`` flag.

The fixture is a small frame built from the simulator's own primitives
(partners, price indices, affinities, examination curve), so the tests run without
the gitignored raw Aviasales feeds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.experiment import (
    TEST_SHARE,
    arm_outcomes,
    arms_from_assignment,
    assign_arms,
    heterogeneity_readout,
    per_user_totals,
    power_readout,
    srm_readout,
)
from src.outcomes import outcomes_under_policy
from src.policies import PolicyContext
from src.serp_sim import (
    PARTNER_BY_NAME,
    SimConfig,
    eta_for_device,
    examination_prob,
    sigmoid,
)

PARTNER_NAMES = tuple(PARTNER_BY_NAME)
#: A fitted-looking serving beta: [intercept, log price ratio, is_cpc]. The tests
#: need a policy that runs, not the production fit.
FAKE_BETA = np.array([-0.30, -0.55, 0.35])


def _synthetic_offers(n_users: int = 900, n_offers: int = 6, seed: int = 5) -> pd.DataFrame:
    """A small PBM-generated frame in the ``serp_offers`` schema.

    Built from the simulator's primitives (partner economics, examination curve,
    relevance logit) so the logged clicks and the re-drawn clicks share one model.
    """
    rng = np.random.default_rng(seed)
    config = SimConfig()
    rows: list[dict] = []
    for u in range(n_users):
        user_id = f"u{u:04d}"
        for s in range(int(rng.integers(1, 4))):
            search_id = f"{user_id}-s{s}"
            device = "mobile" if rng.random() < config.mobile_share else "desktop"
            eta = eta_for_device(device, config)
            month = int(rng.integers(1, 9))
            partners = [PARTNER_NAMES[i] for i in rng.integers(0, len(PARTNER_NAMES), n_offers)]
            prices = (
                6000.0
                * np.array([float(PARTNER_BY_NAME[p]["price_index"]) for p in partners])
                * np.exp(rng.normal(0.0, 0.05, n_offers))
            )
            ratio = prices / prices.min()
            logits = (
                config.base_relevance_logit
                + np.array([float(PARTNER_BY_NAME[p]["affinity"]) for p in partners])
                - config.price_sensitivity * (ratio - 1.0)
            )
            relevance = np.asarray(sigmoid(logits), dtype=float)
            order = np.lexsort((np.arange(n_offers), -relevance))
            for position, i in enumerate(order, start=1):
                partner = partners[i]
                conf = PARTNER_BY_NAME[partner]
                examine = examination_prob(position, eta)
                clicked = bool(rng.random() < examine * relevance[i])
                redirected = clicked
                booked = bool(clicked and rng.random() < 0.30)
                if conf["type"] == "CPA":
                    commission = (
                        float(conf["commission_rate"]) * float(prices[i]) if booked else 0.0
                    )
                else:
                    commission = float(conf["cpc"]) if clicked else 0.0
                rows.append(
                    {
                        "search_id": search_id,
                        "user_id": user_id,
                        "position": position,
                        "partner": partner,
                        "price": float(prices[i]),
                        "commission": commission,
                        "clicked": clicked,
                        "redirected": redirected,
                        "booked": booked,
                        "relevance": float(relevance[i]),
                        "device": device,
                        "booking_month": month,
                        "partner_type": str(conf["type"]),
                        "examine_prob": examine,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def offers() -> pd.DataFrame:
    return _synthetic_offers()


@pytest.fixture(scope="module")
def ctx() -> PolicyContext:
    return PolicyContext(params={"propensity_logistic": FAKE_BETA})


@pytest.fixture(scope="module")
def tagged(offers: pd.DataFrame) -> pd.DataFrame:
    return assign_arms(per_user_totals(offers))


@pytest.fixture(scope="module")
def arms(offers: pd.DataFrame, ctx: PolicyContext, tagged: pd.DataFrame) -> dict:
    return arms_from_assignment(arm_outcomes(offers, ctx), tagged)


# --------------------------------------------------------------------------- #
# Randomisation
# --------------------------------------------------------------------------- #


def test_arms_are_disjoint_by_user(offers: pd.DataFrame, tagged: pd.DataFrame, arms: dict) -> None:
    control = set(arms["control"]["user_id"])
    test = set(arms["test"]["user_id"])

    assert control and test, "both arms must be non-empty"
    assert control.isdisjoint(test), "a user must not appear in both arms"
    assert control | test == set(offers["user_id"]), "every user must be assigned"

    # The unit of randomisation is the user, so the search sets must be disjoint too.
    searches_control = set(arms["control"]["search_id"])
    searches_test = set(arms["test"]["search_id"])
    assert searches_control.isdisjoint(searches_test)
    assert len(searches_control | searches_test) == offers["search_id"].nunique()

    # ... and a user's whole history sits in exactly one arm.
    assert arms["control"].groupby("user_id")["search_id"].nunique().index.isin(control).all()
    assert arms["test"].groupby("user_id")["search_id"].nunique().index.isin(test).all()


# --------------------------------------------------------------------------- #
# Treatment
# --------------------------------------------------------------------------- #


def test_outcomes_differ_between_arms(offers: pd.DataFrame, ctx: PolicyContext) -> None:
    counterfactuals = arm_outcomes(offers, ctx)
    logged = counterfactuals["production"].sort_values(["search_id", "position"], kind="stable")
    treated = counterfactuals["revenue_objective"].sort_values(
        ["search_id", "position"], kind="stable"
    )

    # The order changed: at a given slot the two arms show a different offer.
    moved = (logged["price"].to_numpy() != treated["price"].to_numpy()).mean()
    assert moved > 0.5, f"only {moved:.1%} of slots changed offer -- no treatment"

    # Same offers, same prices: only the display order differs.
    assert np.allclose(
        logged.groupby("search_id")["price"].sum().to_numpy(),
        treated.groupby("search_id")["price"].sum().to_numpy(),
    )

    # Re-drawn outcomes differ in the aggregate, which is what the experiment reads.
    ctr_gap = abs(float(logged["clicked"].mean()) - float(treated["clicked"].mean()))
    revenue_gap = abs(
        float(logged["commission"].sum()) - float(treated["commission"].sum())
    ) / float(logged["commission"].sum())
    assert ctr_gap > 0.005, f"CTR gap {ctr_gap:.4f} is too small to be a treatment"
    assert revenue_gap > 0.01, f"revenue gap {revenue_gap:.2%} is too small to be a treatment"


def test_logged_policy_reproduces_logged_outcomes(offers: pd.DataFrame, ctx: PolicyContext) -> None:
    redrawn = outcomes_under_policy(offers, "logged", ctx, seed=20261020)

    observed = float(offers["clicked"].mean())
    reproduced = float(redrawn["clicked"].mean())
    assert abs(reproduced - observed) <= 0.01, (
        f"logged replay click share {reproduced:.4f} vs observed {observed:.4f}"
    )

    # Keeping the order must also keep the slot geometry exactly.
    logged = offers.sort_values(["search_id", "position"], kind="stable")
    assert np.array_equal(redrawn["position"].to_numpy(), logged["position"].to_numpy())
    assert np.allclose(redrawn["examine_prob"].to_numpy(), logged["examine_prob"].to_numpy())


# --------------------------------------------------------------------------- #
# SRM, with a mutation control
# --------------------------------------------------------------------------- #


def test_srm_detects_deliberately_broken_split(offers: pd.DataFrame) -> None:
    users = per_user_totals(offers)
    n = len(users)

    healthy = users.copy()
    healthy["is_test"] = np.arange(n) % 10 == 0  # exactly the planned 10%
    ok = srm_readout(healthy)
    assert ok["observed_test_share"] == pytest.approx(TEST_SHARE)
    assert not ok["vs_planned"]["mismatch"], "an exact 90/10 split must pass"

    broken = users.copy()
    broken["is_test"] = True  # mutation: everyone lands in the test arm
    bad = srm_readout(broken)
    assert bad["vs_planned"]["mismatch"], "SRM must redden when the split is broken"
    assert bad["vs_planned"]["p_value"] < 1e-6

    half = users.copy()
    half["is_test"] = np.arange(n) % 2 == 0  # 50/50 against a planned 90/10
    assert srm_readout(half)["vs_planned"]["mismatch"]


# --------------------------------------------------------------------------- #
# Power block
# --------------------------------------------------------------------------- #


def test_power_block_has_required_fields(arms: dict) -> None:
    power = power_readout(arms["control"], arms["test"], traffic_searches=12_000)

    assert power["run"]["searches"] > 0
    assert power["at_real_traffic"]["searches"] == 12_000
    assert 0 < power["at_real_traffic"]["mde_rel"] < 1
    assert power["run"]["mde_rel"] > 0

    scenario = power["at_scenario_60000"]
    assert scenario["is_scenario"] is True
    assert scenario["mde_rel"] < power["at_real_traffic"]["mde_rel"], (
        "more traffic must lower the MDE"
    )

    required = power["required_n_per_arm"]
    assert required["n_per_arm_users"] > 0
    assert required["n_per_arm_searches"] > 0
    assert required["observed_effect_rel"] == pytest.approx(power["observed_effect_rel"])

    planned = power["required_n_at_planned_split"]
    assert planned["test_share"] == pytest.approx(TEST_SHARE)
    assert planned["n_total_users"] > required["n_total_users"], (
        "an unequal split must cost more total users than an even one"
    )
    assert set(power["labels"]) == {"reality", "scenario"}


# --------------------------------------------------------------------------- #
# Heterogeneity
# --------------------------------------------------------------------------- #


def test_heterogeneity_is_ctr_only(arms: dict) -> None:
    hetero = heterogeneity_readout(arms["control"], arms["test"])

    assert hetero["metric"] == "ctr"
    assert hetero["underpowered_on_revenue"] is True
    assert hetero["comparisons"] == 2
    assert "no power" in hetero["revenue_note"]

    devices = {row["device"] for row in hetero["by_device"]}
    assert devices == {"mobile", "desktop"}

    for row in hetero["by_device"]:
        assert row["metric"] == "ctr", "the block must not test a revenue effect by device"
        assert row["ci_low_rel"] <= row["diff_rel"] <= row["ci_high_rel"]
        assert 0.0 <= row["p_value"] <= 1.0
        assert row["p_value_bh"] >= row["p_value"] - 1e-12, "BH only ever raises a p-value"
        assert 0.0 <= row["p_value_bh"] <= 1.0
        assert row["mde_rel"] > 0
