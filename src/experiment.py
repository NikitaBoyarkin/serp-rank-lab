"""A/B readout of a ranking change (SERP Rank Lab, Phase 3).

**The test has a treatment.** Arm A is the shipped display order
(``production``); arm B is ``revenue_objective``, the feasible policy that ranks
by ``p_hat * revenue_if_engaged`` on the *same* information set. Every SERP is
re-ranked under each arm and its outcomes are re-drawn from the same
position-based click model (``src/outcomes.py``, PRD section 1.2), under one
shared seed regime. Re-ranking therefore changes *which offer sits in which
slot*, and only that: the same offers, the same prices, the same contracts.

Design
------
* **Randomisation unit:** ``user_id``. A user's searches all land in one arm;
  ``assign_arms`` draws one Bernoulli per user at the planned ``TEST_SHARE``.
  The control arm carries 90% of users (the real Aviasales test-task set-up
  shrinks the test arm to 10% of traffic).
* **Primary metric:** revenue per search, ``R = sum(revenue) / sum(searches)``.
  A ratio metric with the *user* as the cluster: the variance is the delta-method
  per-user variance (``src/stats.py``), never a per-search trial count.
* **SRM** against the planned split, on the randomisation unit.
* **Guardrails**, each with a CI: CTR per search, booking conversion (share of
  searches with at least one booking), mean price in the top 5, share of offers
  priced above their SERP's median. Refund rate is not available: there is no
  join between ``orders`` and the ranking.
* **Stopping rule:** a guardrail whose point estimate is worse than
  ``GUARDRAIL_MAX_REL_CHANGE`` in the harmful direction *and* statistically
  distinguishable from zero blocks the rollout, whatever the revenue says. The CI
  bound is reported next to it so size and evidence stay separable.
* **Power:** the MDE at real traffic, the n/arm the observed effect would need,
  and a labelled 60 000-search scenario. Scenario numbers are computed from the
  measured variance, not from fabricated data.
* **CUPED:** kept, and presented for what it is here -- the covariate has nothing
  to explain (i.i.d. simulator users), and a synthetic positive control proves
  the machinery is alive. Correlation is *not* wired into the simulator.
* **Heterogeneity:** CTR by ``device`` with CIs and a Benjamini-Hochberg
  correction over the two comparisons. Revenue by device is not tested: the
  experiment has no power for it, and saying so is part of the result.
* **Peeking:** the number of interims is capped by the arm size, so 10 looks of
  250 users can no longer re-use the last 250 users of a 2 250-user arm.

Readout order (all seeded and deterministic): SRM, A/A, primary, guardrails,
decision, heterogeneity, power, CUPED, peeking, sizing.

CLI: ``python -m src.experiment [--searches N]`` reads
``data/processed/serp_offers.parquet``, prints the readout, and writes
``reports/experiment.json`` + ``reports/peeking.png``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import stats  # noqa: E402
from src.outcomes import outcomes_under_policy  # noqa: E402
from src.policies import PolicyContext, context_for, train_test_split_searches  # noqa: E402
from src.serp_sim import SERP_OFFERS_PARQUET  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
EXPERIMENT_JSON = REPORTS_DIR / "experiment.json"
PEEKING_PNG = REPORTS_DIR / "peeking.png"
RANKING_REPORT = REPORTS_DIR / "ranking_eval.json"

# --- Fixed, reported seeds and design constants (never left to chance) ------ #
TEST_SHARE = 0.10  # planned allocation: 90% control / 10% test
CONTROL_POLICY = "production"  # arm A: the shipped display order
TEST_POLICY = "revenue_objective"  # arm B: the feasible money objective
REL_MDE = 0.05  # smallest relative lift worth acting on (sizing only)
ALPHA = 0.05
POWER = 0.80
GUARDRAIL_MAX_REL_CHANGE = 0.10  # >10% harm on a guardrail blocks the rollout
SPLIT_SEED = 20261011  # arm assignment
OUTCOME_SEED = 20261020  # outcome re-draw, shared by both arms
MODEL_SEED = 0  # train/test split of searches for the fitted policy
AA_SEED = 101  # A/A pseudo-split
CUPED_SEED = 555  # synthetic positive control only
PEEK_SEED = 909  # peeking streams
PEEK_SIMS = 400
PEEK_BATCHES = 10
PEEK_BATCH_SIZE = 250
PRE_MONTH_MAX = 4  # booking months 1..4 = pre-period; 5..8 = post-period
DEFAULT_SEARCHES = 12_000  # the dataset's traffic: the "reality" point
SCENARIO_SEARCHES = 60_000  # the labelled scenario (needs a re-simulation)


# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #


def per_user_totals(offers: pd.DataFrame) -> pd.DataFrame:
    """Collapse offers to one row per user: ``searches`` and ``revenue``.

    ``searches`` counts the user's distinct ``search_id`` values and ``revenue``
    sums the realised ``commission`` column, so ``sum(revenue)/sum(searches)`` over
    a group is exactly that group's revenue per search.
    """
    grouped = offers.groupby("user_id", sort=True)
    return pd.DataFrame(
        {
            "user_id": grouped.size().index,
            "searches": grouped["search_id"].nunique().to_numpy(),
            "revenue": grouped["commission"].sum().to_numpy(),
        }
    )


def assign_arms(
    users: pd.DataFrame, test_share: float = TEST_SHARE, seed: int = SPLIT_SEED
) -> pd.DataFrame:
    """Tag each user control/test by one Bernoulli draw at ``test_share``.

    One draw per **user**, not per search: a user's whole history lands in one
    arm, which is what makes the user the cluster of the analysis. Simulating the
    assignment (rather than reading a shipped flag) is what makes the SRM check
    meaningful.
    """
    rng = np.random.default_rng(seed)
    out = users.copy()
    out["is_test"] = rng.random(len(out)) < test_share
    return out


def split_arm(arm: pd.DataFrame, seed: int = AA_SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cut one arm into two random halves -- the A/A pseudo-split."""
    rng = np.random.default_rng(seed)
    mask = rng.random(len(arm)) < 0.5
    return arm[mask], arm[~mask]


def _ratio(df: pd.DataFrame) -> float:
    """Pooled revenue per search for a frame carrying ``revenue`` and ``searches``."""
    return float(df["revenue"].sum() / df["searches"].sum())


def arm_outcomes(
    offers: pd.DataFrame, ctx: PolicyContext, *, seed: int = OUTCOME_SEED
) -> dict[str, pd.DataFrame]:
    """Counterfactual outcomes for every offer under both arms.

    Both frames are generated with the **same** seed, so the two arms are the two
    orders of the same offers and the comparison is within-SERP. This is not the
    experiment yet -- it is the pair of potential outcomes the assignment later
    splits.
    """
    return {
        name: outcomes_under_policy(offers, name, ctx, seed=seed)
        for name in (CONTROL_POLICY, TEST_POLICY)
    }


def arms_from_assignment(
    counterfactuals: dict[str, pd.DataFrame], tagged: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Keep each user's rows from their own arm only.

    The counterfactual frames both contain every search. Assignment picks, per
    user, which of the two worlds is observed, so the arms are disjoint on
    ``user_id`` by construction.
    """
    test_users = set(tagged.loc[tagged["is_test"], "user_id"])
    control = counterfactuals[CONTROL_POLICY]
    test = counterfactuals[TEST_POLICY]
    return {
        "control": control[~control["user_id"].isin(test_users)].reset_index(drop=True),
        "test": test[test["user_id"].isin(test_users)].reset_index(drop=True),
    }


def arm_balance(arms: dict[str, pd.DataFrame]) -> dict:
    """Per-arm size, revenue per search and the share of SERPs actually re-ordered."""
    out: dict[str, dict] = {}
    for name, frame in arms.items():
        users = per_user_totals(frame)
        out[name] = {
            "policy": CONTROL_POLICY if name == "control" else TEST_POLICY,
            "n_users": int(len(users)),
            "n_searches": int(frame["search_id"].nunique()),
            "n_offers": int(len(frame)),
            "revenue_per_search": _ratio(users),
        }
    return out


def treatment_check(
    counterfactuals: dict[str, pd.DataFrame], arms: dict[str, pd.DataFrame]
) -> dict:
    """How much of the test arm was actually re-ordered, and by how many slots.

    A guard against a silent no-op: if ``revenue_objective`` reproduced the logged
    order, the "treatment" would be a placebo and the readout would be an A/A test
    wearing a B label. Offers are matched across the two frames on
    ``(search_id, price)`` -- the frames are the same offer set, so the price is a
    stable offer key within a SERP -- and their slot numbers are compared.
    """
    logged = counterfactuals[CONTROL_POLICY][["search_id", "price", "position"]].rename(
        columns={"position": "logged_position"}
    )
    treated = counterfactuals[TEST_POLICY][["search_id", "price", "position"]].rename(
        columns={"position": "treated_position"}
    )
    merged = logged.merge(treated, on=["search_id", "price"], how="inner")
    merged["moved"] = merged["logged_position"] != merged["treated_position"]
    by_search = merged.groupby("search_id", sort=True)["moved"].max()
    shift = (merged["treated_position"] - merged["logged_position"]).abs()
    test_searches = set(arms["test"]["search_id"])
    share = float(by_search.loc[list(test_searches)].mean()) if test_searches else 0.0
    return {
        "share_of_searches_reranked": share,
        "share_of_offers_moved": float(merged["moved"].mean()),
        "mean_abs_position_change": float(shift.mean()),
        "n_offers_matched": int(len(merged)),
        "n_offers_expected": int(len(logged)),
        "note": "arm B must re-order a material share of SERPs, otherwise there is no treatment",
    }


# --------------------------------------------------------------------------- #
# 1-2. SRM and A/A
# --------------------------------------------------------------------------- #


def srm_readout(users: pd.DataFrame) -> dict:
    """Chi-square SRM on the user-level split against the planned allocation."""
    n_control = int((~users["is_test"]).sum())
    n_test = int(users["is_test"].sum())
    planned = stats.srm_check(n_control, n_test, expected_test_share=TEST_SHARE)
    wrong = stats.srm_check(n_control, n_test, expected_test_share=0.5)
    return {
        "unit": "user_id",
        "planned_test_share": TEST_SHARE,
        "n_control": n_control,
        "n_test": n_test,
        "observed_test_share": planned["observed_share"],
        "vs_planned": {**planned, "mismatch": planned["p_value"] < 0.001},
        "vs_fifty_fifty": {**wrong, "mismatch": wrong["p_value"] < 0.001},
    }


def aa_readout(control: pd.DataFrame) -> dict:
    """Split the control arm in two and confirm the ratio test is non-significant."""
    a, b = split_arm(control)
    res = stats.delta_method_test(a["searches"], a["revenue"], b["searches"], b["revenue"])
    return {
        "n_a": int(len(a)),
        "n_b": int(len(b)),
        "ratio_a": res["ratio_a"],
        "ratio_b": res["ratio_b"],
        "diff": res["diff"],
        "p_value": res["p_value"],
        "non_significant": bool(res["p_value"] >= ALPHA),
    }


# --------------------------------------------------------------------------- #
# Ratio-metric helpers (user = cluster)
# --------------------------------------------------------------------------- #


def _ratio_test(
    control: pd.DataFrame,
    test: pd.DataFrame,
    numerator: str,
    denominator: str,
) -> dict:
    """Delta-method test of a ratio metric, summed per user before the test.

    ``control`` / ``test`` are offer-level frames; both are collapsed to
    ``(user_id, sum(numerator), sum(denominator))`` first, so within-user
    correlation is absorbed by the cluster rather than double-counted.
    """
    g_a = control.groupby("user_id", sort=True)[[numerator, denominator]].sum()
    g_b = test.groupby("user_id", sort=True)[[numerator, denominator]].sum()
    res = stats.delta_method_test(
        g_a[denominator].to_numpy(dtype=float),
        g_a[numerator].to_numpy(dtype=float),
        g_b[denominator].to_numpy(dtype=float),
        g_b[numerator].to_numpy(dtype=float),
    )
    control_rate = res["ratio_a"]
    rel = res["diff"] / control_rate if control_rate else float("nan")
    return {
        "control": control_rate,
        "test": res["ratio_b"],
        "diff_abs": res["diff"],
        "diff_rel": rel,
        "se": res["se"],
        "z": res["z"],
        "p_value": res["p_value"],
        "ci_low_abs": res["ci_low"],
        "ci_high_abs": res["ci_high"],
        "ci_low_rel": res["ci_low"] / control_rate if control_rate else float("nan"),
        "ci_high_rel": res["ci_high"] / control_rate if control_rate else float("nan"),
        "n_control_users": int(len(g_a)),
        "n_test_users": int(len(g_b)),
        "cluster": "user_id",
    }


def search_table(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per search with the columns every guardrail needs.

    Aggregating at the search first, then at the user, keeps each search weighted
    equally inside a user -- a user with 6 searches does not outvote a user with 1
    on the per-search metrics.
    """
    frame = rows.copy()
    frame["above_median"] = frame["price"] > frame.groupby("search_id")["price"].transform("median")
    top5 = frame.loc[frame["position"] <= 5].groupby("search_id", sort=True)["price"].mean()
    grouped = frame.groupby("search_id", sort=True)
    table = pd.DataFrame(
        {
            "user_id": grouped["user_id"].first(),
            "device": grouped["device"].first(),
            "clicks": grouped["clicked"].sum(),
            "booked_searches": grouped["booked"].max().astype(int),
            "revenue": grouped["commission"].sum(),
            "offers": grouped.size(),
            "offers_above_median": grouped["above_median"].sum(),
            "top5_price_sum": top5,
            "searches": 1,
        }
    )
    table["top5_price_sum"] = table["top5_price_sum"].fillna(0.0)
    return table.reset_index()


# --------------------------------------------------------------------------- #
# 3. Primary metric: revenue per search
# --------------------------------------------------------------------------- #


def primary_readout(control: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Revenue per search under each arm: effect, CI, p, clustered on the user."""
    a = per_user_totals(control)
    b = per_user_totals(test)
    res = stats.delta_method_test(a["searches"], a["revenue"], b["searches"], b["revenue"])
    control_rps = res["ratio_a"]
    rel = res["diff"] / control_rps if control_rps else float("nan")
    naive = stats.naive_per_unit_ratio_test(
        a["searches"], a["revenue"], b["searches"], b["revenue"]
    )
    return {
        "metric": "revenue per search",
        "definition": "sum(commission) / sum(searches), users summed before the test",
        "control": {
            "policy": CONTROL_POLICY,
            "revenue_per_search": control_rps,
            "n_users": int(len(a)),
            "n_searches": int(a["searches"].sum()),
        },
        "test": {
            "policy": TEST_POLICY,
            "revenue_per_search": res["ratio_b"],
            "n_users": int(len(b)),
            "n_searches": int(b["searches"].sum()),
        },
        "diff_abs": res["diff"],
        "diff_rel": rel,
        "se": res["se"],
        "z": res["z"],
        "p_value": res["p_value"],
        "ci_low_abs": res["ci_low"],
        "ci_high_abs": res["ci_high"],
        "ci_low_rel": res["ci_low"] / control_rps,
        "ci_high_rel": res["ci_high"] / control_rps,
        "alpha": ALPHA,
        "significant": bool(res["p_value"] < ALPHA),
        "cluster": "user_id",
        "naive_per_unit": {
            "diff": naive["diff"],
            "se": naive["se"],
            "p_value": naive["p_value"],
            "significant": naive["significant"],
        },
        "se_ratio_naive_over_delta": naive["se"] / res["se"],
    }


def ratio_metric_readout(control: pd.DataFrame, test: pd.DataFrame) -> dict:
    """App-compatible view of the primary test (naive per-unit vs delta method)."""
    primary = primary_readout(control, test)
    return {
        "control_rps": primary["control"]["revenue_per_search"],
        "test_rps": primary["test"]["revenue_per_search"],
        "relative_lift": primary["diff_rel"],
        "delta_method": {
            "diff": primary["diff_abs"],
            "se": primary["se"],
            "z": primary["z"],
            "p_value": primary["p_value"],
            "ci_low": primary["ci_low_abs"],
            "ci_high": primary["ci_high_abs"],
            "significant": primary["significant"],
        },
        "naive_per_unit": primary["naive_per_unit"],
        "se_ratio_naive_over_delta": primary["se_ratio_naive_over_delta"],
        "p_values_differ": bool(
            abs(primary["naive_per_unit"]["p_value"] - primary["p_value"]) > 1e-6
        ),
    }


# --------------------------------------------------------------------------- #
# 4. Guardrails
# --------------------------------------------------------------------------- #


#: ``(name, label, numerator, denominator, harmful_direction)``. ``down`` means the
#: guardrail is a thing we do not want to lose; ``up`` means a rise is the harm
#: (showing the user more expensive options). CTR uses impressions as the
#: denominator so it is the same number as ``headline.json``'s ``ctr``.
GUARDRAILS: tuple[tuple[str, str, str, str, str], ...] = (
    ("ctr", "CTR (clicks per offer impression)", "clicks", "offers", "down"),
    (
        "booking_conversion",
        "Booking conversion (share of searches with >=1 booking)",
        "booked_searches",
        "searches",
        "down",
    ),
    (
        "mean_price_top5",
        "Mean price in the top 5 slots",
        "top5_price_sum",
        "searches",
        "up",
    ),
    (
        "share_above_route_median",
        "Share of offers priced above their SERP median",
        "offers_above_median",
        "offers",
        "down",
    ),
)


def _guardrail(
    control: pd.DataFrame,
    test: pd.DataFrame,
    numerator: str,
    denominator: str,
    worse: str,
    threshold: float = GUARDRAIL_MAX_REL_CHANGE,
) -> dict:
    """One guardrail with a CI and the stopping rule applied to it.

    The rule is deliberately two-part: a guardrail blocks the rollout when its
    point estimate is worse than ``threshold`` **and** the difference is
    distinguishable from zero (``p < alpha``). A confident but small drop is not a
    breach (size), and a large but noisy drop is not a breach either (evidence);
    both cases are reported so a reviewer can see which one it is.
    """
    res = _ratio_test(control, test, numerator, denominator)
    if worse == "down":
        bad_side_rel = res["ci_high_rel"]
        point_flag = res["diff_rel"] <= -threshold
        ci_beyond = bad_side_rel <= -threshold
    else:
        bad_side_rel = res["ci_low_rel"]
        point_flag = res["diff_rel"] >= threshold
        ci_beyond = bad_side_rel >= threshold
    significant = bool(res["p_value"] < ALPHA)
    return {
        **res,
        "harmful_direction": worse,
        "threshold_rel": threshold,
        "bad_side_ci_rel": bad_side_rel,
        "significant": significant,
        "point_flag": bool(point_flag),
        "ci_beyond_threshold": bool(ci_beyond),
        "breach": bool(point_flag and significant),
        "rule": (
            f"breach when the effect is worse than {threshold:.0%} in the harmful "
            "direction and significant at 5%; the CI bound is reported separately"
        ),
    }


def guardrails_readout(control: pd.DataFrame, test: pd.DataFrame) -> dict:
    """The four guardrails, each with a CI; the stopping rule rides on each one."""
    search_a = search_table(control)
    search_b = search_table(test)
    return {
        name: {
            "label": label,
            "numerator": numerator,
            "denominator": denominator,
            **_guardrail(search_a, search_b, numerator, denominator, worse),
        }
        for name, label, numerator, denominator, worse in GUARDRAILS
    }


# --------------------------------------------------------------------------- #
# 5. Decision
# --------------------------------------------------------------------------- #


def multiplicity_readout(primary: dict, guardrails: dict) -> dict:
    """Holm + BH correction over primary + 4 guardrails; annotates both dicts in place."""
    names = ["primary", *guardrails]
    raw = [primary["p_value"], *(g["p_value"] for g in guardrails.values())]
    holm = _holm_adjust(raw)
    bh = _bh_adjust(raw)

    metrics: dict[str, dict] = {}
    for i, name in enumerate(names):
        significant_raw = bool(raw[i] < ALPHA)
        significant_holm = bool(holm[i] < ALPHA)
        metrics[name] = {
            "p_raw": raw[i],
            "p_holm": holm[i],
            "p_bh": bh[i],
            "significant_raw": significant_raw,
            "significant_holm": significant_holm,
            "significant_bh": bool(bh[i] < ALPHA),
            "flips": significant_raw != significant_holm,
        }

    primary["p_value_holm"] = holm[0]
    primary["p_value_bh"] = bh[0]
    primary["significant_holm"] = metrics["primary"]["significant_holm"]
    primary["significant_bh"] = metrics["primary"]["significant_bh"]
    for i, name in enumerate(guardrails, start=1):
        g = guardrails[name]
        g["p_value_holm"] = holm[i]
        g["p_value_bh"] = bh[i]
        g["significant_holm"] = metrics[name]["significant_holm"]
        g["significant_bh"] = metrics[name]["significant_bh"]
        g["breach_holm"] = bool(g["point_flag"] and g["significant_holm"])
        g["breach_bh"] = bool(g["point_flag"] and g["significant_bh"])

    return {
        "family": names,
        "m": len(names),
        "alpha": ALPHA,
        "method_default": "holm",
        "metrics": metrics,
        "verdict_inputs_changed": [
            name
            for name in names
            if metrics[name]["significant_raw"] != metrics[name]["significant_holm"]
        ],
    }


def _verdict(
    breached: list[str], primary_significant: bool, primary_diff_abs: float
) -> tuple[str, str]:
    """The stopping rule: guardrail breach, then the primary metric."""
    if breached:
        return "do_not_ship", "guardrail breach: " + ", ".join(breached)
    if primary_significant and primary_diff_abs < 0:
        return "do_not_ship", "primary metric significantly worse than production"
    if primary_significant and primary_diff_abs > 0:
        return "ship", "primary metric significantly better, no guardrail breach"
    return "extend", "primary metric not significant at this sample size; see power"


def decision_readout(primary: dict, guardrails: dict) -> dict:
    """Stopping rule first, verdict second.

    A guardrail breach blocks the rollout regardless of the revenue: that is the
    point of a guardrail. Otherwise the verdict follows the primary metric -- and
    an experiment with no significant effect is reported as underpowered, not as
    "no difference". Significance means the Holm-adjusted p-value; the raw verdict
    stays visible as ``verdict_raw``.
    """
    breached = [name for name, g in guardrails.items() if g.get("breach_holm", g["breach"])]
    breached_raw = [name for name, g in guardrails.items() if g["breach"]]
    primary_significant = bool(primary.get("significant_holm", primary["significant"]))
    primary_significant_raw = bool(primary["significant"])
    verdict, reason = _verdict(breached, primary_significant, primary["diff_abs"])
    verdict_raw, _ = _verdict(breached_raw, primary_significant_raw, primary["diff_abs"])
    return {
        "verdict": verdict,
        "reason": reason,
        "verdict_raw": verdict_raw,
        "stopping_rule": (
            f"do not ship if a guardrail is worse than "
            f"{GUARDRAIL_MAX_REL_CHANGE:.0%} relative in its harmful direction and "
            "the difference is significant at 5%"
        ),
        "multiplicity": "holm over primary + 4 guardrails",
        "guardrails_breached": breached,
        "guardrails_breached_raw": breached_raw,
        "primary_significant": primary_significant,
        "primary_significant_raw": primary_significant_raw,
        "primary_diff_rel": primary["diff_rel"],
        "changes_our_mind": "a guardrail breach on the same readout, or a CI that excludes zero",
    }


# --------------------------------------------------------------------------- #
# 6. Heterogeneity by device (CTR only)
# --------------------------------------------------------------------------- #


def _bh_adjust(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values (monotone, capped at 1)."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [1.0] * m
    running = 1.0
    for rank, i in enumerate(reversed(order), start=1):
        k = m - rank + 1  # 1-based rank from the largest p-value downwards
        running = min(running, p_values[i] * m / k)
        adjusted[i] = min(1.0, running)
    return adjusted


def _holm_adjust(p_values: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [1.0] * m
    running = 0.0
    for rank, i in enumerate(order, start=1):
        running = max(running, min(1.0, (m - rank + 1) * p_values[i]))
        adjusted[i] = running
    return adjusted


def _mde_relative(
    baseline: float,
    variance_per_user: float,
    n_users_total: float,
    alpha: float = ALPHA,
    power: float = POWER,
    test_share: float = 0.5,
) -> float:
    """Relative MDE detectable at ``n_users_total`` users, per-user variance ``V``.

    Inverts ``stats.required_sample_size``: ``delta = sqrt(V (1/(1-s) + 1/s)
    (z_a + z_b)^2 / n)``, then ``delta / baseline``. ``s = 0.5`` is the even split
    used for the "how many users would this need" figure.
    """
    from scipy import stats as scipy_stats

    if baseline <= 0 or variance_per_user <= 0 or n_users_total <= 0:
        return float("nan")
    z_a = float(scipy_stats.norm.ppf(1 - alpha / 2))
    z_b = float(scipy_stats.norm.ppf(power))
    delta = np.sqrt(
        variance_per_user
        * (1 / (1 - test_share) + 1 / test_share)
        * (z_a + z_b) ** 2
        / n_users_total
    )
    return float(delta / baseline)


def heterogeneity_readout(control: pd.DataFrame, test: pd.DataFrame) -> dict:
    """CTR effect by ``device`` with CIs and BH over the two comparisons.

    Only CTR is tested. Revenue per search at segment level is far outside the
    experiment's power, so the block says so instead of reporting a segment
    revenue number nobody could act on.
    """
    search_a = search_table(control)
    search_b = search_table(test)
    rows: list[dict] = []
    for device in ("mobile", "desktop"):
        a = search_a.loc[search_a["device"] == device]
        b = search_b.loc[search_b["device"] == device]
        if a.empty or b.empty:
            continue
        res = _ratio_test(a, b, "clicks", "offers")
        g_a = a.groupby("user_id", sort=True)[["clicks", "offers"]].sum()
        variance = stats.ratio_delta_var_perunit(
            g_a["offers"].to_numpy(dtype=float), g_a["clicks"].to_numpy(dtype=float)
        )
        rows.append(
            {
                "device": device,
                "metric": "ctr",
                "n_control_users": res["n_control_users"],
                "n_test_users": res["n_test_users"],
                "n_control_searches": int(a["searches"].sum()),
                "n_test_searches": int(b["searches"].sum()),
                **res,
                "mde_rel": _mde_relative(
                    res["control"],
                    variance,
                    float(res["n_control_users"] + res["n_test_users"]),
                ),
            }
        )
    p_adj = _bh_adjust([r["p_value"] for r in rows])
    for row, p in zip(rows, p_adj, strict=True):
        row["p_value_bh"] = p
        row["significant_bh"] = bool(p < ALPHA)
    return {
        "metric": "ctr",
        "by_device": rows,
        "comparisons": len(rows),
        "correction": "benjamini-hochberg",
        "underpowered_on_revenue": True,
        "revenue_note": (
            "revenue per search is not tested by device: at segment level the "
            "experiment has no power for it (see mde_rel on CTR times the "
            "revenue/CTR variance ratio); reporting the point estimate would invite "
            "a decision the data cannot support"
        ),
    }


# --------------------------------------------------------------------------- #
# 7. Power
# --------------------------------------------------------------------------- #


def power_readout(
    control: pd.DataFrame,
    test: pd.DataFrame,
    *,
    traffic_searches: int = DEFAULT_SEARCHES,
    scenario_searches: int = SCENARIO_SEARCHES,
    searches_per_user_full: float | None = None,
) -> dict:
    """MDE at real traffic, n/arm for the observed effect, and a labelled scenario.

    ``traffic_searches`` is the reality point (the dataset's traffic); the
    ``scenario`` block is what would become detectable at ``scenario_searches``.
    Both are computed from the measured per-user variance, so no data is
    fabricated: the scenario is a statement about detectability, not a result.

    ``searches_per_user_full`` must be the ratio of the *whole* dataset, not of
    the held-out half: the half splits searches, so a user whose searches fall
    on both sides is counted in both arms and the half ratio understates
    searches per user. Extrapolating traffic with the half ratio inflates the
    implied user counts and understates the MDE.
    """
    a = per_user_totals(control)
    b = per_user_totals(test)
    pooled = pd.concat([a, b], ignore_index=True)
    baseline = _ratio(pooled)
    variance = stats.ratio_variance_per_user(a, clicks="revenue", views="searches")
    searches_per_user = float(pooled["searches"].sum() / len(pooled))
    # Extrapolation ratio: whole-dataset searches/user when the caller supplies
    # it, otherwise the (biased) held-out ratio.
    ratio_full = float(searches_per_user_full or searches_per_user)
    observed_rel = float(
        (b["revenue"].sum() / b["searches"].sum()) / (a["revenue"].sum() / a["searches"].sum()) - 1
    )
    run_searches = int(control["search_id"].nunique() + test["search_id"].nunique())
    run_users = int(len(a) + len(b))

    real_users = traffic_searches / ratio_full
    scenario_users = scenario_searches / ratio_full
    mde_real = _mde_relative(baseline, variance, real_users)
    mde_scenario = _mde_relative(baseline, variance, scenario_users)
    mde_run = _mde_relative(baseline, variance, float(run_users))

    need_rel = max(abs(observed_rel), 1e-6)
    need = stats.required_sample_size(baseline, need_rel, variance, ALPHA, POWER, 0.5)
    need_planned = stats.required_sample_size(
        baseline, need_rel, variance, ALPHA, POWER, TEST_SHARE
    )
    return {
        "baseline_rps": baseline,
        "variance_per_user": variance,
        "searches_per_user": searches_per_user,
        "searches_per_user_full_dataset": ratio_full,
        "observed_effect_rel": observed_rel,
        "run": {
            "searches": run_searches,
            "users": run_users,
            "mde_rel": mde_run,
            "is_scenario": False,
            "label": "this readout: held-out searches actually simulated",
        },
        "at_real_traffic": {
            "searches": int(traffic_searches),
            "users": int(round(real_users)),
            "mde_rel": mde_real,
            "mde_abs": mde_real * baseline,
            "is_scenario": bool(traffic_searches > DEFAULT_SEARCHES),
            "label": "real traffic; the MDE a 5% effect would need to clear here",
        },
        "at_scenario_60000": {
            "searches": int(scenario_searches),
            "users": int(round(scenario_users)),
            "mde_rel": mde_scenario,
            "mde_abs": mde_scenario * baseline,
            "is_scenario": True,
            "label": "SCENARIO: requires re-simulating with more searches, not extra data",
        },
        "required_n_per_arm": {
            "observed_effect_rel": observed_rel,
            "n_per_arm_users": float(need["n_test"]),
            "n_total_users": float(need["n_total"]),
            "n_per_arm_searches": float(need["n_test"] * ratio_full),
            "allocation": "50/50 (even split, the cheapest way to size an arm)",
            "alpha": ALPHA,
            "power": POWER,
            "label": "users per arm to detect the observed effect with 80% power",
        },
        "required_n_at_planned_split": {
            "test_share": TEST_SHARE,
            "n_control_users": float(need_planned["n_control"]),
            "n_test_users": float(need_planned["n_test"]),
            "n_total_users": float(need_planned["n_total"]),
            "label": "same effect at the live 90/10 allocation, with its cost multiple",
        },
        "detectable_at_run": bool(abs(observed_rel) >= mde_run),
        "detectable_at_real_traffic": bool(abs(observed_rel) >= mde_real),
        "labels": {
            "reality": "at_real_traffic / run",
            "scenario": "at_scenario_60000 is constructed from the measured variance",
        },
    }


# --------------------------------------------------------------------------- #
# 8. CUPED variance reduction on the logged data
# --------------------------------------------------------------------------- #


def _period_rps(offers: pd.DataFrame, pre: bool) -> pd.DataFrame:
    """Per-user revenue per search in the pre- or post-period slice."""
    months = offers["booking_month"].to_numpy()
    keep = months <= PRE_MONTH_MAX if pre else months > PRE_MONTH_MAX
    sliced = offers.loc[keep]
    grouped = sliced.groupby("user_id", sort=True)
    return pd.DataFrame(
        {
            "user_id": grouped.size().index,
            "searches": grouped["search_id"].nunique().to_numpy(),
            "revenue": grouped["commission"].sum().to_numpy(),
        }
    )


def _cuped_on_pairs(x: np.ndarray, y: np.ndarray) -> dict:
    """theta, variance reduction and CI narrowing for a covariate/outcome pair."""
    theta = stats.cuped_theta(y, x)
    y_adj = stats.cuped_adjust(y, x, theta)
    var_raw = float(np.var(y, ddof=1))
    var_adj = float(np.var(y_adj, ddof=1))
    raw_half = stats.welch_mean_test(y[: len(y) // 2], y[len(y) // 2 :])
    adj_half = stats.welch_mean_test(y_adj[: len(y_adj) // 2], y_adj[len(y_adj) // 2 :])
    corr = float(np.corrcoef(x, y)[0, 1]) if np.var(x) > 0 else 0.0
    ci_raw = raw_half["ci_high"] - raw_half["ci_low"]
    ci_adj = adj_half["ci_high"] - adj_half["ci_low"]
    return {
        "theta": theta,
        "corr_x_y": corr,
        "corr_squared_theory": corr**2,
        "var_raw": var_raw,
        "var_adjusted": var_adj,
        "variance_reduction_pct": (1 - var_adj / var_raw) * 100 if var_raw > 0 else 0.0,
        "ci_half_width_raw": ci_raw,
        "ci_half_width_adjusted": ci_adj,
        "ci_narrowing_pct": (1 - ci_adj / ci_raw) * 100 if ci_raw > 0 else 0.0,
        "n_users": int(len(y)),
    }


def cuped_readout(offers: pd.DataFrame, seed: int = CUPED_SEED) -> dict:
    """CUPED on revenue per search: an honest null plus a positive control.

    Real block: covariate = the user's pre-period revenue per search (months 1-4),
    outcome = the same user's post-period revenue per search (months 5-8). The
    simulator draws users i.i.d., so this block is expected to be near zero -- the
    covariate has nothing to explain. That is the finding, not a failure, and the
    fix is *not* to wire correlation into the simulator (ADR 0002).

    Positive control: a synthetic covariate sharing a latent factor with the
    outcome at rho = 0.6, to show the machinery recovers the theoretical
    ``corr^2`` reduction when a signal exists.
    """
    pre = _period_rps(offers, pre=True).rename(
        columns={"revenue": "rev_pre", "searches": "srch_pre"}
    )
    post = _period_rps(offers, pre=False).rename(
        columns={"revenue": "rev_post", "searches": "srch_post"}
    )
    both = pre.merge(post, on="user_id", how="inner")
    both = both[(both["srch_pre"] > 0) & (both["srch_post"] > 0)]
    real = _cuped_on_pairs(
        (both["rev_pre"] / both["srch_pre"]).to_numpy(),
        (both["rev_post"] / both["srch_post"]).to_numpy(),
    )

    rng = np.random.default_rng(seed)
    n = max(len(both), 2000)
    latent = rng.normal(0, 1, n)
    rho = 0.6
    x_syn = rho * latent + np.sqrt(1 - rho**2) * rng.normal(0, 1, n)
    y_syn = rho * latent + np.sqrt(1 - rho**2) * rng.normal(0, 1, n)
    control = _cuped_on_pairs(x_syn, y_syn)
    return {
        "pre_month_max": PRE_MONTH_MAX,
        "real": {
            **real,
            "covariate": "pre-period revenue per search",
            "verdict": "the covariate has nothing to explain (i.i.d. users)",
        },
        "positive_control": {
            **control,
            "covariate": "synthetic latent (rho = 0.6)",
            "verdict": "machinery alive: a constructed signal is recovered",
        },
        "interpretation": {
            "real": "an honest null: no pre-period signal exists to remove",
            "control": "proves the estimator works when a signal is present",
            "simulator_untouched": True,
            "note": "correlation is deliberately NOT wired into the simulator",
        },
    }


# --------------------------------------------------------------------------- #
# 9. Peeking: fixed-horizon vs always-valid (mSPRT)
# --------------------------------------------------------------------------- #


def peeking_demo(
    users: pd.DataFrame,
    tau: float,
    n_sims: int = PEEK_SIMS,
    n_batches: int = PEEK_BATCHES,
    batch_size: int = PEEK_BATCH_SIZE,
    seed: int = PEEK_SEED,
) -> tuple[pd.DataFrame, dict]:
    """A/A streams under continuous monitoring: fixed-horizon vs mSPRT p-values.

    Each stream shuffles the user pool and cuts it into two arms, then accumulates
    ``batch_size`` users per arm over the interims, recomputing both a
    fixed-horizon delta z p-value and the always-valid mSPRT p-value at every look.

    The number of interims is **capped by the arm size**: with 2 250 users per arm
    the tenth look of ``10 x 250`` would re-measure the same last 250 users, which
    silently inflates nothing but teaches the wrong thing. ``n_batches`` is reduced
    to ``arm_size // batch_size`` and the eff-ective count is reported.
    """
    searches = users["searches"].to_numpy(dtype=float)
    revenue = users["revenue"].to_numpy(dtype=float)
    n = len(users)
    half = (n // 2 // batch_size) * batch_size
    if half < batch_size:
        raise ValueError("not enough users for the requested batch size")

    max_looks = max(1, half // batch_size)
    requested = n_batches
    n_batches = max(1, min(n_batches, max_looks))

    rows: list[dict] = []
    naive_ever = 0
    av_ever = 0
    for s in range(n_sims):
        rng = np.random.default_rng(seed + s)
        order = rng.permutation(n)
        a_idx = order[:half]
        b_idx = order[half : 2 * half]
        min_naive = 1.0
        min_av = 1.0
        for k in range(1, n_batches + 1):
            sl = slice(0, k * batch_size)
            ia, ib = a_idx[sl], b_idx[sl]
            interim = stats.ratio_interim(searches[ia], revenue[ia], searches[ib], revenue[ib])
            _, p_av = stats.msprt_ratio(interim["D"], interim["z"], interim["I"], tau)
            p_naive = stats._two_sided_p_from_z(interim["z"])
            min_naive = min(min_naive, p_naive)
            min_av = min(min_av, p_av)
            rows.append({"sim": s, "n": int(k * batch_size), "p_naive": p_naive, "p_av": p_av})
        naive_ever += int(min_naive <= ALPHA)
        av_ever += int(min_av <= ALPHA)

    frame = pd.DataFrame(rows)
    summary = {
        "n_sims": n_sims,
        "n_batches_requested": int(requested),
        "n_batches": n_batches,
        "batch_size": batch_size,
        "arm_users": int(half),
        "looks_limited": bool(n_batches < requested),
        "duplicated_interims": 0,
        "alpha": ALPHA,
        "tau": tau,
        "naive_peeking_fpr": naive_ever / n_sims,
        "msprt_always_valid_fpr": av_ever / n_sims,
        # No ratio when the always-valid test never false-positived: dividing by
        # zero yields a non-JSON `Infinity`, and "infinite inflation" overstates
        # an estimate built from a handful of looks.
        "inflation_factor": (naive_ever / av_ever) if av_ever else None,
    }
    return frame, summary


def plot_peeking(frame: pd.DataFrame, summary: dict, path: Path = PEEKING_PNG) -> Path:
    """Bar chart of the two false-positive rates plus the per-interim p-value spread."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    rates = [summary["naive_peeking_fpr"] * 100, summary["msprt_always_valid_fpr"] * 100]
    axes[0].bar(
        ["fixed-horizon\n(peeking)", "mSPRT\n(always-valid)"], rates, color=["#b5651d", "#2e6f95"]
    )
    axes[0].axhline(
        ALPHA * 100, color="black", linestyle="--", linewidth=1, label=f"nominal {ALPHA:.0%}"
    )
    axes[0].set_ylabel("P(ever p <= alpha)  [%]")
    axes[0].set_title(
        f"A/A false-positive rate under continuous monitoring "
        f"({summary['n_batches']} looks x {summary['batch_size']})"
    )
    axes[0].legend()
    for i, v in enumerate(rates):
        axes[0].text(i, v, f"{v:.1f}%", ha="center", va="bottom")

    axes[1].hist(frame["p_naive"], bins=25, alpha=0.7, label="fixed-horizon", color="#b5651d")
    axes[1].hist(frame["p_av"], bins=25, alpha=0.7, label="mSPRT", color="#2e6f95")
    axes[1].axvline(ALPHA, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("p-value at an interim")
    axes[1].set_ylabel("interim count")
    axes[1].set_title("Interim p-values (A/A)")
    axes[1].legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# 10. Sizing on the control arm (the sweep table)
# --------------------------------------------------------------------------- #


def sample_size_readout(users: pd.DataFrame) -> dict:
    """Required users at the observed baseline, unequal vs even allocation."""
    baseline = _ratio(users)
    variance = stats.ratio_variance_per_user(users, clicks="revenue", views="searches")
    planned = stats.required_sample_size(baseline, REL_MDE, variance, ALPHA, POWER, TEST_SHARE)
    even = stats.required_sample_size(baseline, REL_MDE, variance, ALPHA, POWER, 0.5)
    return {
        "baseline_rps": baseline,
        "rel_mde": REL_MDE,
        "alpha": ALPHA,
        "power": POWER,
        "variance_per_user": variance,
        "planned_test_share": TEST_SHARE,
        "planned": planned,
        "even": even,
        "cost_multiple": planned["n_total"] / even["n_total"],
        "power_at_planned_n": stats.achieved_power(
            planned["n_total"], baseline, REL_MDE, variance, ALPHA, TEST_SHARE
        ),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _to_native(obj):
    """Recursively convert numpy scalars/arrays to JSON-native Python types."""
    if isinstance(obj, dict):
        return {str(k): _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.experiment",
        description="A/B readout of a ranking change on revenue per search",
    )
    parser.add_argument(
        "--searches",
        type=int,
        default=DEFAULT_SEARCHES,
        help=(
            "traffic figure for the power block (default 12000 = the dataset). "
            "The readout always runs on the held-out searches; this number only "
            "sets where the MDE is extrapolated to."
        ),
    )
    args = parser.parse_args(argv)
    if args.searches < 100:
        parser.error("--searches must be at least 100")

    if not SERP_OFFERS_PARQUET.exists():
        raise FileNotFoundError(
            f"Missing {SERP_OFFERS_PARQUET}. Run `python -m src.serp_sim` first."
        )
    offers = pd.read_parquet(SERP_OFFERS_PARQUET)

    # Arm B's policy is a fitted model, so it may only be scored on searches it
    # has not learned from: fit on one half, run the experiment on the other.
    train, test_offers = train_test_split_searches(offers, test_frac=0.5, seed=MODEL_SEED)
    ctx: PolicyContext = context_for(test_offers, train=train)

    counterfactuals = arm_outcomes(test_offers, ctx)
    users_all = per_user_totals(test_offers)
    tagged = assign_arms(users_all)
    arms = arms_from_assignment(counterfactuals, tagged)
    control, test = arms["control"], arms["test"]

    print("=== experiment: ranking change on revenue per search ===")
    print(f"source:   {SERP_OFFERS_PARQUET}")
    print(
        f"offers:   {len(offers):,}   searches: {offers['search_id'].nunique():,}   "
        f"users: {offers['user_id'].nunique():,}"
    )
    print(
        f"held-out: {len(test_offers):,} offers, {test_offers['search_id'].nunique():,} searches, "
        f"{len(users_all):,} users (arm B is scored on searches it never learned from)"
    )
    print(
        f"seeds:    split={SPLIT_SEED} outcome={OUTCOME_SEED} model={MODEL_SEED} "
        f"aa={AA_SEED} cuped={CUPED_SEED} peek={PEEK_SEED}"
    )
    print(f"arms:     control={CONTROL_POLICY}  test={TEST_POLICY}  unit=user_id\n")

    srm = srm_readout(tagged)
    print("--- 1. SRM (user-level split vs planned 90/10) ---")
    print(
        f"control {srm['n_control']:,} users  test {srm['n_test']:,} users  "
        f"observed test share {srm['observed_test_share']:.4f}"
    )
    print(
        f"vs planned: chi2={srm['vs_planned']['chi2']:.4f} "
        f"p={srm['vs_planned']['p_value']:.4f}  mismatch={srm['vs_planned']['mismatch']}\n"
    )

    control_totals = per_user_totals(control)
    aa = aa_readout(control_totals)
    print("--- 2. A/A test (control split in two) ---")
    print(
        f"n_a {aa['n_a']:,}  n_b {aa['n_b']:,}  "
        f"ratio {aa['ratio_a']:.2f} vs {aa['ratio_b']:.2f}  "
        f"p={aa['p_value']:.3f}  non-significant={aa['non_significant']}\n"
    )

    primary = primary_readout(control, test)
    print("--- 3. Primary: revenue per search (cluster = user_id) ---")
    print(
        f"{CONTROL_POLICY} {primary['control']['revenue_per_search']:,.2f}  "
        f"{TEST_POLICY} {primary['test']['revenue_per_search']:,.2f}  "
        f"effect {primary['diff_abs']:+,.2f} ({primary['diff_rel']:+.2%})"
    )
    print(
        f"95% CI [{primary['ci_low_abs']:+,.2f}, {primary['ci_high_abs']:+,.2f}] "
        f"({primary['ci_low_rel']:+.2%}, {primary['ci_high_rel']:+.2%})  "
        f"p={primary['p_value']:.4f}  significant={primary['significant']}"
    )
    print(
        f"naive per-unit SE / delta SE = {primary['se_ratio_naive_over_delta']:.3f} "
        f"(naive p={primary['naive_per_unit']['p_value']:.4f})\n"
    )

    balance = arm_balance(arms)
    check = treatment_check(counterfactuals, arms)
    print("--- 4. Arms ---")
    for name, block in balance.items():
        print(
            f"{name:<8}{block['policy']:<20} users {block['n_users']:,}"
            f"  searches {block['n_searches']:,}  RPS {block['revenue_per_search']:,.2f}"
        )
    print(
        f"treatment: {check['share_of_searches_reranked']:.1%} of arm B's SERPs re-ordered, "
        f"mean |Δposition| {check['mean_abs_position_change']:.2f}\n"
    )

    guardrails = guardrails_readout(control, test)
    print("--- 5. Guardrails (95% CI; >10% harm blocks) ---")
    for name, g in guardrails.items():
        print(
            f"{name:<26} {g['control']:.4f} -> {g['test']:.4f}  "
            f"{g['diff_rel']:+.2%} [{g['ci_low_rel']:+.2%}, {g['ci_high_rel']:+.2%}]  "
            f"p={g['p_value']:.3f}  breach={g['breach']}"
        )
    print()

    multiplicity = multiplicity_readout(primary, guardrails)
    print("--- 5b. Multiplicity (Holm + BH over primary + 4 guardrails) ---")
    for name in multiplicity["family"]:
        block = multiplicity["metrics"][name]
        flip = "  FLIP" if block["flips"] else ""
        print(
            f"{name:<26} p_raw={block['p_raw']:.4f}  p_holm={block['p_holm']:.4f}  "
            f"p_bh={block['p_bh']:.4f}{flip}"
        )
    print()

    decision = decision_readout(primary, guardrails)
    print("--- 6. Decision ---")
    print(f"verdict: {decision['verdict']}  ({decision['reason']})\n")

    hetero = heterogeneity_readout(control, test)
    print("--- 7. Heterogeneity by device (CTR only, BH over 2) ---")
    for row in hetero["by_device"]:
        print(
            f"{row['device']:<8} CTR {row['control']:.4f} -> {row['test']:.4f}  "
            f"{row['diff_rel']:+.2%} [{row['ci_low_rel']:+.2%}, {row['ci_high_rel']:+.2%}]  "
            f"p={row['p_value']:.3f}  p_bh={row['p_value_bh']:.3f}  "
            f"MDE {row['mde_rel']:.1%}"
        )
    print("(revenue per search is not tested by device: underpowered on revenue)\n")

    power = power_readout(
        control,
        test,
        traffic_searches=args.searches,
        # Whole-dataset ratio, not the held-out one: the split understates it.
        searches_per_user_full=float(
            offers["search_id"].nunique() / max(offers["user_id"].nunique(), 1)
        ),
    )
    print("--- 8. Power ---")
    print(
        f"baseline {power['baseline_rps']:,.2f} RPS  V={power['variance_per_user']:.4f}  "
        f"{power['searches_per_user']:.2f} searches/user"
    )
    print(
        f"at real traffic ({power['at_real_traffic']['searches']:,} searches): "
        f"MDE {power['at_real_traffic']['mde_rel']:.1%} relative; "
        f"observed {power['observed_effect_rel']:+.2%} -> "
        f"detectable at 12k={power['detectable_at_real_traffic']}"
    )
    print(
        f"this run ({power['run']['users']:,} users): MDE {power['run']['mde_rel']:.1%} -> "
        f"detectable={power['detectable_at_run']}"
    )
    print(
        f"scenario (60,000 searches): MDE {power['at_scenario_60000']['mde_rel']:.1%}; "
        f"n/arm for the observed effect "
        f"{power['required_n_per_arm']['n_per_arm_users']:,.0f} users "
        f"({power['required_n_per_arm']['n_per_arm_searches']:,.0f} searches)\n"
    )

    cuped = cuped_readout(offers)
    print("--- 9. CUPED variance reduction (revenue per search) ---")
    for label in ("real", "positive_control"):
        block = cuped[label]
        print(
            f"{label:<16} n={block['n_users']:,}  corr={block['corr_x_y']:.3f}  "
            f"var reduction={block['variance_reduction_pct']:+.1f}%  "
            f"CI narrowing={block['ci_narrowing_pct']:+.1f}%"
        )
    print(f"(real block: {cuped['real']['verdict']}; simulator untouched)\n")

    frame, peek = peeking_demo(users_all, tau=primary["control"]["revenue_per_search"] * REL_MDE)
    print(
        f"--- 10. Peeking (A/A, {peek['n_sims']} streams x {peek['n_batches']} looks "
        f"x {peek['batch_size']}) ---"
    )
    print(
        f"fixed-horizon FPR (ever p<=alpha): {peek['naive_peeking_fpr']:.1%}   "
        f"mSPRT always-valid FPR: {peek['msprt_always_valid_fpr']:.1%}   "
        f"(nominal {ALPHA:.0%})"
    )
    print(
        f"looks capped at arm size: {peek['looks_limited']} "
        f"({peek['n_batches_requested']} requested -> {peek['n_batches']}), "
        f"duplicated interims: {peek['duplicated_interims']}\n"
    )

    sizing = sample_size_readout(control_totals)
    print("--- 11. Sizing at the observed baseline ---")
    print(
        f"90/10 needs {sizing['planned']['n_total']:,.0f} users for a "
        f"{sizing['rel_mde']:.0%} lift; even split {sizing['even']['n_total']:,.0f}\n"
    )

    phase2 = None
    if RANKING_REPORT.exists():
        phase2 = json.loads(RANKING_REPORT.read_text(encoding="utf-8")).get("divergence")

    report = {
        "seeds": {
            "split": SPLIT_SEED,
            "outcome": OUTCOME_SEED,
            "model": MODEL_SEED,
            "aa": AA_SEED,
            "cuped": CUPED_SEED,
            "peek": PEEK_SEED,
        },
        "design": {
            "test_share": TEST_SHARE,
            "rel_mde_for_sizing": REL_MDE,
            "alpha": ALPHA,
            "power": POWER,
            "randomisation_unit": "user_id",
            "control_policy": CONTROL_POLICY,
            "test_policy": TEST_POLICY,
            "guardrail_max_rel_change": GUARDRAIL_MAX_REL_CHANGE,
            "held_out_searches": True,
        },
        "inputs": {
            "offers": int(len(offers)),
            "searches": int(offers["search_id"].nunique()),
            "users": int(offers["user_id"].nunique()),
            "searches_experiment": int(test_offers["search_id"].nunique()),
            "users_experiment": int(len(users_all)),
            "searches_requested": int(args.searches),
        },
        "arms": {
            **balance,
            "randomisation_unit": "user_id",
            "disjoint_by_user": bool(set(control["user_id"]).isdisjoint(set(test["user_id"]))),
            "shared_outcome_seed": OUTCOME_SEED,
            "treatment_check": check,
        },
        "srm": srm,
        "aa_test": aa,
        "primary": primary,
        "guardrails": guardrails,
        "multiplicity": multiplicity,
        "decision": decision,
        "heterogeneity": hetero,
        "power": power,
        "cuped": cuped,
        "peeking": peek,
        "sample_size": sizing,
        "ratio_metric": ratio_metric_readout(control, test),
        "phase2_divergence": phase2,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_JSON.write_text(json.dumps(_to_native(report), indent=2), encoding="utf-8")
    png = plot_peeking(frame, peek)
    print(f"written: {EXPERIMENT_JSON}")
    print(f"written: {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
