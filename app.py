"""Streamlit dashboard for SERP Rank Lab (Phase 3).

    uv run streamlit run app.py --server.headless true

Seven pages, built with the modern ``st.navigation`` / ``st.Page`` API:

* **Funnel** -- search -> click -> redirect -> buy on the results page.
* **Revenue by position** -- the decay of revenue per search down the page.
* **Ranking policies** -- the feasible pair (same information set, revenue vs CTR), the
  exchange rate per forgone click, and the reference oracles behind a `modeled` label.
* **Position bias & partners** -- the examine-probability curve by position
  (mobile vs desktop) and the partner CTR-vs-commission quadrant.
* **Ancillary & refunds** -- the measured order mix, refunds, booking depth and
  price/profit stats.
* **Supply & routes** -- measured airlines, top routes, seasonality and the
  passenger mix.
* **Experiment** -- the rollout readout, verdict first (decision + guardrails), then
  primary metric, power, heterogeneity, SRM, A/A, ratio metric, CUPED, peeking and
  sample size, plus a measured clickstream A/B tab.

Data is read from ``data/processed/`` and ``reports/`` and cached with
``@st.cache_data``. Pages over ``serp_offers`` / ``ranking_eval.json`` are
**modeled** (synthetic); pages over ``reports/business.json`` are **measured**.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from src.business import AB_LABEL, BUSINESS_JSON  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent
SERP_OFFERS = PROJECT_ROOT / "data" / "processed" / "serp_offers.parquet"
PARTNER_QUALITY = PROJECT_ROOT / "data" / "processed" / "partner_quality.parquet"
RANKING_JSON = PROJECT_ROOT / "reports" / "ranking_eval.json"
HEADLINE_JSON = PROJECT_ROOT / "reports" / "headline.json"
EXPERIMENT_JSON = PROJECT_ROOT / "reports" / "experiment.json"


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #


@st.cache_data(show_spinner=False)
def load_offers() -> pd.DataFrame:
    """Load ``serp_offers.parquet`` (cached across reruns)."""
    return pd.read_parquet(SERP_OFFERS)


@st.cache_data(show_spinner=False)
def load_json(path: str) -> dict | None:
    """Load a JSON report if it exists, else ``None``."""
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


@st.cache_data(show_spinner=False)
def load_business() -> dict:
    """Load ``reports/business.json`` (measured islands + clickstream A/B, cached)."""
    return json.loads(BUSINESS_JSON.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_partner_quality() -> pd.DataFrame:
    """Load ``partner_quality.parquet`` (partner-level monetisation SQL view, cached)."""
    return pd.read_parquet(PARTNER_QUALITY)


def _need_offers() -> pd.DataFrame | None:
    if not SERP_OFFERS.exists():
        st.warning(
            "`data/processed/serp_offers.parquet` is missing. Run `uv run python run.py --stage serp`."
        )
        return None
    return load_offers()


def _need_business() -> dict | None:
    if not BUSINESS_JSON.exists():
        st.warning(
            "`reports/business.json` is missing. Run `uv run python run.py --stage business`."
        )
        return None
    return load_business()


def _need_partner_quality() -> pd.DataFrame | None:
    if not PARTNER_QUALITY.exists():
        st.warning(
            "`data/processed/partner_quality.parquet` is missing. "
            "Run `uv run python run.py --stage sql`."
        )
        return None
    return load_partner_quality()


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #


def page_funnel() -> None:
    st.title("Funnel: search -> click -> redirect -> buy")
    offers = _need_offers()
    if offers is None:
        return

    searches = int(offers["search_id"].nunique())
    by_search = offers.groupby("search_id")
    steps = {
        "search": searches,
        "click": int(by_search["clicked"].any().sum()),
        "redirect": int(by_search["redirected"].any().sum()),
        "buy": int(by_search["booked"].any().sum()),
    }
    frame = pd.DataFrame({"step": list(steps), "searches": list(steps.values())})
    frame["share_of_searches"] = frame["searches"] / searches

    left, right = st.columns([3, 2])
    with left:
        fig, ax = plt.subplots(figsize=(7, 3.4))
        ax.bar(frame["step"], frame["searches"], color=["#2e6f95", "#3d8bb5", "#63a6c4", "#8cc0d6"])
        for i, v in enumerate(frame["searches"]):
            ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("searches with >=1 such action")
        ax.set_title("Funnel by search")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.metric("searches", f"{searches:,}")
        st.metric("click-through (search level)", f"{steps['click'] / searches:.1%}")
        st.metric("search -> buy", f"{steps['buy'] / searches:.1%}")
        st.dataframe(frame.set_index("step"), width="stretch")

    st.caption(
        "Offer-impression counts (a search can click several offers): "
        f"clicked={int(offers['clicked'].sum()):,}, "
        f"redirected={int(offers['redirected'].sum()):,}, "
        f"booked={int(offers['booked'].sum()):,}."
    )


def page_revenue_by_position() -> None:
    st.title("Revenue per search by position")
    offers = _need_offers()
    if offers is None:
        return

    n_searches = offers["search_id"].nunique()
    grouped = offers.groupby("position")
    frame = pd.DataFrame(
        {
            "position": grouped.size().index,
            "revenue_per_search": grouped["commission"].sum().to_numpy() / n_searches,
            "click_rate": grouped["clicked"].mean().to_numpy(),
            "impressions": grouped.size().to_numpy(),
        }
    ).sort_values("position")

    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax1.bar(
        frame["position"], frame["revenue_per_search"], color="#2e6f95", label="revenue per search"
    )
    ax1.set_xlabel("position")
    ax1.set_ylabel("revenue per search", color="#2e6f95")
    ax2 = ax1.twinx()
    ax2.plot(
        frame["position"], frame["click_rate"], color="#b5651d", marker="o", label="click rate"
    )
    ax2.set_ylabel("click rate", color="#b5651d")
    ax1.set_title("Monetisation and attention decay down the results page")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    top = frame.iloc[0]["revenue_per_search"]
    tail = frame.iloc[-1]["revenue_per_search"]
    c1, c2, c3 = st.columns(3)
    c1.metric("position 1 revenue/search", f"{top:,.1f}")
    c2.metric(f"position {int(frame.iloc[-1]['position'])} revenue/search", f"{tail:,.1f}")
    c3.metric("position-1 / tail", f"{top / tail:.1f}x" if tail else "n/a")
    st.dataframe(frame, width="stretch", hide_index=True)


def page_ranking_policies() -> None:
    st.title("Ranking policies: the price of ranking by money")

    headline = load_json(str(HEADLINE_JSON))
    if headline is not None:
        hl = headline["headline"]
        st.markdown(
            "Two **feasible** policies on the *same* information set (one feature matrix, one "
            "`beta`, one greedy slot mechanism). Only the utility differs: fitted click "
            "probability vs fitted click probability x commission. The exchange rate below is "
            "what the decision turns on."
        )
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "revenue / search delta",
            f"{hl['revenue_per_search_delta_abs']:+,.2f}",
            delta=f"{hl['revenue_per_search_delta_rel']:+.2%}",
        )
        c2.metric(
            "CTR delta",
            f"{hl['ctr_delta_abs']:+.4f}",
            delta=f"{hl['ctr_delta_rel']:+.2%}",
        )
        c3.metric(
            "breakeven ₽ / forgone click",
            f"{abs(hl['breakeven_rub_per_forgone_click']):,.2f}",
            help="An exchange rate, not a verdict: the value of a click (LTV, brand) is not in the data.",
        )

        rows = [
            {
                "policy": name,
                "information_set": cell["information_set"],
                "ctr": cell["ctr"],
                "revenue_per_search": cell["revenue_per_search"],
                "ndcg_at_10": cell["ndcg_at_10"],
                "conversion": cell["conversion"],
            }
            for name, cell in headline["policies"].items()
        ]
        frame = pd.DataFrame(rows)
        st.dataframe(
            frame,
            width="stretch",
            hide_index=True,
            column_config={
                "ctr": st.column_config.NumberColumn(format="%.4f"),
                "revenue_per_search": st.column_config.NumberColumn(format="%.2f"),
                "ndcg_at_10": st.column_config.NumberColumn(format="%.4f"),
                "conversion": st.column_config.NumberColumn(format="%.4f"),
            },
        )

        fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
        axes[0].bar(frame["policy"], frame["ctr"], color="#3d8bb5")
        axes[0].set_title("CTR (clicks / impressions)")
        axes[1].bar(frame["policy"], frame["revenue_per_search"], color="#b5651d")
        axes[1].set_title("revenue per search (₽)")
        for ax in axes:
            ax.tick_params(axis="x", rotation=15)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        failed = headline["failed_hypothesis"]
        st.info(
            f"**Hypothesis not confirmed:** the learned ranker earns "
            f"{failed['learned_ranker']:,.2f} ₽/search against {failed['price_asc']:,.2f} ₽ for "
            f"plain price sorting — {failed['learned_ranker'] - failed['price_asc']:+,.2f} ₽. "
            "A one-variable policy beats the learned one."
        )
        ceiling = headline["ceiling_oracle"]
        st.caption(
            f"Modeled ceiling (oracle pair): {ceiling['revenue_per_search']:,.2f} ₽/search, "
            f"label `{ceiling['label']}`. Not a shippable result."
        )

    report = load_json(str(RANKING_JSON))
    if report is None:
        st.warning(
            "`reports/ranking_eval.json` is missing. Run `uv run python run.py --stage rank`."
        )
        return

    policies = pd.DataFrame(report["policies"])
    divergence = report["divergence"]

    st.subheader("Reference oracles: better offline is not better online")
    st.markdown(
        "The two reference policies read the simulator's ground truth, so this pair is a "
        "**modeled** statement about the objective, not about a shippable ranker."
    )
    c1, c2 = st.columns(2)
    c1.metric("NDCG delta (relevance - revenue)", f"{divergence['ndcg_delta']:+.4f}")
    c2.metric("revenue/search delta", f"{divergence['revenue_delta']:+,.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    real = policies[policies["policy"] != "delta_relevance_minus_revenue"]
    axes[0].bar(real["policy"], real["ndcg_at_k"], color="#3d8bb5")
    axes[0].set_title(f"NDCG@{divergence['ndcg_k']}")
    axes[1].bar(real["policy"], real["revenue_per_search"], color="#b5651d")
    axes[1].set_title("revenue per search")
    for ax in axes:
        ax.tick_params(axis="x", rotation=12)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.dataframe(policies, width="stretch", hide_index=True)
    ctr = report.get("ctr")
    if ctr:
        st.markdown("**Position bias in the observed CTR**")
        st.dataframe(
            pd.DataFrame(
                {
                    "estimator": ["naive", "IPW (known curve)", "IPW (EM)", "true position-free"],
                    "ctr": [
                        ctr["naive"],
                        ctr["ipw_known_curve"],
                        ctr["ipw_em_estimated"],
                        ctr["true_position_free"],
                    ],
                }
            ),
            width="stretch",
            hide_index=True,
        )


def page_position_bias_partners() -> None:
    st.title("Position bias & partners")
    st.markdown(
        "Attention, not relevance, sets the ceiling on a SERP: the probability a "
        "result is even *examined* collapses down the page, so a slot's value is "
        "as much about position as about the partner in it. The quadrant below "
        "separates partners that earn their slot from those that only occupy it."
    )
    st.caption("Modeled: `serp_offers` and `ranking_eval.json` are synthetic.")

    offers = _need_offers()
    if offers is None:
        return

    bias = (
        offers.groupby(["device", "position"], as_index=False)["examine_prob"]
        .mean()
        .sort_values("position")
    )
    fig = px.line(
        bias,
        x="position",
        y="examine_prob",
        color="device",
        markers=True,
        title="Position bias: mean examine probability by position (mobile vs desktop)",
    )
    st.plotly_chart(fig, width="stretch")

    report = load_json(str(RANKING_JSON))
    if report is not None and report.get("propensity_estimated"):
        propensity = pd.DataFrame(
            {
                "position": [int(k) for k in report["propensity_estimated"]],
                "propensity": list(report["propensity_estimated"].values()),
            }
        ).sort_values("position")
        st.caption("Position bias re-estimated from `ranking_eval.json` (EM propensity).")
        fig = px.line(
            propensity,
            x="position",
            y="propensity",
            markers=True,
            title="Estimated examine propensity by position (ranking_eval.json)",
        )
        st.plotly_chart(fig, width="stretch")

    if "partner_type" in offers.columns:
        st.subheader("CPA vs CPC split")
        split = offers.groupby("partner_type", as_index=False).agg(
            offers=("commission", "size"), revenue=("commission", "sum")
        )
        split["revenue_per_offer"] = split["revenue"] / split["offers"].replace(0, pd.NA)
        st.dataframe(split, width="stretch", hide_index=True)

    partner = _need_partner_quality()
    if partner is None:
        return
    st.subheader("Partner quadrant: CTR vs commission per offer")
    fig = px.scatter(
        partner,
        x="ctr",
        y="commission_per_offer",
        size="searches_present",
        hover_name="partner",
        title="Partner quadrant (bubble = searches present)",
    )
    st.plotly_chart(fig, width="stretch")
    st.dataframe(
        partner[
            [
                "partner",
                "offers",
                "ctr",
                "booking_rate",
                "commission_per_offer",
                "monetisation_rank",
            ]
        ].sort_values("monetisation_rank"),
        width="stretch",
        hide_index=True,
    )


def page_ancillary() -> None:
    st.title("Ancillary & refunds")
    st.markdown(
        "Orders are the realised demand: which ancillary services attach, how "
        "often an order reverses as a refund, and how deep a booking goes. The "
        "refund rate and the near-zero service profit are the two numbers that "
        "set the economics floor."
    )
    st.caption("Measured: the real `orders` island (no foreign key to users or the SERP log).")

    business = _need_business()
    if business is None:
        return
    orders = business["orders"]

    service_mix = pd.DataFrame(orders["service_mix"])
    platform_mix = pd.DataFrame(orders["platform_mix"])
    depth = pd.DataFrame(orders["booking_depth"])
    price_stats = pd.DataFrame(orders["price_stats"])

    refund_rate = float(service_mix["refund_share_of_all_orders"].iloc[0])
    c1, c2, c3 = st.columns(3)
    c1.metric("orders", f"{int(service_mix['count'].sum()):,}")
    c2.metric("refund rate", f"{refund_rate:.1%}")
    c3.metric("services tracked", f"{int(service_mix['service'].nunique()):,}")

    left, right = st.columns(2)
    with left:
        fig = px.bar(
            service_mix.sort_values("count", ascending=False),
            x="service",
            y="count",
            title="Orders by service",
        )
        st.plotly_chart(fig, width="stretch")
    with right:
        fig = px.bar(
            platform_mix.sort_values("count", ascending=False),
            x="order_platform",
            y="count",
            title="Orders by platform",
        )
        st.plotly_chart(fig, width="stretch")

    fig = px.bar(depth, x="bin_left", y="count", title="Booking-depth histogram")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Price and profit stats")
    stats_note = price_stats["note"].iloc[0] if "note" in price_stats.columns else None
    numeric = price_stats.drop(columns=[c for c in ["note"] if c in price_stats.columns])
    st.dataframe(numeric.T.rename(columns={0: "value"}), width="stretch")
    if stats_note:
        st.caption(str(stats_note))
    st.dataframe(service_mix, width="stretch", hide_index=True)


def page_supply_routes() -> None:
    st.title("Supply & routes")
    st.markdown(
        "Supply is concentrated: one destination dominates the leg mix, so the "
        "airline shares, the connection/baggage attach rates and the month-of-year "
        "seasonality all describe a small number of real corridors rather than a "
        "broad network."
    )
    st.caption("Measured: the real `bookings_legs` island (one row per flight leg).")

    business = _need_business()
    if business is None:
        return
    routes = business["routes"]

    airlines = pd.DataFrame(routes["airlines"])
    top_routes = pd.DataFrame(routes["routes"])
    monthly = pd.DataFrame(routes["monthly"])
    passengers = pd.DataFrame(routes["passengers"])

    connection_share = float(top_routes["connection_share"].iloc[0])
    baggage_share = float(top_routes["baggage_share"].iloc[0])
    c1, c2, c3 = st.columns(3)
    c1.metric("legs", f"{int(airlines['count'].sum()):,}")
    c2.metric("connection share", f"{connection_share:.1%}")
    c3.metric("baggage share", f"{baggage_share:.1%}")

    fig = px.bar(
        airlines.sort_values("count", ascending=False),
        x="airline",
        y="count",
        title="Airline share",
    )
    st.plotly_chart(fig, width="stretch")

    top_routes = top_routes.copy()
    top_routes["route"] = (
        top_routes["origin"].astype(str) + " -> " + top_routes["destination"].astype(str)
    )
    fig = px.bar(
        top_routes.sort_values("count"),
        x="count",
        y="route",
        orientation="h",
        title="Top origin -> destination routes",
    )
    st.plotly_chart(fig, width="stretch")

    monthly = monthly.sort_values("booking_month")
    left, right = st.columns(2)
    with left:
        fig = px.bar(
            monthly,
            x="booking_month",
            y="legs",
            title="Seasonality: legs by booking month",
        )
        st.plotly_chart(fig, width="stretch")
    with right:
        fig = px.bar(passengers, x="passengers", y="count", title="Passenger mix")
        st.plotly_chart(fig, width="stretch")

    st.dataframe(top_routes, width="stretch", hide_index=True)


def page_experiment() -> None:
    st.title("Experiment readout")
    report = load_json(str(EXPERIMENT_JSON))
    if report is None:
        st.warning(
            "`reports/experiment.json` is missing. Run `uv run python run.py --stage experiment`."
        )
        return

    tab_ab, tab_click = st.tabs(["Phase 3 A/B", "Clickstream A/B"])

    with tab_ab:
        st.caption("Modeled: the simulated Phase 3 experiment (`reports/experiment.json`).")
        st.caption(
            f"seeds: split={report['seeds']['split']} aa={report['seeds']['aa']} "
            f"cuped={report['seeds']['cuped']} peek={report['seeds']['peek']}"
        )

        # Decision first: the readout exists to produce one verdict.
        decision = report["decision"]
        if decision["verdict"] == "do_not_ship":
            st.error(
                f"**Verdict: `{decision['verdict']}`** — {decision['reason']}. "
                f"Stopping rule: {decision['stopping_rule']}."
            )
        else:
            st.success(f"**Verdict: `{decision['verdict']}`** — {decision['reason']}.")

        primary = report["primary"]
        st.subheader("1. Primary metric: revenue per search")
        c1, c2, c3 = st.columns(3)
        c1.metric("control (production)", f"{primary['control']['revenue_per_search']:,.2f} ₽")
        c2.metric(
            "test (revenue_objective)",
            f"{primary['test']['revenue_per_search']:,.2f} ₽",
            delta=f"{primary['diff_rel']:+.2%}",
        )
        c3.metric(
            "p-value (cluster: user_id)",
            f"{primary['p_value']:.4f}",
            delta="significant" if primary["significant"] else "not significant",
        )
        st.write(
            f"Δ = **{primary['diff_abs']:+,.2f} ₽** per search; 95% CI "
            f"[{primary['ci_low_rel']:+.2%}; {primary['ci_high_rel']:+.2%}]; "
            f"SE = {primary['se']:,.2f}. Definition: `{primary['definition']}`."
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "arm": "control",
                        "policy": report["arms"]["control"]["policy"],
                        "users": report["arms"]["control"]["n_users"],
                        "searches": report["arms"]["control"]["n_searches"],
                        "revenue_per_search": report["arms"]["control"]["revenue_per_search"],
                    },
                    {
                        "arm": "test",
                        "policy": report["arms"]["test"]["policy"],
                        "users": report["arms"]["test"]["n_users"],
                        "searches": report["arms"]["test"]["n_searches"],
                        "revenue_per_search": report["arms"]["test"]["revenue_per_search"],
                    },
                ]
            ),
            width="stretch",
            hide_index=True,
        )

        treatment = report["arms"]["treatment_check"]
        st.caption(
            "Treatment check: "
            f"{treatment['share_of_searches_reranked']:.1%} of searches re-ranked, "
            f"{treatment['share_of_offers_moved']:.1%} of offers moved, "
            f"mean |position change| = {treatment['mean_abs_position_change']:.2f}."
        )

        st.subheader("2. Guardrails (a breach blocks the rollout)")
        guardrails = report["guardrails"]
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "guardrail": g["label"],
                        "control": g["control"],
                        "test": g["test"],
                        "delta_%": g["diff_rel"] * 100,
                        "p_value": g["p_value"],
                        "harmful_direction": g["harmful_direction"],
                        "breach": g["breach"],
                    }
                    for g in guardrails.values()
                ]
            ),
            width="stretch",
            hide_index=True,
            column_config={
                "delta_%": st.column_config.NumberColumn(format="%.2f"),
                "p_value": st.column_config.NumberColumn(format="%.3g"),
            },
        )
        st.caption(decision["changes_our_mind"])

        st.subheader("3. Power: why this effect is hard to detect at real traffic")
        power = report["power"]
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "MDE, this run",
            f"{power['run']['mde_rel']:.2%}",
            help=power["run"]["label"],
        )
        c2.metric(
            "MDE, real traffic",
            f"{power['at_real_traffic']['mde_rel']:.2%}",
            help=power["at_real_traffic"]["label"],
        )
        c3.metric(
            "MDE, 60k scenario",
            f"{power['at_scenario_60000']['mde_rel']:.2%}",
            help=power["at_scenario_60000"]["label"],
        )
        required = power["required_n_per_arm"]
        st.write(
            f"Observed effect {required['observed_effect_rel']:+.2%} would need "
            f"**{required['n_per_arm_users']:,.0f} users per arm** at 80% power. "
            f"At the live 90/10 split: {power['required_n_at_planned_split']['n_control_users']:,.0f} "
            f"control / {power['required_n_at_planned_split']['n_test_users']:,.0f} test."
        )

        st.subheader("4. Heterogeneity by device (CTR only)")
        het = report["heterogeneity"]
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "device": row["device"],
                        "control": row["control"],
                        "test": row["test"],
                        "delta_%": row["diff_rel"] * 100,
                        "ci_low_%": row["ci_low_rel"] * 100,
                        "ci_high_%": row["ci_high_rel"] * 100,
                        "p_value_bh": row["p_value_bh"],
                        "significant_bh": row["significant_bh"],
                    }
                    for row in het["by_device"]
                ]
            ),
            width="stretch",
            hide_index=True,
            column_config={
                "delta_%": st.column_config.NumberColumn(format="%.2f"),
                "ci_low_%": st.column_config.NumberColumn(format="%.2f"),
                "ci_high_%": st.column_config.NumberColumn(format="%.2f"),
            },
        )
        st.caption(het["revenue_note"])

        srm = report["srm"]
        st.subheader("5. Sample ratio mismatch")
        c1, c2, c3 = st.columns(3)
        c1.metric("observed test share", f"{srm['observed_test_share']:.4f}")
        c2.metric("SRM p (vs planned 90/10)", f"{srm['vs_planned']['p_value']:.4f}")
        c3.metric("SRM p (vs assumed 50/50)", f"{srm['vs_fifty_fifty']['p_value']:.2e}")

        aa = report["aa_test"]
        st.subheader("6. A/A test")
        st.write(
            f"Control split in two: p = **{aa['p_value']:.3f}** -> "
            f"non-significant = **{aa['non_significant']}**"
        )

        ratio = report["ratio_metric"]
        st.subheader("7. Ratio metric: naive per-user vs delta method")
        c1, c2 = st.columns(2)
        c1.metric(
            "delta method p",
            f"{ratio['delta_method']['p_value']:.4f}",
            help="correct: user is the cluster",
        )
        c2.metric(
            "naive per-unit p",
            f"{ratio['naive_per_unit']['p_value']:.4f}",
            help="ignores denominator variation",
        )
        st.write(f"SE ratio (naive / delta) = **{ratio['se_ratio_naive_over_delta']:.3f}**")

        cuped = report["cuped"]
        st.subheader("8. CUPED variance reduction")
        st.dataframe(
            pd.DataFrame(
                {
                    "block": ["real pre-period", "positive control"],
                    "n_users": [cuped["real"]["n_users"], cuped["positive_control"]["n_users"]],
                    "corr(X, Y)": [
                        cuped["real"]["corr_x_y"],
                        cuped["positive_control"]["corr_x_y"],
                    ],
                    "variance reduction % (corr^2)": [
                        cuped["real"]["variance_reduction_pct"],
                        cuped["positive_control"]["variance_reduction_pct"],
                    ],
                    "CI narrowing %": [
                        cuped["real"]["ci_narrowing_pct"],
                        cuped["positive_control"]["ci_narrowing_pct"],
                    ],
                }
            ),
            width="stretch",
            hide_index=True,
        )
        st.caption(cuped["interpretation"]["real"])

        peek = report["peeking"]
        st.subheader("9. Peeking: fixed-horizon vs always-valid")
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "fixed-horizon FPR",
            f"{peek['naive_peeking_fpr']:.1%}",
            delta=f"nominal {peek['alpha']:.0%}",
        )
        c2.metric("mSPRT always-valid FPR", f"{peek['msprt_always_valid_fpr']:.1%}")
        c3.metric("looks used", f"{peek['n_batches']} of {peek['n_batches_requested']}")
        st.caption(
            f"mSPRT never crossed in {peek['n_sims']} A/A streams x {peek['n_batches']} looks; "
            f"looks are capped at arm size "
            f"({peek['arm_users']} users / batch {peek['batch_size']}), so no interim is duplicated."
        )
        peeking_png = PROJECT_ROOT / "reports" / "peeking.png"
        if peeking_png.exists():
            st.image(str(peeking_png))

        sizing = report["sample_size"]
        st.subheader("10. Required sample size")
        c1, c2, c3 = st.columns(3)
        c1.metric("baseline revenue/search", f"{sizing['baseline_rps']:,.1f}")
        c2.metric("n needed (90/10)", f"{sizing['planned']['n_total']:,.0f}")
        c3.metric("n needed (50/50)", f"{sizing['even']['n_total']:,.0f}")

    with tab_click:
        business = _need_business()
        if business is None:
            return
        click = business["clickstream_ab"]
        balance = pd.DataFrame(click["balance"])
        per_user = pd.DataFrame(click["per_user"])
        dau = pd.DataFrame(click["dau"])

        st.markdown(
            "The real clickstream A/B is **measured**, but it can only speak at the "
            "**volume** level: the `events` table carries no event taxonomy, so there "
            "is no funnel here -- only per-user activity and daily active users by arm. "
            "The split is a true 50/50, so the arm balance is checkable with an SRM test."
        )
        st.caption(f"Measured: {AB_LABEL}")
        st.caption(
            "Note: `events` has no event taxonomy -- the metric is volume-level "
            "(activity counts), not funnel-level. Events and sessions share the "
            "`user_id` cluster with `serp_offers`."
        )

        srm_p = float(balance["srm_p_value"].iloc[0])
        srm_chi2 = float(balance["srm_chi2"].iloc[0])
        c1, c2 = st.columns(2)
        c1.metric("SRM chi-square (events vs sessions)", f"{srm_chi2:,.2f}")
        c2.metric("SRM p-value", f"{srm_p:.3f}")

        st.subheader("Arm balance (users per source)")
        st.dataframe(balance, width="stretch", hide_index=True)

        st.subheader("Per-user volume by arm (events vs sessions)")
        fig = px.bar(
            per_user,
            x="metric",
            y="mean",
            color="ab_group",
            barmode="group",
            title="Mean per-user volume by arm",
        )
        st.plotly_chart(fig, width="stretch")
        st.dataframe(per_user, width="stretch", hide_index=True)

        st.subheader("Daily active users by arm")
        fig = px.line(
            dau.sort_values("pdate"),
            x="pdate",
            y="dau",
            color="ab_group",
            markers=True,
            title="DAU by pdate and arm",
        )
        st.plotly_chart(fig, width="stretch")


# --------------------------------------------------------------------------- #
# Navigation
# --------------------------------------------------------------------------- #

st.set_page_config(page_title="SERP Rank Lab", layout="wide")

pages = [
    st.Page(page_funnel, title="Funnel", icon=":material/filter_alt:", default=True),
    st.Page(page_revenue_by_position, title="Revenue by position", icon=":material/trending_down:"),
    st.Page(page_ranking_policies, title="Ranking policies", icon=":material/compare_arrows:"),
    st.Page(
        page_position_bias_partners,
        title="Position bias & partners",
        icon=":material/ads_click:",
    ),
    st.Page(page_ancillary, title="Ancillary & refunds", icon=":material/receipt_long:"),
    st.Page(page_supply_routes, title="Supply & routes", icon=":material/public:"),
    st.Page(page_experiment, title="Experiment", icon=":material/science:"),
]

if __name__ == "__main__":
    st.navigation(pages).run()
