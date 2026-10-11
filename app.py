"""Streamlit dashboard for SERP Rank Lab (Phase 3).

    uv run streamlit run app.py --server.headless true

Four pages, built with the modern ``st.navigation`` / ``st.Page`` API:

* **Funnel** -- search -> click -> redirect -> buy on the results page.
* **Revenue by position** -- the decay of revenue per search down the page.
* **Ranking policies** -- the NDCG-vs-revenue divergence from Phase 2.
* **Experiment** -- the Phase 3 A/B readout (SRM, A/A, ratio metric, CUPED,
  peeking, sample size).

Data is read from ``data/processed/`` and ``reports/`` and cached with
``@st.cache_data``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent
SERP_OFFERS = PROJECT_ROOT / "data" / "processed" / "serp_offers.parquet"
RANKING_JSON = PROJECT_ROOT / "reports" / "ranking_eval.json"
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


def _need_offers() -> pd.DataFrame | None:
    if not SERP_OFFERS.exists():
        st.warning(
            "`data/processed/serp_offers.parquet` is missing. Run `uv run python run.py --stage serp`."
        )
        return None
    return load_offers()


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
    st.title("Ranking policies: relevance vs revenue")
    report = load_json(str(RANKING_JSON))
    if report is None:
        st.warning(
            "`reports/ranking_eval.json` is missing. Run `uv run python run.py --stage rank`."
        )
        return

    policies = pd.DataFrame(report["policies"])
    divergence = report["divergence"]

    st.markdown(
        "The headline result: a ranking that is **better offline** (higher NDCG@k) can be "
        "**worse online** (lower revenue per search). NDCG and revenue disagree because the "
        "money-rich partners quote slightly higher prices and rank slightly worse on relevance."
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


def page_experiment() -> None:
    st.title("Experiment readout")
    report = load_json(str(EXPERIMENT_JSON))
    if report is None:
        st.warning(
            "`reports/experiment.json` is missing. Run `uv run python run.py --stage experiment`."
        )
        return

    st.caption(
        f"seeds: split={report['seeds']['split']} aa={report['seeds']['aa']} "
        f"cuped={report['seeds']['cuped']} peek={report['seeds']['peek']}"
    )

    srm = report["srm"]
    st.subheader("1. Sample ratio mismatch")
    c1, c2, c3 = st.columns(3)
    c1.metric("observed test share", f"{srm['observed_test_share']:.4f}")
    c2.metric("SRM p (vs planned 90/10)", f"{srm['vs_planned']['p_value']:.4f}")
    c3.metric("SRM p (vs assumed 50/50)", f"{srm['vs_fifty_fifty']['p_value']:.2e}")

    aa = report["aa_test"]
    st.subheader("2. A/A test")
    st.write(
        f"Control split in two: p = **{aa['p_value']:.3f}** -> "
        f"non-significant = **{aa['non_significant']}**"
    )

    ratio = report["ratio_metric"]
    st.subheader("3. Ratio metric: naive per-user vs delta method")
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
    st.subheader("4. CUPED variance reduction")
    st.dataframe(
        pd.DataFrame(
            {
                "block": ["real pre-period", "positive control"],
                "n_users": [cuped["real"]["n_users"], cuped["positive_control"]["n_users"]],
                "corr(X, Y)": [cuped["real"]["corr_x_y"], cuped["positive_control"]["corr_x_y"]],
                "variance reduction %": [
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

    peek = report["peeking"]
    st.subheader("5. Peeking: fixed-horizon vs always-valid")
    c1, c2 = st.columns(2)
    c1.metric(
        "fixed-horizon FPR",
        f"{peek['naive_peeking_fpr']:.1%}",
        delta=f"nominal {peek['alpha']:.0%}",
    )
    c2.metric("mSPRT always-valid FPR", f"{peek['msprt_always_valid_fpr']:.1%}")
    peeking_png = PROJECT_ROOT / "reports" / "peeking.png"
    if peeking_png.exists():
        st.image(str(peeking_png))

    sizing = report["sample_size"]
    st.subheader("6. Required sample size")
    c1, c2, c3 = st.columns(3)
    c1.metric("baseline revenue/search", f"{sizing['baseline_rps']:,.1f}")
    c2.metric("n needed (90/10)", f"{sizing['planned']['n_total']:,.0f}")
    c3.metric("n needed (50/50)", f"{sizing['even']['n_total']:,.0f}")


# --------------------------------------------------------------------------- #
# Navigation
# --------------------------------------------------------------------------- #

st.set_page_config(page_title="SERP Rank Lab", layout="wide")

pages = [
    st.Page(page_funnel, title="Funnel", icon=":material/filter_alt:", default=True),
    st.Page(page_revenue_by_position, title="Revenue by position", icon=":material/trending_down:"),
    st.Page(page_ranking_policies, title="Ranking policies", icon=":material/compare_arrows:"),
    st.Page(page_experiment, title="Experiment", icon=":material/science:"),
]

st.navigation(pages).run()
