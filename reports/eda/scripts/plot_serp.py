"""serp-rank-lab — EDA figures (matplotlib-scripts contract, light theme).

Saves 5 PNGs (dpi=300) to reports/eda/figures/. Read-only on data.

    /Users/nikitaboarkin/Desktop/00 ide/00 portfolio/.eda-tools/.venv/bin/python \
        reports/eda/scripts/plot_serp.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.family": ["DejaVu Sans", "sans-serif"],
        "axes.unicode_minus": False,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
    }
)

ROOT = Path(__file__).resolve().parents[3]
D = ROOT / "data" / "processed"
OUT = ROOT / "reports" / "eda" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name: str) -> None:
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    offers = pd.read_parquet(D / "serp_offers.parquet")
    orders = pd.read_parquet(D / "orders.parquet")

    # ---------------------------------------------------------------- fig 1
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")

    zero = (offers.commission == 0).mean()
    axs[0].bar(
        ["commission = 0", "commission > 0"],
        [zero * 100, (1 - zero) * 100],
        color=["#b0b7c3", "#1f77b4"],
        edgecolor="black",
    )
    for i, v in enumerate([zero * 100, (1 - zero) * 100]):
        axs[0].text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")
    axs[0].set(
        xlabel="Commission bucket",
        ylabel="Share of offers, %",
        title="A. 94.6% of offers earn zero commission",
    )
    axs[0].set_ylim(0, 105)

    pos = offers.loc[offers.commission > 0, "commission"]
    axs[1].hist(np.log10(pos), bins=45, color="#2ca02c", edgecolor="black", alpha=0.85)
    axs[1].axvline(
        np.log10(pos.median()),
        color="#d62728",
        linestyle="--",
        label=f"median = {pos.median():.0f}",
    )
    axs[1].set(
        xlabel="log10(commission), positive offers only",
        ylabel="Offers",
        title="B. Positive commission is heavy-tailed (skew 7.79)",
    )
    axs[1].legend(frameon=False)

    fig.suptitle(
        "Distribution of `commission` — offer impression grain (n = 120,136)",
        fontsize=14,
        fontweight="bold",
    )
    save(fig, "fig1_commission_distribution.png")

    # ---------------------------------------------------------------- fig 2
    by_pos = offers.groupby("position").commission.agg(["count", "mean", "sum"])
    by_pos["cum"] = by_pos["sum"].cumsum() / by_pos["sum"].sum()

    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
    ax.bar(by_pos.index, by_pos["mean"], color="#1f77b4", edgecolor="black", alpha=0.9)
    ax.set(
        xlabel="SERP position",
        ylabel="Mean commission per offer",
        title="Position 1 earns 47.3% of all commission; value decays faster than position",
    )
    ax2 = ax.twinx()
    ax2.plot(
        by_pos.index,
        by_pos["cum"] * 100,
        color="#d62728",
        marker="o",
        linewidth=2,
        label="cumulative revenue share",
    )
    ax2.axhline(47.3, color="#d62728", linestyle=":", alpha=0.6)
    ax2.set(ylabel="Cumulative commission share, %")
    ax2.set_ylim(0, 105)
    ax2.annotate(
        "pos 1 = 47.3%",
        xy=(1, 47.3),
        xytext=(3.2, 55),
        color="#d62728",
        arrowprops=dict(arrowstyle="->", color="#d62728"),
    )
    ax2.legend(loc="lower right", frameon=False)
    save(fig, "fig2_commission_by_position.png")

    # ---------------------------------------------------------------- fig 3
    by_partner = offers.groupby("partner").commission.sum().sort_values(ascending=False)
    share = by_partner / by_partner.sum() * 100

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    axs[0].barh(
        share.index[::-1],
        share.values[::-1],
        color=["#b0b7c3"] * 3 + ["#1f77b4"] * 5,
        edgecolor="black",
    )
    axs[0].bar_label(axs[0].containers[0], fmt="%.1f%%", padding=2, fontsize=8)
    axs[0].set(
        xlabel="Share of total commission, %",
        title="A. Top-5 partners = 99.3% of commission",
    )
    axs[0].set_xlim(0, 42)

    pt = offers.groupby("partner_type").commission.agg(["mean", "sum"])
    pt_share = pt["sum"] / pt["sum"].sum() * 100
    axs[1].bar(
        pt.index,
        pt_share.values,
        color=["#1f77b4", "#ff7f0e"],
        edgecolor="black",
    )
    for i, v in enumerate(pt_share.values):
        axs[1].text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")
    axs[1].set(
        xlabel="Partner type",
        ylabel="Share of commission, %",
        title="B. CPA = 99.3% of commission; CPC pays per click",
    )
    axs[1].set_ylim(0, 112)
    axs[1].text(
        0.5,
        55,
        "CPC mean = 2.35\nvs CPA mean = 194.3\n(mixed economics)",
        ha="center",
        fontsize=9,
        bbox=dict(boxstyle="round", fc="white", ec="gray"),
    )
    fig.suptitle(
        "Commission concentration — by partner and by partner type",
        fontsize=14,
        fontweight="bold",
    )
    save(fig, "fig3_commission_by_partner.png")

    # ---------------------------------------------------------------- fig 4
    pb = offers.groupby("position").agg(
        examine=("examine_prob", "mean"),
        naive_ctr=("clicked", "mean"),
        relevance=("relevance", "mean"),
    )
    true_ctr = offers.relevance.mean()
    naive_ctr = offers.clicked.mean()

    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
    ax.plot(
        pb.index,
        pb["examine"],
        marker="o",
        color="#1f77b4",
        linewidth=2,
        label="examine prob (1/k)^η",
    )
    ax.plot(
        pb.index,
        pb["naive_ctr"],
        marker="s",
        color="#ff7f0e",
        linewidth=2,
        label="naive CTR by position",
    )
    ax.axhline(
        true_ctr,
        color="#d62728",
        linestyle="--",
        linewidth=2,
        label=f"true click propensity = {true_ctr:.3f} (mean relevance)",
    )
    ax.axhline(
        naive_ctr,
        color="gray",
        linestyle=":",
        label=f"pooled naive CTR = {naive_ctr:.3f}",
    )
    ax.annotate(
        "naive CTR under-states\nclick propensity ~3.9x",
        xy=(8, naive_ctr),
        xytext=(6.5, 0.35),
        color="gray",
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="gray"),
    )
    ax.set(
        xlabel="SERP position",
        ylabel="Probability",
        title="Position bias: raw CTR is biased by construction (0.163 vs 0.629)",
        ylim=(0, 1.05),
    )
    ax.legend(frameon=False, loc="upper right")
    save(fig, "fig4_position_bias.png")

    # ---------------------------------------------------------------- fig 5
    o = orders.set_index("order_date").sort_index()
    wk = o["service_profit"].resample("W").agg(["sum", "count"])
    wk["per_unit"] = wk["sum"] / wk["count"]
    by_month = offers.groupby("booking_month").commission.agg(["sum", "count"])

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    axs[0].plot(wk.index, wk["sum"], marker="o", color="#1f77b4", linewidth=2, label="weekly sum")
    axs[0].plot(
        wk.index,
        wk["per_unit"] * 1000,
        marker="s",
        color="#2ca02c",
        linewidth=1.5,
        alpha=0.8,
        label="per-order profit x1000",
    )
    axs[0].axvline(
        pd.Timestamp("2023-12-24"),
        color="#d62728",
        linestyle="--",
        alpha=0.7,
        label="flagged +56% (flow-driven)",
    )
    axs[0].set(
        xlabel="Week",
        ylabel="service_profit",
        title="A. Measured track — orders, weekly (2023-11 → 2024-01)",
    )
    axs[0].tick_params(axis="x", rotation=45)
    axs[0].legend(frameon=False, fontsize=8)

    axs[1].bar(
        by_month.index,
        by_month["count"],
        color="#1f77b4",
        edgecolor="black",
        alpha=0.9,
    )
    axs[1].set(
        xlabel="booking_month (month-of-year, no year axis)",
        ylabel="Offers",
        title="B. Modeled SERP — flow by month (only 8 months present)",
    )
    axs[1].text(
        0.98,
        0.95,
        "count +84% Jan→Jun\nper-offer flat (±17%)\n→ flow, not behaviour",
        transform=axs[1].transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", fc="white", ec="gray"),
    )
    fig.suptitle("Trend & anomalies — measured vs modeled tracks", fontsize=14, fontweight="bold")
    save(fig, "fig5_trend.png")

    print("wrote 5 figures to", OUT)


if __name__ == "__main__":
    main()
