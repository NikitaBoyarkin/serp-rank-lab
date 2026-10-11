"""Matplotlib figure builders for the SERP Rank Lab report.

Block I (modeled) reads ``serp_offers`` and ``reports/ranking_eval.json``;
Block II (measured) reads the real islands (``orders``, ``bookings_legs``) and
the real clickstream (``events``, ``sessions``). Each builder returns a
``Figure`` and never closes it -- saving is ``save_figure``'s job.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
SQL_DIR = PROJECT_ROOT / "sql"
FIGURES_DIR = REPORTS_DIR / "figures"


def save_figure(fig, name: str) -> Path:
    """Save a figure to reports/figures/<name>.png and return the path."""
    path = FIGURES_DIR / f"{name}.png"
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Block I — modeled (serp_offers + ranking_eval.json)
# --------------------------------------------------------------------------- #


def plot_position_bias(offers: pd.DataFrame):
    """Mean examine probability by position, split by device."""
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for device in ("mobile", "desktop"):
        sub = offers[offers["device"] == device]
        grouped = sub.groupby("position")["examine_prob"].mean()
        ax.plot(list(grouped.index), grouped.to_numpy(), marker="o", label=device)
    ax.set_xlabel("position")
    ax.set_ylabel("mean examine_prob")
    ax.set_title("Position bias (modeled): examine probability by position")
    ax.legend()
    return fig


def plot_funnel_by_position(funnel_df: pd.DataFrame):
    """Impression -> click -> redirect -> book chain by position."""
    fig, ax = plt.subplots(figsize=(9, 3.8))
    positions = funnel_df["position"].to_numpy()
    stages = [
        ("impressions", "impressions"),
        ("clicks", "clicks"),
        ("redirects", "redirects"),
        ("bookings", "book"),
    ]
    width = 0.2
    for i, (col, label) in enumerate(stages):
        ax.bar(positions + (i - 1.5) * width, funnel_df[col].to_numpy(), width=width, label=label)
    ax.set_xlabel("position")
    ax.set_ylabel("offers")
    ax.set_title("Funnel by position (modeled)")
    ax.legend()
    return fig


def plot_partner_quadrant(partner_quality_df: pd.DataFrame):
    """CTR vs commission per offer, bubble-sized by searches present."""
    fig, ax = plt.subplots(figsize=(7, 4.4))
    sizes = partner_quality_df["searches_present"].astype(float)
    scale = sizes / sizes.max() if sizes.max() else sizes
    ax.scatter(
        partner_quality_df["ctr"].to_numpy(),
        partner_quality_df["commission_per_offer"].to_numpy(),
        s=30 + 250 * scale.to_numpy(),
        alpha=0.6,
    )
    for _, row in partner_quality_df.iterrows():
        ax.annotate(str(row["partner"]), (row["ctr"], row["commission_per_offer"]))
    ax.set_xlabel("ctr")
    ax.set_ylabel("commission per offer")
    ax.set_title("Partner quadrant (modeled)")
    return fig


def plot_ranking_divergence(ranking_report: dict):
    """NDCG@k and revenue per search side by side, by ranking policy."""
    policies = ranking_report["policies"]
    names = [p["policy"] for p in policies]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].bar(names, [p["ndcg_at_k"] for p in policies])
    axes[0].set_title("NDCG@k by policy")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(names, [p["revenue_per_search"] for p in policies])
    axes[1].set_title("Revenue per search by policy")
    axes[1].tick_params(axis="x", rotation=20)
    return fig


def plot_cumulative_share(revenue_by_position_df: pd.DataFrame):
    """Cumulative revenue share against position."""
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(
        list(revenue_by_position_df["position"]),
        revenue_by_position_df["cumulative_revenue_share"].to_numpy(),
        marker="o",
    )
    ax.set_xlabel("position")
    ax.set_ylabel("cumulative revenue share")
    ax.set_ylim(0, 1.05)
    ax.set_title("Cumulative revenue share by position (modeled)")
    return fig


# --------------------------------------------------------------------------- #
# Block II — measured (orders, bookings_legs, events, sessions)
# --------------------------------------------------------------------------- #


def plot_orders_mix(orders: pd.DataFrame):
    """Order counts by service, with the missing-label bucket shown."""
    fig, ax = plt.subplots(figsize=(7, 3.6))
    counts = orders["service"].fillna("(none)").value_counts()
    ax.bar(list(counts.index), counts.to_numpy())
    ax.set_xlabel("service")
    ax.set_ylabel("orders")
    ax.set_title("Orders by service (measured)")
    return fig


def plot_refund_rate(orders: pd.DataFrame):
    """Share of orders flagged as refunds."""
    fig, ax = plt.subplots(figsize=(4, 3.2))
    refund = float((orders["service"] == "refund").mean())
    ax.bar(["refund", "other"], [refund, 1 - refund])
    ax.set_ylabel("share of orders")
    ax.set_ylim(0, 1)
    ax.set_title("Refund rate (measured)")
    return fig


def plot_geo(orders: pd.DataFrame):
    """Top origin -> destination country pairs by order count."""
    fig, ax = plt.subplots(figsize=(7, 3.8))
    counts = orders.groupby(["origin_country", "destination_country"]).size()
    counts = counts.sort_values(ascending=False).head(10)
    labels = [f"{o}->{d}" for o, d in counts.index]
    ax.barh(labels[::-1], counts.to_numpy()[::-1])
    ax.set_xlabel("orders")
    ax.set_title("Top country pairs (measured)")
    return fig


def plot_routes(bookings_legs: pd.DataFrame):
    """Top origin -> destination routes by leg count."""
    fig, ax = plt.subplots(figsize=(7, 3.8))
    counts = bookings_legs.groupby(["origin", "destination"]).size()
    counts = counts.sort_values(ascending=False).head(12)
    labels = [f"{o}->{d}" for o, d in counts.index]
    ax.barh(labels[::-1], counts.to_numpy()[::-1])
    ax.set_xlabel("legs")
    ax.set_title("Top routes (measured)")
    return fig


def plot_seasonality(bookings_legs: pd.DataFrame):
    """Leg count by booking month (month-of-year 1-8, not year-month)."""
    fig, ax = plt.subplots(figsize=(7, 3.4))
    counts = bookings_legs["booking_month"].value_counts().sort_index()
    ax.bar(counts.index.astype(str), counts.to_numpy())
    ax.set_xlabel("booking month (1-8)")
    ax.set_ylabel("legs")
    ax.set_title("Seasonality (measured): legs by booking month")
    return fig


def plot_ab_activity(events: pd.DataFrame, sessions: pd.DataFrame):
    """Daily active users by arm, from events and sessions."""
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for group in sorted(events["ab_group"].unique()):
        grouped = events[events["ab_group"] == group].groupby("pdate")["user_id"].nunique()
        ax.plot(list(grouped.index), grouped.to_numpy(), marker="o", label=f"events {group}")
    for group in sorted(sessions["ab_group"].unique()):
        grouped = sessions[sessions["ab_group"] == group].groupby("pdate")["user_id"].nunique()
        ax.plot(
            list(grouped.index),
            grouped.to_numpy(),
            marker="x",
            linestyle="--",
            label=f"sessions {group}",
        )
    ax.set_ylabel("daily active users")
    ax.set_title("Activity-level A/B (measured): DAU by arm")
    ax.legend()
    return fig


# --------------------------------------------------------------------------- #
# Batch save
# --------------------------------------------------------------------------- #


def save_all(**frames) -> list[Path]:
    """Build and save every figure whose inputs are present; skip the rest."""
    orders = frames.get("orders")
    legs = frames.get("bookings_legs")
    events = frames.get("events")
    sessions = frames.get("sessions")
    offers = frames.get("serp_offers")
    funnel = frames.get("funnel_by_position")
    partner = frames.get("partner_quality")
    revenue = frames.get("revenue_by_position")
    ranking = frames.get("ranking_report")

    jobs: list[tuple[object, str]] = []
    if offers is not None:
        jobs.append((plot_position_bias(offers), "position_bias"))
    if funnel is not None:
        jobs.append((plot_funnel_by_position(funnel), "funnel_by_position"))
    if partner is not None:
        jobs.append((plot_partner_quadrant(partner), "partner_quadrant"))
    if ranking is not None:
        jobs.append((plot_ranking_divergence(ranking), "ranking_divergence"))
    if revenue is not None:
        jobs.append((plot_cumulative_share(revenue), "cumulative_revenue_share"))
    if orders is not None:
        jobs.append((plot_orders_mix(orders), "orders_mix"))
        jobs.append((plot_refund_rate(orders), "refund_rate"))
        jobs.append((plot_geo(orders), "geo"))
    if legs is not None:
        jobs.append((plot_routes(legs), "routes"))
        jobs.append((plot_seasonality(legs), "seasonality"))
    if events is not None and sessions is not None:
        jobs.append((plot_ab_activity(events, sessions), "ab_activity"))

    return [save_figure(fig, name) for fig, name in jobs]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(prog="python -m src.plots").parse_args(argv)

    print("=== plots ===")

    def _read(name: str) -> pd.DataFrame | None:
        path = PROCESSED_DIR / f"{name}.parquet"
        return pd.read_parquet(path) if path.exists() else None

    frames: dict[str, object] = {
        "orders": _read("orders"),
        "bookings_legs": _read("bookings_legs"),
        "events": _read("events"),
        "sessions": _read("sessions"),
        "serp_offers": _read("serp_offers"),
    }

    report_path = REPORTS_DIR / "ranking_eval.json"
    if report_path.exists():
        frames["ranking_report"] = json.loads(report_path.read_text(encoding="utf-8"))

    offers = frames.get("serp_offers")
    if offers is not None:
        from src.sql_stage import build as build_views  # local: heavy import

        views = build_views(offers)
        frames["funnel_by_position"] = views["funnel_by_position"]
        frames["partner_quality"] = views["partner_quality"]
        frames["revenue_by_position"] = views["revenue_by_position"]

    paths = save_all(**frames)
    for path in paths:
        print(f"wrote {path.relative_to(PROJECT_ROOT)}")
    print(f"{len(paths)} figures -> {FIGURES_DIR.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
