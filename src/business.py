"""Business metrics for the real-data islands and the real clickstream A/B.

Two kinds of data live in this project and must never be conflated:

* ``orders`` and ``bookings_legs`` are **islands** -- they carry no foreign key
  to ``user_id`` or to each other, so they are described on their own terms,
  never joined.
* ``events`` and ``sessions`` are the **real clickstream** and share the
  ``user_id`` cluster with ``serp_offers``. They carry no event taxonomy, so
  the A/B here is *activity-level*: the primary metric is volume, not a funnel.

``orders``/``bookings_legs``/``events``/``sessions`` are measured; every SERP
ranking or monetisation fact (``serp_offers``) is modeled and lives elsewhere.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
SQL_DIR = PROJECT_ROOT / "sql"

BUSINESS_JSON = REPORTS_DIR / "business.json"

AB_LABEL = "activity-level A/B — no event taxonomy in events, so the primary metric is volume"


# --------------------------------------------------------------------------- #
# Orders island (measured)
# --------------------------------------------------------------------------- #


def orders_metrics(orders: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Service/platform mix, geo pairs, booking depth and price stats."""
    n = len(orders)

    service = orders["service"].fillna("(none)")
    service_mix = (
        service.value_counts(dropna=False).rename_axis("service").reset_index(name="count")
    )
    service_mix["share"] = service_mix["count"] / n
    # A share of ALL orders, not a per-service rate: repeating one number on every
    # row made it read as "refund rate of this service type".
    service_mix["refund_share_of_all_orders"] = float((orders["service"] == "refund").mean())

    platform_mix = (
        orders["order_platform"]
        .fillna("(none)")
        .value_counts(dropna=False)
        .rename_axis("order_platform")
        .reset_index(name="count")
    )
    platform_mix["share"] = platform_mix["count"] / n

    geo = (
        orders.groupby(["origin_country", "destination_country"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    geo["share"] = geo["count"] / n

    depth = orders["booking_depth"].dropna().to_numpy()
    counts, edges = np.histogram(depth, bins=10)
    booking_depth = pd.DataFrame({"bin_left": edges[:-1], "bin_right": edges[1:], "count": counts})

    metrics = ["order_price", "service_price", "service_profit"]
    desc = orders[metrics].describe()
    row = {f"{col}_{stat}": float(desc.loc[stat, col]) for col in metrics for stat in desc.index}
    # service_profit is zero on unlabelled rows, so the all-rows median is ~0 and
    # says nothing. Report it on the rows that actually carry a service instead.
    paid = orders.loc[orders["service"].notna(), "service_profit"]
    profit_median_paid = float(paid.median()) if len(paid) else float("nan")
    profit_median_all = float(orders["service_profit"].median())
    labelled = float(orders["service"].notna().mean())
    row["service_profit_median_labelled"] = profit_median_paid
    row["service_profit_labelled_share"] = labelled
    row["note"] = (
        f"service_profit median={profit_median_paid:.1f} on the {labelled:.0%} of "
        f"orders that carry a service ({profit_median_all:.1f} over all rows, "
        f"because the rest are zero)"
    )
    price_stats = pd.DataFrame([row])

    return {
        "service_mix": service_mix,
        "platform_mix": platform_mix,
        "geo": geo,
        "booking_depth": booking_depth,
        "price_stats": price_stats,
    }


# --------------------------------------------------------------------------- #
# Bookings-legs island (measured)
# --------------------------------------------------------------------------- #


def routes_metrics(bookings_legs: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Airlines, routes, month-of-year seasonality and passenger counts."""
    legs = bookings_legs
    n = len(legs)

    airlines = (
        legs["airline"]
        .fillna("(none)")
        .value_counts(dropna=False)
        .rename_axis("airline")
        .reset_index(name="count")
    )
    airlines["share"] = airlines["count"] / n

    routes = (
        legs.assign(
            origin=legs["origin"].fillna("(none)"),
            destination=legs["destination"].fillna("(none)"),
        )
        .groupby(["origin", "destination"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    routes["share"] = routes["count"] / n
    # LED dominates the destination mix (~94% of legs) -- surfaced here so a
    # notebook can quote it without recomputing.
    routes["connection_share"] = float(legs["is_connection"].mean())
    routes["baggage_share"] = float(legs["has_baggage"].mean())

    monthly = (
        legs["booking_month"]
        .value_counts()
        .sort_index()
        .rename_axis("booking_month")
        .reset_index(name="legs")
    )
    monthly["share"] = monthly["legs"] / n

    passengers = (
        legs["passengers"]
        .value_counts()
        .sort_index()
        .rename_axis("passengers")
        .reset_index(name="count")
    )
    passengers["share"] = passengers["count"] / n

    return {
        "airlines": airlines,
        "routes": routes,
        "monthly": monthly,
        "passengers": passengers,
    }


# --------------------------------------------------------------------------- #
# Clickstream A/B (measured, activity-level)
# --------------------------------------------------------------------------- #


def _per_user_frame(df: pd.DataFrame, metric: str) -> list[dict[str, object]]:
    """Mean/median per-user volume for one clickstream table, by arm."""
    per_user = df.groupby(["ab_group", "user_id"]).size()
    stats = per_user.groupby("ab_group").agg(["mean", "median"])
    return [
        {
            "metric": metric,
            "ab_group": group,
            "mean": float(stats.loc[group, "mean"]),
            "median": float(stats.loc[group, "median"]),
        }
        for group in stats.index
    ]


def clickstream_ab(events: pd.DataFrame, sessions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Arm balance (SRM), per-user volume and DAU -- activity-level, volume-primary."""
    ev_users = events.groupby("ab_group")["user_id"].nunique()
    se_users = sessions.groupby("ab_group")["user_id"].nunique()
    groups = sorted(set(ev_users.index) | set(se_users.index))
    table = np.array([[ev_users.get(g, 0) for g in groups], [se_users.get(g, 0) for g in groups]])
    chi2, p_value, _dof, _expected = chi2_contingency(table)

    rows = [
        {"source": source, "ab_group": group, "users": int(counts.get(group, 0))}
        for source, counts in (("events", ev_users), ("sessions", se_users))
        for group in groups
    ]
    balance = pd.DataFrame(rows)
    balance["srm_chi2"] = float(chi2)
    balance["srm_p_value"] = float(p_value)

    per_user = pd.DataFrame(
        _per_user_frame(events, "events_per_user") + _per_user_frame(sessions, "sessions_per_user")
    )

    dau = events.groupby(["pdate", "ab_group"])["user_id"].nunique().reset_index(name="dau")

    return {"balance": balance, "per_user": per_user, "dau": dau}


# --------------------------------------------------------------------------- #
# Payload assembly
# --------------------------------------------------------------------------- #


def _modeled_summary(
    offers: pd.DataFrame | None,
    funnel_by_position: pd.DataFrame | None,
    revenue_by_position: pd.DataFrame | None,
    ranking_report: dict | None,
) -> dict[str, object]:
    """Small native summary of the modeled SERP block (no frames pass through)."""
    summary: dict[str, object] = {"label": "modeled — serp_offers is synthetic, not measured"}
    if offers is not None:
        summary["offers"] = int(len(offers))
        summary["searches"] = int(offers["search_id"].nunique())
        summary["partners"] = int(offers["partner"].nunique())
    if funnel_by_position is not None:
        summary["funnel_positions"] = int(len(funnel_by_position))
    if revenue_by_position is not None:
        summary["top_position_revenue_share"] = float(
            revenue_by_position["cumulative_revenue_share"].iloc[0]
        )
    if ranking_report is not None and "divergence" in ranking_report:
        summary["divergence"] = ranking_report["divergence"]
    return summary


def build(
    orders: pd.DataFrame,
    bookings_legs: pd.DataFrame,
    events: pd.DataFrame,
    sessions: pd.DataFrame,
    offers: pd.DataFrame | None = None,
    funnel_by_position: pd.DataFrame | None = None,
    revenue_by_position: pd.DataFrame | None = None,
    ranking_report: dict | None = None,
) -> dict[str, object]:
    """Assemble the full payload from the metric groups. Pure: no file IO."""
    return {
        "orders": orders_metrics(orders),
        "routes": routes_metrics(bookings_legs),
        "clickstream_ab": clickstream_ab(events, sessions),
        "modeled": _modeled_summary(
            offers, funnel_by_position, revenue_by_position, ranking_report
        ),
    }


def _to_native(obj):
    """Recursively convert numpy scalars/arrays/frames/dates to JSON-native types."""
    if isinstance(obj, dict):
        return {str(k): _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return _to_native(obj.to_dict(orient="records"))
    if isinstance(obj, np.generic):
        return _to_native(obj.item())
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(prog="python -m src.business").parse_args(argv)

    print("=== business ===")

    def _read(name: str) -> pd.DataFrame | None:
        path = PROCESSED_DIR / f"{name}.parquet"
        return pd.read_parquet(path) if path.exists() else None

    orders = _read("orders")
    bookings_legs = _read("bookings_legs")
    events = _read("events")
    sessions = _read("sessions")
    offers = _read("serp_offers")
    funnel_by_position = _read("funnel_by_position")
    revenue_by_position = _read("revenue_by_position")

    report_path = REPORTS_DIR / "ranking_eval.json"
    ranking_report = (
        json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else None
    )

    payload = build(
        orders,
        bookings_legs,
        events,
        sessions,
        offers,
        funnel_by_position,
        revenue_by_position,
        ranking_report,
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    BUSINESS_JSON.write_text(json.dumps(_to_native(payload), indent=2), encoding="utf-8")

    from src.plots import save_all

    figures = save_all(
        orders=orders,
        bookings_legs=bookings_legs,
        events=events,
        sessions=sessions,
        serp_offers=offers,
        funnel_by_position=funnel_by_position,
        partner_quality=_read("partner_quality"),
        revenue_by_position=revenue_by_position,
        ranking_report=ranking_report,
    )
    for path in figures:
        print(f"figure  {path.relative_to(PROJECT_ROOT)}")

    for group, section in payload.items():
        if isinstance(section, dict):
            for name, frame in section.items():
                rows = len(frame) if hasattr(frame, "__len__") and not isinstance(frame, str) else 1
                print(f"{group:14s} {name:22s} rows: {rows:>7}")
    print(f"wrote {BUSINESS_JSON.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
