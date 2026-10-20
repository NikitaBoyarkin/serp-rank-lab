"""serp-rank-lab — EDA deep checks (Phases 2-5 + README truth check).

Read-only. Prints findings to stdout. Run with the shared EDA venv:

    /Users/nikitaboarkin/Desktop/00 ide/00 portfolio/.eda-tools/.venv/bin/python \
        reports/eda/scripts/analyze_serp.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
D = ROOT / "data" / "processed"


def hr(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def load(name: str) -> pd.DataFrame:
    return pd.read_parquet(D / f"{name}.parquet")


def main() -> None:
    offers = load("serp_offers")
    orders = load("orders")
    sessions = load("sessions")
    events = load("events")
    pq = load("partner_quality")
    fbp = load("funnel_by_position")
    rps = load("revenue_per_search")
    sf = load("search_funnel")

    # ---------------------------------------------------------------- Phase 2
    hr("PHASE 2 — VALIDITY, DUPES, RANGES, LOGIC")
    print("offers full-row dupes:", offers.duplicated().sum())
    print(
        "offers key (search_id,position) dupes:", offers.duplicated(["search_id", "position"]).sum()
    )
    print(
        "offers key (search_id,partner) dupes:", offers.duplicated(["search_id", "partner"]).sum()
    )
    print("offers positions range:", offers.position.min(), "-", offers.position.max())
    print(
        "offers offers-per-search: min/max/mean:",
        offers.groupby("search_id").size().agg(["min", "max", "mean"]).to_dict(),
    )
    print("booking_month range:", offers.booking_month.min(), "-", offers.booking_month.max())
    print(
        "commission >= 0:",
        bool((offers.commission >= 0).all()),
        "| negative:",
        int((offers.commission < 0).sum()),
    )
    print(
        "relevance in [0,1]:",
        bool(offers.relevance.between(0, 1).all()),
        "| examine_prob in [0,1]:",
        bool(offers.examine_prob.between(0, 1).all()),
        "| click_prob in [0,1]:",
        bool(offers.click_prob.between(0, 1).all()),
    )
    print("commission>0 & not booked:", int(((offers.commission > 0) & (~offers.booked)).sum()))
    print(
        "  by partner_type:",
        offers.loc[(offers.commission > 0) & (~offers.booked), "partner_type"]
        .value_counts()
        .to_dict(),
    )
    print(
        "  CPC commission>0 share of CPC:",
        round(offers.loc[offers.partner_type == "CPC", "commission"].gt(0).mean(), 4),
        "| CPC commission>0 & clicked:",
        int(((offers.partner_type == "CPC") & (offers.commission > 0) & offers.clicked).sum()),
    )
    print("booked & not clicked:", int((offers.booked & ~offers.clicked).sum()))
    print("redirected & not clicked:", int((offers.redirected & ~offers.clicked).sum()))
    print("booked & not redirected:", int((offers.booked & ~offers.redirected).sum()))
    print(
        "clicked -> clicked implies examine_prob? mean examine_prob clicked:",
        round(offers.loc[offers.clicked, "examine_prob"].mean(), 4),
    )

    # expected_revenue semantics
    cand = offers.click_prob * offers.commission
    print(
        "corr(expected_revenue, click_prob*commission):",
        round(np.corrcoef(offers.expected_revenue, cand)[0, 1], 6),
    )
    print(
        "expected_revenue == click_prob*commission (allclose, rtol=1e-6):",
        bool(np.allclose(offers.expected_revenue, cand, rtol=1e-6, equal_nan=True)),
    )
    print(
        "expected_revenue == click_prob*commission*booked?:",
        round(np.corrcoef(offers.expected_revenue, offers.click_prob * offers.commission)[0, 1], 6),
    )
    # check relation to relevance (ground truth)
    print(
        "corr(relevance, click_prob):",
        round(np.corrcoef(offers.relevance, offers.click_prob)[0, 1], 6),
    )
    print(
        "mean relevance (ground-truth click propensity, no position):",
        round(offers.relevance.mean(), 6),
    )
    print("naive CTR clicked.mean():", round(offers.clicked.mean(), 6))
    print("mean click_prob:", round(offers.click_prob.mean(), 6))

    # orders validity
    hr("PHASE 2 — ORDERS VALIDITY")
    print("orders full-row dupes:", orders.duplicated().sum())
    print("orders order_id dupes:", orders.duplicated("order_id").sum())
    print(
        "order_price min/max:",
        round(orders.order_price.min(), 2),
        round(orders.order_price.max(), 2),
    )
    print("order_price <= 0:", int((orders.order_price <= 0).sum()))
    print("order_date range:", orders.order_date.min(), "-", orders.order_date.max())
    print(
        "orders is_return values:",
        sorted(orders.is_return.unique().tolist()),
        "| share:",
        round(orders.is_return.mean(), 4),
    )
    print("service_profit < 0:", int((orders.service_profit < 0).sum()))
    print("service null share:", round(orders.service.isna().mean(), 4))

    # sessions / events validity + SRM
    hr("PHASE 2 — SESSIONS / EVENTS / SRM")
    print("sessions range:", sessions.start_ts.min(), "-", sessions.end_ts.max())
    print(
        "sessions dupes:",
        sessions.duplicated().sum(),
        "| neg duration:",
        int((sessions.end_ts < sessions.start_ts).sum()),
    )
    print("events range:", events.ts.min(), "-", events.ts.max())
    print("events dupes:", events.duplicated().sum())
    print("sessions ab_group:", sessions.ab_group.value_counts().to_dict())
    print("events ab_group:", events.ab_group.value_counts().to_dict())
    s = sessions.ab_group.value_counts(normalize=True)
    print("sessions split A/B:", round(s.get("A", 0), 4), round(s.get("B", 0), 4))

    # ---------------------------------------------------------------- Phase 3
    hr("PHASE 3 — CORRELATIONS (offers numeric)")
    num = offers.select_dtypes("number")
    corr = num.corr(numeric_only=True)
    pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if abs(r) >= 0.5:
                pairs.append((cols[i], cols[j], round(r, 4)))
    for a, b, r in sorted(pairs, key=lambda t: -abs(t[2])):
        print(f"  {a:18s} ~ {b:18s} r={r:+.4f}")

    # ---------------------------------------------------------------- Phase 4
    hr("PHASE 4 — METRIC BY SEGMENT")
    total = offers.commission.sum()
    print("total commission:", round(total, 2))
    by_pos = offers.groupby("position").commission.agg(["count", "mean", "sum"])
    by_pos["share"] = by_pos["sum"] / total
    print(
        "position-1 revenue share:",
        round(by_pos.loc[1, "share"], 4),
        "| top-3 share:",
        round(by_pos.loc[1:3, "share"].sum(), 4),
    )
    by_pt = offers.groupby("partner_type").commission.agg(["count", "mean", "sum"])
    by_pt["share"] = by_pt["sum"] / total
    print("partner_type:\n", by_pt.round(3).to_string())
    by_dev = offers.groupby("device").commission.agg(["count", "mean", "sum"])
    print("device mean:\n", by_dev.round(2).to_string())
    by_partner = offers.groupby("partner").commission.agg(["count", "mean", "sum"])
    by_partner["share"] = by_partner["sum"] / total
    by_partner = by_partner.sort_values("sum", ascending=False)
    print("partner:\n", by_partner.round(3).to_string())
    top5 = by_partner["sum"].head(5).sum() / total
    print("top-5 partner commission share:", round(top5, 4))

    # composition artefact check: is position gap explained by partner_type mix?
    hr("PHASE 4b — COMPOSITION CHECK (position gap vs partner_type)")
    ct = pd.crosstab(offers.position, offers.partner_type, normalize="index")
    print("CPA share by position (head):\n", ct["CPA"].head(8).round(3).to_string())
    print("mean commission by position within CPA only:")
    cpa = offers[offers.partner_type == "CPA"]
    print(cpa.groupby("position").commission.mean().round(2).head(8).to_string())
    print("mean commission by position within CPC only:")
    cpc = offers[offers.partner_type == "CPC"]
    print(cpc.groupby("position").commission.mean().round(2).head(8).to_string())

    # ---------------------------------------------------------------- Phase 5
    hr("PHASE 5 — TREND (booking_month seasonality, offers)")
    by_month = offers.groupby("booking_month").commission.agg(["count", "mean", "sum"])
    by_month["per_offer"] = by_month["mean"]
    by_month["delta_sum"] = by_month["sum"].pct_change()
    by_month["delta_per_offer"] = by_month["per_offer"].pct_change()
    print(by_month.round(3).to_string())
    print("mean booking_month by booked vs not:")
    print(offers.groupby("booked").booking_month.mean().round(3).to_string())

    hr("PHASE 5 — ORDERS WEEKLY (measured track)")
    o = orders.set_index("order_date").sort_index()
    wk = o["service_profit"].resample("W").agg(["sum", "count"])
    wk["per_unit"] = wk["sum"] / wk["count"]
    wk["d_sum"] = wk["sum"].pct_change()
    print(wk.round(2).to_string())

    # ---------------------------------------------------------------- Recon
    hr("RECONCILIATION — partner_quality vs serp_offers aggregate")
    agg = (
        offers.groupby("partner")
        .agg(
            offers=("commission", "size"),
            clicks=("clicked", "sum"),
            bookings=("booked", "sum"),
            commission=("commission", "sum"),
            avg_position=("position", "mean"),
        )
        .reset_index()
    )
    m = pq.merge(agg, on="partner", suffixes=("_pq", "_raw"))
    m["d_offers"] = m.offers_pq - m.offers_raw
    m["d_clicks"] = m.clicks_pq - m.clicks_raw
    m["d_bookings"] = m.bookings_pq - m.bookings_raw
    m["d_commission"] = m.commission_pq - m.commission_raw
    m["d_avgpos"] = m.avg_position_pq - m.avg_position_raw
    print(
        m[
            [
                "partner",
                "offers_pq",
                "offers_raw",
                "d_offers",
                "clicks_pq",
                "clicks_raw",
                "d_clicks",
                "bookings_pq",
                "bookings_raw",
                "d_bookings",
                "d_commission",
                "d_avgpos",
            ]
        ]
        .round(4)
        .to_string()
    )
    print("max abs d_commission:", round(m.d_commission.abs().max(), 4))
    print(
        "total commission raw:",
        round(agg.commission.sum(), 2),
        "| pq:",
        round(pq.commission.sum(), 2),
    )

    hr("RECONCILIATION — funnel_by_position vs offers")
    fp = (
        offers.groupby("position")
        .agg(
            impressions=("commission", "size"),
            clicks=("clicked", "sum"),
            redirects=("redirected", "sum"),
            bookings=("booked", "sum"),
            commission=("commission", "sum"),
        )
        .reset_index()
    )
    mm = fbp.merge(fp, on="position", suffixes=("_fbp", "_raw"))
    mm["d_bookings"] = mm.bookings_fbp - mm.bookings_raw
    mm["d_commission"] = mm.booked_commission - mm.commission
    print(
        mm[["position", "bookings_fbp", "bookings_raw", "d_bookings", "d_commission"]]
        .round(3)
        .to_string()
    )
    print(
        "max abs d_bookings:",
        int(mm.d_bookings.abs().max()),
        "| max abs d_commission:",
        round(mm.d_commission.abs().max(), 3),
    )

    hr("RECONCILIATION — revenue_per_search vs offers (by search_id)")
    raw_search = (
        offers.groupby("search_id")
        .agg(
            offers=("commission", "size"),
            clicks=("clicked", "sum"),
            bookings=("booked", "sum"),
            revenue=("commission", "sum"),
        )
        .reset_index()
    )
    rr = rps.merge(raw_search, on="search_id", suffixes=("_rps", "_raw"))
    print("n searches rps:", len(rps), "| raw:", offers.search_id.nunique(), "| merged:", len(rr))
    print("max abs d_revenue:", round((rr.revenue_rps - rr.revenue_raw).abs().max(), 4))
    print("max abs d_clicks:", int((rr.clicks_rps - rr.clicks_raw).abs().max()))
    print("max abs d_bookings:", int((rr.bookings_rps - rr.bookings_raw).abs().max()))
    print("rps logged revenue_per_search:", round(rps.revenue.mean(), 4))

    hr("RECONCILIATION — search_funnel vs offers")
    print("search_funnel rows:", len(sf), "| offers searches:", offers.search_id.nunique())
    print(
        "sf converted share:",
        round(sf.converted.mean(), 4),
        "| booked share (offers):",
        round(offers.groupby("search_id").booked.any().mean(), 4),
    )
    print(
        "sf commission sum:",
        round(sf.commission.sum(), 2),
        "| offers commission sum:",
        round(offers.commission.sum(), 2),
    )

    # ---------------------------------------------------------------- README
    hr("README TRUTH CHECK")
    rj = json.loads((ROOT / "reports" / "ranking_eval.json").read_text())
    pol = {p["policy"]: p for p in rj["policies"]}
    rel = pol["relevance"]["revenue_per_search"]
    rev = pol["revenue"]["revenue_per_search"]
    print(f"relevance rev/search = {rel:.2f} ; revenue rev/search = {rev:.2f}")
    print(f"  ratio-1 = {rev / rel - 1:.4%}  (README says +38%)")
    ct_ = pol["click_trained"]["revenue_per_search"]
    pl = pol["propensity_logistic"]["revenue_per_search"]
    print(
        f"propensity {pl:.2f} vs click_trained {ct_:.2f} -> {pl / ct_ - 1:.4%} (README says +19.3%)"
    )
    print(
        f"NDCG relevance {pol['relevance']['ndcg_at_k']:.4f} vs "
        f"revenue {pol['revenue']['ndcg_at_k']:.4f} -> "
        f"dNDCG {rj['divergence']['ndcg_delta']:.4f}, "
        f"dRev {rj['divergence']['revenue_delta']:.2f}"
    )

    # recompute oracles from raw offers (top-k=10 by relevance / by commission)
    hr("README — ORACLE RE-RANK FROM RAW (k=10)")
    k = 10
    o_sorted = offers.sort_values(["search_id", "relevance"], ascending=[True, False])
    top_rel = o_sorted.groupby("search_id").head(k)
    rev_rel = top_rel.groupby("search_id").commission.sum().mean()
    o_sorted2 = offers.sort_values(["search_id", "commission"], ascending=[True, False])
    top_rev = o_sorted2.groupby("search_id").head(k)
    rev_rev = top_rev.groupby("search_id").commission.sum().mean()
    print(f"recompute relevance-oracle rev/search = {rev_rel:.2f} (json {rel:.2f})")
    print(f"recompute revenue-oracle   rev/search = {rev_rev:.2f} (json {rev:.2f})")

    print(
        "position-1 share recompute:",
        round(offers.loc[offers.position == 1, "commission"].sum() / total, 4),
    )
    print("top-5 partner share recompute:", round(top5, 4))


if __name__ == "__main__":
    main()
