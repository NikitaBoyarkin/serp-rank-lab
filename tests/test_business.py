"""Unit tests for src/business.py over tiny inline frames (no parquet reads).

Each public metric function is pinned: counts/shares, the ``(none)`` bucket,
refund rate, LED dominance, and SRM p-value on a balanced vs skewed split.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from src.business import (
    AB_LABEL,
    build,
    clickstream_ab,
    orders_metrics,
    routes_metrics,
)

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _orders() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "service": ["refund", "changes", None, "refund", "support"],
            "order_price": [100.0, 200.0, 300.0, 400.0, 500.0],
            "service_price": [10.0, 20.0, 30.0, 40.0, 50.0],
            "service_profit": [0.0, 0.0, 0.0, 1.0, 0.0],
            "order_platform": ["ios", "android", "ios", "other", "ios"],
            "origin_country": ["RU", "RU", "KZ", "RU", "KZ"],
            "destination_country": ["TR", "TR", "AE", "AE", "TR"],
            "booking_depth": [1.0, 2.0, 3.0, 2.0, 1.0],
        }
    )


def _legs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "origin": ["SVO", "SVO", "VKO", "DME", "SVO"],
            "destination": ["LED", "LED", "LED", "LED", "AER"],
            "airline": ["DP", "DP", "SU", "FV", "SU"],
            "booking_month": [1, 2, 2, 3, 3],
            "passengers": [1, 2, 2, 3, 1],
            "is_connection": [False, True, False, False, True],
            "has_baggage": [False, False, True, False, True],
        }
    )


def _events() -> pd.DataFrame:
    day = dt.date(2023, 5, 1)
    rows = [{"user_id": f"a{i}", "ab_group": "A", "pdate": day} for i in range(5)]
    rows += [{"user_id": f"a{i}", "ab_group": "A", "pdate": day} for i in range(5)]
    rows += [{"user_id": f"b{i}", "ab_group": "B", "pdate": day} for i in range(5)]
    return pd.DataFrame(rows)


def _sessions() -> pd.DataFrame:
    day = dt.date(2023, 5, 1)
    rows = [{"user_id": f"a{i}", "ab_group": "A", "pdate": day} for i in range(5)]
    rows += [{"user_id": f"b{i}", "ab_group": "B", "pdate": day} for i in range(5)]
    return pd.DataFrame(rows)


def _sessions_skewed() -> pd.DataFrame:
    day = dt.date(2023, 5, 1)
    rows = [{"user_id": f"a{i}", "ab_group": "A", "pdate": day} for i in range(19)]
    rows += [{"user_id": "b0", "ab_group": "B", "pdate": day}]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# orders_metrics
# --------------------------------------------------------------------------- #


def test_orders_service_mix_counts_none_bucket():
    mix = orders_metrics(_orders())["service_mix"].set_index("service")
    assert mix.loc["refund", "count"] == 2
    assert mix.loc["changes", "count"] == 1
    assert mix.loc["(none)", "count"] == 1
    assert mix.loc["support", "count"] == 1


def test_orders_service_mix_shares_sum_to_one():
    mix = orders_metrics(_orders())["service_mix"]
    assert mix["share"].sum() == pytest.approx(1.0)
    assert mix.set_index("service").loc["refund", "share"] == pytest.approx(0.4)


def test_orders_refund_rate_is_share_of_refund_orders():
    mix = orders_metrics(_orders())["service_mix"]
    # A share of all orders, named so it cannot read as a per-service rate.
    assert mix["refund_share_of_all_orders"].iloc[0] == pytest.approx(0.4)


def test_orders_platform_mix_counts_ios():
    plat = orders_metrics(_orders())["platform_mix"].set_index("order_platform")
    assert plat.loc["ios", "count"] == 3
    assert plat.loc["ios", "share"] == pytest.approx(0.6)


def test_orders_geo_top_pair_is_ru_tr():
    geo = orders_metrics(_orders())["geo"]
    top = geo.iloc[0]
    assert top["origin_country"] == "RU"
    assert top["destination_country"] == "TR"
    assert top["count"] == 2
    assert top["share"] == pytest.approx(0.4)


def test_orders_booking_depth_histogram_counts_match_rows():
    hist = orders_metrics(_orders())["booking_depth"]
    assert hist["count"].sum() == 5
    assert len(hist) == 10


def test_orders_price_stats_reports_profit_and_label_note():
    stats = orders_metrics(_orders())["price_stats"]
    assert len(stats) == 1
    assert stats["order_price_mean"].iloc[0] == pytest.approx(300.0)
    note = stats["note"].iloc[0]
    assert "service_profit median" in note
    assert "80%" in note


def test_orders_price_note_quotes_the_real_profit_median():
    """A hardcoded "(~0)" must fail here: the fixture's paid rows are not zero."""
    orders = _orders()
    # Labelled rows (0, 1, 3, 4) carry 100 / 250 / 380 / 400 -> median 315.
    orders["service_profit"] = [100.0, 250.0, 0.0, 380.0, 400.0]
    stats = orders_metrics(orders)["price_stats"]
    note = stats["note"].iloc[0]
    assert "315.0" in note
    assert "(~0)" not in note
    assert stats["service_profit_median_labelled"].iloc[0] == pytest.approx(315.0)


# --------------------------------------------------------------------------- #
# routes_metrics
# --------------------------------------------------------------------------- #


def test_routes_airlines_ranks_dp_and_su():
    airlines = routes_metrics(_legs())["airlines"].set_index("airline")
    assert airlines.loc["DP", "count"] == 2
    assert airlines.loc["SU", "count"] == 2
    assert airlines.loc["DP", "share"] == pytest.approx(0.4)


def test_routes_led_destination_dominance():
    routes = routes_metrics(_legs())["routes"]
    led_share = routes.loc[routes["destination"] == "LED", "share"].sum()
    assert led_share == pytest.approx(0.8)


def test_routes_exposes_connection_and_baggage_share():
    routes = routes_metrics(_legs())["routes"]
    assert routes["connection_share"].iloc[0] == pytest.approx(0.4)
    assert routes["baggage_share"].iloc[0] == pytest.approx(0.4)


def test_routes_monthly_counts_by_month():
    monthly = routes_metrics(_legs())["monthly"].set_index("booking_month")
    assert monthly.loc[2, "legs"] == 2
    assert monthly.loc[1, "legs"] == 1
    assert monthly.loc[3, "legs"] == 2


def test_routes_passengers_value_counts():
    passengers = routes_metrics(_legs())["passengers"].set_index("passengers")
    assert passengers.loc[1, "count"] == 2
    assert passengers.loc[3, "count"] == 1


# --------------------------------------------------------------------------- #
# clickstream_ab
# --------------------------------------------------------------------------- #


def test_clickstream_balance_balanced_split_has_high_p_value():
    balance = clickstream_ab(_events(), _sessions())["balance"]
    assert balance["srm_p_value"].iloc[0] == pytest.approx(1.0)
    events_a = balance[(balance["source"] == "events") & (balance["ab_group"] == "A")]
    assert events_a["users"].iloc[0] == 5


def test_clickstream_balance_skewed_split_has_low_p_value():
    balance = clickstream_ab(_events(), _sessions_skewed())["balance"]
    assert balance["srm_p_value"].iloc[0] < 0.05


def test_clickstream_per_user_mean_and_median_by_arm():
    per_user = clickstream_ab(_events(), _sessions())["per_user"]
    events_a = per_user[(per_user["metric"] == "events_per_user") & (per_user["ab_group"] == "A")]
    events_b = per_user[(per_user["metric"] == "events_per_user") & (per_user["ab_group"] == "B")]
    assert events_a["mean"].iloc[0] == pytest.approx(2.0)
    assert events_a["median"].iloc[0] == pytest.approx(2.0)
    assert events_b["mean"].iloc[0] == pytest.approx(1.0)


def test_clickstream_dau_counts_unique_users_per_day():
    dau = clickstream_ab(_events(), _sessions())["dau"]
    day_rows = dau[dau["pdate"] == dt.date(2023, 5, 1)].set_index("ab_group")
    assert day_rows.loc["A", "dau"] == 5
    assert day_rows.loc["B", "dau"] == 5


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #


def test_ab_label_marks_activity_level_volume_metric():
    assert "activity-level" in AB_LABEL


def test_build_returns_all_metric_groups_and_is_pure():
    offers = pd.DataFrame(
        {
            "search_id": ["s1", "s1", "s2"],
            "partner": ["p1", "p2", "p1"],
            "position": [1, 2, 1],
            "device": ["mobile", "mobile", "desktop"],
            "examine_prob": [1.0, 0.5, 1.0],
        }
    )
    funnel = pd.DataFrame({"position": [1, 2], "bookings": [3, 1]})
    revenue = pd.DataFrame({"position": [1, 2], "cumulative_revenue_share": [0.6, 1.0]})
    ranking = {"divergence": {"ndcg_rises_while_revenue_falls": True}}

    payload = build(
        _orders(),
        _legs(),
        _events(),
        _sessions(),
        offers,
        funnel,
        revenue,
        ranking,
    )

    assert set(payload) == {"orders", "routes", "clickstream_ab", "modeled"}
    assert payload["modeled"]["searches"] == 2
    assert payload["modeled"]["top_position_revenue_share"] == pytest.approx(0.6)
    assert payload["modeled"]["divergence"]["ndcg_rises_while_revenue_falls"] is True
