"""Contract tests for the leg explosion in ``src.data_ingest``.

These tests build tiny ``tmp_path`` fixtures -- they never touch the real
19 MB bookings feed.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.data_ingest import load_bookings, load_events, load_orders

# One direct booking (1 leg) + one connecting booking (2 legs, bag on leg 2).
BOOKINGS_ROWS = [
    {
        "booking_id": "DIRECT01",
        "booking_month": 5,
        "passengers": 1,
        "price": 2460,
        "flights_info": json.dumps(
            [
                {
                    "origin": "VKO",
                    "destination": "LED",
                    "airline": "FV",
                    "baggage": "No baggage",
                }
            ]
        ),
    },
    {
        "booking_id": "CONNECT2",
        "booking_month": 8,
        "passengers": 2,
        "price": 5975,
        "flights_info": json.dumps(
            [
                {
                    "origin": "SVO",
                    "destination": "MCT",
                    "airline": "WY",
                    "baggage": "No baggage",
                },
                {
                    "origin": "MCT",
                    "destination": "DPS",
                    "airline": "WY",
                    "baggage": "With baggage",
                },
            ]
        ),
    },
]

EXPECTED_LEG_COLUMNS = [
    "booking_id",
    "booking_month",
    "passengers",
    "price",
    "leg_index",
    "origin",
    "destination",
    "airline",
    "baggage",
    "is_connection",
    "has_baggage",
]


@pytest.fixture
def bookings_csv(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame(BOOKINGS_ROWS).to_csv(path, index=False)
    return path


def test_two_leg_booking_becomes_two_rows(bookings_csv):
    legs = load_bookings(bookings_csv)

    # 2 bookings, 3 legs total.
    assert len(legs) == 3
    assert sorted(legs["booking_id"].unique()) == ["CONNECT2", "DIRECT01"]

    connecting = legs[legs["booking_id"] == "CONNECT2"]
    assert len(connecting) == 2
    assert connecting["leg_index"].tolist() == [0, 1]


def test_booking_level_fields_are_denormalised_onto_every_leg(bookings_csv):
    legs = load_bookings(bookings_csv)
    connecting = legs[legs["booking_id"] == "CONNECT2"]

    # Booking-level fields repeat unchanged on both legs.
    assert connecting["booking_month"].tolist() == [8, 8]
    assert connecting["passengers"].tolist() == [2, 2]
    assert connecting["price"].tolist() == [5975, 5975]

    # Leg-level fields differ per leg.
    assert connecting["origin"].tolist() == ["SVO", "MCT"]
    assert connecting["destination"].tolist() == ["MCT", "DPS"]


def test_is_connection_flags_only_legs_after_the_first(bookings_csv):
    legs = load_bookings(bookings_csv)

    direct = legs[legs["booking_id"] == "DIRECT01"]
    assert direct["is_connection"].tolist() == [False]

    connecting = legs[legs["booking_id"] == "CONNECT2"]
    assert connecting["is_connection"].tolist() == [False, True]


def test_has_baggage_matches_the_baggage_token(bookings_csv):
    legs = load_bookings(bookings_csv)
    connecting = legs[legs["booking_id"] == "CONNECT2"]

    assert connecting["has_baggage"].tolist() == [False, True]
    assert connecting["baggage"].tolist() == ["No baggage", "With baggage"]


def test_leg_frame_has_the_contracted_schema(bookings_csv):
    legs = load_bookings(bookings_csv)
    assert legs.columns.tolist() == EXPECTED_LEG_COLUMNS


def test_missing_source_file_fails_loudly(tmp_path):
    missing = tmp_path / "nope.csv"
    with pytest.raises(FileNotFoundError, match="Required source dataset not found"):
        load_bookings(missing)


def test_load_orders_coerces_order_date(tmp_path):
    path = tmp_path / "orders.csv"
    pd.DataFrame(
        [
            {
                "order_id": "KPD24O68HE2O",
                "order_price": 6090.0,
                "service": "changes",
                "service_price": 499.0,
                "service_profit": 399.2,
                "origin_country": "RU",
                "destination_country": "RU",
                "order_platform": "desktop",
                "booking_depth": 15.4,
                "is_return": 0,
                "adults": 1.0,
                "children": 0,
                "order_date": "2024-01-05",
            }
        ]
    ).to_csv(path, index=False)

    orders = load_orders(path)
    assert len(orders) == 1
    assert pd.api.types.is_datetime64_any_dtype(orders["order_date"])
    assert orders["order_date"].iloc[0] == pd.Timestamp("2024-01-05")


def test_load_events_parses_hashed_user_id_and_timestamps(tmp_path):
    path = tmp_path / "events.csv"
    pd.DataFrame(
        [
            {
                # The real feed's ids contain '/'.
                "user_id": "2OgK/ukRgvkWAOaa8XaTPg==",
                "ab_group": "A",
                "ts": "2023-05-02 12:24:33.553327",
                "pdate": "2023-05-02",
            }
        ]
    ).to_csv(path, index=False)

    events = load_events(path)
    assert len(events) == 1
    assert events["user_id"].iloc[0] == "2OgK/ukRgvkWAOaa8XaTPg=="
    assert pd.api.types.is_datetime64_any_dtype(events["ts"])
    assert events["ts"].iloc[0] == pd.Timestamp("2023-05-02 12:24:33.553327")


def test_ragged_leg_json_is_preserved_not_imputed(tmp_path):
    """The real feed omits keys on some legs; absent values must stay pd.NA."""
    path = tmp_path / "ragged.csv"
    pd.DataFrame(
        [
            {
                "booking_id": "RAGGED01",
                "booking_month": 3,
                "passengers": 1,
                "price": 1000,
                "flights_info": json.dumps(
                    [
                        # leg 0 omits 'destination', leg 1 omits 'airline'
                        {"origin": "SVO", "airline": "SU", "baggage": "No baggage"},
                        {"origin": "LED", "destination": "AER", "baggage": "No baggage"},
                    ]
                ),
            }
        ]
    ).to_csv(path, index=False)

    legs = load_bookings(path)
    assert len(legs) == 2
    assert pd.isna(legs.loc[0, "destination"])
    assert pd.isna(legs.loc[1, "airline"])
    # Present fields survive untouched.
    assert legs.loc[0, "origin"] == "SVO"
    assert legs.loc[1, "destination"] == "AER"
    # has_baggage is a plain bool even when the baggage key is absent.
    assert legs["has_baggage"].tolist() == [False, False]
    assert legs["has_baggage"].dtype == bool
