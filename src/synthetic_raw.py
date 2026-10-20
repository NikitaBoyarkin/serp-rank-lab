"""Deterministic synthetic stand-in for the three raw Aviasales feeds.

Why this module exists
----------------------
The three source CSVs are third-party interview datasets and can never be
committed. Without them ``data/raw/`` is empty, so a clean clone could not run
``run.py --stage all`` at all: ``src.data_ingest`` fails on the first missing
file. This module regenerates schema-compatible inputs from a seed, so the whole
pipeline -- ingestion, sessionisation, SERP simulation, ranking evaluation,
experiment, SQL views and business metrics -- runs end-to-end with no external
data and no network.

It is **not** a replacement for the real feed and it does not claim to be one:
it is a fixture. Every downstream artifact keeps its existing "modelled vs
measured" labelling (``docs/adr/0001``), and the README says where the numbers
come from. ``src/data_ingest.py`` is left untouched and accepts these files
without modification -- the contract is checked by
``tests/test_reproducibility.py::test_raw_generator_matches_loader_schema``.

What is reused rather than invented
-----------------------------------
* **Route landscape** -- the 20 origin->destination pairs that carry both
  endpoints in ``sql/01_normalize.sql``'s ``route_demand`` view, with their
  real per-route booking counts, average booking prices and within-route log
  price dispersion (``ROUTES``). Freezing them keeps ``serp_sim.load_routes()``
  on the same price levels the project was written against (SVO->LED ~6.0k,
  SVO->HKT ~52.0k).
* **Seasonality** -- :data:`src.serp_sim.MONTH_WEIGHTS`, the real leg-count
  distribution for months 1..8. Imported, not copied, so the simulator and the
  fixture can never drift apart.
* **Corpus sizes** -- 135,508 bookings, 8,966 order rows (8,803 distinct
  ``order_id``, i.e. the documented 163 multi-service duplicates), 44,568
  events, 5,000 users. These are the magnitudes the project and its docs were
  built around; keeping them means a clean clone reproduces the same scale.
* **Distributions** -- passenger mix, airline mix, baggage share, service mix
  and prices, platform mix, country mix, return share (~25.2%), booking-depth
  shape and the 8-day event window with its day/hour profile are all taken from
  the real feeds and frozen below as documented constants.

The one deliberate re-attribution: the real 135,508 bookings include 3,357 whose
leg JSON omitted an endpoint, so they do not belong to any of the 20 complete
pairs. Synthetic legs always carry all keys, so those 3,357 bookings are
allocated pro rata over the three LED routes (the busiest ones) instead of
fabricating malformed JSON. That keeps both the published corpus size and the
20-pair contract.

Determinism
-----------
Everything is driven by ``numpy.random.default_rng`` seeded from
:class:`SyntheticConfig.seed` through three independent streams (bookings,
orders, events). Identical config -> byte-identical CSVs: no ``random``, no
``hash()``, no wall-clock time. ``tests/test_reproducibility.py`` pins this with
sha256 digests.

CLI: ``python -m src.synthetic_raw [--seed 42] [--out-dir data/raw] [--force]``.
``run.py --stage all`` calls it first and skips generation when the feeds are
already present (or when ``AVIASALES_SOURCE_DIR`` points at real ones).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_ingest import SOURCE_FILES, SOURCE_SUBDIRS, source_dir
from src.serp_sim import MONTH_WEIGHTS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# --------------------------------------------------------------------------- #
# Frozen anchors measured once on the real feeds
# --------------------------------------------------------------------------- #

# origin, destination, bookings, avg booking price, within-route log-price sigma.
# The first four are direct routes; the rest are the legs of SVO->hub->HKT
# connecting itineraries and therefore carry equal booking counts per itinerary.
DIRECT_ROUTES: tuple[tuple[str, str, int, float, float], ...] = (
    ("SVO", "LED", 57_836, 6_006.759976, 0.595573),
    ("VKO", "LED", 51_573, 5_299.035503, 0.604677),
    ("DME", "LED", 18_510, 6_105.847488, 0.549512),
    ("SVO", "HKT", 2_694, 52_013.159614, 0.600100),
)

# origin, hub, destination, connecting bookings, avg booking price, price sigma.
CONNECTING_ROUTES: tuple[tuple[str, str, str, int, float, float], ...] = (
    ("SVO", "AUH", "HKT", 468, 59_959.247863, 0.545476),
    ("SVO", "MCT", "HKT", 432, 58_018.305556, 0.436246),
    ("SVO", "TFU", "HKT", 168, 54_277.458333, 0.519387),
    ("SVO", "PVG", "HKT", 156, 58_589.923077, 0.509282),
    ("SVO", "PEK", "HKT", 154, 84_646.519481, 0.567789),
    ("SVO", "XIY", "HKT", 87, 48_706.931034, 0.410979),
    ("SVO", "KJA", "HKT", 39, 52_250.358974, 0.603419),
    ("SVO", "DOH", "HKT", 34, 87_151.176471, 0.659151),
)

# Passenger counts on a booking, from the real legs feed.
PASSENGER_VALUES = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
PASSENGER_WEIGHTS = np.array([107_211, 24_118, 3_983, 1_310, 302, 106, 23, 16, 13], dtype=float)
PASSENGER_WEIGHTS /= PASSENGER_WEIGHTS.sum()

# Airline codes, from the real legs feed (the long tail is pooled).
AIRLINES = np.array(
    [
        "DP",
        "SU",
        "FV",
        "5N",
        "UT",
        "U6",
        "N4",
        "S7",
        "Y7",
        "XT",
        "6R",
        "EY",
        "WY",
        "MU",
        "3U",
        "A4",
        "HU",
        "CA",
        "FM",
        "EO",
        "F7",
        "QR",
        "IO",
        "WZ",
    ]
)
AIRLINE_WEIGHTS = np.array(
    [
        40_111,
        20_984,
        19_542,
        15_328,
        8_247,
        7_942,
        7_863,
        7_464,
        1_778,
        1_486,
        1_133,
        946,
        872,
        377,
        324,
        251,
        166,
        142,
        129,
        78,
        75,
        72,
        58,
        35,
    ],
    dtype=float,
)
AIRLINE_WEIGHTS /= AIRLINE_WEIGHTS.sum()

BAGGAGE_SHARE = 19_869 / 137_082  # real "With baggage" legs
BAGGAGE_TOKENS = ("No baggage", "With baggage")

# Service mix / price ladders from the real orders feed. Prices are discrete in
# the feed, so the ladders are frozen verbatim rather than modelled.
SERVICE_WEIGHTS = {
    None: 7_121,
    "notification": 1_082,
    "changes": 485,
    "refund": 255,
    "support": 23,
}
SERVICE_PRICES = {
    "notification": (149.0, 299.0, 399.0, 499.0),
    "changes": (499.0, 699.0, 1_299.0, 2_999.0),
    "refund": (699.0, 1_599.0, 2_299.0, 6_518.0),
    "support": (459.0, 699.0, 1_479.0),
}
SERVICE_MARGINS = {
    "notification": (0.800, 0.909),
    "changes": (0.800, 0.909),
    "refund": (0.600, 0.682),
    "support": (0.800, 0.909),
}

PLATFORMS = ("desktop", "ios", "android", "ios_web", "android_web", None, "other")
PLATFORM_WEIGHTS = np.array([3_399, 2_684, 1_380, 571, 500, 412, 20], dtype=float)
PLATFORM_WEIGHTS /= PLATFORM_WEIGHTS.sum()

# Country mix as observed, with the long tail kept verbatim rather than pooled
# into an "OTHER" bucket that would surface in the top-10 geo table.
ORIGIN_COUNTRIES = {
    "RU": 7_684,
    "TR": 148,
    "AZ": 117,
    "TH": 92,
    "AM": 90,
    "KZ": 88,
    "UZ": 78,
    "AE": 71,
    "GE": 71,
    "KG": 58,
    "TJ": 57,
    "BY": 37,
    "CN": 31,
    "IL": 30,
    "IT": 24,
    "DE": 20,
    "CY": 19,
    "ME": 17,
    "FR": 16,
    "ES": 15,
    "IN": 14,
    "ID": 13,
    "RS": 13,
    "KR": 12,
    "LK": 12,
    "AT": 10,
    "MD": 9,
    "EG": 9,
    "CH": 9,
    "VN": 9,
    "NY": 6,
    "MV": 6,
    "HU": 5,
    "MY": 5,
    "GR": 5,
    "JP": 5,
    "SA": 5,
    "CZ": 4,
    "SG": 4,
    "BG": 4,
    "GB": 3,
    "CA": 3,
    "NL": 3,
    "LB": 3,
    "MN": 3,
    "PT": 2,
    "AR": 2,
    "BE": 2,
    "RO": 2,
    "SE": 2,
    "PL": 2,
    "CU": 2,
    "DZ": 2,
    "IE": 2,
    "NO": 2,
    "PH": 2,
    "NP": 2,
    "HK": 1,
    "MU": 1,
    "MX": 1,
    "BH": 1,
    "IR": 1,
}

DESTINATION_COUNTRIES = {
    "RU": 6_816,
    "TR": 248,
    "AZ": 189,
    "KG": 185,
    "TJ": 180,
    "UZ": 169,
    "KZ": 142,
    "TH": 127,
    "AE": 126,
    "AM": 124,
    "GE": 97,
    "BY": 52,
    "IT": 45,
    "IL": 35,
    "ID": 34,
    "VN": 30,
    "FR": 25,
    "IN": 24,
    "EG": 21,
    "DE": 20,
    "LK": 19,
    "CN": 19,
    "CY": 18,
    "RS": 18,
    "KR": 16,
    "ES": 14,
    "MD": 13,
    "MV": 11,
    "ME": 11,
    "JP": 11,
    "US": 10,
    "PH": 7,
    "HU": 6,
    "MN": 5,
    "MY": 5,
    "IR": 4,
    "TZ": 4,
    "MX": 4,
    "AT": 4,
    "PK": 4,
    "CZ": 4,
    "PT": 3,
    "NL": 3,
    "GR": 3,
    "SC": 3,
    "ZA": 3,
    "SG": 3,
    "GB": 3,
    "OM": 3,
    "PL": 3,
    "HK": 3,
    "CH": 2,
    "LB": 2,
    "DK": 2,
    "EE": 2,
    "MU": 2,
    "QA": 2,
    "MA": 2,
    "SE": 2,
    "SA": 2,
    "CU": 2,
    "CA": 2,
    "AU": 1,
    "RO": 1,
    "SI": 1,
    "UY": 1,
    "NY": 1,
    "ML": 1,
    "ET": 1,
    "FI": 1,
    "NG": 1,
    "TN": 1,
    "KW": 1,
    "NP": 1,
    "MT": 1,
    "AR": 1,
    "TM": 1,
    "JO": 1,
    "BE": 1,
    "KH": 1,
}


def _country_arrays(table: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    """Turn a ``{country: count}`` table into aligned choice values and weights."""
    codes = np.array(list(table), dtype=object)
    weights = np.array(list(table.values()), dtype=float)
    return codes, weights / weights.sum()


ORIGIN_COUNTRY_VALUES, ORIGIN_COUNTRY_WEIGHTS = _country_arrays(ORIGIN_COUNTRIES)
DESTINATION_COUNTRY_VALUES, DESTINATION_COUNTRY_WEIGHTS = _country_arrays(DESTINATION_COUNTRIES)

ADULTS_VALUES = np.array([1.0, 2.0, 3.0, 4.0])
ADULTS_WEIGHTS = np.array([6_652, 1_408, 189, 61], dtype=float)
ADULTS_WEIGHTS /= ADULTS_WEIGHTS.sum()
ADULTS_MISSING_SHARE = 637 / 8_966

CHILDREN_VALUES = np.array([0, 1, 2, 3])
CHILDREN_WEIGHTS = np.array([8_530, 345, 77, 13], dtype=float)
CHILDREN_WEIGHTS /= CHILDREN_WEIGHTS.sum()

RETURN_SHARE = 0.2523979478028106  # real is_return share

# The real order feed spans 80 days; the event feed covers eight days with a
# clear launch-week profile (days 1-3 heavy, then a long tail).
ORDER_START = date(2023, 11, 1)
ORDER_DAYS = 80

EVENT_DAYS = (
    date(2023, 5, 1),
    date(2023, 5, 2),
    date(2023, 5, 3),
    date(2023, 5, 4),
    date(2023, 5, 5),
    date(2023, 5, 6),
    date(2023, 5, 7),
    date(2023, 5, 8),
)
EVENT_DAY_WEIGHTS = np.array([11_704, 15_239, 13_721, 1_592, 848, 496, 546, 422], dtype=float)
EVENT_DAY_WEIGHTS /= EVENT_DAY_WEIGHTS.sum()

EVENT_HOUR_WEIGHTS = np.array(
    [
        0.005,
        0.009,
        0.012,
        0.014,
        0.020,
        0.028,
        0.047,
        0.059,
        0.065,
        0.063,
        0.068,
        0.072,
        0.064,
        0.066,
        0.063,
        0.057,
        0.056,
        0.053,
        0.058,
        0.045,
        0.034,
        0.022,
        0.012,
        0.008,
    ],
    dtype=float,
)
EVENT_HOUR_WEIGHTS /= EVENT_HOUR_WEIGHTS.sum()

# Session shape: the real feed has ~7.5 events per session under the 30-minute
# inactivity rule, so a session holds 3..10 events a few minutes apart.
SESSION_SIZE_MIN = 3
SESSION_SIZE_MAX = 12
SESSION_GAP_SECONDS = (20, 900)  # within a session: always under 30 minutes
BETWEEN_SESSION_MINUTES = (31, 180)  # always over the 30-minute cutoff

# Largest-remainder allocation of the 3,357 endpoint-less bookings onto the LED
# routes (see the module docstring). Done with integer arithmetic so the total
# is exact regardless of floating-point rounding.
LED_ROUTE_INDICES = (0, 1, 2)


@dataclass(frozen=True)
class SyntheticConfig:
    """Corpus sizes and the seed. Defaults reproduce the documented volumes."""

    seed: int = 42
    bookings: int = 135_508
    orders: int = 8_966
    distinct_orders: int = 8_803
    events: int = 44_568
    users: int = 5_000


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _digest(*parts: object) -> bytes:
    """Deterministic 16-byte digest of ``parts`` (no ``hash()``, no salt)."""
    payload = ":".join(str(p) for p in parts).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=16).digest()


def _booking_id(seed: int, index: int) -> str:
    return _digest("booking", seed, index).hex()


_ORDER_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def _order_id(seed: int, index: int) -> str:
    digest = _digest("order", seed, index)
    return "".join(_ORDER_ALPHABET[b % len(_ORDER_ALPHABET)] for b in digest[:12])


def _user_id(seed: int, index: int) -> str:
    return base64.b64encode(_digest("user", seed, index)).decode("ascii")


def _largest_remainder(total: int, weights: list[int]) -> list[int]:
    """Split ``total`` across ``weights`` with integer parts summing to exactly ``total``."""
    weight_sum = sum(weights)
    exact = [total * w / weight_sum for w in weights]
    floors = [int(x) for x in exact]
    remainder = total - sum(floors)
    order = sorted(range(len(weights)), key=lambda i: (-(exact[i] - floors[i]), i))
    for i in order[:remainder]:
        floors[i] += 1
    return floors


def route_plan(config: SyntheticConfig) -> list[tuple[str, str, int, float, float]]:
    """Expand :data:`DIRECT_ROUTES` to the full corpus size.

    Returns ``(origin, destination, bookings, avg_price, price_sigma)`` per direct
    route after the residual 3,357 bookings are allocated pro rata to the three
    LED routes. Connecting itineraries are planned separately.
    """
    base = [[o, d, n, p, s] for o, d, n, p, s in DIRECT_ROUTES]
    connecting_bookings = sum(row[3] for row in CONNECTING_ROUTES)
    skeleton = sum(row[2] for row in DIRECT_ROUTES) + connecting_bookings
    residual = config.bookings - skeleton
    if residual < 0:
        raise ValueError(f"bookings={config.bookings} is below the 20-route skeleton of {skeleton}")
    weights = [int(base[i][2]) for i in LED_ROUTE_INDICES]
    extra = _largest_remainder(residual, weights)
    for i, add in zip(LED_ROUTE_INDICES, extra, strict=True):
        base[i][2] += add
    return [(str(r[0]), str(r[1]), int(r[2]), float(r[3]), float(r[4])) for r in base]


# --------------------------------------------------------------------------- #
# Bookings feed
# --------------------------------------------------------------------------- #


def generate_bookings(config: SyntheticConfig, rng: np.random.Generator) -> pd.DataFrame:
    """Build ``data.csv``: one row per booking, ``flights_info`` holding its legs."""
    month_values = np.array(sorted(MONTH_WEIGHTS))
    month_weights = np.array([MONTH_WEIGHTS[m] for m in month_values], dtype=float)
    month_weights /= month_weights.sum()

    rows: list[dict[str, object]] = []

    for origin, destination, n, avg_price, sigma in route_plan(config):
        prices = avg_price * np.exp(sigma * rng.normal(size=n))
        # Pin the route mean to the frozen anchor; log-normal shape is preserved.
        prices *= avg_price / float(prices.mean())
        prices = np.maximum(np.rint(prices), 100).astype(np.int64)
        months = rng.choice(month_values, size=n, p=month_weights)
        passengers = rng.choice(PASSENGER_VALUES, size=n, p=PASSENGER_WEIGHTS)
        airlines = rng.choice(AIRLINES, size=n, p=AIRLINE_WEIGHTS)
        baggage = rng.random(size=n) < BAGGAGE_SHARE

        for i in range(n):
            leg = {
                "origin": origin,
                "destination": destination,
                "airline": str(airlines[i]),
                "baggage": BAGGAGE_TOKENS[int(baggage[i])],
            }
            rows.append(
                {
                    "booking_month": int(months[i]),
                    "passengers": int(passengers[i]),
                    "price": int(prices[i]),
                    "flights_info": json.dumps([leg], ensure_ascii=False),
                }
            )

    for origin, hub, destination, n, avg_price, sigma in CONNECTING_ROUTES:
        prices = avg_price * np.exp(sigma * rng.normal(size=n))
        prices *= avg_price / float(prices.mean())
        prices = np.maximum(np.rint(prices), 100).astype(np.int64)
        months = rng.choice(month_values, size=n, p=month_weights)
        passengers = rng.choice(PASSENGER_VALUES, size=n, p=PASSENGER_WEIGHTS)
        airlines = rng.choice(AIRLINES, size=n, p=AIRLINE_WEIGHTS)
        baggage = rng.random(size=(n, 2)) < BAGGAGE_SHARE

        for i in range(n):
            legs = [
                {
                    "origin": origin,
                    "destination": hub,
                    "airline": str(airlines[i]),
                    "baggage": BAGGAGE_TOKENS[int(baggage[i, 0])],
                },
                {
                    "origin": hub,
                    "destination": destination,
                    "airline": str(airlines[i]),
                    "baggage": BAGGAGE_TOKENS[int(baggage[i, 1])],
                },
            ]
            rows.append(
                {
                    "booking_month": int(months[i]),
                    "passengers": int(passengers[i]),
                    "price": int(prices[i]),
                    "flights_info": json.dumps(legs, ensure_ascii=False),
                }
            )

    order = rng.permutation(len(rows))
    bookings = pd.DataFrame(rows).iloc[order].reset_index(drop=True)
    bookings.insert(
        0,
        "booking_id",
        [_booking_id(config.seed, i) for i in range(len(bookings))],
    )
    return bookings.loc[:, ["booking_id", "booking_month", "passengers", "price", "flights_info"]]


# --------------------------------------------------------------------------- #
# Orders feed
# --------------------------------------------------------------------------- #


def _service_price(service: str, rng: np.random.Generator) -> tuple[float, float]:
    """Draw a discrete service price and its profit for one labelled order."""
    ladder = SERVICE_PRICES[service]
    price = float(ladder[int(rng.integers(0, len(ladder)))])
    margins = SERVICE_MARGINS[service]
    margin = float(margins[int(rng.integers(0, len(margins)))])
    return price, round(price * margin, 2)


def generate_orders(config: SyntheticConfig, rng: np.random.Generator) -> pd.DataFrame:
    """Build ``booking_analytics_test_df.csv`` -- paid service orders."""
    services = list(SERVICE_WEIGHTS)
    service_p = np.array([SERVICE_WEIGHTS[s] for s in services], dtype=float)
    service_p /= service_p.sum()

    n_distinct = config.distinct_orders
    n_extra = config.orders - n_distinct
    if n_extra < 0:
        raise ValueError("orders must be >= distinct_orders")

    base_services = rng.choice(np.array(services, dtype=object), size=n_distinct, p=service_p)
    order_prices = np.maximum(500.0, 15_377.0 * np.exp(0.85 * rng.normal(size=n_distinct))).round(6)
    depths = np.clip(6.0 * np.exp(1.25 * rng.normal(size=n_distinct)), 0.0, 330.0)
    platforms = rng.choice(np.array(PLATFORMS, dtype=object), size=n_distinct, p=PLATFORM_WEIGHTS)
    origins = rng.choice(ORIGIN_COUNTRY_VALUES, size=n_distinct, p=ORIGIN_COUNTRY_WEIGHTS)
    destinations = rng.choice(
        DESTINATION_COUNTRY_VALUES, size=n_distinct, p=DESTINATION_COUNTRY_WEIGHTS
    )
    returns = (rng.random(size=n_distinct) < RETURN_SHARE).astype(int)
    adults_missing = rng.random(size=n_distinct) < ADULTS_MISSING_SHARE
    adults = rng.choice(ADULTS_VALUES, size=n_distinct, p=ADULTS_WEIGHTS)
    children = rng.choice(CHILDREN_VALUES, size=n_distinct, p=CHILDREN_WEIGHTS)
    day_offsets = rng.integers(0, ORDER_DAYS, size=n_distinct)

    rows: list[dict[str, object]] = []
    for i in range(n_distinct):
        service = base_services[i]
        labelled = service if isinstance(service, str) else None
        if labelled is None:
            service_price, service_profit = float("nan"), float("nan")
        else:
            service_price, service_profit = _service_price(labelled, rng)
        rows.append(
            {
                "order_id": _order_id(config.seed, i),
                "order_price": float(order_prices[i]),
                "service": labelled,
                "service_price": service_price,
                "service_profit": service_profit,
                "origin_country": str(origins[i]),
                "destination_country": str(destinations[i]),
                "order_platform": (None if platforms[i] is None else str(platforms[i])),
                "booking_depth": round(float(depths[i]), 6),
                "is_return": int(returns[i]),
                "adults": float("nan") if adults_missing[i] else float(adults[i]),
                "children": int(children[i]),
                "order_date": (ORDER_START + timedelta(days=int(day_offsets[i]))).isoformat(),
            }
        )

    # The real feed carries 163 order_ids twice: one purchase, several services.
    duplicates = rng.choice(n_distinct, size=n_extra, replace=False)
    for i in duplicates:
        row = dict(rows[int(i)])
        service = rng.choice(np.array(services, dtype=object), p=service_p)
        labelled = service if isinstance(service, str) else None
        row["service"] = labelled
        if labelled is None:
            row["service_price"] = float("nan")
            row["service_profit"] = float("nan")
        else:
            row["service_price"], row["service_profit"] = _service_price(labelled, rng)
        rows.append(row)

    order = rng.permutation(len(rows))
    frame = pd.DataFrame(rows).iloc[order].reset_index(drop=True)
    return frame.loc[
        :,
        [
            "order_id",
            "order_price",
            "service",
            "service_price",
            "service_profit",
            "origin_country",
            "destination_country",
            "order_platform",
            "booking_depth",
            "is_return",
            "adults",
            "children",
            "order_date",
        ],
    ]


# --------------------------------------------------------------------------- #
# Events feed
# --------------------------------------------------------------------------- #


def _event_counts(config: SyntheticConfig, rng: np.random.Generator) -> np.ndarray:
    """Per-user event counts, each >= 2, summing to exactly ``config.events``."""
    floor = 2
    free = config.events - floor * config.users
    if free < 0:
        raise ValueError("events must be at least 2 per user")
    weights = rng.lognormal(mean=0.0, sigma=0.95, size=config.users)
    probabilities = weights / weights.sum()
    extra = rng.multinomial(free, probabilities)
    return extra + floor


def _user_timestamps(count: int, rng: np.random.Generator) -> list[datetime]:
    """Place ``count`` events into sessions on one or more active days."""
    days = min(count, 1 + int(rng.poisson(0.7)))
    day_indices = rng.choice(
        np.arange(len(EVENT_DAYS)), size=days, replace=False, p=EVENT_DAY_WEIGHTS
    )
    allocation = rng.multinomial(count, np.full(days, 1.0 / days))

    stamps: list[datetime] = []
    for day_index, n in zip(day_indices, allocation, strict=True):
        remaining = int(n)
        cursor_seconds = int(
            rng.choice(np.arange(24), p=EVENT_HOUR_WEIGHTS) * 3600 + rng.integers(0, 3600)
        )
        while remaining > 0:
            size = min(remaining, int(rng.integers(SESSION_SIZE_MIN, SESSION_SIZE_MAX + 1)))
            moment = cursor_seconds
            for _ in range(size):
                seconds = moment % (24 * 3600)
                micro = int(rng.integers(0, 1_000_000))
                stamps.append(
                    datetime.combine(EVENT_DAYS[int(day_index)], datetime.min.time())
                    + timedelta(seconds=seconds, microseconds=micro)
                )
                moment += int(rng.integers(*SESSION_GAP_SECONDS))
            remaining -= size
            # Next session starts well after the 30-minute inactivity cutoff.
            cursor_seconds += int(rng.integers(*BETWEEN_SESSION_MINUTES)) * 60
    return sorted(stamps)


def generate_events(config: SyntheticConfig, rng: np.random.Generator) -> pd.DataFrame:
    """Build ``task_1_events.csv`` -- 5,000 users, half in each A/B arm."""
    counts = _event_counts(config, rng)
    user_ids = [_user_id(config.seed, i) for i in range(config.users)]
    groups = ["A" if i % 2 == 0 else "B" for i in range(config.users)]

    rows: list[dict[str, object]] = []
    for i, count in enumerate(counts):
        for stamp in _user_timestamps(int(count), rng):
            rows.append(
                {
                    "user_id": user_ids[i],
                    "ab_group": groups[i],
                    "ts": stamp.strftime("%Y-%m-%d %H:%M:%S.%f"),
                    "pdate": stamp.date().isoformat(),
                }
            )

    frame = pd.DataFrame(rows)
    return frame.loc[:, ["user_id", "ab_group", "ts", "pdate"]]


# --------------------------------------------------------------------------- #
# Writers / CLI
# --------------------------------------------------------------------------- #


def target_paths(out_dir: Path | str = RAW_DIR) -> dict[str, Path]:
    """Paths of the three feeds under ``out_dir``, mirroring ``data_ingest``."""
    root = Path(out_dir)
    return {kind: root / SOURCE_SUBDIRS[kind] / SOURCE_FILES[kind] for kind in SOURCE_FILES}


def generate_raw(
    out_dir: Path | str = RAW_DIR,
    config: SyntheticConfig | None = None,
    *,
    force: bool = True,
) -> dict[str, int]:
    """Write the three feeds under ``out_dir``; returns ``{feed: row_count}``."""
    config = config or SyntheticConfig()
    paths = target_paths(out_dir)
    if not force and all(path.exists() for path in paths.values()):
        return {}

    bookings = generate_bookings(config, np.random.default_rng([config.seed, 1]))
    orders = generate_orders(config, np.random.default_rng([config.seed, 2]))
    events = generate_events(config, np.random.default_rng([config.seed, 3]))

    for path, frame in (
        (paths["bookings"], bookings),
        (paths["orders"], orders),
        (paths["events"], events),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)

    return {
        "bookings": len(bookings),
        "orders": len(orders),
        "events": len(events),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.synthetic_raw",
        description="Regenerate schema-compatible synthetic raw feeds under data/raw/.",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default 42)")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="destination directory (default: data/raw, or AVIASALES_SOURCE_DIR when set)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing feeds even when all three are present",
    )
    args = parser.parse_args(argv)

    out_dir = args.out_dir if args.out_dir is not None else source_dir()
    paths = target_paths(out_dir)
    if not args.force and all(path.exists() for path in paths.values()):
        print("=== synthetic_raw ===")
        print(f"feed directory: {out_dir}")
        print("all three feeds present -- nothing generated (use --force to overwrite)")
        return 0

    print("=== synthetic_raw ===")
    print(f"feed directory: {out_dir}")
    print(f"seed:           {args.seed}")
    counts = generate_raw(out_dir, SyntheticConfig(seed=args.seed), force=True)
    for feed, rows in counts.items():
        print(f"{feed:<10} {rows:>10,} rows -> {paths[feed]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
