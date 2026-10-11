"""SERP offer simulator for SERP Rank Lab (Phase 2).

Materialises the ``serp_offers`` table that ``sql/03_funnel.sql`` was written
against: one row per ``(search_id, position)`` offer on a search results page.

The simulation is **anchored on the real data**, never invented:

* **Routes** are sampled from the ``route_demand`` view defined in
  ``sql/01_normalize.sql``. That view has 39 origin->destination pairs, of which
  **20 carry both endpoints**; a SERP needs a concrete route, so the simulator
  samples those 20.
* **Price levels** are derived from each pair's ``avg_booking_price`` (real
  per-route averages, e.g. SVO->LED ~6006, SVO->HKT ~52013) scaled by a
  partner-specific index and a log-normal noise term.
* **Seasonality** samples ``booking_month`` with the real leg-count distribution
  (months 1..8: 13204, 10878, 12982, 14633, 20041, 22532, 21987, 20825 legs).
* **Users** are sampled from the 5,000 distinct real ``user_id`` values in the
  events feed.

Click generation is a **position-bias model (PBM)**:

    P(click | offer at position k) = P(examine at k) * P(relevant | offer)

with the standard decaying examination curve ``P(examine at k) = (1/k)**eta``
(so ``P(examine at 1) = 1``), and

    logit(P(relevant)) = base_relevance_logit + partner_affinity
                         - price_sensitivity * (price / min_price_in_serp - 1)

so a cheaper offer in its SERP is more relevant, and a partner-quality term
separates partners at equal price. Booking is only possible if the offer was
clicked and then redirected.

-------------------------------------------------------------------------------
Partner economics (the exact parameters; two economic types)
-------------------------------------------------------------------------------

**CPA partners** earn ``commission_rate * price`` **only when booked**.
**CPC partners** are paid a flat ``cpc`` **per click**, booked or not. The
``commission`` column holds *realised* partner revenue for the offer row: the
CPA commission on a booked row, or the CPC fee on a clicked row, and 0 otherwise.
That makes ``sum(commission)`` over a SERP the site's revenue for that search.

===========================  =====  ========  ===========  ========  ======  ====
partner                      type   rate/cpc  price_index  affinity  redir.  book
===========================  =====  ========  ===========  ========  ======  ====
OTA Global                   CPA    0.05      0.96         1.10      0.58    0.44
SkyTour                      CPA    0.07      0.99         0.80      0.55    0.40
AviaCenter                   CPA    0.09      1.02         0.45      0.52    0.36
FlyDirect                    CPA    0.11      1.05         0.10      0.50    0.33
TripMasters                  CPA    0.13      1.09        -0.35      0.47    0.30
CloudTrip                    CPC    22.0      0.97         0.95      0.56    0.39
MetaFly                      CPC    38.0      1.01         0.50      0.53    0.35
GoFare                       CPC    58.0      1.06         0.05      0.49    0.31
===========================  =====  ========  ===========  ========  ======  ====

``price_index`` and ``affinity`` are deliberately anti-correlated with the
money a partner pays (rate / cpc): the commission-rich partners quote slightly
higher prices and rank slightly worse on relevance. That is the modelling
assumption that makes relevance and revenue genuinely conflict -- it is an
assumption, documented here, not an empirical finding.

Other documented parameters (all overridable through :class:`SimConfig`):
``eta_mobile = 1.30`` decays faster than ``eta_desktop = 1.00`` (users scroll
less on mobile); ``mobile_share = 0.62``; offers per SERP uniform on 5..15; distinct partners per
SERP uniform on 3..6, capped at ``offers - 1`` (always fewer than the offer
count, so partners repeat across positions and ``partner_quality`` is
meaningful);
``price_sigma = 0.05``; ``money_tilt = 0.50`` blends relevance with normalised
expected revenue to pick the *displayed* order (the shipped policy -- money-
leaning but not pure); ``redirect_price_sens = 0.15`` and
``book_price_sens = 0.25`` make pricier offers convert less; ``month_multiplier``
is a documented mild seasonal booking factor.

``position`` is the displayed order (the shipped blend). ``relevance`` is
carried as an extra column so Phase 3 can build an alternative ranking.

CLI: ``python -m src.serp_sim`` writes ``data/processed/serp_offers.parquet``.
Everything is driven by one seeded ``numpy`` generator, so two runs with the
same ``--seed`` are byte-identical.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SQL_DIR = PROJECT_ROOT / "sql"

BOOKINGS_PARQUET = PROCESSED_DIR / "bookings_legs.parquet"
ORDERS_PARQUET = PROCESSED_DIR / "orders.parquet"
EVENTS_PARQUET = PROCESSED_DIR / "events.parquet"
SERP_OFFERS_PARQUET = PROCESSED_DIR / "serp_offers.parquet"

# The contract columns sql/03_funnel.sql selects by name, in contract order.
CONTRACT_COLUMNS = [
    "search_id",
    "user_id",
    "position",
    "partner",
    "price",
    "commission",
    "clicked",
    "redirected",
    "booked",
]

# Extra columns carried for later phases (03_funnel selects by name, so extras
# are safe). relevance is the intrinsic relevance Phase 3 re-ranks on.
EXTRA_COLUMNS = [
    "relevance",
    "device",
    "booking_month",
    "origin",
    "destination",
    "partner_type",
    "price_rank",
    "examine_prob",
    "click_prob",
    "expected_revenue",
]

OUTPUT_COLUMNS = CONTRACT_COLUMNS + EXTRA_COLUMNS

# --------------------------------------------------------------------------- #
# Partner economics (see the module docstring table)
# --------------------------------------------------------------------------- #

PARTNERS: list[dict[str, object]] = [
    {
        "partner": "OTA Global",
        "type": "CPA",
        "commission_rate": 0.05,
        "price_index": 0.96,
        "affinity": 1.10,
        "redirect_rate": 0.58,
        "book_rate": 0.44,
    },
    {
        "partner": "SkyTour",
        "type": "CPA",
        "commission_rate": 0.07,
        "price_index": 0.99,
        "affinity": 0.80,
        "redirect_rate": 0.55,
        "book_rate": 0.40,
    },
    {
        "partner": "AviaCenter",
        "type": "CPA",
        "commission_rate": 0.09,
        "price_index": 1.02,
        "affinity": 0.45,
        "redirect_rate": 0.52,
        "book_rate": 0.36,
    },
    {
        "partner": "FlyDirect",
        "type": "CPA",
        "commission_rate": 0.11,
        "price_index": 1.05,
        "affinity": 0.10,
        "redirect_rate": 0.50,
        "book_rate": 0.33,
    },
    {
        "partner": "TripMasters",
        "type": "CPA",
        "commission_rate": 0.13,
        "price_index": 1.09,
        "affinity": -0.35,
        "redirect_rate": 0.47,
        "book_rate": 0.30,
    },
    {
        "partner": "CloudTrip",
        "type": "CPC",
        "cpc": 22.0,
        "price_index": 0.97,
        "affinity": 0.95,
        "redirect_rate": 0.56,
        "book_rate": 0.39,
    },
    {
        "partner": "MetaFly",
        "type": "CPC",
        "cpc": 38.0,
        "price_index": 1.01,
        "affinity": 0.50,
        "redirect_rate": 0.53,
        "book_rate": 0.35,
    },
    {
        "partner": "GoFare",
        "type": "CPC",
        "cpc": 58.0,
        "price_index": 1.06,
        "affinity": 0.05,
        "redirect_rate": 0.49,
        "book_rate": 0.31,
    },
]

PARTNER_BY_NAME: dict[str, dict[str, object]] = {p["partner"]: p for p in PARTNERS}
PARTNER_NAMES = np.array([p["partner"] for p in PARTNERS])

ETA_DESKTOP = 1.00
ETA_MOBILE = 1.30

# Real booking_month leg counts (months 1..8) -- sampling weights, verbatim.
MONTH_WEIGHTS = {1: 13204, 2: 10878, 3: 12982, 4: 14633, 5: 20041, 6: 22532, 7: 21987, 8: 20825}

# Documented mild seasonal booking factor (modelling assumption, not measured).
MONTH_MULTIPLIER = np.array(
    [0.98, 0.97, 1.00, 1.01, 1.03, 1.05, 1.06, 1.04, 1.00, 1.00, 0.99, 0.99]
)


# --------------------------------------------------------------------------- #
# Click-model primitives (shared with src.ranking_eval)
# --------------------------------------------------------------------------- #


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """Logistic function ``1 / (1 + exp(-x))``."""
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def examination_curve(n_positions: int, eta: float) -> np.ndarray:
    """PBM examination probabilities ``P(examine at k) = (1/k)**eta`` for k=1..n.

    ``P(examine at 1)`` is exactly 1.0; the curve is strictly decreasing for
    ``eta > 0``. This is the propensity an inverse-propensity correction
    divides by.
    """
    if n_positions < 1:
        raise ValueError("n_positions must be >= 1")
    k = np.arange(1, n_positions + 1, dtype=float)
    return (1.0 / k) ** eta


def examination_prob(position: int, eta: float) -> float:
    """``P(examine)`` at a single 1-based ``position``."""
    if position < 1:
        raise ValueError("position is 1-based and must be >= 1")
    return float((1.0 / position) ** eta)


def eta_for_device(device: str, config: SimConfig) -> float:
    """Return the examination decay for a device label."""
    return config.eta_mobile if device == "mobile" else config.eta_desktop


def revenue_if_engaged(
    partner: str,
    price: float,
    partner_by_name: dict[str, dict[str, object]] | None = None,
) -> float:
    """Expected partner revenue given the offer was clicked.

    * CPA: ``commission_rate * price * redirect_rate * book_rate`` (commission
      only lands on a booking).
    * CPC: ``cpc`` (paid on the click itself).

    ``partner_by_name`` overrides the module-level :data:`PARTNERS` mapping; the
    calibrator uses it to re-anchor the economics without editing this file.
    """
    p = (partner_by_name or PARTNER_BY_NAME)[partner]
    if p["type"] == "CPA":
        return (
            float(p["commission_rate"]) * price * float(p["redirect_rate"]) * float(p["book_rate"])
        )
    return float(p["cpc"])


def _clamp(x: float, lo: float = 0.02, hi: float = 0.95) -> float:
    return float(min(max(x, lo), hi))


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SimConfig:
    """Every knob of the simulation. Defaults match the module docstring."""

    seed: int = 42
    n_searches: int = 12_000
    offers_min: int = 5
    offers_max: int = 15
    partners_min: int = 3
    partners_max: int = 6
    eta_desktop: float = ETA_DESKTOP
    eta_mobile: float = ETA_MOBILE
    mobile_share: float = 0.62
    base_relevance_logit: float = 0.30
    price_sensitivity: float = 1.80
    price_sigma: float = 0.05
    redirect_price_sens: float = 0.15
    book_price_sens: float = 0.25
    money_tilt: float = 0.50
    month_multiplier: tuple[float, ...] = field(
        default_factory=lambda: tuple(float(x) for x in MONTH_MULTIPLIER)
    )
    months: tuple[int, ...] = field(default_factory=lambda: tuple(sorted(MONTH_WEIGHTS)))
    month_weights: tuple[float, ...] = field(
        default_factory=lambda: tuple(
            MONTH_WEIGHTS[m] / sum(MONTH_WEIGHTS.values()) for m in sorted(MONTH_WEIGHTS)
        )
    )


# --------------------------------------------------------------------------- #
# Real-data loaders
# --------------------------------------------------------------------------- #


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run `python -m src.data_ingest --stage all` first."
        )
    return path


def load_routes() -> pd.DataFrame:
    """Return the real ``route_demand`` pairs that have both endpoints.

    Reuses ``sql/01_normalize.sql`` (the source of truth for the view) rather
    than re-deriving it. Of the view's 39 pairs, 20 carry both an ``origin`` and
    a ``destination``; only those can host a SERP.
    """
    con = duckdb.connect()
    try:
        for name, path in (
            ("bookings_legs", BOOKINGS_PARQUET),
            ("orders", ORDERS_PARQUET),
            ("events", EVENTS_PARQUET),
        ):
            con.register(name, pd.read_parquet(_require(path)))
        con.execute((SQL_DIR / "01_normalize.sql").read_text(encoding="utf-8"))
        routes = con.execute(
            """
            SELECT origin, destination, avg_booking_price
            FROM route_demand
            WHERE origin IS NOT NULL AND destination IS NOT NULL
            ORDER BY origin, destination
            """
        ).df()
    finally:
        con.close()
    return routes


def load_user_ids() -> np.ndarray:
    """Return the 5,000 distinct real ``user_id`` values from the events feed."""
    events = pd.read_parquet(_require(EVENTS_PARQUET), columns=["user_id"])
    return np.asarray(sorted(events["user_id"].unique()))


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #


def simulate_serps(
    routes: pd.DataFrame,
    user_ids: np.ndarray | list[str],
    config: SimConfig | None = None,
    partners: list[dict[str, object]] | None = None,
) -> pd.DataFrame:
    """Generate the ``serp_offers`` frame (one row per offer).

    Deterministic: one ``numpy`` generator seeded from ``config.seed`` drives
    every random draw, in a fixed order, with stable tie-breaks -- identical
    inputs give identical output.

    ``partners`` defaults to the module-level :data:`PARTNERS`, so the default
    path is unchanged; a table with the same keys re-anchors the economics.
    """
    config = config or SimConfig()
    partner_table = PARTNERS if partners is None else list(partners)
    partner_by_name = {str(p["partner"]): p for p in partner_table}
    partner_names = np.array([str(p["partner"]) for p in partner_table])
    rng = np.random.default_rng(config.seed)

    routes = routes.reset_index(drop=True)
    if routes.empty:
        raise ValueError("routes must contain at least one real origin->destination pair")
    user_ids = np.asarray(user_ids)
    if user_ids.size == 0:
        raise ValueError("user_ids must be non-empty")

    month_weights = np.asarray(config.month_weights, dtype=float)
    months = np.asarray(config.months, dtype=int)
    month_multiplier = np.asarray(config.month_multiplier, dtype=float)

    rows: list[dict[str, object]] = []
    for s in range(config.n_searches):
        route = routes.iloc[int(rng.integers(0, len(routes)))]
        origin = str(route["origin"])
        destination = str(route["destination"])
        base_price = float(route["avg_booking_price"])

        month = int(rng.choice(months, p=month_weights))
        device = "mobile" if rng.random() < config.mobile_share else "desktop"
        eta = eta_for_device(device, config)

        n_offers = int(rng.integers(config.offers_min, config.offers_max + 1))
        # Strictly fewer distinct partners than offers, so partners repeat.
        n_partners = min(
            int(rng.integers(config.partners_min, config.partners_max + 1)), n_offers - 1
        )
        partner_pool = rng.choice(partner_names, size=n_partners, replace=False)
        offers_partner = [
            str(partner_pool[int(rng.integers(0, n_partners))]) for _ in range(n_offers)
        ]

        prices = np.array(
            [
                base_price
                * float(partner_by_name[p]["price_index"])
                * float(np.exp(rng.normal(0.0, config.price_sigma)))
                for p in offers_partner
            ]
        )
        price_ratio = prices / prices.min()
        affinities = np.array([float(partner_by_name[p]["affinity"]) for p in offers_partner])
        logits = (
            config.base_relevance_logit
            + affinities
            - config.price_sensitivity * (price_ratio - 1.0)
        )
        relevance = np.asarray(sigmoid(logits), dtype=float)

        rev_potential = np.array(
            [
                revenue_if_engaged(p, pr, partner_by_name)
                for p, pr in zip(offers_partner, prices, strict=True)
            ]
        )

        # Displayed order = shipped money-leaning blend (documented).
        spread = rev_potential.max() - rev_potential.min()
        rev_norm = (
            (rev_potential - rev_potential.min()) / spread if spread > 0 else np.zeros(n_offers)
        )
        blend = relevance + config.money_tilt * rev_norm
        order = np.lexsort((np.arange(n_offers), -blend))

        price_rank = np.empty(n_offers, dtype=int)
        price_order = np.lexsort((np.arange(n_offers), prices))
        for rank, oi in enumerate(price_order, start=1):
            price_rank[oi] = rank

        examine = examination_curve(n_offers, eta)
        search_id = f"SRCH{s:06d}"
        # A SERP belongs to exactly one user: draw once per search, not per offer.
        user_id = str(user_ids[int(rng.integers(0, len(user_ids)))])

        for pos, oi in enumerate(order, start=1):
            partner = offers_partner[oi]
            p_conf = partner_by_name[partner]
            p_examine = float(examine[pos - 1])
            p_click = p_examine * float(relevance[oi])

            clicked = bool(rng.random() < p_click)
            redirected = False
            booked = False
            commission = 0.0

            if clicked:
                ratio_excess = float(price_ratio[oi]) - 1.0
                p_redirect = _clamp(
                    float(p_conf["redirect_rate"])
                    * (1.0 - config.redirect_price_sens * ratio_excess)
                )
                redirected = bool(rng.random() < p_redirect)
                if redirected:
                    p_book = _clamp(
                        float(p_conf["book_rate"])
                        * (1.0 - config.book_price_sens * ratio_excess)
                        * float(month_multiplier[month - 1])
                    )
                    booked = bool(rng.random() < p_book)
                if p_conf["type"] == "CPA":
                    if booked:
                        commission = float(p_conf["commission_rate"]) * float(prices[oi])
                else:  # CPC: paid on the click, booked or not
                    commission = float(p_conf["cpc"])

            rows.append(
                {
                    "search_id": search_id,
                    "user_id": user_id,
                    "position": pos,
                    "partner": partner,
                    "price": float(prices[oi]),
                    "commission": commission,
                    "clicked": clicked,
                    "redirected": redirected,
                    "booked": booked,
                    "relevance": float(relevance[oi]),
                    "device": device,
                    "booking_month": month,
                    "origin": origin,
                    "destination": destination,
                    "partner_type": str(p_conf["type"]),
                    "price_rank": int(price_rank[oi]),
                    "examine_prob": p_examine,
                    "click_prob": p_click,
                    "expected_revenue": p_click * float(rev_potential[oi]),
                }
            )

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def write_serp_offers(df: pd.DataFrame, path: Path = SERP_OFFERS_PARQUET) -> Path:
    """Write the frame to parquet and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.serp_sim",
        description="Simulate serp_offers from the real route/price pairs.",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default 42)")
    parser.add_argument("--searches", type=int, default=12_000, help="number of SERPs")
    args = parser.parse_args(argv)

    config = SimConfig(seed=args.seed, n_searches=args.searches)
    routes = load_routes()
    user_ids = load_user_ids()

    print("=== serp_sim ===")
    print(f"real route pairs:  {len(routes)} (both endpoints)")
    print(f"real users:        {len(user_ids):,}")
    print(f"searches:          {config.n_searches:,}  seed={config.seed}")

    offers = simulate_serps(routes, user_ids, config)
    target = write_serp_offers(offers)

    per_search = offers.groupby("search_id").size()
    revenue = float(offers["commission"].sum())
    print(f"rows written:      {len(offers):,} -> {target}")
    print(
        f"offers per SERP:   min={int(per_search.min())} max={int(per_search.max())} "
        f"mean={per_search.mean():.2f}"
    )
    print(f"click rate:        {offers['clicked'].mean():.4f}")
    print(f"book rate:         {offers['booked'].mean():.4f}")
    print(f"revenue per search: {revenue / config.n_searches:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
