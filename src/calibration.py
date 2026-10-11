"""Re-anchoring the SERP simulator on observable anchors (Phase 4).

``docs/adr/0001`` draws a hard line: ``serp_offers`` is *modelled*, the orders
and legs feeds are *measured*, and the two are never fused. That line leaves an
obvious hole -- nothing checks that the simulator's parameters are anywhere near
the world they claim to model. This module closes it without crossing the line:
it reads anchors **out of the real islands only**, and writes them **into the
simulator's knobs only**. No row of real data ever enters ``serp_offers``.

What is actually observable
----------------------------
The islands are ``orders`` (order value, ancillary service economics) and
``bookings_legs`` (per-leg price, route, month). Neither carries a click or a
partner, so the simulator's click model, partner table and commission rates are
**not identifiable** from them. We do not pretend otherwise. The anchors that
*are* observable, and the knobs they bind:

========================================  ====================================
anchor                                    knob
========================================  ====================================
``within_route_log_price_sigma``          ``SimConfig.price_sigma``
``month_share``                           ``SimConfig.months/month_weights``
``order_price_median`` / quantiles        reported only (level check)
``service_attach_rate`` / margin          reported only (unit economics)
========================================  ====================================

The first is the substantive one. A SERP is drawn on a single real route, so the
price spread a user actually sees is the spread *within* a route -- not the
spread across routes, which is dominated by SVO->LED vs SVO->HKT and would
calibrate the noise term to nonsense. ``price_sigma`` is therefore fitted so the
simulated within-SERP dispersion matches the measured within-route dispersion,
after subtracting the part contributed by the partner index spread.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd

from src.serp_sim import PARTNERS, SimConfig


@dataclass(frozen=True)
class Anchors:
    """Anchors measured from the real islands. Every field is a measurement."""

    n_orders: int
    n_legs: int
    order_price_median: float
    order_price_q10: float
    order_price_q90: float
    within_route_log_price_sigma: float
    service_attach_rate: float
    service_margin_median: float
    month_share: dict[int, float]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def within_route_price_sigma(legs: pd.DataFrame, min_legs: int = 30) -> float:
    """Leg-weighted mean of ``std(log price)`` measured *within* a route.

    Routes with fewer than ``min_legs`` legs are dropped: their standard
    deviation is noise, not a measurement.
    """
    legs = legs.dropna(subset=["origin", "destination", "price"])
    legs = legs[legs["price"] > 0]
    if legs.empty:
        raise ValueError("no usable legs")
    grouped = legs.assign(log_price=np.log(legs["price"].astype(float))).groupby(
        ["origin", "destination"], sort=True
    )["log_price"]
    sizes = grouped.size()
    keep = sizes[sizes >= min_legs].index
    if len(keep) == 0:
        raise ValueError(f"no route has at least {min_legs} legs")
    sd = grouped.std(ddof=1).loc[keep]
    weights = sizes.loc[keep].astype(float)
    return float((sd * weights).sum() / weights.sum())


def anchors_from_islands(orders: pd.DataFrame, legs: pd.DataFrame, min_legs: int = 30) -> Anchors:
    """Measure the observable anchors off ``orders`` and ``bookings_legs``."""
    price = orders["order_price"].to_numpy(dtype=float)
    price = price[np.isfinite(price) & (price > 0)]
    if price.size == 0:
        raise ValueError("orders carries no usable order_price")

    attach = orders["service_price"].to_numpy(dtype=float) > 0
    margin = orders.loc[attach, "service_profit"].to_numpy(dtype=float) / orders.loc[
        attach, "service_price"
    ].to_numpy(dtype=float)
    margin = margin[np.isfinite(margin)]

    months = legs["booking_month"].to_numpy()
    unique, counts = np.unique(months, return_counts=True)
    total = float(counts.sum())

    return Anchors(
        n_orders=int(len(orders)),
        n_legs=int(len(legs)),
        order_price_median=float(np.median(price)),
        order_price_q10=float(np.quantile(price, 0.10)),
        order_price_q90=float(np.quantile(price, 0.90)),
        within_route_log_price_sigma=within_route_price_sigma(legs, min_legs),
        service_attach_rate=float(attach.mean()),
        service_margin_median=float(np.median(margin)) if margin.size else float("nan"),
        month_share={int(m): float(c / total) for m, c in zip(unique, counts, strict=True)},
    )


def partner_index_sigma(partners: list[dict[str, object]] | None = None) -> float:
    """Std of ``log(price_index)`` across the partner table."""
    table = partners or PARTNERS
    index = np.array([float(p["price_index"]) for p in table], dtype=float)
    if np.any(index <= 0):
        raise ValueError("price_index must be strictly positive")
    return float(np.std(np.log(index), ddof=1))


def anchor_config(
    anchors: Anchors,
    base: SimConfig | None = None,
    partners: list[dict[str, object]] | None = None,
) -> SimConfig:
    """A :class:`SimConfig` whose price noise is fitted to the measured spread.

    ``price_sigma`` is the only free parameter that moves the within-SERP price
    dispersion, so it is solved for: the squared dispersion a SERP shows is
    ``price_sigma**2 + log-index spread**2``, hence the subtraction below.
    Month weights are re-derived from the measured leg shares (they should land
    close to the values hard-coded in ``serp_sim``; the anchored run uses the
    measured ones).
    """
    base = base or SimConfig()
    index_sigma = partner_index_sigma(partners)
    variance = anchors.within_route_log_price_sigma**2 - index_sigma**2
    months = tuple(sorted(m for m, share in anchors.month_share.items() if share > 0))
    if not months:
        raise ValueError("anchors carry no months")
    shares = tuple(anchors.month_share[m] for m in months)
    total = float(sum(shares))
    return replace(
        base,
        price_sigma=float(np.sqrt(max(variance, 1e-6))),
        months=months,
        month_weights=tuple(s / total for s in shares),
    )


def anchor_partners(partners: list[dict[str, object]] | None = None) -> list[dict[str, object]]:
    """Re-centre the partner price indices so the table is neutral on average.

    The absolute price level is anchored on the real per-route
    ``avg_booking_price``; ``price_index`` only expresses *relative* partner
    positioning, so a table whose median index is not 1.0 silently shifts every
    simulated price. Dividing by the median is a positive scaling, so the
    ranking of partners by price and the documented price/money anti-correlation
    both survive untouched.
    """
    table = [dict(p) for p in (partners or PARTNERS)]
    median = float(np.median([float(p["price_index"]) for p in table]))
    if median <= 0:
        raise ValueError("median price_index must be positive")
    for p in table:
        p["price_index"] = float(p["price_index"]) / median
    return table


def build_anchored(
    orders: pd.DataFrame,
    legs: pd.DataFrame,
    base: SimConfig | None = None,
    partners: list[dict[str, object]] | None = None,
    min_legs: int = 30,
) -> tuple[Anchors, SimConfig, list[dict[str, object]]]:
    """One call returning ``(anchors, anchored config, anchored partner table)``."""
    anchors = anchors_from_islands(orders, legs, min_legs=min_legs)
    table = anchor_partners(partners)
    return anchors, anchor_config(anchors, base=base, partners=table), table
