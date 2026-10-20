"""Outcome generation under a counterfactual ranking (PRD section 1.2).

The rows in ``serp_offers`` carry the *logged* order's outcomes: ``clicked``,
``redirected``, ``booked`` and ``commission`` were drawn while the offers sat in
the slots the shipped policy put them in. Reading those columns as the outcome of
a different order is the bug this module removes -- an arm that re-ranks a SERP
but keeps the logged clicks has no treatment at all, and an "A/B test" built on it
compares two random subsets of the same data.

:func:`outcomes_under_policy` therefore re-ranks every SERP with a policy from
:data:`src.policies.POLICIES` (or ``"logged"`` to keep the current order),
recomputes ``position`` (1..n) and ``examine_prob`` for the new slot, and re-draws
the position-based click model with the *same* conditional probabilities the
simulator uses (``src/serp_sim.py:470-499``):

    P(click | slot k)   = P(examine at k) * relevance
    P(redirect | click) = clamp(redirect_rate * (1 - redirect_price_sens * (price/min - 1)))
    P(book | redirect)  = clamp(book_rate * (1 - book_price_sens * (price/min - 1))
                                * month_multiplier[month])

No formula is copied from the simulator: ``examination_curve``, ``eta_for_device``,
``_clamp``, ``revenue_if_engaged`` and :class:`~src.serp_sim.SimConfig` are
imported. ``relevance`` and the per-SERP price ratio are properties of the *offer
set*, not of the order, so they are carried over unchanged -- the treatment is the
slot, not the offer. A policy that re-ranks does not re-price: the same offers,
shown in a different order.

Determinism: one generator per ``(seed, search_id)``, consumed in displayed order,
so the output does not depend on the order of rows in the input frame. Same
``(offers, policy, ctx, seed)`` -> identical frame, row for row.
"""

from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

from src.policies import POLICIES, PolicyContext
from src.serp_sim import (
    SimConfig,
    _clamp,
    eta_for_device,
    examination_curve,
    revenue_if_engaged,
)

#: Special policy name: leave the logged display order alone and only re-draw.
LOGGED_POLICY = "logged"

#: Columns the re-draw needs. ``relevance`` is the simulator's intrinsic click
#: propensity; it is a property of the offer, so it survives a re-rank.
REQUIRED_COLUMNS = (
    "search_id",
    "user_id",
    "position",
    "partner",
    "price",
    "relevance",
    "device",
    "booking_month",
)


def _search_rng(seed: int, search_id: str) -> np.random.Generator:
    """Per-search generator; ``hash()`` is salted per process, ``crc32`` is not."""
    return np.random.default_rng(int(zlib.crc32(f"{seed}|{search_id}".encode())))


def _commission(
    partner_conf: dict[str, object], price: float, *, clicked: bool, booked: bool
) -> float:
    """Realised partner revenue for one row, exactly as ``serp_sim`` books it.

    CPA partners are paid only on a booking; CPC partners are paid on the click,
    booked or not.
    """
    if partner_conf["type"] == "CPA":
        return float(partner_conf["commission_rate"]) * price if booked else 0.0
    return float(partner_conf["cpc"]) if clicked else 0.0


def _policy_order(frame: pd.DataFrame, policy: str, ctx: PolicyContext) -> np.ndarray:
    """Display order (row indices) for one SERP, validated as a permutation."""
    n = len(frame)
    if policy == LOGGED_POLICY:
        return np.arange(n, dtype=int)
    order = np.asarray(POLICIES[policy](frame, ctx), dtype=int)
    if order.shape != (n,) or sorted(order.tolist()) != list(range(n)):
        raise ValueError(f"policy {policy!r} did not return a permutation of {n} offers")
    return order


def outcomes_under_policy(
    offers: pd.DataFrame,
    policy: str,
    ctx: PolicyContext,
    *,
    seed: int,
) -> pd.DataFrame:
    """Re-rank each SERP by ``policy``, then re-draw the PBM outcomes.

    Returns one row per offer, ordered by ``(search_id, position)``, with:
    ``search_id``, ``user_id``, ``position`` (1..n), ``examine_prob`` (recomputed
    for the new slot), ``clicked``, ``redirected``, ``booked``, ``commission``.
    The remaining simulator columns are carried along so the frame stays
    self-consistent (``click_prob`` and ``expected_revenue`` are recomputed too).

    Deterministic for a given ``(policy, seed)``. ``policy="logged"`` keeps the
    current order and only re-draws the outcomes, which is the check that the
    generator reproduces the logged click rate.
    """
    if policy != LOGGED_POLICY and policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; use one of {sorted(POLICIES)} or 'logged'")
    missing = [c for c in REQUIRED_COLUMNS if c not in offers.columns]
    if missing:
        raise ValueError(f"offers is missing required columns: {missing}")
    if offers.empty:
        raise ValueError("offers must not be empty")

    config = SimConfig(eta_desktop=ctx.eta_desktop, eta_mobile=ctx.eta_mobile)
    frames: list[pd.DataFrame] = []

    for search_id, group in offers.groupby("search_id", sort=True):
        # The policy contract: rows in logged display order, re-indexed 0..n-1.
        frame = group.sort_values("position", kind="stable").reset_index(drop=True)
        n = len(frame)

        order = _policy_order(frame, policy, ctx)

        # Slot of every offer under the new order (1-based), then P(examine) there.
        position = np.empty(n, dtype=int)
        position[order] = np.arange(1, n + 1)
        eta = eta_for_device(str(frame["device"].iloc[0]), config)
        examine = examination_curve(n, eta)[position - 1]

        prices = frame["price"].to_numpy(dtype=float)
        price_ratio = prices / prices.min()
        relevance = frame["relevance"].to_numpy(dtype=float)
        month = int(frame["booking_month"].iloc[0])
        month_factor = float(config.month_multiplier[month - 1])
        partner = frame["partner"].astype(str).to_numpy()

        rng = _search_rng(seed, str(search_id))
        clicked = np.zeros(n, dtype=bool)
        redirected = np.zeros(n, dtype=bool)
        booked = np.zeros(n, dtype=bool)
        commission = np.zeros(n, dtype=float)

        # Draw in displayed order: the first offer of the page is drawn first.
        for row in order:
            p_conf = ctx.partner_by_name[partner[row]]
            # ``examine`` is indexed per row, so read the row -- not the loop slot.
            p_click = float(examine[row]) * relevance[row]
            if rng.random() >= p_click:
                continue
            clicked[row] = True

            excess = float(price_ratio[row]) - 1.0
            p_redirect = _clamp(
                float(p_conf["redirect_rate"]) * (1.0 - config.redirect_price_sens * excess)
            )
            if rng.random() < p_redirect:
                redirected[row] = True
                p_book = _clamp(
                    float(p_conf["book_rate"])
                    * (1.0 - config.book_price_sens * excess)
                    * month_factor
                )
                booked[row] = bool(rng.random() < p_book)
            commission[row] = _commission(
                p_conf, float(prices[row]), clicked=True, booked=bool(booked[row])
            )

        out = frame.copy()
        out["position"] = position
        out["examine_prob"] = examine
        out["clicked"] = clicked
        out["redirected"] = redirected
        out["booked"] = booked
        out["commission"] = commission
        out["click_prob"] = examine * relevance
        out["expected_revenue"] = out["click_prob"] * np.array(
            [
                revenue_if_engaged(p, pr, ctx.partner_by_name)
                for p, pr in zip(partner, prices, strict=True)
            ],
            dtype=float,
        )
        frames.append(out)

    result = pd.concat(frames, ignore_index=True)
    return result.sort_values(["search_id", "position"], kind="stable").reset_index(drop=True)
