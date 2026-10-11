"""Ranking policies for SERP Rank Lab (Phase 4).

Phase 2 compared exactly two rankings: the relevance *oracle* and the
revenue-greedy optimum. Both are references -- they are built from the
simulator's own ground truth -- so the headline "offline relevance and money
diverge" could be waved away as "you compared the ideal order with the greedy
one". This module adds policies that a real team could actually ship, so the
divergence can be re-checked on realistic rankers.

Policy contract
---------------
A policy is ``policy(frame, ctx) -> np.ndarray`` where ``frame`` is one search's
offers **in logged display order** with a 0..n-1 index, and the return value is
the row indices in the order the policy wants to display them. Policies never
mutate the frame.

What each policy is allowed to see
----------------------------------
``relevance`` is the simulator's ground-truth click propensity. Only the two
*reference* policies may read it. The realistic ones must not -- that is the
whole point -- and ``tests/test_policies.py`` pins it.

===========================  ==============  ==================================
policy                       sees relevance  role
===========================  ==============  ==================================
``relevance``                yes             oracle reference (NDCG = 1.0)
``revenue``                  yes             revenue-optimal reference
``price_asc``                no              trivial realistic baseline
``click_trained``            no              logistic click model on raw logs
``propensity_logistic``      no              same model, IPW-weighted logs
===========================  ==============  ==================================

The last two share one fitter and differ only in the sample weights, which makes
them a controlled pair: the only thing that changes is whether the position bias
in the training signal is corrected.

No scikit-learn: the fit is L-BFGS on a weighted Bernoulli log-likelihood via
``scipy.optimize``, ~30 lines.
"""

from __future__ import annotations

import zlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.serp_sim import (
    ETA_DESKTOP,
    ETA_MOBILE,
    PARTNER_BY_NAME,
    examination_prob,
    revenue_if_engaged,
)


@dataclass(frozen=True)
class PolicyContext:
    """Everything a policy may consult that is not the offer row itself."""

    eta_desktop: float = ETA_DESKTOP
    eta_mobile: float = ETA_MOBILE
    seed: int = 0
    partner_by_name: dict[str, dict[str, object]] = field(default_factory=lambda: PARTNER_BY_NAME)
    params: dict[str, object] = field(default_factory=dict)

    def eta(self, device: str) -> float:
        """Examination decay for a device label."""
        return self.eta_mobile if device == "mobile" else self.eta_desktop


def _order(scores: np.ndarray) -> np.ndarray:
    """Descending order of ``scores`` with a stable, index-based tie-break."""
    return np.lexsort((np.arange(len(scores)), -np.asarray(scores, dtype=float)))


def rev_engaged(frame: pd.DataFrame, ctx: PolicyContext) -> np.ndarray:
    """Per-row ``revenue_if_engaged`` under the context's partner table."""
    return np.array(
        [
            revenue_if_engaged(str(p), float(pr), ctx.partner_by_name)
            for p, pr in zip(frame["partner"], frame["price"], strict=True)
        ],
        dtype=float,
    )


def _search_seed(frame: pd.DataFrame, ctx: PolicyContext) -> int:
    """Deterministic per-search seed (``hash()`` is salted per process)."""
    sid = str(frame["search_id"].iloc[0]) if "search_id" in frame else ""
    return int(zlib.crc32(f"{ctx.seed}|{sid}".encode()))


# --------------------------------------------------------------------------- #
# Reference policies -- built from the simulator's ground truth
# --------------------------------------------------------------------------- #


def relevance(frame: pd.DataFrame, ctx: PolicyContext) -> np.ndarray:
    """ORACLE: sort by intrinsic relevance. ``NDCG@k`` is 1.0 by construction."""
    return _order(frame["relevance"].to_numpy(dtype=float))


def revenue(frame: pd.DataFrame, ctx: PolicyContext) -> np.ndarray:
    """ORACLE: greedy slot assignment maximising expected revenue.

    For slot ``k`` pick the offer maximising ``P(examine at k) * relevance *
    revenue_if_engaged``. Deterministic: ties break on the row index.
    """
    eta = ctx.eta(str(frame["device"].iloc[0]))
    rel = frame["relevance"].to_numpy(dtype=float)
    rev = rev_engaged(frame, ctx)
    remaining = list(range(len(frame)))
    order: list[int] = []
    for pos in range(1, len(frame) + 1):
        e = examination_prob(pos, eta)
        best_i, best_v = -1, -1.0
        for i in remaining:
            v = e * rel[i] * rev[i]
            if best_i < 0 or v > best_v + 1e-12:
                best_i, best_v = i, v
        remaining.remove(best_i)
        order.append(best_i)
    return np.array(order)


# --------------------------------------------------------------------------- #
# Realistic policies -- no access to ``relevance``
# --------------------------------------------------------------------------- #


def price_asc(frame: pd.DataFrame, ctx: PolicyContext) -> np.ndarray:
    """Cheapest offer first. The baseline a naive team ships."""
    return np.lexsort((np.arange(len(frame)), frame["price"].to_numpy(dtype=float)))


def click_features(frame: pd.DataFrame) -> np.ndarray:
    """Serving features of the click model: ``[1, log price ratio, is_cpc]``.

    Deliberately excludes the position: a production ranker scores candidates
    before they have a slot. The bias that leaves behind is what
    :func:`fit_click_models` documents.
    """
    price = frame["price"].to_numpy(dtype=float)
    ratio = np.log(price / price.min())
    is_cpc = (frame["partner_type"].to_numpy() == "CPC").astype(float)
    return np.column_stack([np.ones(len(frame)), ratio, is_cpc])


def _stacked(frame: pd.DataFrame, ctx: PolicyContext, with_position: bool) -> np.ndarray:
    x = click_features(frame)
    if with_position:
        pos = frame["position"].to_numpy(dtype=float)
        x = np.column_stack([x, np.log(pos)])
    return x


def fit_click_models(
    frames: Sequence[pd.DataFrame],
    ctx: PolicyContext | None = None,
    l2: float = 1.0,
    max_iter: int = 300,
) -> dict[str, np.ndarray]:
    """Fit the logistic click model on logs, with and without IPW correction.

    Both fits use the *same* rows and the *same* features; the only difference is
    the sample weight. The position enters as a training column so the fit can
    absorb the examination bias -- a click model that has never seen position
    would be a straw man, and would misattribute the top-slot clicks to price.

    * ``click_trained``       -- weight 1 per impression: the naive production fit.
      The fitted price coefficient is confounded by position bias.
    * ``propensity_logistic`` -- weight ``1 / examine_prob``: Horvitz-Thompson
      weighted, so the price coefficient is estimated as if every slot were seen.

    Returns serving-time coefficients only (``[1, log price ratio, is_cpc]``).
    """
    ctx = ctx or PolicyContext()
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ws: list[np.ndarray] = []
    for frame in frames:
        xs.append(_stacked(frame, ctx, with_position=True))
        ys.append(frame["clicked"].to_numpy(dtype=float))
        ws.append(1.0 / frame["examine_prob"].to_numpy(dtype=float))
    x = np.vstack(xs)
    y = np.concatenate(ys)
    w_ipw = np.concatenate(ws)

    def weights(ipw: bool) -> np.ndarray:
        return w_ipw if ipw else np.ones_like(w_ipw)

    def fit_one(ipw: bool) -> np.ndarray:
        w = weights(ipw)

        def objective(beta: np.ndarray) -> float:
            z = x @ beta
            return -float(
                np.sum(w * (y * z - np.logaddexp(0.0, z))) - 0.5 * l2 * beta[1:] @ beta[1:]
            )

        def gradient(beta: np.ndarray) -> np.ndarray:
            p = 1.0 / (1.0 + np.exp(-(x @ beta)))
            g = x.T @ (w * (p - y))
            g[1:] += l2 * beta[1:]
            return g

        result = minimize(
            objective,
            np.zeros(x.shape[1]),
            jac=gradient,
            method="L-BFGS-B",
            options={"maxiter": max_iter},
        )
        # Serving features are the first three columns; position is dropped.
        return np.asarray(result.x[:3], dtype=float)

    return {"click_trained": fit_one(False), "propensity_logistic": fit_one(True)}


def click_trained(frame: pd.DataFrame, ctx: PolicyContext) -> np.ndarray:
    """Rank by the click model fitted on raw (position-biased) logs."""
    beta = ctx.params.get("click_trained")
    if beta is None:
        raise ValueError("ctx.params['click_trained'] is missing; run fit_click_models()")
    return _order(click_features(frame) @ np.asarray(beta, dtype=float))


def propensity_logistic(frame: pd.DataFrame, ctx: PolicyContext) -> np.ndarray:
    """Rank by the click model fitted on IPW-weighted logs (bias corrected)."""
    beta = ctx.params.get("propensity_logistic")
    if beta is None:
        raise ValueError("ctx.params['propensity_logistic'] is missing; run fit_click_models()")
    return _order(click_features(frame) @ np.asarray(beta, dtype=float))


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

POLICIES: dict[str, Callable[[pd.DataFrame, PolicyContext], np.ndarray]] = {
    "relevance": relevance,
    "revenue": revenue,
    "price_asc": price_asc,
    "click_trained": click_trained,
    "propensity_logistic": propensity_logistic,
}

# The oracle pair. ``compare_rankings`` defaults to these so a caller with a
# bare offer frame still gets the Phase 2 headline; the realistic policies need
# ``partner_type`` / ``examine_prob`` / a fitted model, so they are opt-in.
DEFAULT_POLICIES = ("relevance", "revenue")
REALISTIC_POLICIES = ("price_asc", "click_trained", "propensity_logistic")
DELTA_POLICY = "delta_relevance_minus_revenue"


def train_test_split_searches(
    offers_df: pd.DataFrame, test_frac: float = 0.5, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split offers into disjoint train/test halves **by ``search_id``**.

    A click model must never be fitted and scored on the same search, so the
    split is on searches, not rows: ``set(train.search_id) & set(test.search_id)``
    is empty. Deterministic for a given seed.
    """
    searches = np.array(sorted(offers_df["search_id"].unique()))
    if len(searches) < 2:
        raise ValueError("need at least two searches to split")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(searches)
    n_test = max(1, int(round(len(searches) * test_frac)))
    test_ids = set(shuffled[:n_test].tolist())
    is_test = offers_df["search_id"].isin(test_ids)
    return offers_df[~is_test].copy(), offers_df[is_test].copy()


def context_for(
    offers_df: pd.DataFrame,
    train: pd.DataFrame | None = None,
    seed: int = 0,
    partner_by_name: dict[str, dict[str, object]] | None = None,
) -> PolicyContext:
    """Build a :class:`PolicyContext`, fitting the click models on ``train``."""
    ctx = PolicyContext(
        seed=seed,
        partner_by_name=partner_by_name or PARTNER_BY_NAME,
        params={},
    )
    if train is not None:
        frames = [g for _, g in train.groupby("search_id", sort=True)]
        params = fit_click_models(frames, ctx)
        ctx = replace(ctx, params=params)
    return ctx
