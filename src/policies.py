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
``production``               no              the logged (shipped) display order
``price_asc``                no              trivial realistic baseline
``click_trained``            no              logistic click model on raw logs
``propensity_logistic``      no              same model, IPW-weighted logs
``ctr_objective``            no              feasible: rank by fitted click prob
``revenue_objective``        no              feasible: fitted click prob x money
===========================  ==============  ==================================

``click_trained`` / ``propensity_logistic`` share one fitter and differ only in
the sample weights, which makes them a controlled pair: the only thing that
changes is whether the position bias in the training signal is corrected.

``ctr_objective`` / ``revenue_objective`` are the *feasible pair* the headline
compares. They are deliberately built as one experiment with two objectives:

* **one training set** -- the IPW-corrected fit (``propensity_logistic``), the
  bias-corrected one; both policies read the *same* ``beta``;
* **one feature matrix** -- :func:`click_features`, i.e. ``[1, log price ratio,
  is_cpc]``, all of it observable before a slot is assigned;
* **one slot mechanism** -- :func:`_greedy_slot_assignment`, the same
  ``examine_prob(k)`` greedy used by :func:`revenue`.

So the only difference between them is the utility: ``p_hat`` versus
``p_hat * revenue_if_engaged``. Neither reads ``relevance``, ``click_prob`` or
``expected_revenue``; the money term comes from the partner *contract*
(rate / cpc / redirect / book rates, see
:func:`src.serp_sim.revenue_if_engaged`). Any revenue gap between them is
therefore attributable to the objective, not to privileged information.

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


def _greedy_slot_assignment(
    frame: pd.DataFrame, ctx: PolicyContext, utility: np.ndarray
) -> np.ndarray:
    """Greedy slot assignment: for slot ``k`` pick ``argmax examine_prob(k) * utility``.

    This is *the* slot mechanism of the module: :func:`revenue`,
    :func:`ctr_objective` and :func:`revenue_objective` all go through it, so
    they differ in the utility only -- never in how attention decay is handled.
    Deterministic: ties break on the lowest row index.
    """
    eta = ctx.eta(str(frame["device"].iloc[0]))
    utility = np.asarray(utility, dtype=float)
    remaining = list(range(len(frame)))
    order: list[int] = []
    for pos in range(1, len(frame) + 1):
        e = examination_prob(pos, eta)
        best_i, best_v = -1, -np.inf
        for i in remaining:
            v = e * float(utility[i])
            if best_i < 0 or v > best_v + 1e-12:
                best_i, best_v = i, v
        remaining.remove(best_i)
        order.append(best_i)
    return np.array(order, dtype=int)


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
    rel = frame["relevance"].to_numpy(dtype=float)
    return _greedy_slot_assignment(frame, ctx, rel * rev_engaged(frame, ctx))


# --------------------------------------------------------------------------- #
# Realistic policies -- no access to ``relevance``
# --------------------------------------------------------------------------- #


def price_asc(frame: pd.DataFrame, ctx: PolicyContext) -> np.ndarray:
    """Cheapest offer first. The baseline a naive team ships."""
    return np.lexsort((np.arange(len(frame)), frame["price"].to_numpy(dtype=float)))


def production(frame: pd.DataFrame, ctx: PolicyContext) -> np.ndarray:
    """The shipped display order: the identity permutation of the logged SERP.

    ``serp_offers.position`` *is* the money-tilted blend the product shipped, so
    re-displaying it unchanged is the "do nothing" arm. Consults no column at
    all -- it is the cleanest possible ``logged`` information set.

    Caveat worth stating wherever this arm is quoted: the blend that produced the
    logged order was built *from the simulator's* ``relevance``. Re-displaying it
    needs no truth, so the arm is legitimate; but its quality is lent by a
    modelling input a real product would not have, so "the new ranker loses to
    production" is partly an artefact of the simulation, not a production fact.
    """
    return np.arange(len(frame), dtype=int)


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
# The feasible pair -- one fit, one feature matrix, one slot mechanism
# --------------------------------------------------------------------------- #

#: The one fitted model both feasible policies are built on. Chosen deliberately:
#: it is the IPW-corrected fit (sample weight ``1 / examine_prob``), i.e. the one
#: whose price coefficient is not confounded by position bias. Using the naive
#: fit would import a known bias into a "what would shipping this cost" number.
FITTED_MODEL = "propensity_logistic"


def fitted_click_probability(frame: pd.DataFrame, ctx: PolicyContext) -> np.ndarray:
    """``p_hat`` -- the fitted click probability of each offer, position excluded.

    Same matrix as both feasible policies and as the trained ranker
    (:func:`click_features`), same ``beta`` and therefore one shared number per
    offer. If the coefficients are missing this raises instead of silently
    falling back to a different model.
    """
    beta = ctx.params.get(FITTED_MODEL)
    if beta is None:
        raise ValueError(
            f"ctx.params['{FITTED_MODEL}'] is missing; run fit_click_models() "
            "and build the context with context_for(..., train=train)"
        )
    z = click_features(frame) @ np.asarray(beta, dtype=float)
    return np.asarray(1.0 / (1.0 + np.exp(-z)), dtype=float)


def ctr_objective(frame: pd.DataFrame, ctx: PolicyContext) -> np.ndarray:
    """FEASIBLE: rank by ``p_hat`` -- maximise clicks, ignore the money.

    Reads ``partner_type`` and ``price`` only (through :func:`click_features`).
    The slot mechanism is :func:`_greedy_slot_assignment`, identical to
    :func:`revenue`.

    Because ``examine_prob(k)`` is common to every candidate within a slot, this
    greedy reduces to sorting by ``p_hat`` -- and ``p_hat`` is a monotone function
    of the fitted linear score, so the resulting order coincides with
    :func:`propensity_logistic`. That is an identity, not a duplication: the CTR
    objective *is* the bias-corrected learned ranker, now stated with the same slot
    mechanism as the money objective it is compared against.
    """
    return _greedy_slot_assignment(frame, ctx, fitted_click_probability(frame, ctx))


def revenue_objective(frame: pd.DataFrame, ctx: PolicyContext) -> np.ndarray:
    """FEASIBLE: rank by ``p_hat * revenue_if_engaged`` -- maximise money.

    The money term is the *contract* value of an engaged click
    (:func:`rev_engaged`: CPA ``rate * price * redirect * book``, CPC ``cpc``),
    not the simulator's ``expected_revenue``. Identical ``X``, identical
    ``beta`` and identical slot mechanism as :func:`ctr_objective`; the utility
    is the only difference, which is what makes the revenue/CTR trade-off
    attributable to the objective.
    """
    utility = fitted_click_probability(frame, ctx) * rev_engaged(frame, ctx)
    return _greedy_slot_assignment(frame, ctx, utility)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

POLICIES: dict[str, Callable[[pd.DataFrame, PolicyContext], np.ndarray]] = {
    "relevance": relevance,
    "revenue": revenue,
    "production": production,
    "price_asc": price_asc,
    "click_trained": click_trained,
    "propensity_logistic": propensity_logistic,
    "ctr_objective": ctr_objective,
    "revenue_objective": revenue_objective,
}

#: What each policy is allowed to know. ``oracle`` = built from the simulator's
#: ground truth (``relevance`` / ``expected_revenue``) and therefore unusable as
#: a headline; ``logged`` = only columns a production ranker has at serving time.
INFORMATION_SET: dict[str, str] = {
    "relevance": "oracle",
    "revenue": "oracle",
    "production": "logged",
    "price_asc": "logged",
    "click_trained": "logged",
    "propensity_logistic": "logged",
    "ctr_objective": "logged",
    "revenue_objective": "logged",
}

# The two policies the headline compares: same information, different objective.
FEASIBLE_POLICIES = ("ctr_objective", "revenue_objective")
#: The reference whose number is the "ceiling" appendix (+38% in the old README).
ORACLE_POLICIES = ("relevance", "revenue")

# The oracle pair. ``compare_rankings`` defaults to these so a caller with a
# bare offer frame still gets the Phase 2 headline; the realistic policies need
# ``partner_type`` / ``examine_prob`` / a fitted model, so they are opt-in.
# ``DEFAULT_POLICIES`` stays on the oracles on purpose: it is what the
# "ceiling (modeled)" appendix reproduces, never the headline.
DEFAULT_POLICIES = ORACLE_POLICIES
REALISTIC_POLICIES = ("production", "price_asc", "click_trained", "propensity_logistic")
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
