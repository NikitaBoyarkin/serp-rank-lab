"""Ranking evaluation for SERP Rank Lab (Phase 2).

Implements the offline and online measurement of a ranking policy and the
project's headline demonstration: **a ranking that improves offline relevance
(NDCG) can lose money (revenue per search)**.

Metric definitions (exact formulas)
-----------------------------------

**Observed (naive) CTR** over a set of impressions::

    naive_ctr = mean(clicked)

This is biased low whenever clicks are logged behind a position-biased display,
because offers sitting below the first slot are examined less often.

**Unbiased CTR via inverse propensity weighting (IPW)** -- Horvitz-Thompson::

    unbiased_ctr = mean( clicked_i / propensity(position_i) )

where ``propensity(k) = P(examine at k)`` from the PBM. Correcting each click by
the inverse of its examination probability re-weights down-position clicks back
to what they would have been under full examination, recovering the
position-free click rate. Using a *uniform* propensity (the classic bug) makes
``unbiased_ctr`` collapse back to ``naive_ctr`` and leaves the bias in place.

**Propensity recovery from logs** (EM, the standard PBM alternation) assumes the
product model ``P(click | impression at position k) = e_k * r_g`` and alternates::

    e_k = sum(clicks at k) / sum(r_g over impressions at k)
    r_g = sum(clicks for group g) / sum(e_{position} for group g)

where ``g`` is a *repeated-document group* (here the partner). Grouping is what
makes the fit identified -- with a single Bernoulli observation per offer the
model is degenerate. ``e_1`` is fixed to 1 to pin the scale (the model is only
identified up to ``e * c, r / c``).

**NDCG@k** (linear gain = relevance, log2 discount)::

    DCG@k  = sum_{i=1..k} rel_i / log2(i + 1)
    IDCG@k = same, with rel sorted descending
    NDCG@k = DCG@k / IDCG@k

**MRR**: ``1 / rank`` of the first item with ``relevance >= threshold``, else 0.

**Revenue per search** = mean over searches of the realised partner revenue
(CPA commission on bookings + CPC fee on clicks), i.e. ``sum(commission)``.

**Expected revenue per search** under an alternative ranking is computed
deterministically from the same click model::

    E[revenue] = sum_k P(examine at k) * relevance_at_k * revenue_if_engaged_at_k

where ``revenue_if_engaged`` is the CPA commission on a booking or the CPC fee
on a click. Revenue-aware ranking uses a greedy slot assignment: for position
1..n pick the remaining offer maximising
``P(examine at k) * relevance * revenue_if_engaged``.

CLI: ``python -m src.ranking_eval`` reads ``data/processed/serp_offers.parquet``
and writes ``reports/ranking_eval.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.serp_sim import (
    ETA_DESKTOP,
    ETA_MOBILE,
    SERP_OFFERS_PARQUET,
    examination_prob,
    revenue_if_engaged,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORT_JSON = REPORTS_DIR / "ranking_eval.json"


# --------------------------------------------------------------------------- #
# CTR estimators
# --------------------------------------------------------------------------- #


def naive_ctr(clicked: np.ndarray | list[bool]) -> float:
    """``mean(clicked)`` -- the position-biased observed click rate."""
    return float(np.mean(np.asarray(clicked, dtype=float)))


def unbiased_ctr(
    clicked: np.ndarray | list[bool],
    positions: np.ndarray | list[int] | None = None,
    propensity: dict[int, float] | np.ndarray | list[float] | None = None,
    eta: float | None = None,
) -> float:
    """Inverse-propensity-corrected click rate ``mean(clicked / propensity(k))``.

    ``propensity`` may be a mapping ``{position: P(examine)}`` (e.g. estimated
    from logs), a per-row array of propensities (e.g. one curve per device), or
    omit it and pass an ``eta`` to use the controlled ``(1/k)**eta`` curve.
    """
    clicked_arr = np.asarray(clicked, dtype=float)
    if propensity is not None:
        if isinstance(propensity, dict):
            if positions is None:
                raise ValueError("positions are required with a position-keyed propensity")
            p = np.array([propensity[int(k)] for k in positions], dtype=float)
        else:
            p = np.asarray(propensity, dtype=float)
            if p.shape != clicked_arr.shape:
                raise ValueError("per-row propensity must match clicked in length")
    elif eta is not None:
        if positions is None:
            raise ValueError("positions are required with an eta")
        p = np.array([examination_prob(int(k), eta) for k in positions], dtype=float)
    else:
        raise ValueError("provide a propensity (mapping or array) or an eta")
    if np.any(p <= 0):
        raise ValueError("propensity must be strictly positive")
    return float(np.mean(clicked_arr / p))


def estimate_propensity_em(
    clicked: np.ndarray | list[bool],
    positions: np.ndarray | list[int],
    groups: np.ndarray | list,
    n_iter: int = 200,
    tol: float = 1e-10,
) -> dict[int, float]:
    """Estimate the examination curve ``{position: e_k}`` from logged clicks.

    Alternating PBM fit of ``P(click) = e_k * r_g`` on the product model, where
    ``r_g`` is a *relevance shared by the repeated-document group* ``groups``
    (e.g. the partner, which recurs across positions and searches). Grouping is
    what makes the fit identified: with one Bernoulli observation per offer the
    model is degenerate and EM collapses to ``e_k = 1``. The curve is only
    identified up to scale, so ``e_1`` is pinned to 1.
    """
    clicked = np.asarray(clicked, dtype=float)
    positions = np.asarray(positions, dtype=int)
    groups = np.asarray(groups)
    if clicked.size == 0:
        raise ValueError("no impressions to estimate from")
    k_max = int(positions.max())
    uniq = np.unique(groups)
    group_index = {g: i for i, g in enumerate(uniq)}
    gi = np.array([group_index[g] for g in groups], dtype=int)

    r = np.array([clicked[gi == j].mean() for j in range(len(uniq))], dtype=float)
    r = np.clip(r, 1e-3, 1.0)
    e = np.ones(k_max + 1, dtype=float)
    for _ in range(n_iter):
        previous = e.copy()
        for k in range(1, k_max + 1):
            mask = positions == k
            denom = float(r[gi[mask]].sum())
            e[k] = float(clicked[mask].sum()) / denom if denom > 0 else previous[k]
        e = np.clip(e, 1e-6, 1.0)
        e[1] = 1.0
        for j in range(len(uniq)):
            mask = gi == j
            e_sum = float(e[positions[mask]].sum())
            r[j] = float(clicked[mask].sum()) / e_sum if e_sum > 0 else r[j]
        r = np.clip(r, 1e-6, 1.0)
        if float(np.max(np.abs(e - previous))) < tol:
            break
    return {k: float(e[k]) for k in range(1, k_max + 1)}


# --------------------------------------------------------------------------- #
# Offline relevance metrics
# --------------------------------------------------------------------------- #


def ndcg_at_k(relevances: np.ndarray | list[float], k: int) -> float:
    """NDCG@k with linear gain and log2 discount (see module docstring)."""
    rel = np.asarray(relevances, dtype=float)
    if rel.size == 0 or k < 1:
        return 0.0
    top = rel[:k]
    dcg = float(np.sum(top / np.log2(np.arange(2, top.size + 2))))
    ideal = np.sort(rel)[::-1][:k]
    idcg = float(np.sum(ideal / np.log2(np.arange(2, ideal.size + 2))))
    return dcg / idcg if idcg > 0 else 0.0


def mrr(relevances: np.ndarray | list[float], threshold: float = 0.5) -> float:
    """Reciprocal rank of the first item with ``relevance >= threshold``."""
    for rank, rel in enumerate(np.asarray(relevances, dtype=float), start=1):
        if rel >= threshold:
            return 1.0 / rank
    return 0.0


# --------------------------------------------------------------------------- #
# Ranking policies
# --------------------------------------------------------------------------- #


def _greedy_revenue_ranking(offers: list[dict], eta: float) -> list[dict]:
    """Greedy slot assignment maximising expected revenue (deterministic ties)."""
    remaining = sorted(offers, key=lambda o: o["key"])
    ranked: list[dict] = []
    for pos in range(1, len(offers) + 1):
        e = examination_prob(pos, eta)
        best: dict | None = None
        best_value = -1.0
        for offer in remaining:
            value = e * offer["relevance"] * offer["rev_engaged"]
            if best is None or value > best_value + 1e-12:
                best, best_value = offer, value
        assert best is not None
        remaining.remove(best)
        ranked.append(best)
    return ranked


def _ranking_expected_metrics(ranked: list[dict], eta: float) -> tuple[float, float]:
    """(expected revenue, expected clicks) for a ranked list under the PBM."""
    revenue = 0.0
    clicks = 0.0
    for pos, offer in enumerate(ranked, start=1):
        p_click = examination_prob(pos, eta) * offer["relevance"]
        clicks += p_click
        revenue += p_click * offer["rev_engaged"]
    return revenue, clicks


def _search_offers(frame: pd.DataFrame) -> list[dict]:
    """Turn one search's rows into offer dicts with intrinsic relevance."""
    ordered = frame.sort_values("position")
    offers = []
    for key, (rel, partner, price) in enumerate(
        zip(ordered["relevance"], ordered["partner"], ordered["price"], strict=True)
    ):
        offers.append(
            {
                "key": key,
                "relevance": float(rel),
                "rev_engaged": revenue_if_engaged(str(partner), float(price)),
            }
        )
    return offers


def compare_rankings(offers_df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """Headline comparison: relevance-optimal vs revenue-optimal ranking.

    Both policies re-rank the *same* offer set within each search. NDCG@k is the
    offline metric; expected revenue per search is the online metric. Returns a
    tidy frame with one row per policy plus a ``delta_relevance_minus_revenue``
    row (relevance minus revenue).
    """
    totals = {
        "relevance": {"ndcg": 0.0, "revenue": 0.0, "clicks": 0.0},
        "revenue": {"ndcg": 0.0, "revenue": 0.0, "clicks": 0.0},
    }
    n_searches = 0
    for _, frame in offers_df.groupby("search_id", sort=True):
        device = str(frame["device"].iloc[0])
        eta = ETA_MOBILE if device == "mobile" else ETA_DESKTOP
        offers = _search_offers(frame)

        relevance_ranked = sorted(offers, key=lambda o: (-o["relevance"], o["key"]))
        revenue_ranked = _greedy_revenue_ranking(offers, eta)

        for name, ranked in (("relevance", relevance_ranked), ("revenue", revenue_ranked)):
            totals[name]["ndcg"] += ndcg_at_k([o["relevance"] for o in ranked], k)
            revenue, clicks = _ranking_expected_metrics(ranked, eta)
            totals[name]["revenue"] += revenue
            totals[name]["clicks"] += clicks
        n_searches += 1

    if n_searches == 0:
        raise ValueError("offers_df must contain at least one search")

    rows = []
    for name in ("relevance", "revenue"):
        rows.append(
            {
                "policy": name,
                "ndcg_at_k": totals[name]["ndcg"] / n_searches,
                "revenue_per_search": totals[name]["revenue"] / n_searches,
                "clicks_per_search": totals[name]["clicks"] / n_searches,
            }
        )
    rows.append(
        {
            "policy": "delta_relevance_minus_revenue",
            "ndcg_at_k": totals["relevance"]["ndcg"] / n_searches
            - totals["revenue"]["ndcg"] / n_searches,
            "revenue_per_search": totals["relevance"]["revenue"] / n_searches
            - totals["revenue"]["revenue"] / n_searches,
            "clicks_per_search": totals["relevance"]["clicks"] / n_searches
            - totals["revenue"]["clicks"] / n_searches,
        }
    )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Team-draft interleaving
# --------------------------------------------------------------------------- #


def team_draft_interleaving(
    rank_a: list, rank_b: list, rng: np.random.Generator
) -> list[tuple[object, str]]:
    """Team-draft interleaving of two rankings over the same offer set.

    Each round both teams contribute their top remaining item, in a random order,
    so each team gets equal exposure. A drafted offer leaves BOTH rankers' remaining
    lists, so every offer is shown exactly once. Returns ``[(offer, "A"|"B"), ...]``
    in display order.
    """
    if set(rank_a) != set(rank_b):
        raise ValueError("both rankings must cover the same offer set")
    a, b = list(rank_a), list(rank_b)
    interleaved: list[tuple[object, str]] = []
    while a or b:
        teams = []
        if a:
            teams.append("A")
        if b:
            teams.append("B")
        rng.shuffle(teams)
        for team in teams:
            source = a if team == "A" else b
            if not source:
                continue
            item = source.pop(0)
            # The drafted offer is already placed, so drop it from the other
            # ranker too: no offer is displayed twice.
            if item in a:
                a.remove(item)
            if item in b:
                b.remove(item)
            interleaved.append((item, team))
    return interleaved


def team_draft_preference(
    interleaved: list[tuple[object, str]],
    relevance: dict,
    eta: float,
    rng: np.random.Generator,
) -> str:
    """Simulate one user on an interleaved list; credit clicks to the source team."""
    credits = {"A": 0, "B": 0}
    for pos, (item, team) in enumerate(interleaved, start=1):
        p_click = examination_prob(pos, eta) * float(relevance[item])
        if rng.random() < p_click:
            credits[team] += 1
    if credits["A"] > credits["B"]:
        return "A"
    if credits["B"] > credits["A"]:
        return "B"
    return "tie"


def interleaving_experiment(
    rank_a: list,
    rank_b: list,
    relevance: dict,
    eta: float,
    n_users: int = 4000,
    seed: int = 7,
) -> dict:
    """Aggregate team-draft interleaving over ``n_users`` simulated users.

    Returns win counts, win rates and the decision (A wins if it collects more
    user wins than B).
    """
    rng = np.random.default_rng(seed)
    wins = {"A": 0, "B": 0, "tie": 0}
    for _ in range(n_users):
        interleaved = team_draft_interleaving(rank_a, rank_b, rng)
        wins[team_draft_preference(interleaved, relevance, eta, rng)] += 1
    decision = "A" if wins["A"] > wins["B"] else ("B" if wins["B"] > wins["A"] else "tie")
    return {
        "users": n_users,
        "wins": wins,
        "win_rate_a": wins["A"] / n_users,
        "win_rate_b": wins["B"] / n_users,
        "decision": decision,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _to_native(obj):
    """Recursively convert numpy scalars/arrays to JSON-native Python types."""
    if isinstance(obj, dict):
        return {str(k): _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.ranking_eval",
        description="Evaluate ranking policies over serp_offers and write reports/ranking_eval.json",
    )
    parser.add_argument("--k", type=int, default=10, help="NDCG cutoff (default 10)")
    args = parser.parse_args(argv)

    if not SERP_OFFERS_PARQUET.exists():
        raise FileNotFoundError(
            f"Missing {SERP_OFFERS_PARQUET}. Run `python -m src.serp_sim` first."
        )
    offers = pd.read_parquet(SERP_OFFERS_PARQUET)

    print("=== ranking_eval ===")
    print(f"source:    {SERP_OFFERS_PARQUET}")
    print(f"offers:    {len(offers):,}   searches: {offers['search_id'].nunique():,}")
    print(f"NDCG cutoff k = {args.k}\n")

    summary = compare_rankings(offers, k=args.k)
    logged_revenue = float(offers.groupby("search_id")["commission"].sum().mean())

    # --- IPW / propensity demonstration on the logged clicks ---
    clicked = offers["clicked"].to_numpy()
    positions = offers["position"].to_numpy()
    eta_rows = np.where(offers["device"].to_numpy() == "mobile", ETA_MOBILE, ETA_DESKTOP)
    propensity_rows = (1.0 / positions) ** eta_rows  # controlled per-device curve
    naive = naive_ctr(clicked)
    corrected_known = unbiased_ctr(clicked, propensity=propensity_rows)
    estimated = estimate_propensity_em(clicked, positions, offers["partner"].to_numpy())
    corrected_em = unbiased_ctr(clicked, positions, propensity=estimated)
    true_rel = float(offers["relevance"].mean())

    # --- interleaving decision vs the shipped revenue-leaning display ---
    first_search = offers["search_id"].iloc[0]
    frame = offers[offers["search_id"] == first_search]
    device = str(frame["device"].iloc[0])
    eta_dev = ETA_MOBILE if device == "mobile" else ETA_DESKTOP
    keys = list(range(len(frame)))
    rel_map = {i: float(r) for i, r in enumerate(frame.sort_values("position")["relevance"])}
    by_rel = sorted(keys, key=lambda i: (-rel_map[i], i))
    offer_dicts = _search_offers(frame)
    by_rev = [o["key"] for o in _greedy_revenue_ranking(offer_dicts, eta_dev)]
    inter = interleaving_experiment(by_rel, by_rev, rel_map, eta_dev)

    divergence = {
        "ndcg_k": args.k,
        "ndcg_relevance": float(summary.loc[0, "ndcg_at_k"]),
        "ndcg_revenue": float(summary.loc[1, "ndcg_at_k"]),
        "revenue_per_search_relevance": float(summary.loc[0, "revenue_per_search"]),
        "revenue_per_search_revenue": float(summary.loc[1, "revenue_per_search"]),
        "ndcg_delta": float(summary.loc[2, "ndcg_at_k"]),
        "revenue_delta": float(summary.loc[2, "revenue_per_search"]),
        "ndcg_rises_while_revenue_falls": bool(
            summary.loc[2, "ndcg_at_k"] > 0 and summary.loc[2, "revenue_per_search"] < 0
        ),
    }

    report = {
        "k": args.k,
        "offers": int(len(offers)),
        "searches": int(offers["search_id"].nunique()),
        "logged_revenue_per_search": logged_revenue,
        "policies": summary.to_dict(orient="records"),
        "divergence": divergence,
        "ctr": {
            "naive": naive,
            "true_position_free": true_rel,
            "ipw_known_curve": corrected_known,
            "ipw_em_estimated": corrected_em,
            "eta_mobile": ETA_MOBILE,
            "eta_desktop": ETA_DESKTOP,
        },
        "propensity_estimated": estimated,
        "interleaving_example": inter,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(_to_native(report), indent=2), encoding="utf-8")

    print(summary.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
    print()
    print("--- CTR: is the observed click rate biased? ---")
    print(f"naive observed CTR:        {naive:.4f}  (biased low by position bias)")
    print(f"IPW-corrected (known curve): {corrected_known:.4f}")
    print(f"IPW-corrected (EM estimate): {corrected_em:.4f}")
    print(f"true position-free CTR:    {true_rel:.4f}  (mean intrinsic relevance)")
    print()
    print("--- headline divergence ---")
    print(
        f"relevance policy:  NDCG@{args.k}={divergence['ndcg_relevance']:.4f}  "
        f"revenue/search={divergence['revenue_per_search_relevance']:,.2f}"
    )
    print(
        f"revenue policy:    NDCG@{args.k}={divergence['ndcg_revenue']:.4f}  "
        f"revenue/search={divergence['revenue_per_search_revenue']:,.2f}"
    )
    print(
        f"delta (rel - rev): NDCG {divergence['ndcg_delta']:+.4f}  "
        f"revenue {divergence['revenue_delta']:+,.2f}/search"
    )
    print(f"NDCG rises while revenue falls: {divergence['ndcg_rises_while_revenue_falls']}")
    print()
    print(
        f"interleaving ({inter['users']} users, first SERP): decision={inter['decision']} "
        f"A={inter['win_rate_a']:.3f} B={inter['win_rate_b']:.3f}"
    )
    print(f"\nwritten: {REPORT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
