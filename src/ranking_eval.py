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
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.policies import (
    DEFAULT_POLICIES,
    DELTA_POLICY,
    POLICIES,
    REALISTIC_POLICIES,
    PolicyContext,
    context_for,
    rev_engaged,
    train_test_split_searches,
)
from src.serp_sim import (
    ETA_DESKTOP,
    ETA_MOBILE,
    SERP_OFFERS_PARQUET,
    SimConfig,
    examination_prob,
    load_routes,
    load_user_ids,
    simulate_serps,
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


def search_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """One search's offers in logged display order, re-indexed 0..n-1.

    This is the frame the policies rank over: row ``i`` is the offer that was
    displayed at position ``i + 1``. Re-indexing is what lets a policy return a
    permutation of ``range(len(frame))``.
    """
    return frame.sort_values("position").reset_index(drop=True)


def _offer_arrays(frame: pd.DataFrame, ctx: PolicyContext) -> tuple[np.ndarray, np.ndarray]:
    """Per-row ``(relevance, revenue_if_engaged)`` in logged display order."""
    return frame["relevance"].to_numpy(dtype=float), rev_engaged(frame, ctx)


def _expected_metrics(
    order: np.ndarray, rel: np.ndarray, rev: np.ndarray, eta: float
) -> tuple[float, float]:
    """``(expected revenue, expected clicks)`` for a displayed order under the PBM.

    ``P(click at slot k) = P(examine at k) * relevance``, and revenue lands only
    if the offer is engaged.
    """
    order = np.asarray(order, dtype=int)
    e = np.array([examination_prob(k, eta) for k in range(1, len(order) + 1)], dtype=float)
    p_click = e * rel[order]
    return float((p_click * rev[order]).sum()), float(p_click.sum())


def compare_rankings(
    offers_df: pd.DataFrame,
    k: int = 10,
    policies: Sequence[str] | None = None,
    ctx: PolicyContext | None = None,
    eta_mobile: float = ETA_MOBILE,
    eta_desktop: float = ETA_DESKTOP,
) -> pd.DataFrame:
    """Policy table: NDCG@k and expected revenue per search, one row per policy.

    Every policy re-ranks the *same* offer set within each search, so the
    comparison is apples to apples. NDCG@k is the offline metric; expected revenue
    per search is the online one. A ``delta_relevance_minus_revenue`` row is
    appended whenever both references are present.

    Defaults to the two Phase 2 reference policies. The realistic rankers are
    opt-in because they need ``partner_type``/``examine_prob`` columns and fitted
    coefficients on ``ctx``.
    """
    names = tuple(policies) if policies is not None else DEFAULT_POLICIES
    ctx = ctx or PolicyContext()
    if (eta_mobile, eta_desktop) != (ctx.eta_mobile, ctx.eta_desktop):
        ctx = replace(ctx, eta_mobile=eta_mobile, eta_desktop=eta_desktop)

    totals = {name: {"ndcg": 0.0, "revenue": 0.0, "clicks": 0.0} for name in names}
    n_searches = 0
    for _, group in offers_df.groupby("search_id", sort=True):
        frame = search_frame(group)
        eta = ctx.eta(str(frame["device"].iloc[0]))
        rel, rev = _offer_arrays(frame, ctx)
        for name in names:
            order = np.asarray(POLICIES[name](frame, ctx), dtype=int)
            totals[name]["ndcg"] += ndcg_at_k(rel[order], k)
            revenue, clicks = _expected_metrics(order, rel, rev, eta)
            totals[name]["revenue"] += revenue
            totals[name]["clicks"] += clicks
        n_searches += 1

    if n_searches == 0:
        raise ValueError("offers_df must contain at least one search")

    rows = [
        {
            "policy": name,
            "ndcg_at_k": totals[name]["ndcg"] / n_searches,
            "revenue_per_search": totals[name]["revenue"] / n_searches,
            "clicks_per_search": totals[name]["clicks"] / n_searches,
        }
        for name in names
    ]
    if "relevance" in names and "revenue" in names:
        rows.append(
            {
                "policy": DELTA_POLICY,
                "ndcg_at_k": (totals["relevance"]["ndcg"] - totals["revenue"]["ndcg"]) / n_searches,
                "revenue_per_search": (
                    totals["relevance"]["revenue"] - totals["revenue"]["revenue"]
                )
                / n_searches,
                "clicks_per_search": (totals["relevance"]["clicks"] - totals["revenue"]["clicks"])
                / n_searches,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Sensitivity of the divergence to the model's own parameters
# --------------------------------------------------------------------------- #

ETA_GRID = (0.6, 0.8, 1.0, 1.2, 1.4, 1.6)
TILT_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
SWEEP_SEARCHES = 2000


def _sample_searches(offers_df: pd.DataFrame, n_searches: int, seed: int) -> pd.DataFrame:
    """Deterministic random subset of whole searches -- never a partial SERP."""
    ids = np.array(sorted(offers_df["search_id"].unique()))
    if len(ids) <= n_searches:
        return offers_df
    rng = np.random.default_rng(seed)
    picked = set(rng.choice(ids, size=n_searches, replace=False).tolist())
    return offers_df[offers_df["search_id"].isin(picked)]


def sensitivity_sweep(
    offers_df: pd.DataFrame,
    k: int = 10,
    ctx: PolicyContext | None = None,
    eta_grid: Sequence[float] = ETA_GRID,
    tilt_grid: Sequence[float] = TILT_GRID,
    n_searches: int = SWEEP_SEARCHES,
    seed: int = 0,
    routes: pd.DataFrame | None = None,
    user_ids: np.ndarray | list[str] | None = None,
) -> dict[str, object]:
    """Ask whether the divergence is an artefact of the simulator's own knobs.

    Two axes, because there are two candidate explanations:

    * ``eta`` -- the examination curve. Re-scored on the *same* offers, so if the
      gap were a property of the curve, some point on the grid would close it.
      Both device curves are set to the same value, making the axis one number.
    * ``money_tilt`` -- the shipped display blend. This one cannot be re-scored in
      place: it decides which offers were ever displayed, so the SERPs must be
      regenerated. What it changes is the *logged* order, so this axis reports how
      the shipped blend itself trades NDCG against revenue.

    The verdict is either an eta band where the gap survives, or an explicit
    "does not vanish on this grid".
    """
    ctx = ctx or PolicyContext()
    sub = _sample_searches(offers_df, n_searches, seed)
    n_used = int(sub["search_id"].nunique())

    eta_rows: list[dict[str, object]] = []
    for eta in eta_grid:
        table = compare_rankings(
            sub,
            k=k,
            policies=("relevance", "revenue"),
            ctx=ctx,
            eta_mobile=float(eta),
            eta_desktop=float(eta),
        ).set_index("policy")
        delta_ndcg = float(table.loc[DELTA_POLICY, "ndcg_at_k"])
        delta_rev = float(table.loc[DELTA_POLICY, "revenue_per_search"])
        eta_rows.append(
            {
                "eta": float(eta),
                "ndcg_delta": delta_ndcg,
                "revenue_delta": delta_rev,
                "divergence_holds": bool(delta_ndcg > 0 and delta_rev < 0),
            }
        )

    surviving = [float(row["eta"]) for row in eta_rows if row["divergence_holds"]]
    if len(surviving) == len(eta_rows):
        verdict = (
            f"does not vanish on eta in [{min(eta_grid)}, {max(eta_grid)}]: NDCG rises "
            "while revenue falls at every grid point"
        )
    elif surviving:
        verdict = (
            f"holds only for eta in [{min(surviving)}, {max(surviving)}]; "
            f"vanishes outside that band (grid {list(eta_grid)})"
        )
    else:
        verdict = "vanishes across the whole eta grid"

    tilt_rows: list[dict[str, object]] = []
    if routes is not None and user_ids is not None:
        for tilt in tilt_grid:
            sim = simulate_serps(
                routes, user_ids, SimConfig(n_searches=n_searches, money_tilt=float(tilt))
            )
            ndcg_sum = 0.0
            rev_sum = 0.0
            count = 0
            for _, group in sim.groupby("search_id", sort=True):
                frame = search_frame(group)
                eta = ctx.eta(str(frame["device"].iloc[0]))
                rel, rev = _offer_arrays(frame, ctx)
                ndcg_sum += ndcg_at_k(rel, k)
                revenue, _clicks = _expected_metrics(np.arange(len(frame)), rel, rev, eta)
                rev_sum += revenue
                count += 1
            tilt_rows.append(
                {
                    "money_tilt": float(tilt),
                    "logged_ndcg_at_k": ndcg_sum / count,
                    "logged_revenue_per_search": rev_sum / count,
                }
            )

    return {
        "k": k,
        "n_searches": n_used,
        "eta": eta_rows,
        "money_tilt": tilt_rows,
        "verdict": verdict,
    }


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

    # A fitted ranker must not be scored on the searches it learned from, so the
    # split is on ``search_id``: fit on one half, score every policy on the other.
    train, test = train_test_split_searches(offers, test_frac=0.5, seed=0)
    ctx = context_for(test, train=train)
    print(
        f"train searches: {train['search_id'].nunique():,}   "
        f"test searches:  {test['search_id'].nunique():,}"
    )
    for name, beta in ctx.params.items():
        beta = np.asarray(beta, dtype=float)
        note = "position-biased fit" if name == "click_trained" else "IPW fit"
        print(f"  {name:<20} price={beta[1]:+.3f}  cpc={beta[2]:+.3f}   ({note})")
    print()

    summary = compare_rankings(test, k=args.k, policies=POLICIES, ctx=ctx)
    by_policy = summary.set_index("policy")

    def cell(name: str, column: str) -> float:
        """Key-based lookup -- never ``.loc[i]``, so adding a policy cannot renumber."""
        return float(by_policy.loc[name, column])

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
    first_search = test["search_id"].iloc[0]
    frame = search_frame(test[test["search_id"] == first_search])
    eta_dev = ctx.eta(str(frame["device"].iloc[0]))
    rel_map = {i: float(r) for i, r in enumerate(frame["relevance"])}
    by_rel = [int(i) for i in POLICIES["relevance"](frame, ctx)]
    by_rev = [int(i) for i in POLICIES["revenue"](frame, ctx)]
    inter = interleaving_experiment(by_rel, by_rev, rel_map, eta_dev)

    divergence = {
        "ndcg_k": args.k,
        "ndcg_relevance": cell("relevance", "ndcg_at_k"),
        "ndcg_revenue": cell("revenue", "ndcg_at_k"),
        "revenue_per_search_relevance": cell("relevance", "revenue_per_search"),
        "revenue_per_search_revenue": cell("revenue", "revenue_per_search"),
        "ndcg_delta": cell(DELTA_POLICY, "ndcg_at_k"),
        "revenue_delta": cell(DELTA_POLICY, "revenue_per_search"),
        "ndcg_rises_while_revenue_falls": bool(
            cell(DELTA_POLICY, "ndcg_at_k") > 0 and cell(DELTA_POLICY, "revenue_per_search") < 0
        ),
    }

    # The point of the realistic rankers: the divergence is not an artefact of
    # comparing the oracle with the greedy optimum.
    realistic = [
        {
            "policy": name,
            "ndcg_at_k": cell(name, "ndcg_at_k"),
            "revenue_per_search": cell(name, "revenue_per_search"),
            "ndcg_below_oracle": bool(cell(name, "ndcg_at_k") < divergence["ndcg_relevance"]),
        }
        for name in REALISTIC_POLICIES
    ]

    try:
        sensitivity = sensitivity_sweep(
            offers, k=args.k, ctx=ctx, routes=load_routes(), user_ids=load_user_ids()
        )
        tilt_note = ""
    except (FileNotFoundError, OSError, ValueError) as exc:
        # Routes/users are only needed to regenerate SERPs for the tilt axis;
        # without them the eta axis still runs.
        sensitivity = sensitivity_sweep(offers, k=args.k, ctx=ctx)
        tilt_note = f"money_tilt axis skipped ({type(exc).__name__}: {exc})"

    report = {
        "k": args.k,
        "offers": int(len(offers)),
        "searches": int(offers["search_id"].nunique()),
        "train_searches": int(train["search_id"].nunique()),
        "test_searches": int(test["search_id"].nunique()),
        "logged_revenue_per_search": logged_revenue,
        "policies": summary.to_dict(orient="records"),
        "divergence": divergence,
        "realistic_policies": realistic,
        "sensitivity": sensitivity,
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
    print("--- reference divergence (both policies see ground truth) ---")
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
    print("--- realistic policies (no access to the relevance column) ---")
    for row in realistic:
        print(
            f"{row['policy']:<20} NDCG@{args.k}={row['ndcg_at_k']:.4f}  "
            f"revenue/search={row['revenue_per_search']:,.2f}"
        )
    print()
    print("--- does the divergence survive the model's own parameters? ---")
    for row in sensitivity["eta"]:
        print(
            f"eta={row['eta']:.2f}  NDCG delta={row['ndcg_delta']:+.4f}  "
            f"revenue delta={row['revenue_delta']:+,.2f}  holds={row['divergence_holds']}"
        )
    print(f"verdict: {sensitivity['verdict']}")
    if tilt_note:
        print(tilt_note)
    for row in sensitivity["money_tilt"]:
        print(
            f"shipped tilt={row['money_tilt']:.2f}  logged NDCG@{args.k}="
            f"{row['logged_ndcg_at_k']:.4f}  "
            f"logged revenue/search={row['logged_revenue_per_search']:,.2f}"
        )
    print()
    print(
        f"interleaving ({inter['users']} users, first SERP): decision={inter['decision']} "
        f"A={inter['win_rate_a']:.3f} B={inter['win_rate_b']:.3f}"
    )
    print(f"\nwritten: {REPORT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
