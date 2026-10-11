"""A/B readout for a ranking change (SERP Rank Lab, Phase 3).

The metric under test is **revenue per search** -- a *ratio metric*, not a
per-user mean. ``R = sum(revenue) / sum(searches)``; the user is the unit of
randomisation, so its variance is estimated per user with the delta method
(``src/stats.py``), never by treating each search as an independent trial.

Readout order (all seeded and deterministic):

1. **SRM** -- chi-square check of a simulated 90/10 split against the planned
   90/10 allocation (the real Aviasales test-task set-up shrinks the test arm to
   10% of traffic).
2. **AA test** -- one arm cut randomly in two; the ratio test must be
   non-significant.
3. **Ratio metric** -- naive per-user ratio test vs the delta-method test, side
   by side, to show the gap the naive unit assumption opens.
4. **CUPED** -- variance reduction from a pre-period covariate (each user's
   pre-period revenue per search, on a booking-month split). Two blocks are
   reported: the *real* measurement, and a *synthetic positive control* whose
   covariate is correlated by construction, because the simulator draws users
   i.i.d. and so carries little genuine pre-period signal.
5. **Sequential / always-valid (mSPRT)** -- the peeking problem on this metric:
   many A/A streams, the fixed-horizon p-value recomputed at every interim (its
   false-positive rate inflates far above alpha) against the mSPRT always-valid
   p-value (which stays near alpha). Returns a tidy frame.
6. **Required sample size** for the observed baseline at a realistic MDE with the
   planned unequal allocation.

CLI: ``python -m src.experiment`` reads ``data/processed/serp_offers.parquet``,
prints the readout, and writes ``reports/experiment.json`` + ``reports/peeking.png``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import stats  # noqa: E402
from src.serp_sim import SERP_OFFERS_PARQUET  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
EXPERIMENT_JSON = REPORTS_DIR / "experiment.json"
PEEKING_PNG = REPORTS_DIR / "peeking.png"
RANKING_REPORT = REPORTS_DIR / "ranking_eval.json"

# --- Fixed, reported seeds and design constants (never left to chance) ------ #
TEST_SHARE = 0.10  # planned allocation: 90% control / 10% test
REL_MDE = 0.05  # smallest relative lift worth acting on
ALPHA = 0.05
SPLIT_SEED = 20261011  # arm assignment
AA_SEED = 101  # A/A pseudo-split
CUPED_SEED = 555  # synthetic positive control only
PEEK_SEED = 909  # peeking streams
PEEK_SIMS = 400
PEEK_BATCHES = 10
PEEK_BATCH_SIZE = 250
PRE_MONTH_MAX = 4  # booking months 1..4 = pre-period; 5..8 = post-period


# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #


def per_user_totals(offers: pd.DataFrame) -> pd.DataFrame:
    """Collapse ``serp_offers`` to one row per user: ``searches`` and ``revenue``.

    ``searches`` counts the user's distinct ``search_id`` values and ``revenue``
    sums the realised ``commission`` column, so ``sum(revenue)/sum(searches)`` over
    a group is exactly that group's revenue per search.
    """
    grouped = offers.groupby("user_id", sort=True)
    return pd.DataFrame(
        {
            "user_id": grouped.size().index,
            "searches": grouped["search_id"].nunique().to_numpy(),
            "revenue": grouped["commission"].sum().to_numpy(),
        }
    )


def assign_arms(
    users: pd.DataFrame, test_share: float = TEST_SHARE, seed: int = SPLIT_SEED
) -> pd.DataFrame:
    """Tag each user control/test by an independent Bernoulli draw at ``test_share``.

    Simulating the assignment (rather than reading a shipped flag) is what makes
    the SRM check meaningful: the split is drawn at the planned share and then
    tested against it.
    """
    rng = np.random.default_rng(seed)
    out = users.copy()
    out["is_test"] = rng.random(len(out)) < test_share
    return out


def split_arm(arm: pd.DataFrame, seed: int = AA_SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cut one arm into two random halves -- the A/A pseudo-split."""
    rng = np.random.default_rng(seed)
    mask = rng.random(len(arm)) < 0.5
    return arm[mask], arm[~mask]


def _ratio(df: pd.DataFrame) -> float:
    """Pooled revenue per search for a frame carrying ``revenue`` and ``searches``."""
    return float(df["revenue"].sum() / df["searches"].sum())


# --------------------------------------------------------------------------- #
# 1-2. SRM and A/A
# --------------------------------------------------------------------------- #


def srm_readout(users: pd.DataFrame) -> dict:
    """Chi-square SRM on the simulated split against the planned allocation."""
    n_control = int((~users["is_test"]).sum())
    n_test = int(users["is_test"].sum())
    planned = stats.srm_check(n_control, n_test, expected_test_share=TEST_SHARE)
    wrong = stats.srm_check(n_control, n_test, expected_test_share=0.5)
    return {
        "planned_test_share": TEST_SHARE,
        "n_control": n_control,
        "n_test": n_test,
        "observed_test_share": planned["observed_share"],
        "vs_planned": {**planned, "mismatch": planned["p_value"] < 0.001},
        "vs_fifty_fifty": {**wrong, "mismatch": wrong["p_value"] < 0.001},
    }


def aa_readout(control: pd.DataFrame) -> dict:
    """Split the control arm in two and confirm the ratio test is non-significant."""
    a, b = split_arm(control)
    res = stats.delta_method_test(a["searches"], a["revenue"], b["searches"], b["revenue"])
    return {
        "n_a": int(len(a)),
        "n_b": int(len(b)),
        "ratio_a": res["ratio_a"],
        "ratio_b": res["ratio_b"],
        "diff": res["diff"],
        "p_value": res["p_value"],
        "non_significant": bool(res["p_value"] >= ALPHA),
    }


# --------------------------------------------------------------------------- #
# 3. Ratio metric: naive per-unit vs delta method
# --------------------------------------------------------------------------- #


def ratio_metric_readout(control: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Compare the naive per-user ratio test with the delta-method test."""
    delta = stats.delta_method_test(
        control["searches"], control["revenue"], test["searches"], test["revenue"]
    )
    naive = stats.naive_per_unit_ratio_test(
        control["searches"], control["revenue"], test["searches"], test["revenue"]
    )
    return {
        "control_rps": delta["ratio_a"],
        "test_rps": delta["ratio_b"],
        "relative_lift": delta["diff"] / delta["ratio_a"],
        "delta_method": {
            "diff": delta["diff"],
            "se": delta["se"],
            "z": delta["z"],
            "p_value": delta["p_value"],
            "ci_low": delta["ci_low"],
            "ci_high": delta["ci_high"],
            "significant": delta["significant"],
        },
        "naive_per_unit": {
            "diff": naive["diff"],
            "se": naive["se"],
            "p_value": naive["p_value"],
            "significant": naive["significant"],
        },
        "se_ratio_naive_over_delta": naive["se"] / delta["se"],
        "p_values_differ": bool(abs(naive["p_value"] - delta["p_value"]) > 1e-6),
    }


# --------------------------------------------------------------------------- #
# 4. CUPED variance reduction on the ratio metric
# --------------------------------------------------------------------------- #


def _period_rps(offers: pd.DataFrame, pre: bool) -> pd.DataFrame:
    """Per-user revenue per search in the pre- or post-period slice."""
    months = offers["booking_month"].to_numpy()
    keep = months <= PRE_MONTH_MAX if pre else months > PRE_MONTH_MAX
    sliced = offers.loc[keep]
    grouped = sliced.groupby("user_id", sort=True)
    return pd.DataFrame(
        {
            "user_id": grouped.size().index,
            "searches": grouped["search_id"].nunique().to_numpy(),
            "revenue": grouped["commission"].sum().to_numpy(),
        }
    )


def _cuped_on_pairs(x: np.ndarray, y: np.ndarray) -> dict:
    """theta, variance reduction and CI narrowing for a covariate/outcome pair."""
    theta = stats.cuped_theta(y, x)
    y_adj = stats.cuped_adjust(y, x, theta)
    var_raw = float(np.var(y, ddof=1))
    var_adj = float(np.var(y_adj, ddof=1))
    raw_half = stats.welch_mean_test(y[: len(y) // 2], y[len(y) // 2 :])
    adj_half = stats.welch_mean_test(y_adj[: len(y_adj) // 2], y_adj[len(y_adj) // 2 :])
    corr = float(np.corrcoef(x, y)[0, 1]) if np.var(x) > 0 else 0.0
    ci_raw = raw_half["ci_high"] - raw_half["ci_low"]
    ci_adj = adj_half["ci_high"] - adj_half["ci_low"]
    return {
        "theta": theta,
        "corr_x_y": corr,
        "corr_squared_theory": corr**2,
        "var_raw": var_raw,
        "var_adjusted": var_adj,
        "variance_reduction_pct": (1 - var_adj / var_raw) * 100 if var_raw > 0 else 0.0,
        "ci_half_width_raw": ci_raw,
        "ci_half_width_adjusted": ci_adj,
        "ci_narrowing_pct": (1 - ci_adj / ci_raw) * 100 if ci_raw > 0 else 0.0,
        "n_users": int(len(y)),
    }


def cuped_readout(offers: pd.DataFrame, seed: int = CUPED_SEED) -> dict:
    """CUPED on revenue per search: real pre-period covariate plus a positive control.

    Real block: covariate = the user's pre-period revenue per search (months 1-4),
    outcome = the same user's post-period revenue per search (months 5-8), over the
    users present in both periods. The simulator draws users i.i.d., so this block
    is expected to be near zero -- an honest null, not a failure.

    Positive control: a synthetic covariate sharing a latent factor with the
    outcome at a known correlation, to show the machinery recovers the theoretical
    ``corr^2`` reduction when a signal exists.
    """
    pre = _period_rps(offers, pre=True).rename(
        columns={"revenue": "rev_pre", "searches": "srch_pre"}
    )
    post = _period_rps(offers, pre=False).rename(
        columns={"revenue": "rev_post", "searches": "srch_post"}
    )
    both = pre.merge(post, on="user_id", how="inner")
    both = both[(both["srch_pre"] > 0) & (both["srch_post"] > 0)]
    real = _cuped_on_pairs(
        (both["rev_pre"] / both["srch_pre"]).to_numpy(),
        (both["rev_post"] / both["srch_post"]).to_numpy(),
    )

    rng = np.random.default_rng(seed)
    n = max(len(both), 2000)
    latent = rng.normal(0, 1, n)
    rho = 0.6
    x_syn = rho * latent + np.sqrt(1 - rho**2) * rng.normal(0, 1, n)
    y_syn = rho * latent + np.sqrt(1 - rho**2) * rng.normal(0, 1, n)
    control = _cuped_on_pairs(x_syn, y_syn)
    return {
        "pre_month_max": PRE_MONTH_MAX,
        "real": {**real, "covariate": "pre-period revenue per search"},
        "positive_control": {**control, "covariate": "synthetic latent (rho = 0.6)"},
    }


# --------------------------------------------------------------------------- #
# 5. Peeking: fixed-horizon vs always-valid (mSPRT)
# --------------------------------------------------------------------------- #


def peeking_demo(
    users: pd.DataFrame,
    tau: float,
    n_sims: int = PEEK_SIMS,
    n_batches: int = PEEK_BATCHES,
    batch_size: int = PEEK_BATCH_SIZE,
    seed: int = PEEK_SEED,
) -> tuple[pd.DataFrame, dict]:
    """A/A streams under continuous monitoring: fixed-horizon vs mSPRT p-values.

    Each stream shuffles the user pool and cuts it into two arms, then accumulates
    ``batch_size`` users per arm over ``n_batches`` interims, recomputing both a
    fixed-horizon delta z p-value and the always-valid mSPRT p-value at every look.
    Returns ``(frame, summary)`` where ``frame`` is one tidy row per interim with
    ``sim, n, p_naive, p_av`` and ``summary`` counts the streams that ever crossed
    alpha under each rule.
    """
    searches = users["searches"].to_numpy(dtype=float)
    revenue = users["revenue"].to_numpy(dtype=float)
    n = len(users)
    half = (n // 2 // batch_size) * batch_size
    if half < batch_size:
        raise ValueError("not enough users for the requested batch size")

    rows: list[dict] = []
    naive_ever = 0
    av_ever = 0
    for s in range(n_sims):
        rng = np.random.default_rng(seed + s)
        order = rng.permutation(n)
        a_idx = order[:half]
        b_idx = order[half : 2 * half]
        min_naive = 1.0
        min_av = 1.0
        for k in range(1, n_batches + 1):
            sl = slice(0, k * batch_size)
            ia, ib = a_idx[sl], b_idx[sl]
            interim = stats.ratio_interim(searches[ia], revenue[ia], searches[ib], revenue[ib])
            _, p_av = stats.msprt_ratio(interim["D"], interim["z"], interim["I"], tau)
            p_naive = stats._two_sided_p_from_z(interim["z"])
            min_naive = min(min_naive, p_naive)
            min_av = min(min_av, p_av)
            rows.append({"sim": s, "n": int(k * batch_size), "p_naive": p_naive, "p_av": p_av})
        naive_ever += int(min_naive <= ALPHA)
        av_ever += int(min_av <= ALPHA)

    frame = pd.DataFrame(rows)
    summary = {
        "n_sims": n_sims,
        "n_batches": n_batches,
        "batch_size": batch_size,
        "alpha": ALPHA,
        "tau": tau,
        "naive_peeking_fpr": naive_ever / n_sims,
        "msprt_always_valid_fpr": av_ever / n_sims,
        "inflation_factor": (naive_ever / av_ever) if av_ever else float("inf"),
    }
    return frame, summary


def plot_peeking(frame: pd.DataFrame, summary: dict, path: Path = PEEKING_PNG) -> Path:
    """Bar chart of the two false-positive rates plus the per-interim p-value spread."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    rates = [summary["naive_peeking_fpr"] * 100, summary["msprt_always_valid_fpr"] * 100]
    axes[0].bar(
        ["fixed-horizon\n(peeking)", "mSPRT\n(always-valid)"], rates, color=["#b5651d", "#2e6f95"]
    )
    axes[0].axhline(
        ALPHA * 100, color="black", linestyle="--", linewidth=1, label=f"nominal {ALPHA:.0%}"
    )
    axes[0].set_ylabel("P(ever p <= alpha)  [%]")
    axes[0].set_title("A/A false-positive rate under continuous monitoring")
    axes[0].legend()
    for i, v in enumerate(rates):
        axes[0].text(i, v, f"{v:.1f}%", ha="center", va="bottom")

    axes[1].hist(frame["p_naive"], bins=25, alpha=0.7, label="fixed-horizon", color="#b5651d")
    axes[1].hist(frame["p_av"], bins=25, alpha=0.7, label="mSPRT", color="#2e6f95")
    axes[1].axvline(ALPHA, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("p-value at an interim")
    axes[1].set_ylabel("interim count")
    axes[1].set_title("Interim p-values (A/A)")
    axes[1].legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# 6. Required sample size
# --------------------------------------------------------------------------- #


def sample_size_readout(users: pd.DataFrame) -> dict:
    """Required users at the observed baseline, unequal vs even allocation."""
    baseline = _ratio(users)
    variance = stats.ratio_variance_per_user(users, clicks="revenue", views="searches")
    planned = stats.required_sample_size(baseline, REL_MDE, variance, ALPHA, 0.8, TEST_SHARE)
    even = stats.required_sample_size(baseline, REL_MDE, variance, ALPHA, 0.8, 0.5)
    return {
        "baseline_rps": baseline,
        "rel_mde": REL_MDE,
        "alpha": ALPHA,
        "power": 0.8,
        "variance_per_user": variance,
        "planned_test_share": TEST_SHARE,
        "planned": planned,
        "even": even,
        "cost_multiple": planned["n_total"] / even["n_total"],
        "power_at_planned_n": stats.achieved_power(
            planned["n_total"], baseline, REL_MDE, variance, ALPHA, TEST_SHARE
        ),
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
        prog="python -m src.experiment",
        description="A/B readout of a ranking change on revenue per search",
    )
    parser.parse_args(argv)

    if not SERP_OFFERS_PARQUET.exists():
        raise FileNotFoundError(
            f"Missing {SERP_OFFERS_PARQUET}. Run `python -m src.serp_sim` first."
        )
    offers = pd.read_parquet(SERP_OFFERS_PARQUET)
    users = per_user_totals(offers)
    tagged = assign_arms(users)
    control = tagged[~tagged["is_test"]]
    test = tagged[tagged["is_test"]]

    baseline_rps = _ratio(users)
    tau = baseline_rps * REL_MDE  # mSPRT prior SD in the same units as the lift

    print("=== experiment: ranking change on revenue per search ===")
    print(f"source:   {SERP_OFFERS_PARQUET}")
    print(
        f"offers:   {len(offers):,}   searches: {offers['search_id'].nunique():,}   users: {len(users):,}"
    )
    print(f"seeds:    split={SPLIT_SEED} aa={AA_SEED} cuped={CUPED_SEED} peek={PEEK_SEED}\n")

    srm = srm_readout(tagged)
    print("--- 1. SRM (simulated 90/10 split) ---")
    print(
        f"control {srm['n_control']:,}  test {srm['n_test']:,}  "
        f"observed test share {srm['observed_test_share']:.4f}"
    )
    print(
        f"vs planned 90/10: chi2={srm['vs_planned']['chi2']:.4f} "
        f"p={srm['vs_planned']['p_value']:.4f}  mismatch={srm['vs_planned']['mismatch']}"
    )
    print(
        f"vs assumed 50/50: chi2={srm['vs_fifty_fifty']['chi2']:.1f} "
        f"p={srm['vs_fifty_fifty']['p_value']:.2e}  mismatch={srm['vs_fifty_fifty']['mismatch']}\n"
    )

    aa = aa_readout(control)
    print("--- 2. A/A test (control split in two) ---")
    print(
        f"n_a {aa['n_a']:,}  n_b {aa['n_b']:,}  "
        f"ratio {aa['ratio_a']:.2f} vs {aa['ratio_b']:.2f}  "
        f"p={aa['p_value']:.3f}  non-significant={aa['non_significant']}\n"
    )

    ratio = ratio_metric_readout(control, test)
    print("--- 3. Ratio metric: naive per-user vs delta method ---")
    print(
        f"control RPS {ratio['control_rps']:.2f}  test RPS {ratio['test_rps']:.2f}  "
        f"relative lift {ratio['relative_lift']:+.2%}"
    )
    print(
        f"delta method:  se={ratio['delta_method']['se']:.3f}  "
        f"p={ratio['delta_method']['p_value']:.4f}  "
        f"CI [{ratio['delta_method']['ci_low']:.2f}, {ratio['delta_method']['ci_high']:.2f}]"
    )
    print(
        f"naive per-unit: se={ratio['naive_per_unit']['se']:.3f}  "
        f"p={ratio['naive_per_unit']['p_value']:.4f}"
    )
    print(f"SE ratio (naive/delta) = {ratio['se_ratio_naive_over_delta']:.3f}\n")

    cuped = cuped_readout(offers)
    print("--- 4. CUPED variance reduction (revenue per search) ---")
    for label in ("real", "positive_control"):
        block = cuped[label]
        print(
            f"{label:<16} n={block['n_users']:,}  corr={block['corr_x_y']:.3f}  "
            f"var reduction={block['variance_reduction_pct']:+.1f}%  "
            f"CI narrowing={block['ci_narrowing_pct']:+.1f}%"
        )
    print("(real block: simulator users are i.i.d., so a near-zero reduction is expected)\n")

    frame, peek = peeking_demo(users, tau=tau)
    print(f"--- 5. Peeking (A/A, {peek['n_sims']} streams x {peek['n_batches']} interims) ---")
    print(
        f"fixed-horizon FPR (ever p<=alpha): {peek['naive_peeking_fpr']:.1%}   "
        f"mSPRT always-valid FPR: {peek['msprt_always_valid_fpr']:.1%}   "
        f"(nominal {ALPHA:.0%})"
    )
    print(f"inflation factor: {peek['inflation_factor']:.1f}x\n")

    sizing = sample_size_readout(users)
    print("--- 6. Required sample size ---")
    print(
        f"baseline RPS {sizing['baseline_rps']:.2f}  V={sizing['variance_per_user']:.4f}  "
        f"MDE {sizing['rel_mde']:.0%} relative"
    )
    print(
        f"planned 90/10: n_total={sizing['planned']['n_total']:,.0f} "
        f"(control {sizing['planned']['n_control']:,.0f} + test {sizing['planned']['n_test']:,.0f})"
    )
    print(
        f"even 50/50:    n_total={sizing['even']['n_total']:,.0f}  "
        f"({sizing['cost_multiple']:.2f}x vs even)\n"
    )

    phase2 = None
    if RANKING_REPORT.exists():
        phase2 = json.loads(RANKING_REPORT.read_text(encoding="utf-8")).get("divergence")

    report = {
        "seeds": {
            "split": SPLIT_SEED,
            "aa": AA_SEED,
            "cuped": CUPED_SEED,
            "peek": PEEK_SEED,
        },
        "design": {"test_share": TEST_SHARE, "rel_mde": REL_MDE, "alpha": ALPHA},
        "inputs": {
            "offers": int(len(offers)),
            "searches": int(offers["search_id"].nunique()),
            "users": int(len(users)),
        },
        "srm": srm,
        "aa_test": aa,
        "ratio_metric": ratio,
        "cuped": cuped,
        "peeking": peek,
        "sample_size": sizing,
        "phase2_divergence": phase2,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_JSON.write_text(json.dumps(_to_native(report), indent=2), encoding="utf-8")
    png = plot_peeking(frame, peek)
    print(f"written: {EXPERIMENT_JSON}")
    print(f"written: {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
