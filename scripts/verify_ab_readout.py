"""Independent re-check of ``reports/experiment.json``.

Rebuilds the arms from ``serp_offers.parquet`` and recomputes the readout, then
compares every stored number against the recomputed one. Read-only: it never
writes a report and never touches the stored JSON.

The primary SE is re-derived twice: once through ``src.stats`` and once through a
hand-written delta-method formula, so a formula bug cannot hide behind a shared
helper.

    uv run python scripts/verify_ab_readout.py

``tests/test_ab_readout_reproducible.py`` imports ``recompute`` and
``comparisons`` from this file, so the same check runs in the suite.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import experiment as xp  # noqa: E402
from src.policies import context_for, train_test_split_searches  # noqa: E402
from src.serp_sim import SERP_OFFERS_PARQUET  # noqa: E402

REPORT = PROJECT_ROOT / "reports" / "experiment.json"
REL_TOL = 1e-6


def manual_delta_se(x: np.ndarray, y: np.ndarray) -> float:
    """SE of sum(y)/sum(x) from the delta method, written without ``src.stats``."""
    n = len(x)
    mean_x = x.mean()
    ratio = y.mean() / mean_x
    var_x = x.var(ddof=1)
    var_y = y.var(ddof=1)
    cov = np.cov(x, y, ddof=1)[0, 1]
    return math.sqrt((var_y + ratio**2 * var_x - 2 * ratio * cov) / mean_x**2 / n)


def bh_adjust(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg, implemented locally so it is not the project's helper."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [1.0] * m
    running = 1.0
    for rank, idx in enumerate(reversed(order), start=1):
        k = m - rank + 1
        running = min(running, p_values[idx] * m / k)
        adjusted[idx] = min(1.0, running)
    return adjusted


def naive_two_proportion(x_a: int, n_a: int, x_b: int, n_b: int) -> dict:
    """Unclustered two-proportion z-test, for contrast with the clustered test."""
    p_a, p_b = x_a / n_a, x_b / n_b
    pooled = (x_a + x_b) / (n_a + n_b)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    z = (p_b - p_a) / se
    p = math.erfc(abs(z) / math.sqrt(2))
    return {"rate_a": p_a, "rate_b": p_b, "z": z, "p_value": p, "se": se}


def recompute() -> dict:
    """Rebuild the arms from the parquet and recompute the whole readout in memory."""
    offers = pd.read_parquet(SERP_OFFERS_PARQUET)
    train, held_out = train_test_split_searches(offers, test_frac=0.5, seed=xp.MODEL_SEED)
    ctx = context_for(held_out, train=train)
    counterfactuals = xp.arm_outcomes(held_out, ctx)
    tagged = xp.assign_arms(xp.per_user_totals(held_out))
    arms = xp.arms_from_assignment(counterfactuals, tagged)
    control, test = arms["control"], arms["test"]

    primary = xp.primary_readout(control, test)
    a, b = xp.per_user_totals(control), xp.per_user_totals(test)
    manual_se = math.sqrt(
        manual_delta_se(a["searches"].to_numpy(float), a["revenue"].to_numpy(float)) ** 2
        + manual_delta_se(b["searches"].to_numpy(float), b["revenue"].to_numpy(float)) ** 2
    )
    guardrails = xp.guardrails_readout(control, test)

    search_a, search_b = xp.search_table(control), xp.search_table(test)
    naive = {
        "ctr": naive_two_proportion(
            int(search_a["clicks"].sum()),
            int(search_a["offers"].sum()),
            int(search_b["clicks"].sum()),
            int(search_b["offers"].sum()),
        ),
        "booking_conversion": naive_two_proportion(
            int(search_a["booked_searches"].sum()),
            int(search_a["searches"].sum()),
            int(search_b["booked_searches"].sum()),
            int(search_b["searches"].sum()),
        ),
    }

    return {
        "srm": xp.srm_readout(tagged),
        "primary": primary,
        "manual_se": manual_se,
        "guardrails": guardrails,
        "decision": xp.decision_readout(primary, guardrails),
        "naive": naive,
        "bh_p": dict(zip(list(guardrails), bh_adjust([g["p_value"] for g in guardrails.values()]),
                         strict=True)),
    }


def comparisons(stored: dict, fresh: dict) -> list[tuple[str, float, float, str]]:
    """Every stored-vs-recomputed pair, each tagged PASS/FAIL at ``REL_TOL``."""
    rows: list[tuple[str, float, float, str]] = []

    def compare(label: str, old: float, new: float) -> None:
        if old == 0:
            ok = abs(new) < REL_TOL
        else:
            ok = abs(new - old) / abs(old) < REL_TOL
        rows.append((label, old, new, "PASS" if ok else "FAIL"))

    compare("srm.n_control", stored["srm"]["n_control"], fresh["srm"]["n_control"])
    compare("srm.n_test", stored["srm"]["n_test"], fresh["srm"]["n_test"])
    compare(
        "srm.chi2_vs_planned", stored["srm"]["vs_planned"]["chi2"], fresh["srm"]["vs_planned"]["chi2"]
    )
    compare(
        "srm.p_vs_planned", stored["srm"]["vs_planned"]["p_value"], fresh["srm"]["vs_planned"]["p_value"]
    )

    old_primary, new_primary = stored["primary"], fresh["primary"]
    compare(
        "primary.control_rps",
        old_primary["control"]["revenue_per_search"],
        new_primary["control"]["revenue_per_search"],
    )
    compare(
        "primary.test_rps",
        old_primary["test"]["revenue_per_search"],
        new_primary["test"]["revenue_per_search"],
    )
    for key in ("diff_abs", "se", "p_value", "ci_low_abs", "ci_high_abs"):
        compare(f"primary.{key}", old_primary[key], new_primary[key])
    compare("primary.se_manual_delta", old_primary["se"], fresh["manual_se"])

    for name, new_g in fresh["guardrails"].items():
        old_g = stored["guardrails"][name]
        for key in ("control", "test", "diff_rel", "p_value"):
            compare(f"guardrail.{name}.{key}", old_g[key], new_g[key])

    return rows


def main() -> int:
    stored = json.loads(REPORT.read_text())
    fresh = recompute()
    rows = comparisons(stored, fresh)

    width = max(len(r[0]) for r in rows) + 2
    print("=" * 78)
    print("REPRODUCIBILITY CHECK — reports/experiment.json vs fresh recompute")
    print("=" * 78)
    for label, old, new, status in rows:
        print(f"{label:<{width}}stored {old:>16.6f}   fresh {new:>16.6f}   {status}")
    failures = sum(1 for r in rows if r[3] == "FAIL")
    print(f"\n{len(rows) - failures}/{len(rows)} checks pass; tolerance {REL_TOL:g} relative")

    print("\n--- Decision reproduced ---")
    print(f"stored verdict : {stored['decision']['verdict']} ({stored['decision']['reason']})")
    print(f"fresh verdict  : {fresh['decision']['verdict']} ({fresh['decision']['reason']})")

    print("\n--- Guardrail multiple comparisons (BH over 4) ---")
    for name, g in fresh["guardrails"].items():
        adj = fresh["bh_p"][name]
        print(
            f"{name:<26} raw p={g['p_value']:.6g}  BH p={adj:.6g}  "
            f"breach(rule)={g['breach']}  breach(BH)={bool(g['point_flag'] and adj < xp.ALPHA)}"
        )

    print("\n--- Clustered (delta) vs unclustered (two-proportion) SE ---")
    for name, n in fresh["naive"].items():
        g = fresh["guardrails"][name]
        print(
            f"{name:<20} delta z={g['z']:+.3f} p={g['p_value']:.3g}   "
            f"naive z={n['z']:+.3f} p={n['p_value']:.3g}   "
            f"naive/delta SE={n['se'] / g['se']:.3f}"
        )

    ok = failures == 0 and fresh["decision"]["verdict"] == stored["decision"]["verdict"]
    print("\nRESULT:", "reproduced" if ok else "DIVERGED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
