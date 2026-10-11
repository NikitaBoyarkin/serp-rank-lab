"""Behavioural checks that bind ``src.stats`` to known-correct answers.

Three kinds of test:

1. **Real reproduction** -- the vendored sample-size and ratio-test functions are
   run against the sibling project's real data files (read by absolute path, never
   copied here) and must reproduce its documented answers. Skipped cleanly when
   those files are absent, so the suite stays green elsewhere.
2. **CUPED mutation control** -- the adjustment must lower variance on a
   correlated covariate and must NOT lower it on a decoupled (shuffled) one; a
   no-op adjustment fails the first assertion.
3. **Peeking** -- the fixed-horizon false-positive rate under continuous
   monitoring must exceed the always-valid (mSPRT) rate.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import stats
from src.experiment import peeking_demo

# Sibling project: the Aviasales CTR A/B test-task project (read-only, not vendored).
# Point AVIASALES_AB_CTR_DIR at it to exercise the reproduction tests; they skip when
# it is unset or absent, so the suite stays green on a bare clone.
SIBLING_ENV = "AVIASALES_AB_CTR_DIR"
SIBLING = Path(os.environ[SIBLING_ENV]).expanduser() if os.environ.get(SIBLING_ENV) else None
HISTORY_CSV = SIBLING / "data" / "history_control.csv" if SIBLING else None
RESULTS_CSV = SIBLING / "data" / "experiment_results.csv" if SIBLING else None

# Documented answers from that project's README (rel_mde=10%, alpha=0.05, power=0.80).
DOCUMENTED_N_TOTAL = 3223.0  # 2,900 control + 322 test at 90/10
DOCUMENTED_DELTA_P = 0.0915
DOCUMENTED_NAIVE_P = 0.0460


def _require_sibling_data() -> None:
    if SIBLING is None or not (HISTORY_CSV.exists() and RESULTS_CSV.exists()):
        pytest.skip(f"{SIBLING_ENV} unset or sibling project data absent under {SIBLING}")


# --------------------------------------------------------------------------- #
# 1. Real reproduction against the sibling project
# --------------------------------------------------------------------------- #


def test_required_sample_size_reproduces_sibling_answer() -> None:
    """90/10 sizing at the real baseline -> 3,223 total (2,900 control + 322 test)."""
    _require_sibling_data()
    history = pd.read_csv(HISTORY_CSV)
    baseline = stats.ctr(history)
    variance = stats.ratio_variance_per_user(history)

    res = stats.required_sample_size(
        baseline, 0.10, variance, alpha=0.05, power=0.80, test_share=0.10
    )

    assert res["n_total"] == pytest.approx(DOCUMENTED_N_TOTAL, rel=0.01)
    assert round(res["n_control"]) == 2900
    assert round(res["n_test"]) == 322


def test_delta_method_ratio_test_reproduces_sibling_pvalue() -> None:
    """User-level delta-method p on the real experiment -> ~0.0915, not a win."""
    _require_sibling_data()
    results = pd.read_csv(RESULTS_CSV)
    control = results[results["group"] == "A"]
    test = results[results["group"] == "B"]

    delta = stats.ratio_metric_test(control, test)
    naive = stats.naive_binomial_test(control, test)

    assert delta["p_value"] == pytest.approx(DOCUMENTED_DELTA_P, abs=0.002)
    assert naive["p_value"] == pytest.approx(DOCUMENTED_NAIVE_P, abs=0.002)
    # The clustering penalty: the delta-method SE must exceed the view-level one.
    assert delta["se"] > naive["se"]


def test_srm_check_uses_the_planned_allocation() -> None:
    """A designed 90/10 split passes a 90/10 check and fails a 50/50 one."""
    planned = stats.srm_check(4000, 444, expected_test_share=0.10)
    wrong = stats.srm_check(4000, 444, expected_test_share=0.50)
    assert planned["p_value"] > 0.05
    assert wrong["p_value"] < 0.001


def test_multi_arm_srm_matches_two_arm_case() -> None:
    """``srm_test`` on two arms agrees with ``srm_check`` on the same counts."""
    two = stats.srm_test([4000, 444], [0.9, 0.1])
    check = stats.srm_check(4000, 444, expected_test_share=0.10)
    assert two["p_value"] == pytest.approx(check["p_value"], rel=1e-6)


def test_achieved_power_is_exported() -> None:
    """The source omitted ``achieved_power`` from ``__all__``; here it is exported."""
    assert "achieved_power" in stats.__all__
    power = stats.achieved_power(3223, 0.30, 0.10, 0.033, test_share=0.10)
    assert power == pytest.approx(0.80, abs=0.03)


# --------------------------------------------------------------------------- #
# 2. CUPED mutation control
# --------------------------------------------------------------------------- #


def _correlated_sample(
    n: int = 4000, rho: float = 0.6, seed: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    latent = rng.normal(0, 1, n)
    x = rho * latent + np.sqrt(1 - rho**2) * rng.normal(0, 1, n)
    y = rho * latent + np.sqrt(1 - rho**2) * rng.normal(0, 1, n)
    return x, y


def test_cuped_reduces_variance_on_correlated_covariate() -> None:
    x, y = _correlated_sample()
    theta = stats.cuped_theta(y, x)
    y_adj = stats.cuped_adjust(y, x, theta)
    var_raw = float(np.var(y, ddof=1))
    var_adj = float(np.var(y_adj, ddof=1))
    # Strictly lower, and close to the theoretical corr^2 reduction.
    assert var_adj < var_raw
    assert (1 - var_adj / var_raw) == pytest.approx(float(np.corrcoef(x, y)[0, 1]) ** 2, abs=0.03)


def test_cuped_does_not_reduce_variance_on_decoupled_covariate() -> None:
    """Mutation control: a shuffled covariate must NOT lower variance.

    If the adjustment were a no-op the correlated test above would fail; if it
    adjusted on noise it would (wrongly) reduce variance here.
    """
    x, y = _correlated_sample()
    rng = np.random.default_rng(11)
    x_shuffled = rng.permutation(x)
    theta = stats.cuped_theta(y, x_shuffled)
    y_adj = stats.cuped_adjust(y, x_shuffled, theta)
    var_raw = float(np.var(y, ddof=1))
    var_adj = float(np.var(y_adj, ddof=1))
    assert theta == pytest.approx(0.0, abs=0.05)
    assert var_adj >= var_raw * 0.99


# --------------------------------------------------------------------------- #
# 3. Peeking: fixed-horizon vs always-valid
# --------------------------------------------------------------------------- #


@pytest.fixture
def synthetic_users() -> pd.DataFrame:
    """Per-user (searches, revenue) with i.i.d. draws -- an A/A world."""
    rng = np.random.default_rng(7)
    n = 4000
    searches = 1 + rng.poisson(1.5, n)
    revenue = searches * rng.lognormal(mean=3.0, sigma=0.7, size=n)
    return pd.DataFrame({"searches": searches.astype(float), "revenue": revenue})


def test_peeking_inflates_fixed_horizon_fpr_but_not_msprt(synthetic_users: pd.DataFrame) -> None:
    baseline = synthetic_users["revenue"].sum() / synthetic_users["searches"].sum()
    tau = baseline * 0.05
    _, summary = peeking_demo(synthetic_users, tau=tau, n_sims=200, n_batches=8, batch_size=200)
    assert summary["naive_peeking_fpr"] > summary["msprt_always_valid_fpr"]
    assert summary["naive_peeking_fpr"] > summary["alpha"]
