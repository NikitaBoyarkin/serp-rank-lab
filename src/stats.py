"""Consolidated statistics library for SERP Rank Lab (Phase 3).

Vendored, de-duplicated into one module. Nothing is imported from the sibling
projects at runtime: each formula was re-read from its source and rewritten here,
with the exact expression in the docstring.

Source files and origin projects
--------------------------------
1. ``aviasales-ab-ctr/utils/stats.py`` (project: aviasales-ab-ctr) --
   ``ctr``, ``ratio_variance_per_user``, ``ratio_standard_error``,
   ``ratio_metric_test``, ``naive_binomial_test``, ``bootstrap_lift``,
   ``required_sample_size`` (keeps unequal allocation via ``test_share``),
   ``achieved_power``, ``srm_check``.
   Known gotcha in that source: ``achieved_power`` is defined and printed but is
   missing from ``__all__``. Fixed here -- every public name is in ``__all__``.
2. ``ab_test/scripts/cuped.py`` (project: ab_test) --
   ``cuped_theta``, ``cuped_adjust``, ``welch_mean_test``.
3. ``ab_test/scripts/delta_method_ratio.py`` (project: ab_test) --
   ``ratio_delta_se``, ``delta_method_test``, ``naive_per_unit_ratio_test``.
4. ``ab_test/scripts/sequential_ratio.py`` (project: ab_test) --
   ``ratio_delta_var_perunit``, ``ratio_interim``, ``msprt_ratio``, ``run_stream``.
5. ``ab_test/scripts/srm_test.py`` (project: ab_test) -- ``srm_test``.

Duplicates resolved (one formula, one home)
-------------------------------------------
* ``ratio_delta_var_perunit`` (array interface) is the canonical per-unit
  delta-method variance. ``ratio_variance_per_user`` is the *same* expression
  behind a pandas interface; it now delegates instead of re-deriving it.
* ``delta_method_test`` (array interface) is the canonical ratio test;
  ``ratio_metric_test`` delegates and reshapes the result.
* ``srm_test`` (k arms, ``scipy.stats.chisquare``) is canonical; ``srm_check`` is
  the two-arm special case kept because the readout calls it by that name.

Two execution notes carried from the sources
--------------------------------------------
* A ratio metric ``R = sum(y) / sum(x)`` has asymptotic variance estimated from
  *per-unit* pairs, not by treating each unit of ``x`` as an independent trial.
  The user is the cluster in both source projects and in this one.
* ``scipy.stats`` is imported lazily inside the few functions that need it, so the
  module imports without the heavy stack when only the closed-form estimators are
  used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    # ratio-metric estimators and tests
    "ctr",
    "ratio_delta_var_perunit",
    "ratio_variance_per_user",
    "ratio_delta_se",
    "ratio_standard_error",
    "ratio_metric_test",
    "delta_method_test",
    "naive_binomial_test",
    "naive_per_unit_ratio_test",
    "bootstrap_lift",
    # sizing / power / allocation
    "required_sample_size",
    "achieved_power",
    "srm_check",
    "srm_test",
    # CUPED variance reduction
    "cuped_theta",
    "cuped_adjust",
    "welch_mean_test",
    # sequential / always-valid
    "ratio_interim",
    "msprt_ratio",
    "run_stream",
]


# --------------------------------------------------------------------------- #
# Ratio-metric estimators (delta method)
# --------------------------------------------------------------------------- #


def ctr(df: pd.DataFrame, clicks: str = "clicks", views: str = "views") -> float:
    """Pooled ratio metric ``sum(clicks) / sum(views)``.

    ``clicks`` is the numerator column and ``views`` the denominator column; for
    revenue-per-search the caller passes ``clicks="revenue"``, ``views="searches"``.
    """
    return float(df[clicks].sum() / df[views].sum())


def ratio_delta_var_perunit(x, y) -> float:
    """Per-unit delta-method variance ``(Var(y) + R^2 Var(x) - 2 R Cov(x,y)) / mean(x)^2``.

    ``R = mean(y) / mean(x)`` is the ratio. Multiplying by ``1/n`` gives
    ``Var(R_hat)``. ``x`` is per-unit denominator, ``y`` per-unit numerator.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mean_x = x.mean()
    if mean_x == 0:
        raise ValueError("denominator x has mean zero; the ratio is undefined")
    ratio = y.mean() / mean_x
    var_x = np.var(x, ddof=1)
    var_y = np.var(y, ddof=1)
    cov = np.cov(x, y, ddof=1)[0, 1]
    return float((var_y + ratio**2 * var_x - 2 * ratio * cov) / mean_x**2)


def ratio_variance_per_user(
    df: pd.DataFrame, clicks: str = "clicks", views: str = "views"
) -> float:
    """Return ``V`` with ``Var(R_hat) ~= V / n`` (delegates to the array form).

    Same expression as ``ratio_delta_var_perunit(views, clicks)``: the residual
    ``Var(clicks - R * views) / mean(views)^2`` expands to it. ``V`` does not
    depend on ``n``, which is what lets one history estimate size any group.
    """
    return ratio_delta_var_perunit(df[views].to_numpy(), df[clicks].to_numpy())


def ratio_delta_se(x, y) -> float:
    """Asymptotic SE of ``R = sum(y)/sum(x)``: ``sqrt(ratio_delta_var_perunit / n)``."""
    n = len(np.asarray(x))
    if n < 2:
        raise ValueError("at least two per-unit observations are required")
    return float(np.sqrt(ratio_delta_var_perunit(x, y) / n))


def ratio_standard_error(df: pd.DataFrame, clicks: str = "clicks", views: str = "views") -> float:
    """SE of the pooled ratio for one group (delegates to ``ratio_delta_se``)."""
    return ratio_delta_se(df[views].to_numpy(), df[clicks].to_numpy())


def _two_sided_p_from_z(z: float) -> float:
    """``2 * P(Z > |z|)`` under the standard normal."""
    from scipy import stats

    return float(2 * stats.norm.sf(abs(z)))


def delta_method_test(x_a, y_a, x_b, y_b, alpha: float = 0.05) -> dict:
    """Two-sample delta-method test of ``R_B - R_A`` for ``R = sum(y)/sum(x)``.

    ``SE = sqrt(se_a^2 + se_b^2)`` from ``ratio_delta_se`` per group; ``z = diff/SE``.
    """
    from scipy import stats

    x_a, y_a = np.asarray(x_a, float), np.asarray(y_a, float)
    x_b, y_b = np.asarray(x_b, float), np.asarray(y_b, float)
    r_a = float(y_a.mean() / x_a.mean())
    r_b = float(y_b.mean() / x_b.mean())
    se = float(np.sqrt(ratio_delta_se(x_a, y_a) ** 2 + ratio_delta_se(x_b, y_b) ** 2))
    diff = r_b - r_a
    z = diff / se
    z_crit = float(stats.norm.ppf(0.975))
    p = _two_sided_p_from_z(z)
    return {
        "ratio_a": r_a,
        "ratio_b": r_b,
        "diff": float(diff),
        "se": se,
        "z": float(z),
        "p_value": p,
        "ci_low": float(diff - z_crit * se),
        "ci_high": float(diff + z_crit * se),
        "significant": bool(p < alpha),
    }


def ratio_metric_test(
    control: pd.DataFrame,
    test: pd.DataFrame,
    clicks: str = "clicks",
    views: str = "views",
) -> dict:
    """Two-sided delta-method test (delegates to ``delta_method_test``)."""
    res = delta_method_test(
        control[views].to_numpy(),
        control[clicks].to_numpy(),
        test[views].to_numpy(),
        test[clicks].to_numpy(),
    )
    c_a = res["ratio_a"]
    return {
        "control": c_a,
        "test": res["ratio_b"],
        "diff": res["diff"],
        "abs_lift": res["diff"] / c_a if c_a else float("nan"),
        "se": res["se"],
        "z": res["z"],
        "p_value": res["p_value"],
        "ci_low": res["ci_low"],
        "ci_high": res["ci_high"],
    }


def naive_binomial_test(
    control: pd.DataFrame,
    test: pd.DataFrame,
    clicks: str = "clicks",
    views: str = "views",
) -> dict:
    """View-level two-proportion test -- the *wrong* unit, kept for contrast.

    ``SE = sqrt(p_pool (1 - p_pool) (1/n_a + 1/n_b))`` with ``n`` counted in
    *views*. Ignores that clicks cluster inside users, so it is anti-conservative.
    """
    n_a, n_b = float(control[views].sum()), float(test[views].sum())
    c_a, c_b = ctr(control, clicks, views), ctr(test, clicks, views)
    p_pool = (control[clicks].sum() + test[clicks].sum()) / (n_a + n_b)
    se = float(np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b)))
    z = (c_b - c_a) / se if se else float("nan")
    return {"diff": float(c_b - c_a), "se": se, "z": float(z), "p_value": _two_sided_p_from_z(z)}


def naive_per_unit_ratio_test(x_a, y_a, x_b, y_b, alpha: float = 0.05) -> dict:
    """Naive: form the per-unit ratio ``r_i = y_i / x_i`` then Welch t-test.

    Biased whenever the denominator ``x`` varies across units, because it weights
    a user with one unit as heavily as a user with many. Compared against the
    delta method to show the gap.
    """
    from scipy import stats

    x_a, y_a = np.asarray(x_a, float), np.asarray(y_a, float)
    x_b, y_b = np.asarray(x_b, float), np.asarray(y_b, float)
    if np.any(x_a == 0) or np.any(x_b == 0):
        raise ValueError("naive per-unit ratio is undefined for a zero denominator")
    r_a, r_b = y_a / x_a, y_b / x_b
    res = stats.ttest_ind(r_b, r_a, equal_var=False, alternative="two-sided")
    se = float(np.sqrt(np.var(r_b, ddof=1) / len(r_b) + np.var(r_a, ddof=1) / len(r_a)))
    return {
        "ratio_a": float(r_a.mean()),
        "ratio_b": float(r_b.mean()),
        "diff": float(r_b.mean() - r_a.mean()),
        "se": se,
        "p_value": float(res.pvalue),
        "significant": bool(res.pvalue < alpha),
    }


def bootstrap_lift(
    control: pd.DataFrame,
    test: pd.DataFrame,
    n_boot: int = 5000,
    seed: int = 42,
    clicks: str = "clicks",
    views: str = "views",
) -> dict:
    """User-level bootstrap CI for ``R_B - R_A``; resamples users with replacement."""
    rng = np.random.default_rng(seed)
    a_c, a_v = control[clicks].to_numpy(), control[views].to_numpy()
    b_c, b_v = test[clicks].to_numpy(), test[views].to_numpy()
    n_a, n_b = len(a_c), len(b_c)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        ia = rng.integers(0, n_a, n_a)
        ib = rng.integers(0, n_b, n_b)
        diffs[i] = b_c[ib].sum() / b_v[ib].sum() - a_c[ia].sum() / a_v[ia].sum()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "se": float(diffs.std(ddof=1)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "excludes_zero": bool(lo > 0 or hi < 0),
    }


# --------------------------------------------------------------------------- #
# Sample sizing, power, allocation
# --------------------------------------------------------------------------- #


def required_sample_size(
    baseline: float,
    rel_mde: float,
    variance_per_user: float,
    alpha: float = 0.05,
    power: float = 0.8,
    test_share: float = 0.5,
) -> dict:
    """Users needed for a ratio metric, arbitrary allocation.

    ``n_total = V * (1/(1-s) + 1/s) * (z_{1-a/2} + z_power)^2 / (baseline*rel_mde)^2``
    with ``s = test_share``. Unequal allocation inflates the total because the SE
    depends on ``1/n_control + 1/n_test``: shrinking one arm costs more than the
    traffic it frees.
    """
    from scipy import stats

    if not 0 < test_share < 1:
        raise ValueError("test_share must be strictly between 0 and 1")
    z_a = float(stats.norm.ppf(1 - alpha / 2))
    z_b = float(stats.norm.ppf(power))
    delta = baseline * rel_mde
    n_total = (
        variance_per_user * (1 / (1 - test_share) + 1 / test_share) * (z_a + z_b) ** 2 / delta**2
    )
    return {
        "n_total": float(n_total),
        "n_control": float(n_total * (1 - test_share)),
        "n_test": float(n_total * test_share),
        "delta_abs": float(delta),
    }


def achieved_power(
    n_total: float,
    baseline: float,
    rel_mde: float,
    variance_per_user: float,
    alpha: float = 0.05,
    test_share: float = 0.5,
) -> float:
    """Power reached at ``n_total``: ``Phi(baseline*rel_mde / SE - z_{1-a/2})``.

    Present in the source module but missing from its ``__all__``; exported here.
    """
    from scipy import stats

    z_a = float(stats.norm.ppf(1 - alpha / 2))
    se = float(
        np.sqrt(variance_per_user * (1 / (n_total * (1 - test_share)) + 1 / (n_total * test_share)))
    )
    return float(stats.norm.cdf(baseline * rel_mde / se - z_a))


def srm_check(n_control: int, n_test: int, expected_test_share: float = 0.5) -> dict:
    """Two-arm chi-square SRM check against the *planned* split.

    ``chi2 = sum((observed - expected)^2 / expected)`` with one degree of freedom.
    A designed 90/10 rollout fails a 50/50 check by construction, so the expected
    share must be passed in, never assumed.
    """
    from scipy import stats

    n = n_control + n_test
    expected = np.array([n * (1 - expected_test_share), n * expected_test_share])
    observed = np.array([n_control, n_test], dtype=float)
    chi2 = float(((observed - expected) ** 2 / expected).sum())
    return {
        "chi2": chi2,
        "p_value": float(stats.chi2.sf(chi2, df=1)),
        "observed_share": n_test / n,
    }


def srm_test(observed, expected_ratio, alpha: float = 0.001) -> dict:
    """k-arm chi-square goodness-of-fit SRM test.

    ``chi2, p = scipy.stats.chisquare(observed, expected_ratio * sum(observed))``;
    flags a mismatch when ``p < alpha``. Default ``alpha = 0.001`` is stricter than
    the usual test level because SRM false alarms are costly at scale.
    """
    from scipy import stats

    observed = np.asarray(observed, dtype=float)
    expected_ratio = np.asarray(expected_ratio, dtype=float)
    if not np.isclose(expected_ratio.sum(), 1.0):
        raise ValueError(f"expected_ratio must sum to 1, got {expected_ratio.sum()}")
    expected = expected_ratio * observed.sum()
    chi2, p = stats.chisquare(observed, expected)
    return {
        "observed": observed.tolist(),
        "expected_ratio": expected_ratio.tolist(),
        "observed_ratio": (observed / observed.sum()).tolist(),
        "chi2": float(chi2),
        "p_value": float(p),
        "srm_detected": bool(p < alpha),
        "alpha": float(alpha),
    }


# --------------------------------------------------------------------------- #
# CUPED variance reduction
# --------------------------------------------------------------------------- #


def cuped_theta(y, x) -> float:
    """Regression coefficient ``Cov(Y, X) / Var(X)`` of the outcome on the covariate.

    Estimated from the pooled (control + test) pre/post pairs, so treatment cannot
    contaminate it. Returns 0 when ``Var(X) == 0`` (a constant covariate carries no
    information) instead of propagating ``nan``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    var_x = float(np.var(x, ddof=1))
    if not var_x > 0:
        return 0.0
    cov = float(np.cov(x, y, ddof=1)[0, 1])
    return float(cov / var_x)


def cuped_adjust(y, x, theta) -> np.ndarray:
    """Adjusted outcome ``Y_adj = Y - theta * (X - mean(X))`` (mean from the pooled X)."""
    x = np.asarray(x, dtype=float)
    return np.asarray(y, dtype=float) - theta * (x - x.mean())


def welch_mean_test(a, b) -> dict:
    """Welch two-sample test of ``mean(b) - mean(a)`` with unequal variances.

    ``SE = sqrt(Var(a)/(n_a-1 ... ))`` computed as
    ``sqrt(var_a/n_a + var_b/n_b)`` with sample variances; CI is normal-based.
    """
    from scipy import stats

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    _, p = stats.ttest_ind(a, b, equal_var=False)
    diff = float(np.mean(b) - np.mean(a))
    se = float(np.sqrt(np.var(a, ddof=1) / len(a) + np.var(b, ddof=1) / len(b)))
    z = float(stats.norm.ppf(0.975))
    return {
        "point": diff,
        "se": se,
        "ci_low": float(diff - z * se),
        "ci_high": float(diff + z * se),
        "p_value": float(p),
        "var_a": float(np.var(a, ddof=1)),
        "var_b": float(np.var(b, ddof=1)),
    }


# --------------------------------------------------------------------------- #
# Sequential / always-valid testing (mSPRT)
# --------------------------------------------------------------------------- #


def ratio_interim(x_a, y_a, x_b, y_b) -> dict:
    """One interim look: ``D = R_B - R_A``, ``z = D sqrt(I)``, ``I = n / V``.

    ``V = ratio_delta_var_perunit(A) + ratio_delta_var_perunit(B)`` is the per-unit
    variance of the difference; ``I`` is the Fisher information.
    """
    x_a, y_a = np.asarray(x_a, float), np.asarray(y_a, float)
    x_b, y_b = np.asarray(x_b, float), np.asarray(y_b, float)
    r_a = float(y_a.mean() / x_a.mean())
    r_b = float(y_b.mean() / x_b.mean())
    d = r_b - r_a
    v = ratio_delta_var_perunit(x_a, y_a) + ratio_delta_var_perunit(x_b, y_b)
    n = len(x_a)
    info = n / v
    return {
        "D": float(d),
        "z": float(d * np.sqrt(info)),
        "I": float(info),
        "R_a": r_a,
        "R_b": r_b,
        "se": float(np.sqrt(v / n)),
    }


def msprt_ratio(D: float, z: float, info: float, tau: float) -> tuple[float, float]:
    """Always-valid mSPRT p-value for an effect with normal prior ``N(0, tau^2)``.

    ``Lambda = (1 + tau^2 I)^(-1/2) * exp(tau^2 z^2 I / (2 (1 + tau^2 I)))``, then
    ``p = min(1, 1 / Lambda)``. ``tau`` is the prior SD of the effect in the same
    natural units as ``D`` (set it near the MDE you care about). Returns
    ``(Lambda, p)``.
    """
    lam = (1 + tau**2 * info) ** (-0.5) * np.exp(tau**2 * z**2 * info / (2 * (1 + tau**2 * info)))
    return float(lam), float(min(1.0, 1.0 / lam))


def run_stream(
    n_batches: int,
    batch_size: int,
    base_rate: float,
    rel_lift: float,
    tau: float,
    seed: int = 0,
) -> list[dict]:
    """Stream ``n_batches`` of synthetic per-user ``(denominator, numerator)`` pairs.

    At each interim accumulate the batch and record the fixed-horizon delta z and
    the always-valid mSPRT p-value, so the peeking gap can be compared. The
    synthetic generator keeps per-unit denominator heterogeneity (log-normal
    denominator), which is what makes the ratio metric non-trivial.
    """
    rng = np.random.default_rng(seed)
    xa_all = np.array([], dtype=float)
    ya_all = np.array([], dtype=float)
    xb_all = np.array([], dtype=float)
    yb_all = np.array([], dtype=float)
    records: list[dict] = []

    def group(rate: float) -> tuple[np.ndarray, np.ndarray]:
        x = rng.lognormal(mean=3.5, sigma=0.6, size=batch_size).astype(int) + 1
        y = rng.binomial(x, rate)
        return x, y

    for k in range(1, n_batches + 1):
        xa, ya = group(base_rate)
        xb, yb = group(base_rate * (1 + rel_lift))
        xa_all = np.concatenate([xa_all, xa])
        ya_all = np.concatenate([ya_all, ya])
        xb_all = np.concatenate([xb_all, xb])
        yb_all = np.concatenate([yb_all, yb])
        r = ratio_interim(xa_all, ya_all, xb_all, yb_all)
        _, p_av = msprt_ratio(r["D"], r["z"], r["I"], tau)
        records.append(
            {
                "n": k * batch_size,
                **r,
                "p_av": p_av,
                "p_naive": _two_sided_p_from_z(r["z"]),
            }
        )
    return records
