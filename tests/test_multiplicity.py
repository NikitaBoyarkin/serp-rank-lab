"""Tests for the 5-test multiplicity correction (primary + 4 guardrails).

Pure functions on synthetic dicts: no data dependency, so the correction is pinned
even when ``data/processed/`` is absent.
"""

from __future__ import annotations

import pytest

from src.experiment import (
    ALPHA,
    _bh_adjust,
    _holm_adjust,
    decision_readout,
    multiplicity_readout,
)

FAMILY = [
    "primary",
    "ctr",
    "booking_conversion",
    "mean_price_top5",
    "share_above_route_median",
]


def _guardrail(p: float, *, point_flag: bool = False) -> dict:
    return {
        "p_value": p,
        "point_flag": point_flag,
        "significant": bool(p < ALPHA),
        "breach": bool(point_flag and p < ALPHA),
    }


def _family(primary_p: float = 0.001, **p_values: float) -> tuple[dict, dict]:
    primary = {
        "p_value": primary_p,
        "diff_abs": 0.4,
        "diff_rel": 0.1,
        "significant": bool(primary_p < ALPHA),
    }
    raw = dict.fromkeys(FAMILY[1:], 0.6)
    raw.update(p_values)
    return primary, {name: _guardrail(p) for name, p in raw.items()}


def test_holm_adjust_known_vector() -> None:
    raw = [0.01, 0.04, 0.03, 0.20, 0.50]
    adjusted = _holm_adjust(raw)
    assert adjusted == pytest.approx([0.05, 0.12, 0.12, 0.40, 0.50])
    ordered = [a for _, a in sorted(zip(raw, adjusted, strict=True))]
    assert ordered == sorted(ordered)
    assert all(a <= 1.0 for a in adjusted)


def test_holm_adjust_caps_at_one() -> None:
    assert _holm_adjust([0.5, 0.5, 0.5, 0.5, 0.5]) == [1.0] * 5


def test_holm_and_bh_are_monotone_in_sorted_order() -> None:
    raw = [0.001, 0.2, 0.04, 0.03, 0.5]
    for adjust in (_holm_adjust, _bh_adjust):
        adjusted = adjust(raw)
        ordered = [a for _, a in sorted(zip(raw, adjusted, strict=True))]
        assert ordered == sorted(ordered)
        assert all(0.0 <= a <= 1.0 for a in adjusted)


def test_multiplicity_readout_annotates_the_family() -> None:
    primary, guardrails = _family(primary_p=0.001, ctr=0.03)
    result = multiplicity_readout(primary, guardrails)

    assert result["family"] == FAMILY
    assert result["m"] == 5
    assert result["alpha"] == ALPHA
    assert result["method_default"] == "holm"

    ctr = result["metrics"]["ctr"]
    assert ctr["significant_raw"] is True
    assert ctr["significant_holm"] is False
    assert ctr["flips"] is True
    assert "ctr" in result["verdict_inputs_changed"]
    assert result["metrics"]["primary"]["significant_holm"] is True

    # In-place annotation matches the existing row["significant_bh"] style.
    assert guardrails["ctr"]["p_value_holm"] == pytest.approx(0.12)
    assert guardrails["ctr"]["significant_holm"] is False
    assert guardrails["ctr"]["breach_holm"] is False
    assert primary["significant_holm"] is True


def test_decision_readout_uses_corrected_significance() -> None:
    primary, guardrails = _family(primary_p=0.001)
    guardrails["ctr"] = _guardrail(0.03, point_flag=True)
    multiplicity_readout(primary, guardrails)

    assert guardrails["ctr"]["breach"] is True
    assert guardrails["ctr"]["breach_holm"] is False

    decision = decision_readout(primary, guardrails)
    assert decision["guardrails_breached"] == []
    assert decision["guardrails_breached_raw"] == ["ctr"]
    assert decision["verdict"] == "ship"
    assert decision["verdict_raw"] == "do_not_ship"
    assert decision["primary_significant"] is True
    assert decision["primary_significant_raw"] is True
    assert decision["multiplicity"] == "holm over primary + 4 guardrails"
