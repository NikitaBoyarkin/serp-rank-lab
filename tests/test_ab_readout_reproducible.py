"""Binds ``reports/experiment.json`` to a fresh recomputation.

``.claude/rules/claim-mechanism.md`` (project rule mirrored in this repo's style):
a claim with no mechanism is prose. The readout claims a verdict; until this file
existed, nothing went red if that claim drifted from the data. The recompute costs
one arm rebuild (~40 s), so it is a module-scoped fixture -- not one per test.

Skipped when the gitignored processed artifact is absent, like the other
data-dependent suites here.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT = PROJECT_ROOT / "data" / "processed" / "serp_offers.parquet"
REPORT = PROJECT_ROOT / "reports" / "experiment.json"

pytestmark = pytest.mark.skipif(
    not (ARTIFACT.exists() and REPORT.exists()),
    reason="needs data/processed/serp_offers.parquet and reports/experiment.json",
)


def _load_verifier():
    """Load ``scripts/verify_ab_readout.py`` by path -- ``scripts`` is not a package."""
    path = PROJECT_ROOT / "scripts" / "verify_ab_readout.py"
    spec = importlib.util.spec_from_file_location("verify_ab_readout", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


verifier = _load_verifier()


@pytest.fixture(scope="module")
def fresh() -> dict:
    return verifier.recompute()


@pytest.fixture(scope="module")
def stored() -> dict:
    return json.loads(REPORT.read_text())


def test_every_stored_number_reproduces(stored: dict, fresh: dict) -> None:
    """28 stored values, each within 1e-6 relative of the recomputed one."""
    rows = verifier.comparisons(stored, fresh)
    failed = [row for row in rows if row[3] == "FAIL"]
    assert not failed, "\n".join(
        f"{lbl}: stored {old:g} vs fresh {new:g}" for lbl, old, new, _ in failed
    )
    # A silently dropped comparison would keep this test green; pin the coverage.
    assert len(rows) >= 28


def test_manual_delta_se_agrees_with_the_project_helper(fresh: dict) -> None:
    """The clustered SE is not an artefact of one implementation of the formula."""
    assert fresh["manual_se"] == pytest.approx(fresh["primary"]["se"], rel=1e-9)


def test_verdict_is_do_not_ship_on_the_guardrails(stored: dict, fresh: dict) -> None:
    assert fresh["decision"]["verdict"] == "do_not_ship"
    assert fresh["decision"]["verdict"] == stored["decision"]["verdict"]
    assert set(fresh["decision"]["guardrails_breached"]) == {"ctr", "booking_conversion"}


def test_guardrail_breaches_survive_bh_correction(fresh: dict) -> None:
    """Four guardrails are tested at once; the breach cannot be a multiplicity fluke."""
    for name in ("ctr", "booking_conversion"):
        guardrail = fresh["guardrails"][name]
        assert guardrail["point_flag"] and guardrail["significant"]
        assert fresh["bh_p"][name] < verifier.xp.ALPHA
    for name in ("mean_price_top5", "share_above_route_median"):
        assert fresh["guardrails"][name]["breach"] is False


def test_primary_is_underpowered_at_this_sample_size(stored: dict) -> None:
    """The honest reading is "not detected", not "no effect": MDE exceeds the effect."""
    power = stored["power"]
    assert power["detectable_at_run"] is False
    assert power["run"]["mde_rel"] > abs(power["observed_effect_rel"])


def test_srm_is_judged_against_the_planned_split(fresh: dict) -> None:
    """A 90/10 rollout checked against 50/50 fails by construction; the planned share is what counts."""
    srm = fresh["srm"]
    assert srm["planned_test_share"] == pytest.approx(0.1)
    assert srm["vs_planned"]["mismatch"] is False
