"""Smoke tests for src/plots.py: every builder returns a Figure; saving writes a file.

No pixel assertions -- these pin that builders construct without error and that
``save_figure`` lands a PNG under the (monkeypatched) figures directory.
"""

from __future__ import annotations

import datetime as dt

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.figure import Figure

from src import plots
from src.plots import (
    plot_ab_activity,
    plot_cumulative_share,
    plot_funnel_by_position,
    plot_geo,
    plot_orders_mix,
    plot_partner_quadrant,
    plot_position_bias,
    plot_ranking_divergence,
    plot_refund_rate,
    plot_routes,
    plot_seasonality,
    plot_sensitivity_sweep,
    save_all,
    save_figure,
)


def _offers() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "position": [1, 2, 1, 2],
            "device": ["mobile", "mobile", "desktop", "desktop"],
            "examine_prob": [1.0, 0.5, 1.0, 0.6],
        }
    )


def _funnel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "position": [1, 2],
            "impressions": [100, 100],
            "clicks": [60, 30],
            "redirects": [30, 12],
            "bookings": [10, 4],
        }
    )


def _partner_quality() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "partner": ["p1", "p2", "p3"],
            "ctr": [0.4, 0.2, 0.1],
            "commission_per_offer": [300.0, 200.0, 100.0],
            "searches_present": [5000, 3000, 1000],
        }
    )


def _revenue_by_position() -> pd.DataFrame:
    return pd.DataFrame({"position": [1, 2, 3], "cumulative_revenue_share": [0.5, 0.8, 1.0]})


def _ranking_report() -> dict:
    return {
        "policies": [
            {"policy": "relevance", "ndcg_at_k": 1.0, "revenue_per_search": 869.5},
            {"policy": "revenue", "ndcg_at_k": 0.93, "revenue_per_search": 1193.7},
        ]
    }


def _sensitivity() -> dict:
    return {
        "eta": [
            {"eta": 0.6, "ndcg_delta": 0.069, "revenue_delta": -309.7, "divergence_holds": True},
            {"eta": 1.6, "ndcg_delta": 0.069, "revenue_delta": -305.3, "divergence_holds": True},
        ],
        "money_tilt": [
            {"money_tilt": 0.0, "logged_ndcg_at_k": 1.0, "logged_revenue_per_search": 901.2},
            {"money_tilt": 1.0, "logged_ndcg_at_k": 0.913, "logged_revenue_per_search": 1196.4},
        ],
    }


def _orders() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "service": ["refund", "changes", None, "support"],
            "origin_country": ["RU", "RU", "KZ", "KZ"],
            "destination_country": ["TR", "AE", "TR", "AE"],
        }
    )


def _legs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "origin": ["SVO", "SVO", "VKO"],
            "destination": ["LED", "LED", "AER"],
            "booking_month": [1, 2, 2],
        }
    )


def _events() -> pd.DataFrame:
    day = dt.date(2023, 5, 1)
    return pd.DataFrame(
        [
            {"user_id": "a0", "ab_group": "A", "pdate": day},
            {"user_id": "a1", "ab_group": "A", "pdate": day},
            {"user_id": "b0", "ab_group": "B", "pdate": day},
        ]
    )


def _sessions() -> pd.DataFrame:
    day = dt.date(2023, 5, 1)
    return pd.DataFrame(
        [
            {"user_id": "a0", "ab_group": "A", "pdate": day},
            {"user_id": "b0", "ab_group": "B", "pdate": day},
        ]
    )


def _check(fig) -> None:
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_position_bias_returns_figure():
    _check(plot_position_bias(_offers()))


def test_plot_funnel_by_position_returns_figure():
    _check(plot_funnel_by_position(_funnel()))


def test_plot_partner_quadrant_returns_figure():
    _check(plot_partner_quadrant(_partner_quality()))


def test_plot_ranking_divergence_returns_figure():
    _check(plot_ranking_divergence(_ranking_report()))


def test_plot_sensitivity_sweep_returns_figure():
    _check(plot_sensitivity_sweep(_ranking_report() | {"sensitivity": _sensitivity()}))


def test_plot_cumulative_share_returns_figure():
    _check(plot_cumulative_share(_revenue_by_position()))


def test_plot_orders_mix_returns_figure():
    _check(plot_orders_mix(_orders()))


def test_plot_refund_rate_returns_figure():
    _check(plot_refund_rate(_orders()))


def test_plot_geo_returns_figure():
    _check(plot_geo(_orders()))


def test_plot_routes_returns_figure():
    _check(plot_routes(_legs()))


def test_plot_seasonality_returns_figure():
    _check(plot_seasonality(_legs()))


def test_plot_ab_activity_returns_figure():
    _check(plot_ab_activity(_events(), _sessions()))


def test_save_figure_writes_png_under_figures_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(plots, "FIGURES_DIR", tmp_path)
    path = save_figure(plot_orders_mix(_orders()), "orders_mix")
    assert path == tmp_path / "orders_mix.png"
    assert path.exists()


def test_save_all_skips_missing_inputs_and_writes_present_ones(tmp_path, monkeypatch):
    monkeypatch.setattr(plots, "FIGURES_DIR", tmp_path)
    paths = save_all(orders=_orders(), bookings_legs=_legs())
    names = {p.name for p in paths}
    assert names == {
        "orders_mix.png",
        "refund_rate.png",
        "geo.png",
        "routes.png",
        "seasonality.png",
    }
    assert all(p.exists() for p in paths)


def test_save_all_with_no_inputs_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(plots, "FIGURES_DIR", tmp_path)
    assert save_all() == []


def test_save_all_writes_sweep_only_when_sensitivity_present(tmp_path, monkeypatch):
    monkeypatch.setattr(plots, "FIGURES_DIR", tmp_path)
    plain = {p.name for p in save_all(ranking_report=_ranking_report())}
    assert "sensitivity_sweep.png" not in plain

    report = _ranking_report() | {"sensitivity": _sensitivity()}
    swept = {p.name for p in save_all(ranking_report=report)}
    assert "sensitivity_sweep.png" in swept


@pytest.mark.parametrize("name", ["position_bias", "orders_mix"])
def test_save_figure_path_is_inside_figures_dir(tmp_path, monkeypatch, name):
    monkeypatch.setattr(plots, "FIGURES_DIR", tmp_path)
    fig = plot_position_bias(_offers()) if name == "position_bias" else plot_orders_mix(_orders())
    assert save_figure(fig, name).parent == tmp_path
