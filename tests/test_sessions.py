"""Contract tests for the sessionisation rule in ``sql/02_sessions.sql``.

The fixture below is hand-written to exercise BOTH halves of the rule:

* a 10-minute gap on the SAME date must NOT split,
* a 10-minute gap ACROSS a date boundary MUST split,
* a 45-minute gap on the SAME date MUST split.

These expected boundaries are the contract for the 30-minute rule.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.sessions import build_sessions

USER = "user-1"
OTHER_USER = "2OgK/ukRgvkWAOaa8XaTPg=="

# One user, six events. See the assertions for the expected split points.
EVENTS = [
    # -- session 1: opens, then a 10-min same-date gap that must NOT split --
    ("2023-05-02 12:00:00", "2023-05-02"),
    ("2023-05-02 12:10:00", "2023-05-02"),
    # -- 45 minutes of inactivity on the SAME date -> session 2 --
    ("2023-05-02 12:55:00", "2023-05-02"),
    # -- 11 hours later, still the SAME date -> session 3 --
    ("2023-05-02 23:55:00", "2023-05-02"),
    # -- 10-min gap but the DATE CHANGES -> session 4 --
    ("2023-05-03 00:05:00", "2023-05-03"),
    # -- 15-min same-date gap -> still session 4 --
    ("2023-05-03 00:20:00", "2023-05-03"),
]


@pytest.fixture
def events_df():
    rows = [{"user_id": USER, "ab_group": "A", "ts": ts, "pdate": pdate} for ts, pdate in EVENTS]
    # A second user with a single event: sessionisation must not leak across users.
    rows.append(
        {
            "user_id": OTHER_USER,
            "ab_group": "B",
            "ts": "2023-05-02 12:05:00",
            "pdate": "2023-05-02",
        }
    )
    frame = pd.DataFrame(rows)
    frame["ts"] = pd.to_datetime(frame["ts"])
    frame["pdate"] = pd.to_datetime(frame["pdate"]).dt.date
    return frame


def test_gap_of_45_minutes_on_same_date_splits(events_df):
    sessions = build_sessions(events_df)
    first = sessions[(sessions["user_id"] == USER)].iloc[0]

    # 12:00 and 12:10 collapse; 12:55 starts a new session.
    assert first["start_ts"] == pd.Timestamp("2023-05-02 12:00:00")
    assert first["end_ts"] == pd.Timestamp("2023-05-02 12:10:00")


def test_gap_of_10_minutes_across_date_boundary_splits(events_df):
    sessions = build_sessions(events_df)
    user_sessions = sessions[sessions["user_id"] == USER].reset_index(drop=True)

    # The 23:55 -> 00:05 pair is only 10 minutes apart but crosses midnight.
    assert user_sessions.loc[2, "end_ts"] == pd.Timestamp("2023-05-02 23:55:00")
    assert user_sessions.loc[3, "start_ts"] == pd.Timestamp("2023-05-03 00:05:00")


def test_gap_of_10_minutes_on_same_date_does_not_split(events_df):
    sessions = build_sessions(events_df)
    first = sessions[sessions["user_id"] == USER].iloc[0]

    assert first["start_ts"] == pd.Timestamp("2023-05-02 12:00:00")
    assert first["end_ts"] == pd.Timestamp("2023-05-02 12:10:00")


def test_expected_session_boundaries_are_exact(events_df):
    sessions = build_sessions(events_df)
    user_sessions = sessions[sessions["user_id"] == USER].reset_index(drop=True)

    assert len(user_sessions) == 4

    expected = [
        ("2023-05-02 12:00:00", "2023-05-02 12:10:00", "2023-05-02"),
        ("2023-05-02 12:55:00", "2023-05-02 12:55:00", "2023-05-02"),
        ("2023-05-02 23:55:00", "2023-05-02 23:55:00", "2023-05-02"),
        ("2023-05-03 00:05:00", "2023-05-03 00:20:00", "2023-05-03"),
    ]
    actual = [
        (
            row.start_ts.strftime("%Y-%m-%d %H:%M:%S"),
            row.end_ts.strftime("%Y-%m-%d %H:%M:%S"),
            str(row.pdate),
        )
        for row in user_sessions.itertuples()
    ]
    assert actual == expected


def test_sessions_do_not_leak_across_users(events_df):
    sessions = build_sessions(events_df)

    assert set(sessions["user_id"]) == {USER, OTHER_USER}

    other = sessions[sessions["user_id"] == OTHER_USER]
    assert len(other) == 1
    assert other["ab_group"].iloc[0] == "B"
    assert other["start_ts"].iloc[0] == pd.Timestamp("2023-05-02 12:05:00")
    assert other["end_ts"].iloc[0] == pd.Timestamp("2023-05-02 12:05:00")


def test_output_has_the_contracted_columns(events_df):
    sessions = build_sessions(events_df)
    assert sessions.columns.tolist() == ["user_id", "ab_group", "start_ts", "end_ts", "pdate"]


def test_single_event_user_yields_one_session(events_df):
    sessions = build_sessions(events_df)
    other = sessions[sessions["user_id"] == OTHER_USER]
    assert len(other) == 1


def test_sessions_are_chronological_per_user(events_df):
    sessions = build_sessions(events_df)
    user_sessions = sessions[sessions["user_id"] == USER]
    starts = user_sessions["start_ts"].tolist()
    assert starts == sorted(starts)
    assert all(
        end >= start
        for start, end in zip(user_sessions["start_ts"], user_sessions["end_ts"], strict=True)
    )
