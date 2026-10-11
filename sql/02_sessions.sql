-- 02_sessions.sql
-- Sessionisation of the raw event stream.
--
-- Input relation:  events (user_id, ab_group, ts, pdate)
-- Output columns:  user_id, ab_group, start_ts, end_ts, pdate
--
-- Assignment rule, applied exactly:
--   * a new session starts after 30 minutes of inactivity (gap > 1800s), AND
--   * a session is broken whenever the calendar date changes, even when the
--     gap is under 30 minutes.
--
-- Implementation: lag() to measure the gap, a CASE flag, then a cumulative
-- sum() over that flag to number the sessions. No Python loop.

WITH ordered AS (
    SELECT
        user_id,
        ab_group,
        ts,
        pdate,
        lag(ts)    OVER (PARTITION BY user_id ORDER BY ts) AS prev_ts,
        lag(pdate) OVER (PARTITION BY user_id ORDER BY ts) AS prev_pdate
    FROM events
),
flagged AS (
    SELECT
        *,
        CASE
            -- first touch ever -> session opener
            WHEN prev_ts IS NULL THEN 1
            -- date boundary -> always a new session
            WHEN pdate IS DISTINCT FROM prev_pdate THEN 1
            -- more than 30 minutes of inactivity -> new session
            WHEN (epoch(ts) - epoch(prev_ts)) > 1800 THEN 1
            ELSE 0
        END AS is_new_session
    FROM ordered
),
numbered AS (
    SELECT
        *,
        sum(is_new_session) OVER (
            PARTITION BY user_id
            ORDER BY ts
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS session_number
    FROM flagged
)
SELECT
    user_id,
    any_value(ab_group) AS ab_group,
    min(ts)             AS start_ts,
    max(ts)             AS end_ts,
    pdate
FROM numbered
GROUP BY user_id, session_number, pdate
ORDER BY user_id, start_ts
