-- 01_normalize.sql
-- Normalisation layer for SERP Rank Lab.
--
-- Inputs (registered by the pipeline before this file runs):
--   bookings_legs : leg-level output of src.data_ingest.load_bookings
--   orders        : paid-service orders feed
--   events        : raw product event feed
--
-- Outputs: three views on the leg / order / event grain, with derived window
-- columns that every later phase builds on.

-- ---------------------------------------------------------------------------
-- Flight legs: per-leg route, plus booking-level rollups and sequence context.
-- ---------------------------------------------------------------------------

CREATE OR REPLACE VIEW normalized_flights AS
WITH legs AS (
    SELECT
        booking_id,
        booking_month,
        passengers,
        price                                   AS booking_price,
        price / NULLIF(passengers, 0)           AS price_per_passenger,
        leg_index,
        origin,
        destination,
        airline,
        baggage,
        is_connection,
        has_baggage
    FROM bookings_legs
),
sequenced AS (
    SELECT
        *,
        count(*) OVER (PARTITION BY booking_id)                        AS leg_count,
        lag(origin)      OVER (PARTITION BY booking_id ORDER BY leg_index) AS prev_origin,
        lag(destination) OVER (PARTITION BY booking_id ORDER BY leg_index) AS prev_destination,
        sum(has_baggage::INT) OVER (
            PARTITION BY booking_id ORDER BY leg_index
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        )                                                              AS baggage_legs_so_far
    FROM legs
)
SELECT
    *,
    -- A leg is a true connection when the previous leg landed somewhere else.
    (leg_index > 0 AND prev_destination IS DISTINCT FROM origin)       AS is_real_connection,
    (leg_index = leg_count - 1)                                        AS is_final_leg,
    (baggage_legs_so_far > 0)                                          AS baggage_by_now
FROM sequenced;

-- Route-level demand: how many bookings and passengers each origin->destination
-- pair attracts, ranked inside the origin for Phase 2's SERP construction.
CREATE OR REPLACE VIEW route_demand AS
SELECT
    origin,
    destination,
    count(DISTINCT booking_id)                                          AS bookings,
    sum(passengers)                                                     AS passengers,
    avg(booking_price)                                                  AS avg_booking_price,
    count(*) FILTER (WHERE is_connection)                               AS connection_legs,
    count(*) FILTER (WHERE has_baggage)                                 AS baggage_legs,
    rank() OVER (PARTITION BY origin ORDER BY count(DISTINCT booking_id) DESC) AS demand_rank_in_origin
FROM normalized_flights
GROUP BY origin, destination;

-- ---------------------------------------------------------------------------
-- Orders: booking depth buckets and per-service monetisation.
-- ---------------------------------------------------------------------------

CREATE OR REPLACE VIEW normalized_orders AS
SELECT
    order_id,
    order_date,
    date_trunc('month', order_date)                                     AS order_month,
    order_price,
    service,
    service_price,
    service_profit,
    service_profit / NULLIF(service_price, 0)                           AS service_margin,
    origin_country,
    destination_country,
    order_platform,
    booking_depth,
    (is_return = 1)                                                     AS is_return,
    adults,
    children,
    (adults + children)                                                 AS party_size,
    CASE
        WHEN booking_depth < 7   THEN '0-7d'
        WHEN booking_depth < 30  THEN '7-30d'
        WHEN booking_depth < 90  THEN '30-90d'
        ELSE '90d+'
    END                                                                 AS booking_depth_bucket,
    count(*) OVER (PARTITION BY order_id)                               AS services_on_order,
    sum(service_profit) OVER (PARTITION BY order_id)                    AS order_service_profit
FROM orders;

-- ---------------------------------------------------------------------------
-- Events: attach the previous-touch gap used by the sessionisation rule.
-- ---------------------------------------------------------------------------

CREATE OR REPLACE VIEW normalized_events AS
SELECT
    user_id,
    ab_group,
    ts,
    pdate,
    lag(ts) OVER (PARTITION BY user_id ORDER BY ts)                     AS prev_ts,
    lag(pdate) OVER (PARTITION BY user_id ORDER BY ts)                  AS prev_pdate,
    epoch(ts) - epoch(lag(ts) OVER (PARTITION BY user_id ORDER BY ts))  AS gap_seconds
FROM events;
