-- 04_revenue_per_search.sql
-- Metasearch economics over the SERP offer log.
--
-- Input relation:  serp_offers
--   (search_id, user_id, position, partner, price, commission,
--    clicked, redirected, booked)
--   `commission` is realised partner revenue for the row: the CPA commission on
--   a booked offer, or the CPC fee on a clicked offer, else 0. So sum(commission)
--   over a SERP is the site's revenue for that search.
--
-- Outputs:
--   revenue_per_search  -- one row per search_id: revenue, offers, clicks,
--                          bookings, revenue per click and per offer.
--   revenue_by_position -- one row per position: the aggregate that makes the
--                          "revenue decays with position" claim queryable.

-- ---------------------------------------------------------------------------
-- Per-search monetisation.
-- ---------------------------------------------------------------------------

CREATE OR REPLACE VIEW revenue_per_search AS
SELECT
    search_id,
    any_value(user_id)                                              AS user_id,
    count(*)                                                        AS offers,
    count(*) FILTER (WHERE clicked)                                 AS clicks,
    count(*) FILTER (WHERE redirected)                              AS redirects,
    count(*) FILTER (WHERE booked)                                  AS bookings,
    sum(commission)                                                 AS revenue,
    sum(commission) / NULLIF(count(*) FILTER (WHERE clicked), 0)    AS revenue_per_click,
    sum(commission) / NULLIF(count(*), 0)                           AS revenue_per_offer,
    count(*) FILTER (WHERE clicked)::DOUBLE / NULLIF(count(*), 0)   AS click_rate,
    count(*) FILTER (WHERE booked)::DOUBLE / NULLIF(count(*), 0)    AS booking_rate,
    -- CPC revenue is paid on the click; CPA revenue only on the booking.
    sum(commission) FILTER (WHERE clicked)                          AS click_revenue,
    sum(commission) FILTER (WHERE booked)                           AS booked_revenue
FROM serp_offers
GROUP BY search_id;

-- ---------------------------------------------------------------------------
-- Per-position monetisation: the decay curve the ranking debate turns on.
-- ---------------------------------------------------------------------------

CREATE OR REPLACE VIEW revenue_by_position AS
SELECT
    position,
    count(DISTINCT search_id)                                       AS impressions,
    count(*)                                                        AS offers,
    count(*) FILTER (WHERE clicked)                                 AS clicks,
    count(*) FILTER (WHERE booked)                                  AS bookings,
    sum(commission)                                                 AS revenue,
    sum(commission) / NULLIF(count(*), 0)                           AS revenue_per_offer,
    sum(commission) / NULLIF(count(DISTINCT search_id), 0)          AS revenue_per_impression,
    count(*) FILTER (WHERE clicked)::DOUBLE
        / NULLIF(count(DISTINCT search_id), 0)                      AS ctr,
    count(*) FILTER (WHERE booked)::DOUBLE
        / NULLIF(count(DISTINCT search_id), 0)                      AS bookings_per_impression,
    -- Share of all SERP revenue collected at this position and above.
    sum(sum(commission)) OVER (
        ORDER BY position ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) / NULLIF(sum(sum(commission)) OVER (), 0)                     AS cumulative_revenue_share
FROM serp_offers
GROUP BY position
ORDER BY position;
