-- 03_funnel.sql
-- SERP funnel: search -> click -> redirect -> buy.
--
-- Input relation:  serp_offers
--   (search_id, user_id, position, partner, price, commission,
--    clicked, redirected, booked)
--
-- NOTE: serp_offers does not exist yet. Phase 2 (the SERP ranking simulator)
-- creates it. This SQL is written now so the funnel contract is fixed before
-- the simulator lands; it is executed only in later phases.

-- ---------------------------------------------------------------------------
-- Per-position funnel: where do users stop engaging down the SERP?
-- ---------------------------------------------------------------------------

CREATE OR REPLACE VIEW funnel_by_position AS
WITH counts AS (
    SELECT
        position,
        count(DISTINCT search_id)                                   AS impressions,
        count(*)                                                    AS offers,
        count(*) FILTER (WHERE clicked)                             AS clicks,
        count(*) FILTER (WHERE redirected)                          AS redirects,
        count(*) FILTER (WHERE booked)                              AS bookings,
        sum(commission) FILTER (WHERE booked)                       AS booked_commission,
        avg(price)                                                  AS avg_price
    FROM serp_offers
    GROUP BY position
)
SELECT
    position,
    impressions,
    offers,
    clicks,
    redirects,
    bookings,
    booked_commission,
    avg_price,
    clicks::DOUBLE    / NULLIF(impressions, 0)                      AS ctr,
    redirects::DOUBLE / NULLIF(clicks, 0)                           AS click_to_redirect,
    bookings::DOUBLE  / NULLIF(redirects, 0)                        AS redirect_to_book,
    bookings::DOUBLE  / NULLIF(impressions, 0)                      AS impression_to_book,
    -- Cumulative reach: how many of the top-N offers got any engagement at all.
    sum(clicks) OVER (ORDER BY position ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
        / NULLIF(sum(impressions) OVER (), 0)                       AS cumulative_click_share
FROM counts;

-- ---------------------------------------------------------------------------
-- Partner-level monetisation: which partners earn their slot?
-- ---------------------------------------------------------------------------

CREATE OR REPLACE VIEW partner_quality AS
WITH partner_slots AS (
    SELECT
        partner,
        count(*)                                                    AS offers,
        avg(position)                                               AS avg_position,
        count(*) FILTER (WHERE clicked)                             AS clicks,
        count(*) FILTER (WHERE booked)                              AS bookings,
        sum(commission)                                             AS commission,
        sum(commission) FILTER (WHERE booked)                       AS booked_commission,
        count(DISTINCT search_id)                                   AS searches_present
    FROM serp_offers
    GROUP BY partner
)
SELECT
    *,
    clicks::DOUBLE   / NULLIF(offers, 0)                            AS ctr,
    bookings::DOUBLE / NULLIF(offers, 0)                            AS booking_rate,
    booked_commission / NULLIF(offers, 0)                           AS commission_per_offer,
    booked_commission / NULLIF(searches_present, 0)                 AS commission_per_search,
    -- Rank partners by realised revenue per slot they occupied.
    rank() OVER (ORDER BY booked_commission / NULLIF(offers, 0) DESC) AS monetisation_rank
FROM partner_slots;

-- ---------------------------------------------------------------------------
-- Per-search funnel shape: did this SERP convert at all, and at what depth?
-- ---------------------------------------------------------------------------

CREATE OR REPLACE VIEW search_funnel AS
WITH search_rollup AS (
    SELECT
        search_id,
        user_id,
        count(*)                                                    AS offers,
        min(position) FILTER (WHERE clicked)                        AS first_clicked_position,
        min(position) FILTER (WHERE booked)                         AS first_booked_position,
        count(*) FILTER (WHERE clicked)                             AS clicks,
        count(*) FILTER (WHERE booked)                              AS bookings,
        sum(commission) FILTER (WHERE booked)                       AS commission
    FROM serp_offers
    GROUP BY search_id, user_id
)
SELECT
    *,
    (clicks > 0)                                                    AS engaged,
    (bookings > 0)                                                  AS converted,
    CASE
        WHEN bookings > 0 THEN 'booked'
        WHEN clicks > 0   THEN 'clicked_no_book'
        WHEN offers > 0   THEN 'no_click'
        ELSE 'empty_serp'
    END                                                             AS funnel_outcome,
    -- Depth of the winning result relative to the SERP it sat in.
    first_booked_position::DOUBLE / NULLIF(offers, 0)               AS winning_depth_ratio
FROM search_rollup;
