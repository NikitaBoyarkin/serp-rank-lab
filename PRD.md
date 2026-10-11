# PRD -- SERP Rank Lab

## Problem

A flight metasearch product shows a ranked list of offers on its search results
page. Ranking is the core product surface: it decides what a user sees first, and
it decides which partner earns the commission. Yet ranking changes are routinely
shipped on anecdote -- "the top result looks bad" -- because the two effects are
measured separately, if at all: a relevance effect on the user and a revenue
effect for the marketplace.

This lab exists to make both measurable on one dataset: the same search can be
scored for user-side funnel quality (does the user click, redirect, book?) and for
marketplace-side monetisation (commission per slot).

## Target user

A Product Analyst on the Aviasales SERP. Their recurring questions:

* Where does the funnel leak on the results page -- impressions, clicks, redirects
  or bookings?
* Which partners occupy expensive top positions without converting?
* How deep must a user scroll before the winning offer appears, and does that
  depth change between mobile and desktop?
* Did the last ranking change move conversion, and did it move revenue in the same
  direction?

## Success = metrics this lab demonstrates

| Metric | Grain | Phase |
|---|---|---|
| Booking legs per booking, connection share | booking / leg | 1 (done) |
| Sessions per user, session length, events per session | user / session | 1 (done) |
| CTR by SERP position | position | 2 |
| Click -> redirect -> book conversion by position | position | 2 |
| Commission per offer slot, by partner | partner | 2 |
| Winning-offer depth ratio per search | search | 2 |
| Ranking-policy uplift (conversion and commission) | experiment arm | 3 |
| Guardrail: session-level booking rate stability | experiment arm | 3 |

## Scope -- three phases

1. **Data foundation.** Ingest the three real feeds; explode the booking feed to
   one row per flight leg; normalise legs, orders and events into DuckDB views;
   sessionise the event stream with the 30-minute inactivity + date-boundary rule;
   fix the funnel SQL contract over `serp_offers`.
2. **SERP ranking simulator + evaluation.** Materialise `serp_offers` (search_id,
   user_id, position, partner, price, commission, clicked, redirected, booked)
   from the leg-level data, define and implement two or more ranking policies, and
   evaluate them against the funnel and monetisation views in `sql/03_funnel.sql`.
3. **A/B experiment + dashboard.** Run the ranking policies as arms against the
   real `ab_group` assignment in the events feed, check for sample-ratio mismatch,
   measure the primary and guardrail metrics, and ship a dashboard over the
   resulting aggregates.

## Out of scope

* Any production deployment or live traffic.
* Pricing, availability or inventory modelling -- offers come from the historical
  feed, not a live supplier API.
* User-level personalisation or ML ranking models; the simulator uses explicit,
  inspectable policy functions, not a learned ranker.
* Return-ticket modelling: the booking feed contains one-way itineraries only, and
  the orders feed's `is_return` flag is used as a segmentation attribute, not as a
  reconstructed itinerary.
* Committing any raw third-party booking data to this repository.
