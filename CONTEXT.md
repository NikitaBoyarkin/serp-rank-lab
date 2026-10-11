# CONTEXT — SERP Rank Lab glossary

Terms used across this lab, with their **precise** meaning here. This file defines
vocabulary only — no implementation detail, no spec, no code. Where a term appears
in the project tells you which dataset it belongs to.

## Money terms

- **revenue** — RESERVED: the **partner commission** the site earns on `serp_offers`
  offers only. CPA commission on a booked offer, or the CPC fee on a clicked one, and
  0 otherwise; `sum(revenue)` over a SERP is the site's earnings for that search.
  Appears: `serp_offers.commission`; `revenue / search` in `src/ranking_eval.py`.
- **order_value** — the real `order_price` of an order in the orders feed. It is
  **not** revenue: it is a booking's price, unrelated to partner commission.
  Appears: `orders.order_price` (view `normalized_orders`).
- **service_profit** — profit on a single paid ancillary service inside an `orders`
  row. Median ~0 and only ~20% of orders carry a labelled profit.
  Appears: `orders.service_profit`, `orders.service_margin` (view `normalized_orders`).

## Position-bias terms

- **position bias** — the tendency of users to examine and click top slots far more
  than lower ones, so naive click rates understate true click propensity.
  Appears: `src/serp_sim.py`, `src/ranking_eval.py`.
- **examine_prob** — `P(examine at position k)`, the probability a user looks at a
  slot; modelled as the decaying curve `(1/k)^eta`.
  Appears: `serp_offers.examine_prob`, `examination_curve` in `src/serp_sim.py`.
- **propensity** — the per-position inverse-weight used for correction; here it is
  `P(examine at k)` (the same quantity as `examine_prob`).
  Appears: `propensity` argument of `unbiased_ctr` in `src/ranking_eval.py`.
- **expected_revenue** — deterministically modelled revenue for an offer:
  `P(click) * revenue_if_engaged`, i.e. `click_prob` times the CPA commission or CPC
  fee. Appears: `serp_offers.expected_revenue`; `src/ranking_eval.py`.

## Estimation terms

- **IPW** — inverse-propensity weighting (Horvitz–Thompson): correct each click by
  `1 / propensity(position)` to recover a position-free click rate.
  Appears: `unbiased_ctr` in `src/ranking_eval.py`.
- **EM** — expectation-maximisation estimate of the examination curve from the logs
  alone, alternating an examination estimate and a relevance estimate.
  Appears: `estimate_propensity_em` in `src/ranking_eval.py`.
- **CUPED** — variance reduction using a pre-period covariate; here the pre-period
  revenue per search, with a shuffled covariate as mutation control.
  Appears: `src/experiment.py`.
- **mSPRT** — mixture sequential probability ratio test, an always-valid sequential
  test used to quantify the peeking penalty.
  Appears: `src/experiment.py`.
- **SRM** — sample ratio mismatch: a chi-square check that the observed arm split
  matches the planned allocation.
  Appears: `src/experiment.py`.

## Ranking / evaluation terms

- **NDCG@k** — normalised discounted cumulative gain at cutoff k; graded relevance
  with a log2 discount, normalised by the ideal ordering.
  Appears: `src/ranking_eval.py`.
- **MRR** — mean reciprocal rank; `1 / rank` of the first item above a relevance
  threshold, else 0. Appears: `src/ranking_eval.py`.
- **interleaving** — team-draft interleaving: blend two rankers' results and credit
  each click to the ranker that contributed the clicked slot; a position-bias-robust
  online comparison. Appears: `src/ranking_eval.py`.

## Entity terms

- **partner** — the **partner name** (the metasearch supplier, e.g. `OTA Global`)
  that an offer links out to. Appears: `serp_offers.partner`.
- **partner_type** — the economic type of a partner, **CPA vs CPC**; a different
  attribute from `partner` (which is the name). Appears: `serp_offers.partner_type`.

## Provenance terms

- **modeled vs measured** — `serp_offers` is **modeled / synthetic** (simulated
  offers built on *real* `user_id` values); `orders`, `bookings_legs`, `events` and
  `sessions` are **measured / real**. The two are never fused. Appears: throughout —
  see `docs/adr/0001-modeled-vs-measured.md`.

## Data sources

Only `user_id` joins the three clickstream relations: `events` ↔ `sessions` ↔
`serp_offers`. `orders` and `bookings_legs` are **islands** — they carry no foreign
key to the clickstream, and `orders` does not overlap the clickstream in time.
