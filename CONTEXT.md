# CONTEXT — SERP Rank Lab glossary

Terms used across this lab, with their **precise** meaning here. This file defines
vocabulary only — no implementation detail, no spec, no code. Where a term appears
in the project tells you which dataset it belongs to.

**Formal definitions, formulas and `file:line` sources live in
[`docs/metric-definitions.md`](docs/metric-definitions.md).** This file is the short
vocabulary; that file is the contract.

## Money terms

- **revenue** — RESERVED: the **partner commission** the site earns on `serp_offers`
  offers only. CPA commission on a booked offer, or the CPC fee on a clicked one, and
  0 otherwise; `sum(revenue)` over a SERP is the site's earnings for that search.
  Appears: `serp_offers.commission`; `revenue / search` in `src/ranking_eval.py`.
- **commission** — the column that *is* revenue at the offer row; the two words are used
  interchangeably, but "revenue" always means this and never the ticket price.
- **revenue per search** — `sum(commission) / count(distinct search_id)`; the primary
  metric of ranking and of the experiment readout. A ratio metric, so its variance needs
  the delta method with `user_id` as the cluster.
- **expected_revenue** — deterministically modelled revenue for an offer:
  `P(click) * revenue_if_engaged`, i.e. `click_prob` times the CPA commission or CPC
  fee. It is a simulator ground truth, so only oracles may read it.
  Appears: `serp_offers.expected_revenue`; `src/ranking_eval.py`.
- **order_value** — the real `order_price` of an order in the orders feed. It is
  **not** revenue: it is a booking's price, unrelated to partner commission.
  Appears: `orders.order_price` (view `normalized_orders`).
- **service_profit** — profit on a single paid ancillary service inside an `orders`
  row. Labelled on only ~20% of rows (1,830 of 8,966); among those the median is 399.2 ₽.
  The unlabelled rows are **not** zeros, so any full-table profit read describes labelled
  services only. Appears: `orders.service_profit`, `orders.service_margin` (view
  `normalized_orders`).
- **breakeven ₽ per forgone click** — `Δrevenue_per_search / Δclicks_per_search`,
  reported as a rate (981.83 ₽ on the headline pair). It is an exchange rate against an
  external value of a click (LTV, brand), which the data does not contain.

## Experiment terms

- **guardrail** — a metric that can block a rollout regardless of the primary metric:
  CTR, booking conversion, mean price in the top 5, share of offers above their SERP
  median. Refund rate is **not** one: there is no join between `orders` and the ranking.
- **stopping rule** — do not ship if a guardrail is worse than 10% relative in its
  harmful direction **and** significant after Holm correction over the family of five
  tests (primary + four guardrails); see `multiplicity` in `reports/experiment.json`.
  The CI bound is reported separately from the point estimate.
- **CUPED** — variance reduction using a pre-period covariate; here the pre-period
  revenue per search, with a constructed positive control. Because users are i.i.d., the
  real reduction is ~0 (`corr² = 0.004%`); see `docs/adr/0002`.
- **mSPRT** — mixture sequential probability ratio test, an always-valid sequential
  test used to quantify the peeking penalty.
- **SRM** — sample ratio mismatch: a chi-square check that the observed arm split
  matches the planned allocation.

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

## Estimation terms

- **IPW** — inverse-propensity weighting (Horvitz–Thompson): correct each click by
  `1 / propensity(position)` to recover a position-free click rate.
  Appears: `unbiased_ctr` in `src/ranking_eval.py`.
- **EM** — expectation-maximisation estimate of the examination curve from the logs
  alone, alternating an examination estimate and a relevance estimate.
  Appears: `estimate_propensity_em` in `src/ranking_eval.py`.

## Ranking / evaluation terms

- **NDCG@k** — normalised discounted cumulative gain at cutoff k; graded relevance
  with a log2 discount, normalised by the ideal ordering.
  Appears: `src/ranking_eval.py`.
- **MRR** — mean reciprocal rank; `1 / rank` of the first item whose relevance clears
  the threshold (`MRR_THRESHOLD = 0.5`), else 0. Computed for every policy and reported
  in `ranking_eval.json` under `policies` and `mrr_by_policy`.
- **information set** — what a policy is allowed to read. `logged` = observable from
  logs only (price, partner type, position). `relevance` and `expected_revenue` are
  simulator ground truth and belong to oracles only; the headline pair is `logged`.
- **interleaving** — team-draft interleaving: blend two rankers' results and credit
  each click to the ranker that contributed the clicked slot. In this lab it is a
  single-SERP demonstration, **not** a recommended online method.
- **oracle vs feasible** — an oracle reads ground truth; a feasible policy reads only
  observed data. Only feasible policies can enter `headline.json`.

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

`orders` is **order × service**, not one row per order: 8 966 rows, 8 803 unique
`order_id`, i.e. **163 duplicated `order_id` rows**. Any per-order metric must
deduplicate by `order_id`; counting rows as orders inflates multi-service orders.
