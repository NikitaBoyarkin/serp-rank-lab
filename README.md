# SERP Rank Lab

A flight-metasearch **search-results-page** analytics lab: ranking, position bias,
partner monetisation and A/B readout for the surface where a metasearch product
actually earns.

A metasearch site sells no tickets. Its revenue is a partner commission (CPA per
booking) or a cost-per-click, multiplied by **which position** the offer occupied
on the results page. So the two questions that dominate the role are:

1. **Ranking** — which offer deserves which position, and how deep do users
   actually scroll before they convert?
2. **Monetisation** — which partners and which positions earn their slot, and how
   much revenue does a given ranking policy produce per search?

This lab builds the measurement substrate for both, then runs the experiment
machinery on top of it.

## What it does

| Layer | Files | Role |
|---|---|---|
| Data | `src/data_ingest.py`, `sql/01_normalize.sql` | booking feed → one row per flight leg; orders; raw events |
| Sessions | `src/sessions.py`, `sql/02_sessions.sql` | 30-minute inactivity + date-boundary session rule |
| SERP | `src/serp_sim.py` | synthetic results pages over **real** route/price pairs; position-bias click model; CPA/CPC partner economics |
| Evaluation | `src/ranking_eval.py`, `sql/03_funnel.sql`, `sql/04_revenue_per_search.sql` | unbiased CTR, NDCG/MRR, interleaving, revenue per search |
| Experiment | `src/stats.py`, `src/experiment.py` | SRM, A/A, delta method, CUPED, mSPRT, peeking |
| Surface | `app.py`, `run.py` | Streamlit dashboard; single pipeline entry point |

## Results

All numbers below are reproducible from a fixed seed (`uv run python run.py --stage all`).

### Ranking: the offline metric and the money disagree

| Policy | NDCG@10 | Revenue / search | Clicks / search |
|---|---|---|---|
| Relevance-first | **1.000** | 869.5 | 1.763 |
| Revenue-first | 0.931 | **1,193.7** | 1.596 |
| Δ (relevance − revenue) | **+0.069** | **−324.2** | +0.168 |

Ranking by relevance maximises NDCG *and* clicks, while ranking by revenue earns
**38% more per search**. Optimising the offline metric would have shipped the
worse policy — which is exactly why the ranking team's north star has to be a
revenue-per-search ratio metric, not an offline relevance score.

### Position bias is real and correctable

| Quantity | Value |
|---|---|
| Naive CTR (clicks / impressions) | 0.163 |
| True position-free CTR | 0.629 |
| IPW-corrected, known examination curve | 0.625 |
| IPW-corrected, curve estimated by EM | 0.629 |

The naive CTR under-states the true click propensity by **~4×**, because top
positions get examined far more often. Inverse-propensity weighting against the
position examination curve `(1/k)^η` recovers it; the EM variant recovers it
without knowing the curve in advance.

### Experiment readout

| Check | Result | Reading |
|---|---|---|
| SRM vs planned 90/10 | χ² = 0.119, **p = 0.730** | Split is sound |
| SRM vs a naive 50/50 assumption | χ² = 2947, p ≈ 0 | The assumption, not the split, is wrong |
| A/A test | **p = 0.721** | No false positive on a null difference |
| Ratio metric, delta method | diff −77.2, SE 69.0, p = 0.264 | Not significant |
| Same metric, naive per-unit | SE 88.5, p = 0.449 | Naive SE is **1.28× too wide** |
| CUPED, real pre-period covariate | corr −0.006 → **+0.0%** variance cut | i.i.d. users: nothing to exploit |
| CUPED, correlated positive control | corr 0.335 → **+11.2%** variance cut | Recovers the theoretical ρ² reduction |
| Peeking, fixed-horizon α = 0.05 | FPR **14.5%** over 10 looks | 2.9× the nominal rate |
| Peeking, mSPRT always-valid | FPR **0.25%** | Inflation factor 58× |

### Required sample size

| Allocation | n total | Control | Test |
|---|---|---|---|
| 90/10 (planned) | **55,196** | 49,676 | 5,520 |
| 50/50 | 19,871 | 9,935 | 9,935 |

The 90/10 split costs **2.78×** the traffic of an even split for the same power —
the real price of protecting the overwhelming majority of users from a change.

## Data

| Dataset | File | Rows | Grain |
|---|---|---|---|
| Bookings (Aviastats) | `data.csv` | 135,508 | booking, exploded to 137,082 leg rows |
| Paid-service orders | `booking_analytics_test_df.csv` | 8,966 | order × service |
| Product events | `task_1_events.csv` | 44,568 | raw event (user × timestamp) |

The 135,508 bookings carry 137,082 legs; 1,574 bookings contain a connection.
24 airlines, 11 origins and 10 destinations appear. The leg JSON is **ragged** —
1,679 legs have no `airline`, 1,697 no `origin`, 1,696 no `destination` and 1,772
no `baggage` key. Absent `origin`/`destination`/`airline` are preserved as `pd.NA`
(nullable `string`), never imputed; `baggage` alone defaults to `"No baggage"` so
`has_baggage` stays a non-nullable boolean.

**Raw third-party data is NOT committed to this repository.** These are booking
records supplied for an interview assignment. Only the *loader*
(`src/data_ingest.py`), the synthetic SERP generator, and the SQL are here; the
`data/processed/*.parquet`, `data/raw/*` and `reports/*` outputs are gitignored.

To run the pipeline against the real feeds, point the loader at them:

```bash
export AVIASALES_SOURCE_DIR="/path/to/Aviasales"   # holds the 3 subdirectories
```

The directory must contain `Aviastats Аналитик/`, `Booking — Продуктовый аналитик/`
and `Продуктовый аналитик-2/`. Alternatively pass `--source-dir`, or drop the
files under `data/raw/` and rely on the default. Without the feeds, the ingest
stage fails loudly (by design) — but the SERP, ranking and experiment stages run
entirely on the committed synthetic layer.

One option test (`tests/test_stats.py`) cross-checks the vendored statistics
against a sibling project's real answer; it skips unless you set
`AVIASALES_AB_CTR_DIR` to that project.

## Methods

- **Position-bias model (PBM).** Examination probability decays with position as
  `(1/k)^η`, `η = 1.3` mobile / `1.0` desktop. Clicks are `examined × relevance`.
- **Unbiased CTR.** Horvitz–Thompson inverse-propensity weighting, with the curve
  either known or fitted by EM.
- **Ranking metrics.** NDCG@10 on graded relevance, MRR, plus team-draft
  interleaving as the online (position-bias-robust) alternative.
- **Ratio metrics.** Revenue per search is a ratio of sums, so it needs the delta
  method for its variance — the naive per-unit test is both wrong and conservative
  here.
- **CUPED.** Pre-period revenue per search as the covariate, with a shuffled
  covariate as a mutation control.
- **Sequential testing.** Fixed-horizon p-values under repeated looks are compared
  against an always-valid mSPRT to quantify the peeking penalty.

## Quick start

```bash
uv sync --all-groups
uv run python run.py --stage all      # ingest → sessions → serp → rank → experiment
uv run pytest -q                      # 46 tests
uv run ruff check . && uv run ruff format --check .
uv run streamlit run app.py           # dashboard on :8501
```

`--stage` accepts `ingest`, `sessions`, `serp`, `rank` (alias `eval`), `experiment`
or `all`.

## Status

All three phases complete.

| Phase | Scope | State |
|---|---|---|
| 1 | Data foundation: ingest, leg explosion, normalisation SQL, sessionisation | Done |
| 2 | SERP ranking simulator over `serp_offers` + ranking evaluation | Done |
| 3 | A/B experiment on the ranking policy + dashboard | Done |

Verified: `uv sync --all-groups` resolves; `uv run python run.py --stage all` runs
end-to-end; `uv run pytest -q` is **46 passed** (2 of them skip without the optional
sibling data); `uv run ruff check .` and `uv run ruff format --check .` are clean;
`uv run streamlit run app.py` serves on `localhost:8501`.

**Not verified / inference:** the synthetic SERP layer resembling real production
traffic is an *inference*, not a measurement — no real impression log exists for
this surface. `sql/03_funnel.sql` is written against the `serp_offers` relation and
executes when that Parquet is registered, but it is not wired into `run.py`.
`sql/01_normalize.sql` and `sql/02_sessions.sql` **are** executed by the pipeline.

## Layout

```
serp-rank-lab/
├── pyproject.toml
├── .gitignore
├── README.md
├── PRD.md
├── run.py                  # pipeline entry point
├── app.py                  # Streamlit dashboard
├── src/
│   ├── data_ingest.py      # loaders + leg explosion + parquet writer + CLI
│   ├── sessions.py         # DuckDB-backed sessionisation
│   ├── serp_sim.py         # synthetic results pages + PBM click model
│   ├── ranking_eval.py     # unbiased CTR, NDCG/MRR, interleaving
│   ├── stats.py            # vendored experiment statistics
│   └── experiment.py       # SRM, A/A, delta method, CUPED, mSPRT
├── sql/
│   ├── 01_normalize.sql    # flight legs / orders / events normalisation views
│   ├── 02_sessions.sql     # 30-min inactivity + date-boundary session rule
│   ├── 03_funnel.sql       # search → click → redirect → buy
│   └── 04_revenue_per_search.sql
├── tests/                  # 46 tests
├── data/
│   ├── raw/                # .gitkeep only — real data is not committed
│   └── processed/          # generated parquet (gitignored)
└── reports/                # generated charts and JSON (gitignored)
```

## Provenance

The statistics in `src/stats.py` are vendored from the author's own
`ab_test/scripts/` toolkit and cross-checked against a sibling A/B project's
documented answers (see `tests/test_stats.py`). Everything else is written for
this lab.

Licensed under the MIT License — see `LICENSE`.
