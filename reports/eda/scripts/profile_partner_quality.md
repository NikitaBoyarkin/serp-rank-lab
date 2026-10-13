# EDA Report

**Rows:** 8 · **Columns:** 13 · **Memory:** 0.0 MB

## Profile

| Column | dtype |
|---|---|
| partner | str |
| offers | int64 |
| avg_position | float64 |
| clicks | int64 |
| bookings | int64 |
| commission | float64 |
| booked_commission | float64 |
| searches_present | int64 |
| ctr | float64 |
| booking_rate | float64 |
| commission_per_offer | float64 |
| commission_per_search | float64 |
| monetisation_rank | int64 |

## Data Quality

- Duplicated rows: **0**
- Constant columns: **0**
- Пропусков нет.

## Numeric Distributions

| Column | n | mean | median | std | p25 | p75 | p95 | outliers | skew |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| offers | 8 | 15017.0 | 15027.5 | 202.7018 | 14897.25 | 15188.25 | 15241.2 | 0.0% | -0.4082 |
| avg_position | 8 | 6.0041 | 5.4787 | 2.501 | 4.0597 | 8.126 | 9.2889 | 0.0% | 0.2672 |
| clicks | 8 | 2448.375 | 1756.5 | 1810.6902 | 1071.25 | 3569.25 | 5184.85 | 0.0% | 0.8693 |
| bookings | 8 | 521.0 | 322.0 | 458.9821 | 185.75 | 740.75 | 1244.5 | 0.0% | 1.0725 |
| commission | 8 | 1838441.1378 | 1821643.266 | 1815764.1098 | 40937.0 | 2908360.1173 | 4349322.8807 | 0.0% | 0.5882 |
| booked_commission | 8 | 1827688.3878 | 1821643.266 | 1827983.1683 | 7082.5 | 2908360.1173 | 4349322.8807 | 0.0% | 0.5734 |
| searches_present | 8 | 5866.625 | 5859.5 | 44.3684 | 5847.25 | 5885.0 | 5930.35 | 0.0% | 0.2307 |
| ctr | 8 | 0.1631 | 0.1175 | 0.1207 | 0.0708 | 0.2374 | 0.3455 | 0.0% | 0.8787 |
| booking_rate | 8 | 0.0347 | 0.0215 | 0.0305 | 0.0125 | 0.0492 | 0.0829 | 0.0% | 1.0778 |
| commission_per_offer | 8 | 121.8292 | 121.6289 | 122.2224 | 0.4791 | 193.6034 | 290.1065 | 0.0% | 0.5898 |
| commission_per_search | 8 | 311.3251 | 311.609 | 311.3977 | 1.2181 | 495.2531 | 739.7563 | 0.0% | 0.5783 |
| monetisation_rank | 8 | 4.5 | 4.5 | 2.4495 | 2.75 | 6.25 | 7.65 | 0.0% | 0.0 |

## Categorical Slices

**partner** — кардинальность 8

| Value | Count | Share |
|---|---:|---:|
| SkyTour | 1 | 12.5% |
| OTA Global | 1 | 12.5% |
| AviaCenter | 1 | 12.5% |
| FlyDirect | 1 | 12.5% |
| TripMasters | 1 | 12.5% |
| MetaFly | 1 | 12.5% |
| GoFare | 1 | 12.5% |
| CloudTrip | 1 | 12.5% |

## Correlations

| Pair | r | Note |
|---|---:|---|
| commission ~ booked_commission | 1.0 | multicollinearity |
| commission ~ commission_per_search | 1.0 | multicollinearity |
| booked_commission ~ commission_per_search | 1.0 | multicollinearity |
| commission_per_offer ~ commission_per_search | 1.0 | multicollinearity |
| clicks ~ ctr | 0.9999 | multicollinearity |
| bookings ~ booking_rate | 0.9999 | multicollinearity |
| commission ~ commission_per_offer | 0.9999 | multicollinearity |
| booked_commission ~ commission_per_offer | 0.9999 | multicollinearity |
| clicks ~ booking_rate | 0.9878 | multicollinearity |
| clicks ~ bookings | 0.9864 | multicollinearity |
| ctr ~ booking_rate | 0.9863 | multicollinearity |
| bookings ~ ctr | 0.9846 | multicollinearity |

## Metric by Segment

### partner (overall mean = 1838441.1378, gap x179.8058)

| Value | Count | Mean | Sum | Share | vs overall |
|---|---:|---:|---:|---:|---:|
| SkyTour | 1 | 5055420.108 | 5055420.11 | 34.4% | +175.0% |
| OTA Global | 1 | 3037999.4586 | 3037999.46 | 20.7% | +65.2% |
| AviaCenter | 1 | 2865147.0035 | 2865147.0 | 19.5% | +55.9% |
| FlyDirect | 1 | 2312325.056 | 2312325.06 | 15.7% | +25.8% |
| TripMasters | 1 | 1330961.476 | 1330961.48 | 9.0% | -27.6% |
| GoFare | 1 | 43094.0 | 43094.0 | 0.3% | -97.7% |
| MetaFly | 1 | 34466.0 | 34466.0 | 0.2% | -98.1% |
| CloudTrip | 1 | 28116.0 | 28116.0 | 0.2% | -98.5% |


## Findings

- **[WARN]** commission ~ booked_commission: r=1.0 — мультиколлинеарность или дубль признака
- **[WARN]** commission ~ commission_per_search: r=1.0 — мультиколлинеарность или дубль признака
- **[WARN]** booked_commission ~ commission_per_search: r=1.0 — мультиколлинеарность или дубль признака
- **[WARN]** commission_per_offer ~ commission_per_search: r=1.0 — мультиколлинеарность или дубль признака
- **[WARN]** clicks ~ ctr: r=0.9999 — мультиколлинеарность или дубль признака
- **[WARN]** bookings ~ booking_rate: r=0.9999 — мультиколлинеарность или дубль признака
- **[WARN]** commission ~ commission_per_offer: r=0.9999 — мультиколлинеарность или дубль признака
- **[WARN]** booked_commission ~ commission_per_offer: r=0.9999 — мультиколлинеарность или дубль признака
- **[WARN]** clicks ~ booking_rate: r=0.9878 — мультиколлинеарность или дубль признака
- **[WARN]** clicks ~ bookings: r=0.9864 — мультиколлинеарность или дубль признака
- **[WARN]** ctr ~ booking_rate: r=0.9863 — мультиколлинеарность или дубль признака
- **[WARN]** bookings ~ ctr: r=0.9846 — мультиколлинеарность или дубль признака
- **[INFO]** Разрыв по partner: x179.8058 между сегментами — кандидат в главный вывод

## What We Did Not Cover

- Причинность: корреляции не доказывают влияние.
- Внешние источники и бизнес-контекст, которых нет в датасете.
- Формальные тесты гипотез (это CDA / A-B testing).

## Reproducibility

```bash
uv run python .claude/skills/eda/scripts/eda_profile.py --input <dataset> \
    [--metric <col>] [--segments <a,b>] [--time-col <col>] [--top <n>]
```
