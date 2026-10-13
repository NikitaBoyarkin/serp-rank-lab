# EDA Report

**Rows:** 5924 · **Columns:** 5 · **Memory:** 0.55 MB

## Profile

| Column | dtype |
|---|---|
| user_id | str |
| ab_group | str |
| start_ts | datetime64[us] |
| end_ts | datetime64[us] |
| pdate | object |

## Data Quality

- Duplicated rows: **0**
- Constant columns: **0**
- Пропусков нет.

**Datetime columns (excluded from categorical):** `start_ts`, `end_ts`, `pdate`

## Numeric Distributions

_Числовых колонок нет._

## Categorical Slices

**user_id** — кардинальность 5000

| Value | Count | Share |
|---|---:|---:|
| rBMuVGPMG/d27AAgCCmNAg== | 9 | 0.1% |
| rBMI3mQ5BbJ1OgAfBvpFAg== | 8 | 0.1% |
| Cmo4IGRQ3VwO1AAgCOGHAg== | 7 | 0.1% |
| rBQjTWQjGL5ccAAfBI/YAg== | 7 | 0.1% |
| xppk/u0RbGIWAKC+6VfbPg== | 7 | 0.1% |
| CtYgeWA5KnJw5eIOA5yFAg== | 6 | 0.1% |
| rBMs7WQZhxem+wAfDV2jAg== | 6 | 0.1% |
| rBMs9GGSAMiLyAAgBKYjAg== | 6 | 0.1% |
| rBQjL2N6a8sCEwAgGEszAg== | 6 | 0.1% |
| rBQjxWI5Ac8LowAgFUW1Ag== | 6 | 0.1% |

**ab_group** — кардинальность 2

| Value | Count | Share |
|---|---:|---:|
| B | 2977 | 50.2% |
| A | 2947 | 49.8% |

## Correlations

_Пар с |r| >= 0.5 нет._

## Findings

- **[WARN]** user_id: кардинальность 5000 (84% строк) — похоже на ID, не группировать

## What We Did Not Cover

- Причинность: корреляции не доказывают влияние.
- Внешние источники и бизнес-контекст, которых нет в датасете.
- Формальные тесты гипотез (это CDA / A-B testing).

## Reproducibility

```bash
uv run python .claude/skills/eda/scripts/eda_profile.py --input <dataset> \
    [--metric <col>] [--segments <a,b>] [--time-col <col>] [--top <n>]
```
