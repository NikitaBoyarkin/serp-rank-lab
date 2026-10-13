# EDA Report

**Rows:** 44568 · **Columns:** 4 · **Memory:** 3.78 MB

## Profile

| Column | dtype |
|---|---|
| user_id | str |
| ab_group | str |
| ts | datetime64[us] |
| pdate | object |

## Data Quality

- Duplicated rows: **0**
- Constant columns: **0**
- Пропусков нет.

**Datetime columns (excluded from categorical):** `ts`, `pdate`

## Numeric Distributions

_Числовых колонок нет._

## Categorical Slices

**user_id** — кардинальность 5000

| Value | Count | Share |
|---|---:|---:|
| rBMsVWPitiocbQAfe/TFAg== | 196 | 0.4% |
| Cmo4IGRQ3VwO1AAgCOGHAg== | 194 | 0.4% |
| rBMI3mQ5BbJ1OgAfBvpFAg== | 160 | 0.4% |
| rBMuVGPMG/d27AAgCCmNAg== | 150 | 0.3% |
| rBQjTWQjGL5ccAAfBI/YAg== | 137 | 0.3% |
| rBQjL2N6a8sCEwAgGEszAg== | 116 | 0.3% |
| CmoCrGRPsl4NiwAeAyN/Ag== | 105 | 0.2% |
| rBMs7WQZhxem+wAfDV2jAg== | 103 | 0.2% |
| CmsWkmRQ6zxWwwAhAzI0Ag== | 96 | 0.2% |
| CtYgeWBoJT0CtZ+3BBW3Ag== | 94 | 0.2% |

**ab_group** — кардинальность 2

| Value | Count | Share |
|---|---:|---:|
| A | 22360 | 50.2% |
| B | 22208 | 49.8% |

## Correlations

_Пар с |r| >= 0.5 нет._

## Findings

- **[INFO]** user_id: высокая кардинальность 5000 — для срезов свернуть в топ-N + «прочее»

## What We Did Not Cover

- Причинность: корреляции не доказывают влияние.
- Внешние источники и бизнес-контекст, которых нет в датасете.
- Формальные тесты гипотез (это CDA / A-B testing).

## Reproducibility

```bash
uv run python .claude/skills/eda/scripts/eda_profile.py --input <dataset> \
    [--metric <col>] [--segments <a,b>] [--time-col <col>] [--top <n>]
```
