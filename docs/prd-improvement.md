# PRD — усиление `serp-rank-lab` как продуктового кейса

**Статус:** на утверждении. Код не меняется до подтверждения.
**Дата:** 2026-10-20
**Автор:** сессия грилинга (11 решений зафиксированы, см. §0)

---

## 0. Принятые решения

| Ось | Решение |
|---|---|
| Аудитория | Product Analyst в маркетплейс/метапоиск. Доказательство — доведение анализа до решения, не набор методов |
| Бизнес-вопрос кейса | сколько выручки даёт ранжирование по деньгам и чем платим в UX |
| Заголовок | две **feasible**-политики на одинаковом информационном множестве: `+X% revenue/search`, `−Y% CTR`, breakeven в рублях на потерянный клик |
| Судьба `+38%` | приложение «потолок (modeled)» с явной подписью «сконструировано» |
| Второй вывод | «обученный ранкер проигрывает сортировке по цене» — отдельная секция «гипотеза не подтвердилась» |
| Эксперимент | настоящее воздействие: arm A = прод-политика, arm B = новая; рандомизация по `user_id`, кластеризация SE, guardrails, правило остановки |
| Размер эксперимента | два слоя: описательный на 12 000 поисков, эксперимент на мощном прогоне, с явной подписью какой из них сценарий |
| CUPED | честный нулевой результат + синтетический positive control; ADR о том, что сигнал не выдумываем |
| Guardrails | CTR, конверсия в бронь, средняя цена в топ-5, доля офферов дороже медианы маршрута. Refund rate — нельзя (нет join) |
| Воспроизводимость | синтетический raw с нуля по seed; `--stage all` на чистом клоне; golden hash |
| Чистка | MRR считаем, `calibration` подключаем, interleaving вырезаем из рекомендаций, 6 фактических ошибок чиним |
| Поверхность | README → Streamlit → `docs/decision-memo.md` → ноутбуки |
| Портфолио | целимся в headline-набор; промоут только после зелёного гейта (claims + чистый клон) |

**Инвариант порядка:** правда → глубина → нарратив → упаковка. Ни один REQ из «нарратива» не исполняется, пока не закрыты REQ-T (правда).

---

## 1. Замороженные интерфейсы

Волна 1 запускается параллельно, поэтому контракты фиксируются **до** правок.

### 1.1 `reports/headline.json` — единственный источник чисел для README и кейса

```json
{
  "generated_from": {"searches": 12000, "seed": 42, "commit": "<sha>"},
  "policies": {
    "ctr_objective":     {"ctr": 0.0, "revenue_per_search": 0.0, "ndcg_at_10": 0.0, "conversion": 0.0},
    "revenue_objective": {"ctr": 0.0, "revenue_per_search": 0.0, "ndcg_at_10": 0.0, "conversion": 0.0},
    "production":        {"ctr": 0.0, "revenue_per_search": 0.0, "ndcg_at_10": 0.0, "conversion": 0.0},
    "price_asc":         {"ctr": 0.0, "revenue_per_search": 0.0, "ndcg_at_10": 0.0, "conversion": 0.0}
  },
  "headline": {
    "revenue_per_search_delta_abs": 0.0,
    "revenue_per_search_delta_rel": 0.0,
    "ctr_delta_abs": 0.0,
    "ctr_delta_rel": 0.0,
    "breakeven_rub_per_forgone_click": 0.0,
    "information_set": "logged"
  },
  "ceiling_oracle": {
    "revenue_per_search": 0.0,
    "label": "modeled",
    "note": "affinity/price_index anti-correlation is a modelling assumption (src/serp_sim.py:57-61)"
  },
  "failed_hypothesis": {
    "learned_ranker": 0.0,
    "price_asc": 0.0,
    "statement": "learned ranker loses to price sorting"
  }
}
```

Правила: `information_set` у обеих headline-политик обязан быть `logged`; оракулы в блок `headline` не попадают никогда.

### 1.2 `outcomes_under_policy` — генерация исходов под новым порядком

```python
def outcomes_under_policy(
    offers: pd.DataFrame,
    policy: str,
    ctx: "PolicyContext",
    *,
    seed: int,
) -> pd.DataFrame:
    """Re-rank each SERP by ``policy``, then re-draw the PBM outcomes.

    Returns one row per offer with: search_id, user_id, position (1..n),
    examine_prob (recomputed for the new slot), clicked, redirected, booked,
    commission. Deterministic for a given (policy, seed).
    """
```

Почему это обязательно: сейчас исходы в `serp_offers` — свойства *логированного* порядка. Без пересчёта «рука B» не имеет своих кликов, и эксперимент снова превращается в сплит одинаковых юзеров.

### 1.3 `policy_scores(offers, ctx) -> dict[str, np.ndarray]`

Обе headline-политики обязаны строиться из **одной** матрицы признаков и одного вектора коэффициентов; различие — только множитель полезности. Это и есть доказуемое «одинаковое информационное множество».

---

## 2. Правда

### REQ-T01 — Две feasible-политики на одинаковой информации
**Приоритет:** P0 (без этого заголовок остаётся фикцией)
**Доказательство:** `src/policies.py:244-253` (в реестре только оракулы `relevance`/`revenue` и слабые `click_trained`/`propensity_logistic`); `README.md:3,127-144`
**Что делаем:** добавляем `ctr_objective` и `revenue_objective`. Обе используют fitted CTR-модель на логах (`fit_click_models`, IPW-коррекция), одну матрицу `X`; `ctr_objective` ранжирует по `p̂`, `revenue_objective` — по `p̂ × commission_if_engaged`. Ни одна не читает `relevance` и `expected_revenue`.

**Симметрия (обязательна):** обе политики применяют **один и тот же** механизм назначения слотов — greedy по `examine_prob(k)`, как в `revenue()` (`src/policies.py:107`). Различие только в полезности: `p̂` против `p̂ × commission`. Если одна политика учитывает затухание внимания, а другая нет, сравнение снова нечестное.
**Acceptance:**
- `reports/ranking_eval.json` содержит блок `headline` со схемой §1.1;
- тест `test_feasible_policies_share_information_set`: одинаковый `X`, одинаковый `β`, отличается только множитель;
- тест `test_headline_policies_do_not_read_ground_truth`: на фрейме без колонок `relevance`/`expected_revenue` обе политики строят порядок;
- тест `test_oracles_never_enter_headline_block`.
**Проверка:** `uv run pytest tests/test_policies.py -k "feasible or ground_truth or headline" -v && uv run python -m src.ranking_eval`

### REQ-T02 — Breakeven как решаемое число
**Приоритет:** P0
**Что делаем:** `breakeven_rub_per_forgone_click = Δrevenue_per_search / Δclicks_per_search`. Формулировка в отчёте: «каждый потерянный клик должен приносить ≥ N ₽ выручки, иначе сделка невыгодна». Явная сноска: пересчёт клика в деньги требует внешнего курса (LTV/бренд-риск), которого в данных нет — поэтому число дано как курс, а не как вердикт.
**Acceptance:** поле присутствует, знак согласован со знаками Δ (тест на арифметику), в README оно объяснено одной фразой.
**Проверка:** `uv run pytest tests/test_ranking_eval.py -k breakeven -v`

### REQ-T03 — `+38%` понижается до потолка
**Приоритет:** P0 (бэклог портфолио B42)
**Доказательство:** `src/serp_sim.py:57-61`; `README.md:3`; `NikitaBoyarkin.github.io/src/content/projects/serp-rank.md:13`
**Что делаем:** число переезжает в `ceiling_oracle` с меткой `modeled` и прямым текстом, что конфликт релевантности и денег сконструирован антикорреляцией. В заголовке README его нет.
**Acceptance:** `rg -n "38%" README.md docs/` не находит вхождений вне блока «потолок»; в `headline.json` поле `ceiling_oracle.label == "modeled"`.
**Проверка:** `rg -n "38" README.md`

### REQ-T04 — Шесть фактических ошибок
**Приоритет:** P1
Список и доказательства:
1. MRR заявлен, не считается — `README.md:65,85,385`, `docs/diagrams/serp-rank-lab.html`; функция есть `src/ranking_eval.py:207`, вызовов нет.
2. «Монотонно» — `README.md:165`; данные дают провал на tilt 0.75 (`reports/ranking_eval.json:136-141`).
3. «58×» из одного события 1/400 — `README.md:250,260,338`; нужен интервал (0.06%–1.4%).
4. CUPED «ρ²» — `README.md:248`; контроль возвращает `corr²` = ρ⁴.
5. Устаревший status-блок: `sql/03` **подключён** (`run.py:45`, `src/sql_stage.py:26`), Phase 4 **в основном подключён** (`src/ranking_eval.py:572`); мёртв только `calibration.py` — `README.md:426,436-437`.
6. Перепутанные колонки в докстринге — `src/data_ingest.py:105-108` vs фактические 1 679 airline / 1 697 origin.
**Acceptance:** MRR в `ranking_eval.json`; каждое утверждение в README совпадает с артефактом; тест `test_readme_numbers_match_artifacts` сверяет 5 headline-чисел.
**Проверка:** `uv run pytest tests/test_claims.py -v`

### REQ-T05 — Claims в портфолио
**Приоритет:** P0 (ось «правда» структурно не выполнена: у serp ноль claim'ов)
**Что делаем:** 5 claim-функций в `scripts/check_claims.py`: headline Δrevenue, ΔCTR, naive vs true CTR, peeking FPR, logged revenue/search. Читают закоммиченные `reports/*.json`.
**Acceptance:** `make check` зелёный на чистом клоне; каждая claim падает при подмене числа.
**Проверка:** `cd .. && make check`

### REQ-T06 — JSON-отчёты коммитятся
**Приоритет:** P0 (иначе claims нечего читать на чистом клоне)
**Что делаем:** `reports/*.json` выходят из `.gitignore:14`, коммитятся; `reports/headline.json` — новый.
**Acceptance:** `git ls-files reports/` содержит JSON; claim проходит после `git clone` в пустую папку.

---

## 3. Глубина

### REQ-D01 — Настоящее воздействие
**Приоритет:** P0 (главная дыра: сейчас «A/B» без воздействия)
**Доказательство:** `src/experiment.py:91-103` (руки — независимый Бернулли), `:159-188` (сравнение случайных подмножеств), `ab_group` не читается ни разу
**Что делаем:** arm A — переранжирование `production` (money_tilt=0.5), arm B — `revenue_objective`; назначение по `user_id` (юзер целиком в одной руке); исходы через §1.2; первичная метрика revenue/search с кластеризованной по юзеру ошибкой; guardrails из §0.
**Acceptance:**
- `reports/experiment.json` содержит `arms`, `primary` (эффект + CI + p), `guardrails` (4 метрики с CI), `decision`;
- тест `test_arms_are_disjoint_by_user`;
- тест `test_outcomes_differ_between_arms` (иначе воздействия нет);
- тест `test_srm_detects_a_deliberately_broken_split` (мутационный контроль: ломаем сплит — SRM краснеет).
**Проверка:** `uv run python -m src.experiment && uv run pytest tests/test_experiment.py -v`

### REQ-D02 — Мощность как вывод, а не как параметр
**Приоритет:** P1
**Что делаем:** прогон эксперимента на `--searches 60000` (сценарий) и на 12 000 (реальность); в отчёт `power`: MDE при 12k, требуемое n/arm для наблюдаемого эффекта, явная подпись какой прогон сценарий. Формулировка вывода: «при реальном трафике такой эффект недетектируем — вот сколько нужно».
**Acceptance:** `power` блок с тремя числами и подписью; README цитирует именно `power.at_real_traffic`.
**Проверка:** `uv run python -m src.experiment --searches 60000`

### REQ-D03 — CUPED: честный нуль + ADR
**Приоритет:** P1
**Доказательство:** pre/post корреляция `−0.0065 → 0.00%` сокращения; юзеры i.i.d. (`src/experiment.py:17-21,239-243`)
**Что делаем:** оставляем как есть, но переписываем подачу: «ковариате нечего объяснять», positive control (ρ=0.6 → 11.2%) доказывает, что машинерия жива. Новый `docs/adr/0002-idempotent-users-no-cuped-signal.md` с запретом «вшить корреляцию, чтобы CUPED заработал».
**Acceptance:** ADR существует; отчёт содержит `cuped.real` и `cuped.control` с подписью; README не обещает выигрыш.
**Проверка:** `uv run pytest tests/test_stats.py -k cuped -v`

### REQ-D04 — Гетерогенность по устройству, только CTR
**Приоритет:** P1
**Доказательство:** 7 484 mobile / 4 516 desktop поисков; MDE по CTR 1.7%/2.2%, по revenue/search — 13.4%/17.3% (недетектируемо)
**Что делаем:** эффект по `device` с CI и поправкой BH на два сравнения; отдельным абзацем — «по деньгам на сегмент мощности нет, поэтому вывод только по CTR».
**Acceptance:** блок `heterogeneity` с `mde` и `underpowered_on_revenue: true`.
**Проверка:** `uv run pytest tests/test_experiment.py -k heterogeneity -v`

### REQ-D05 — Определения метрик и правило остановки
**Приоритет:** P1
**Что делаем:** `docs/metric-definitions.md`: revenue vs order_value vs service_profit vs commission vs expected_revenue — с формулой и источником; guardrail-набор и правило остановки (guardrail падает >X% → не катим независимо от выручки); строка «refund rate не может быть guardrail'ом: нет join между `orders` и ранжированием».
**Acceptance:** документ есть, `CONTEXT.md` ссылается на него; каждое определение имеет `file:line` источника.

---

## 4. Нарратив

### REQ-N01 — README переписывается вокруг решения
**Приоритет:** P1 (после REQ-T*, D*)
**Структура:** Ситуация → Вопрос → Решение (таблица: что катим, что нет, почему) → Цена решения (breakeven) → Почему не оракулы (информационное множество) → Гипотеза не подтвердилась (ранкер < price_asc) → Эксперимент и его мощность → Ограничения (синтетика как свойство) → Приложение «потолок».
**Acceptance:** первые 40 строк отвечают «что решили и чем платим»; ни одного числа без источника в `headline.json`/`experiment.json`.

### REQ-N02 — `docs/decision-memo.md` на одну страницу
Вопрос, рекомендация, доказательство, цена, что сделаем следующим шагом, что изменит наше мнение. Формата memo в портфолио нет ни у одного проекта.

### REQ-N03 — Страница кейса RU+EN
Переписать `NikitaBoyarkin.github.io/src/content/projects{,-en}/serp-rank.md`: убрать оракульный заголовок, поставить решение, синхронизировать `impact`/`faq`/`bet`/`rejected`. Демо (`public/demos/serp-rank/index.html`) генерировать из закоммиченных данных, а не держать ручным.
**Acceptance:** EN/RU паритет по структуре и числам; `scripts/content-drift-audit.mjs` зелёный.

### REQ-N04 — Диаграмма
`docs/diagrams/serp-rank-lab.html`: убрать MRR-обещание (или добавить MRR в поток), заменить `+38% REVENUE` на headline-число с меткой `logged`.

---

## 5. Упаковка и воспроизводимость

### REQ-P01 — Чистый клон работает
**Приоритет:** P0 (бэклог B27)
**Доказательство:** `data/raw/` пуст, `data/processed` в `.gitignore:11`, исходники вне репо; `README.md:120,373-374` утверждает обратное
**Что делаем:** `src/synthetic_raw.py` — schema-совместимая генерация `bookings_legs`/`orders`/`events` из seed (маршруты, цены, сезонность — по тем же весам, что уже зашиты: `MONTH_WEIGHTS`, `load_routes`); `run.py --stage all` вызывает её, если `data/raw` пуст; golden sha256 на `serp_offers`.
**Acceptance:** в пустой папке `git clone && uv sync && uv run python run.py --stage all` завершается успешно; тест `test_pipeline_runs_from_empty_data_dir`; golden hash совпадает.
**Проверка:** `cd $(mktemp -d) && git clone <repo> . && uv sync && uv run python run.py --stage all`

### REQ-P02 — CI: линт + чистый клон
**Приоритет:** P0 (CI красный с 17.10)
**Что делаем:** чиним 6 ruff-ошибок (`reports/eda/scripts/analyze_serp.py:36,39`; `plot_serp.py:118,163,217,280`); второй джоб — чистый клон + `--stage all` + pytest.
**Acceptance:** `uv run ruff check . && uv run ruff format --check .` чисты; оба джоба зелёные.

### REQ-P03 — Ноутбуки перезапускаемы
`jupyter`/`ipython` в dev-зависимости; ноутбуки перезапущены; prose исправлен (3 политики → 5).

### REQ-P04 — Бэклог портфолио
B27 → DONE после REQ-P01; B42 → DONE после REQ-T03/N03. Строки с `file:line`.

---

## 6. Red flags

| ID | Что | Доказательство | Действие |
|---|---|---|---|
| REQ-R01 | `calibration.py` — целый мёртвый модуль | `src/calibration.py`, импортируется только тестами | подключить стадией `calibrate` (валидация восстановления параметров симулятора) |
| REQ-R02 | Interleaving меряет клики, но рекомендуется как онлайн-метод | `src/ranking_eval.py:593-599` (один SERP); `README.md:334,386` | убрать из рекомендаций; оставить как демонстрацию с явной сноской; из headline убрать |
| REQ-R03 | Δ-строка нарисована как политика (отрицательный бар) | `src/plots.py:98-106` vs корректный `app.py:215` | починить |
| REQ-R04 | 163 дубликата `order_id` не задокументированы | `orders.parquet` | строка в `CONTEXT.md` |
| REQ-R05 | Peeking: 10×250 = 2 500, но в руке 2 250 — последние «взгляды» дублируются | `src/experiment.py:300-316` | чинить или явно ограничить число взглядов |

---

## 7. Порядок исполнения

**Волна 0 (серийно, оркестратор):** этот PRD + §1 контракты. Гейт: подтверждение пользователя.

**Волна 1 (параллельно, 3 агента, непересекающиеся файлы):**
- **A1 · воспроизводимость:** `src/synthetic_raw.py`, `run.py`, `.gitignore`, `tests/test_reproducibility.py`, `.github/workflows/ci.yml`
- **A2 · политики и оценка:** `src/policies.py`, `src/ranking_eval.py`, `tests/test_policies.py`, `tests/test_ranking_eval.py`, `reports/headline.json`
- **A3 · эксперимент:** `src/experiment.py`, `src/outcomes.py` (новый, §1.2), `tests/test_experiment.py`

**Волна 2 (после того как числа появились):**
- **A4 · нарратив:** `README.md`, `docs/decision-memo.md`, `docs/metric-definitions.md`, `CONTEXT.md`, `docs/adr/0002`, `src/plots.py`, `app.py`
- **A5 · портфолио:** claims в `scripts/check_claims.py`, страница кейса RU/EN, бэклог B27/B42

**Волна 3 (верификатор, producer ≠ verifier):** чистый клон, claims, сверка README ↔ JSON, мутационная проверка каждого нового теста (сломай вещь — тест обязан покраснеть), `ruff`, `pytest`.

**Правила для агентов:** не запускать `git` (коммитит оркестратор), не трогать файлы чужих осей, отчёт = `file:line` + severity, никаких вопросов пользователю.

---

## 8. Что НЕ делаем

- Когорты/LTV: в `orders` нет ключа пользователя, в `events` одна неделя.
- Сквозная воронка поиск → бронь: нет FK, `booked` — флаг симулятора.
- Refund rate как guardrail: нет связи с ранжированием.
- Вшивание корреляции в симулятор, чтобы «заработал» CUPED.
- Раздувание данных ради значимости.
- Переписывание interleaving на N SERP (полдня ради абзаца; вместо этого — честная сноска и вывод из рекомендаций).
