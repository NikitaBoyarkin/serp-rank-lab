# 0002 — i.i.d. users make CUPED a null result, and the simulator stays that way

## Status

Accepted.

## Context

The experiment in `reports/experiment.json` reports two CUPED blocks:

| Block | corr(X, Y) | Variance reduction |
|---|---:|---:|
| Real pre-period covariate (revenue per search, up to 4 pre-months) | **−0.0065** | **0.004%** |
| Positive control (constructed latent, ρ = 0.6) | 0.3353 | **11.24%** |

The real covariate explains nothing. The reason is structural, not an estimator bug: each
search draws its own user independently (`src/serp_sim.py:466-468`), so a user's pre-period
revenue carries no information about that user's in-period revenue. There is no stable
user-level effect to absorb.

CUPED's gain is exactly `corr(X, Y)²` in variance terms (`src/experiment.py:778`). With
`corr = −0.0065` that is 4.2e-05, i.e. 0.004% — indistinguishable from zero. The positive
control recovers 11.24%, which is `0.3353² = 11.24%` of the variance: the machinery works
whenever a signal exists to be removed.

## Decision

Report the real block as an **honest null** and keep the positive control as proof that the
estimator is alive. Do **not** wire correlation between the pre-period covariate and the
outcome into the simulator so that CUPED "works".

The simulator draws users i.i.d. by design: it models a search surface, not a panel of
repeat customers. Injecting a user-level autocorrelation would change the estimand of every
other result in the lab — the variance of the primary metric, the MDE, the sample-size table,
and the peeking simulation — in exchange for a nicer-looking CUPED number. That is tuning the
data-generating process to flatter an estimator, which is the same failure mode as picking a
metric after seeing the result.

## Consequences

- The lab can say precisely **when** CUPED pays: when a pre-period covariate actually
  correlates with the outcome. On this surface it does not, and the report says so.
- `effect of CUPED = 0` is a finding about the data, not a bug to fix. The positive control
  separates "estimator broken" from "nothing to explain".
- The honest cost: CUPED cannot be sold as a variance win in this case. Any power or
  sample-size claim in `reports/experiment.json` is computed **without** CUPED reduction.
- Anyone extending the simulator with returning users must treat it as a new data-generating
  process: re-run the headline, the experiment, the power block and the sample-size table,
  and label the change. Quietly adding user persistence to shrink a CI is out of bounds.
- Reference: `src/experiment.py:778` (the `corr²` identity), `:789-830` (the two blocks and
  their verdicts), `src/serp_sim.py` (users drawn per search, not per panel).
