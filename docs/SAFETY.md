# Phase 3: does the policy know when it is about to fail?

Command: `lacc safety evaluate` (settings in the `safety` section of `configs/policy_gru.yaml`).

## Inputs

- **Test replay** from Phase 2b: the GRU ensemble drives 4,037 test events behind recorded leaders; every step stores gap, speed, acceleration and the ensemble's disagreement (epistemic std).
- **Validation replay**: the same for the 1,167 validation events, created automatically on first run. Used only to set thresholds.

## Labels (as in the experiment protocol)

- **Violation step:** collision, TTC < 1 s, or time headway < 0.3 s (ego speed > 1 m/s).
- **Failure event:** a run of violation steps; it starts at its *onset*.
- **Positive step:** a failure onset occurs within the next 3 s (1.5 s and 5 s as sensitivity). Steps already in violation are excluded, because announcing an ongoing failure is trivial.

## Monitors

| Monitor | Raw signal | Meaning |
|---|---|---|
| uncertainty | ensemble std of predicted acceleration | "the models disagree" |
| ttc | 1 / TTC | "closing in fast" |
| thw | 1 / time headway | "following too close" |
| kinematic | TTC or headway | the conventional engineering monitor |
| combined | uncertainty or kinematic | does uncertainty add anything? |

Each raw signal is converted to its **percentile among normal driving steps on validation**, so all monitors share one scale. A monitor made of several signals takes the highest percentile. All monitors use the same delayed observation as the policy (0.5 s old).

## Thresholds: equal alarm budget

A monitor that alarms all the time catches every failure, so monitors must be compared at an equal cost. The threshold of each monitor is set on **validation** so that it is in alarm during a fixed share of normal driving time: 1 %, **2 % (main)** or 5 %. Test data never influences thresholds.

## Metrics

**RQ1 (ranking, offline):** AUROC and AUPRC of each monitor for "failure within the horizon", with 95 % bootstrap confidence intervals that resample whole events.

**RQ2 (warning, offline, per budget):**
- share of failures with at least one alarm in the 3 s before onset
- median lead time (first alarm to onset)
- false alarms per hour: alarm episodes during normal driving with no failure in the following 3 s
- complementarity: failures warned only by uncertainty, only by kinematic, by both, by neither

**RQ3 (closed loop, test):** when the monitor alarms, the fallback controller (IDM calibrated on spacing) takes over for at least 2 s and as long as the alarm lasts. Compared setups:

| Setup | Protocol baseline |
|---|---|
| fallback IDM only | 1 |
| GRU only | 2 |
| GRU + kinematic monitor | 3 |
| GRU + uncertainty monitor | 4 |
| GRU + combined monitor | 5 |

Reported: collisions, events with violation, spacing RMSE against the human (how human-like the driving stays), RMS jerk (comfort), interventions per hour, share of time in fallback.

## Outputs

| File | Content |
|---|---|
| `reports/safety/safety_metrics.json` | Thresholds, all offline and closed-loop metrics |
| `reports/safety/safety_report.png` | AUROC with CIs, warned share vs false alarms, closed-loop comparison, an example collision |
| `data/processed/ngsim/replay_val_traces.npz` | Validation replay (local only) |

## Limitations

- In-distribution only: training, validation and test all come from NGSIM. Whether uncertainty becomes more useful on unfamiliar data is tested with highD and the scenario search (Phases 4–5).
- The deep ensemble captures epistemic uncertainty only; a confidently wrong policy is invisible to it (see `tests/test_safety.py`).
- Failures are defined by the same kinematic quantities (TTC, headway) that the kinematic monitor watches. This favours the kinematic monitor by construction, and is reported as such.
