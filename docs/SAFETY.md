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

## Results (4,037 test events, 62 h; thresholds from 1,167 validation events)

### RQ1: ranking of pre-failure moments (failure onset within 3 s; 3,586 positive steps, rate 0.17 %)

| Monitor | AUROC [95 % CI] | AUPRC [95 % CI] |
|---|---|---|
| uncertainty | 0.720 [0.692–0.758] | 0.0084 [0.0068–0.0117] |
| ttc | 0.769 [0.738–0.800] | 0.0201 [0.0147–0.0277] |
| thw | 0.893 [0.873–0.915] | 0.0414 [0.0254–0.0719] |
| kinematic (ttc or thw) | 0.880 [0.862–0.903] | 0.0404 [0.0291–0.0584] |
| combined (+ uncertainty) | 0.873 [0.856–0.898] | 0.0371 [0.0266–0.0540] |

Ensemble disagreement is clearly informative (AUROC well above 0.5, AUPRC about 5× the base rate), but it ranks pre-failure moments worse than time headway alone, and the confidence intervals do not overlap with the kinematic monitor.

### RQ2: warnings at an alarm budget of 2 % of normal driving

| Monitor | Failures warned in the 3 s before onset | Median lead time | False alarms per hour | Alarm time on test |
|---|---|---|---|---|
| uncertainty | 49.3 % | 0.9 s | 76.0 | 0.7 % |
| kinematic | 93.4 % | 1.0 s | 42.0 | 2.0 % |
| combined | 90.4 % | 0.8 s | 49.7 | 1.4 % |

Of 136 failure events, 1 was warned only by uncertainty, 61 only by the kinematic monitor, 66 by both and 8 by neither. Ensemble disagreement was lower on test than on validation (mean 0.038 vs 0.045 m/s²), so the uncertainty monitor used only 0.7 % alarm time on test; its alarms are also more fragmented (more, shorter episodes), which raises its false-alarm count.

### RQ3: closed loop, alarm hands control to the fallback (≥ 2 s)

| Setup | Collisions | Events with violation | Spacing RMSE vs human | RMS jerk | Interventions per hour | Time in fallback |
|---|---|---|---|---|---|---|
| fallback IDM only | 0 | 6 | 4.96 m | 1.52 m/s³ | – | – |
| GRU only | 30 | 125 | 4.32 m | 0.79 m/s³ | – | – |
| GRU + kinematic | **0** | **8** | 4.61 m | 1.66 m/s³ | 38.5 | 2.8 % |
| GRU + uncertainty | 9 | 67 | 4.48 m | 1.37 m/s³ | 36.4 | 3.2 % |
| GRU + combined | 0 | 11 | 4.59 m | 1.61 m/s³ | 35.8 | 3.2 % |

### Answers

1. **RQ1/RQ2:** On in-distribution traffic, ensemble disagreement does rise before failures, but it is clearly weaker than a TTC/headway monitor and almost never catches a failure the kinematic monitor misses (1 of 136).
2. **RQ3:** A hybrid of the learned policy, a kinematic monitor and a rule-based fallback removes all 30 collisions and reaches nearly the fallback's safety (8 vs 6 violation events), while staying more human-like than the fallback alone (spacing RMSE 4.61 vs 4.96 m). The fallback is active 2.8 % of the time.
3. Adding uncertainty to the kinematic monitor at the same alarm budget does not help (11 vs 8 violation events): the budget is split across more signals.
4. **Cost:** hard switching between controllers doubles RMS jerk compared with the GRU alone (1.66 vs 0.79 m/s³). Blending instead of switching is an obvious improvement.

These are in-distribution results. The open question carried into Phases 4–5 is whether disagreement becomes more useful in situations unlike the training data, where kinematic signals may react late.

## Limitations

- In-distribution only: training, validation and test all come from NGSIM. Whether uncertainty becomes more useful on unfamiliar data is tested with highD and the scenario search (Phases 4–5).
- The deep ensemble captures epistemic uncertainty only; a confidently wrong policy is invisible to it (see `tests/test_safety.py`).
- Failures are defined by the same kinematic quantities (TTC, headway) that the kinematic monitor watches. This favours the kinematic monitor by construction, and is reported as such.
