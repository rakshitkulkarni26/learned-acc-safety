# Experiment protocol

Written **before** any experiment is run, so results cannot quietly shape the definitions.
Any later change is recorded in the change log at the bottom with a reason.

## Research questions

**RQ1 – Prediction.** Does the ensemble uncertainty of a learned car-following policy rise before a safety violation?

**RQ2 – Timeliness.** Is the warning early enough for an intervention to prevent the violation?

**RQ3 – Added value.** Does uncertainty-triggered intervention improve closed-loop safety beyond a conventional monitor (TTC / time headway) alone, and at what comfort and false-alarm cost?

**RQ4 – Efficient testing.** How many simulations do grid, Monte Carlo and Bayesian-optimisation search need to find failures that meet a minimum plausibility threshold?

## What the learned model is (and is not)

The learned policy is trained by **behaviour cloning** on human car-following trajectories: it predicts the acceleration a human driver applied, given the recent traffic state. It is **not** a validated or safe ACC. It is the system under test.

## Definitions

| Term | Definition (v0) |
|---|---|
| Collision | Bumper-to-bumper gap ≤ 0 m |
| Safety violation (failure) | Collision **or** TTC < 1.0 s **or** time headway < 0.3 s (at ego speed > 1 m/s) |
| Prediction horizon | A warning at time *t* counts as correct if a failure occurs in (*t*, *t* + 3 s]. 3 s is an initial experimental choice, not a standard; 1.5 s and 5 s reported as sensitivity |
| Failure event | Consecutive time steps in violation form one event (event-level evaluation, not per-step) |
| Plausibility | Log-likelihood of a scenario's parameters under distributions fitted to naturalistic data. Used as a **constraint** (above a fixed percentile), never multiplied into the objective |
| Uncertainty | Standard deviation of the acceleration predicted by an ensemble of independently trained models (epistemic only; aleatoric uncertainty is not captured) |

## Data splits

Split by **recording / location** first, then by vehicle ID. No vehicle appears in more than one split. Overlapping sliding windows never cross a split boundary.

## Baselines

1. IDM (transparent rule-based car-following)
2. Learned policy without any monitor
3. Learned policy + TTC/headway monitor (no uncertainty)
4. Learned policy + uncertainty monitor
5. Learned policy + both monitors

## Metrics

- Failure prediction: AUROC, AUPRC (event level)
- Early warning: median and 10th-percentile lead time; missed failures
- Safety: collisions, violations, min TTC, min gap
- Comfort: RMS jerk, max deceleration
- Cost of warnings: false alarms and interventions per hour of driving
- Uncertainty reported with bootstrap 95% confidence intervals where feasible

## Safety shield assumptions

The shield is a safe-distance check **inspired by** RSS (Shalev-Shwartz et al., 2017); it is not a full RSS implementation. It assumes: known ego reaction time, known minimum guaranteed ego braking, a worst-case lead braking rate, perfect perception of gap and speeds. Its guarantees hold only under these assumptions, and tests check exactly that.

## Claims this project will not make

- No claim of ISO 21448 (SOTIF) compliance; the work is *SOTIF-oriented* validation.
- NGSIM → highD is reported as **cross-dataset generalisation**, not a pure geographic shift.
- Novelty lies in the controlled comparison and the reproducible pipeline, not in a new algorithm.

## Change log

| Date | Change | Reason |
|---|---|---|
| 2026-10-20 | v0 created | – |
| 2026-10-21 | Fallback for RQ3 is IDM calibrated on closed-loop spacing instead of an RSS-inspired braking shield | Simpler, and empirically safe on this data before any RQ3 experiment: 0 collisions and 6 violation events in 4,037 test events (Phase 2b). The RSS-inspired shield remains possible future work. |
| 2026-10-21 | All monitors use the same delayed observation as the policy (input window ends 0.5 s before the decision) | Equal information timing, so monitors are compared on information content, not on sensor latency |
| 2026-10-21 | Thresholds are set as an alarm budget (share of normal driving time in alarm) on the validation replay: 1 %, 2 % (main), 5 % | Makes monitors with different scales comparable; thresholds never see test data |
| 2026-10-21 | "Kinematic" monitor = TTC or time headway (max of their validation percentiles); "combined" = kinematic or uncertainty | Matches the protocol's baselines 3–5 at an equal alarm budget |
