# Phase 2: learned car-following policy

Settings live in `configs/policy_gru.yaml`. This file explains the choices.

## What is learned

**Behaviour cloning:** a neural network learns to predict the acceleration a human driver applied, given what that driver could see. It imitates humans; it is the *system under test*, not a proposed safe ACC.

## Samples

```
event (one ego following one leader, ≥ 15 s)
  └─ sample every 0.5 s:
       input  = last 3.0 s of [gap, ego speed, relative speed]   (30 steps × 3 features)
       target = ego acceleration 0.5 s after the last input step
```

| Choice | Value | Reason |
|---|---|---|
| Features | gap, ego speed, relative speed (leader − ego) | What a driver or radar sees; the classic inputs of car-following models. Own past acceleration is deliberately **not** an input: models given it tend to copy it forward instead of reacting to the leader (the "copycat" problem in behaviour cloning). |
| History | 3.0 s | Long enough to see the leader braking or accelerating |
| Delay | 0.5 s | (1) Human reaction time. (2) Smoothed speeds use a centred 1 s window, so a value at time t contains position data up to t + 0.5 s. Ending the input 0.5 s before the target keeps all future information out of the inputs. |
| Stride | 0.5 s | Neighbouring samples are almost identical at 0.1 s; 0.5 s keeps the dataset small without losing variety |
| Artefacts | excluded as **targets** (3.4 % of samples) | Constant-speed stretches in NGSIM are processing artefacts (docs/DATA.md). They may still appear inside input windows. |

Windows never cross events, and each inherits its event's split, so the leakage-safe split from Phase 1 holds.

Resulting dataset: **573k train, 100k validation, 420k test windows.**

## Model

- **GRU** (gated recurrent unit), 1 layer, 64 hidden units: reads the 30-step history, outputs a **mean** acceleration and a **variance**.
- Trained with the **Gaussian negative log-likelihood**, so the variance learns how unpredictable the driver is in that situation (*aleatoric* uncertainty).
- **Deep ensemble:** 5 networks trained from different random seeds. Where their means disagree, the models lack knowledge (*epistemic* uncertainty). Research questions 1–3 are about the epistemic part.
- Adam, learning rate 1e-3, batch 1024, up to 15 epochs, early stopping on validation NLL (patience 3), gradient clipping at 1.0. Inputs and targets are standardised with training-set statistics.

## Baselines (open loop, same windows)

| Baseline | Why |
|---|---|
| Predict 0 | Sanity floor: any useful model must beat it |
| IDM, textbook parameters | Shows how far an uncalibrated physical model is from this data |
| IDM, calibrated on train | The fair baseline: least-squares fit of the five IDM parameters on training accelerations |

Calibrated IDM on NGSIM: RMSE 0.76 (val) / 0.73 (test) m/s², versus 0.84 / 0.81 for predicting 0, and above 2 for textbook IDM. Calibrated parameters: T ≈ 0.69 s, s0 ≈ 1.5 m, a ≈ 0.21 m/s², b ≈ 2.0 m/s². This reflects dense, slow traffic.

Note: car-following models are usually calibrated on simulated *spacing*, not on one-step acceleration. Closed-loop calibration and evaluation follow in Phase 2b.

## Outputs

| File | Content |
|---|---|
| `data/processed/ngsim/windows.npz` | Samples (local only, ~400 MB) |
| `models/gru_ensemble/` | `member_k.pt` weights + `ensemble.json` (architecture, normalisation) |
| `reports/policy_gru/policy_metrics.json` | Baselines and ensemble metrics on val and test, training histories |
| `reports/policy_gru/policy_report.png` | Training curves, RMSE comparison, one test event with uncertainty bands |

## Metrics reported

RMSE and MAE of acceleration; Gaussian NLL using total variance; mean epistemic and aleatoric standard deviation; Spearman rank correlation between epistemic std and absolute error. The last is a first, open-loop hint for RQ1; the actual RQ1–3 analysis is closed loop (Phase 3).

## Phase 2b: closed-loop replay

`lacc policy replay` replays every test event: the leader follows its recording, and a controller drives the ego.

- The first 3.5 s are the recorded human trajectory (warm-up), so the controller starts with a full input window.
- From then on the controller acts every 0.1 s on the same delayed 3 s window as in training; acceleration is limited to [−8, +4] m/s².
- Compared controllers: IDM calibrated on acceleration (from Phase 2), **IDM calibrated on closed-loop spacing** (300 training events, first 60 s, Powell search, collision penalty; the standard calibration approach), and the GRU ensemble.
- Per event: spacing and speed RMSE against the human, collision, minimum gap and TTC, and safety violations exactly as defined in the experiment protocol. The human's own violations on the same events are reported for reference.
- GRU traces (gap, speed, acceleration, epistemic std per step) are saved for Phase 3.

A unit test confirms that replaying the human's recorded accelerations reproduces the recorded gap exactly, so the simulator is consistent with the data.

### Preliminary observation (trial models: 2 members, 100k windows, 4 epochs; 300 test events)

| Controller | Spacing RMSE mean / median | Collisions | Events with violation |
|---|---|---|---|
| IDM, acceleration-calibrated | 20.3 / 11.1 m | 1 | 56 |
| IDM, spacing-calibrated | 5.2 / 3.9 m | 0 | 1 |
| GRU ensemble | 4.5 / 3.7 m | 21 | 36 |
| Human (recorded) | – | 0 | 22 |

The learned policy is the most human-like on average but collides in about 7 % of events, mostly when approaching a stopped or slow leader in stop-and-go traffic (it brakes at most about −2 m/s², more gently than the human) or by creeping into a stopped leader at very low speed. Both are typical behaviour-cloning weaknesses: regression towards average behaviour and no notion of "hold still". In the 3 s before a collision, ensemble disagreement was about twice its average. These numbers come from trial models and are replaced by the full-ensemble results.
