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

### Results (5-member ensemble, 15 epochs; all 4,037 test events, 62 h)

Open loop (acceleration RMSE, m/s²): predict 0: 0.84 / 0.81 (val / test); IDM textbook: 2.19 / 2.13; IDM calibrated: 0.76 / 0.73; **GRU ensemble: 0.55 / 0.50**. Mean epistemic std 0.045 / 0.038 m/s²; Spearman correlation between epistemic std and absolute error 0.16 on both splits. Validation NLL was still decreasing at epoch 15, so longer training may help further.

Closed loop:

| Controller | Spacing RMSE mean / median | Speed RMSE | Collisions | Events with violation |
|---|---|---|---|---|
| IDM, acceleration-calibrated | 19.0 / 10.0 m | 1.87 m/s | 1 | 824 |
| IDM, spacing-calibrated | 5.0 / 3.8 m | 0.98 m/s | 0 | 6 |
| **GRU ensemble** | **4.3 / 3.6 m** | **0.81 m/s** | **30** | 125 |
| Human (recorded) | – | – | 0 | 198 |

The learned policy is the most human-like controller and has fewer violation events than the recorded humans, but it collides in 0.7 % of events, while spacing-calibrated IDM never does. A trial ensemble (2 members, 4 epochs, 100k windows) had collided in 7 % of a 300-event subset; full training reduced this to 0.7 % (2 of the same 300 events). Character of the 30 collisions: mostly low-speed contact in stop-and-go traffic. In 19 the leader had stopped and the ego was below 5 km/h; the median closing speed at impact is 3.5 km/h (90th percentile 6.5 km/h, maximum 19.7 km/h). In the 3 s before impact the policy braked about as hard as the human did in the same situation (median minimum acceleration −2.85 vs −2.72 m/s²), so the failures are mainly stopping slightly too late or creeping forward, not a lack of braking authority. (The trial ensemble additionally showed too-gentle braking; full training removed most of that.)
