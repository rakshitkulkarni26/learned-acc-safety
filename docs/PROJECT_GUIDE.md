# Project guide

A plain-language walk through the whole project: what it is for, what each part does, and where things stand. Read this first.

## 1. The big idea

Car makers are starting to use machine learning inside driving functions such as Adaptive Cruise Control (ACC). The hard part is proving such a learned controller is safe: it has no explicit rules to inspect, and it may behave badly in situations nobody tested.

This project asks two questions:

1. **Does the learned controller "know" when it is about to fail?** We train several copies of the same neural network. Where they disagree about what to do, the model is *uncertain*. If uncertainty rises a few seconds before dangerous situations, it can serve as an early warning that hands control to a simple, rule-based safety system.
2. **Can we find dangerous situations efficiently,** using only realistic test scenarios rather than physically absurd ones?

It continues my master's thesis, where I tested an existing neural-network ACC in closed loop. Here I train the model myself and study when and why it fails.

## 2. Tools

| Tool | What it is | Why |
|---|---|---|
| GitHub repository | Public project page | Shows code and progress |
| Git | Version control (`git add`, `git commit`, `git push`) | Every change is a saved, traceable snapshot |
| Virtual environment `C:\venvs\lacc` | Private Python installation for this project | No clashes with other software |
| `pip install -e ".[dev]"` | Installs the project as a package with its dependencies | Makes the `lacc` command work; code edits apply immediately |
| pytest | Runs the automated tests | `26 passed` = all checks succeed |
| GitHub Actions (CI) | Re-runs lint, tests and the demo on a clean Linux machine on every push | Proves the code works beyond one laptop |
| Location `C:\dev` | Project folder outside OneDrive | OneDrive would upload the 1.5 GB dataset and can corrupt git folders |

## 3. Phase 0: the test environment (done)

| File | Role |
|---|---|
| `src/lacc/sim.py` | **Simulator.** Ego car follows a lead car on a straight road. Every 0.1 s: give the controller gap, speeds and own acceleration; apply its command with a 0.3 s actuator delay and limits (+2.5 / −8 m/s², no reversing); update positions. Deterministic: same inputs, same result. |
| `src/lacc/controllers.py` | **IDM baseline.** Intelligent Driver Model: accelerate towards a desired speed, brake harder the closer the gap gets to a desired gap that grows with speed and closing speed. The transparent reference for the learned model. |
| `src/lacc/scenarios.py` | **Test situations.** Lead car cruises, brakes at X m/s² for Y s, then holds speed. A scenario is a handful of numbers, so thousands can be searched automatically later. |
| `src/lacc/metrics.py` | **KPIs.** Collision, minimum gap, TTC (gap ÷ closing speed), time headway (gap ÷ own speed), maximum braking, jerk. |
| `src/lacc/cli.py` | **The `lacc` command.** `lacc demo` runs one scenario and saves a plot and KPIs. |
| `tests/` | **Automatic checks.** Determinism, limits, IDM survives normal braking, and a *known-failure* case (weak brakes + hard-braking leader must collide) proving collision detection works. |
| `docs/EXPERIMENT_PROTOCOL.md` | **Rules written before any results:** what counts as a failure, warning horizon, data splits, baselines, claims we will not make. Prevents bending definitions later. |
| `configs/demo.yaml` | All numbers in one place, not hidden in code. |

## 4. Phase 1: real driving data (done)

NGSIM is a free US government dataset: cameras filmed highways US-101 and I-80 and tracked every vehicle 10 times per second. The raw data has problems; each pipeline step fixes one.

| Step | Problem | Solution | File |
|---|---|---|---|
| Load | 1.5 GB, feet, city streets mixed in | Keep the two highways, convert to metres | `data/ngsim.py` |
| Recordings | Each highway has three 15-min recordings and **vehicle IDs restart in each** | Identify the recording from its start time; warn unless 3 per highway are found | `data/ngsim.py` |
| Smoothing | Camera positions jitter; differentiating twice gives ±30 m/s² nonsense | Savitzky-Golay filter; speed and acceleration from the smooth curve; 1 s cut at each end where the filter is unreliable | `data/smoothing.py` |
| Events | Only clean car-following is wanted | Same leader, same lane, ≥ 15 s; reject impossible values (tracking errors) | `data/events.py` |
| Splits | Same car in training and test makes results look too good (**data leakage**) | Last recording per highway = test; the rest split by car, never by time window | `data/splits.py` |
| Report | Need to see whether the data makes sense | Counts per step and distribution plots | `data/pipeline.py` |

`tests/_ngsim_synth.py` builds fake data in NGSIM's exact format with deliberate traps (reused IDs, back-to-back recordings, duplicates, a lane change, an off-ramp car). Because the true answers are known, the tests check every step. `docs/DATA.md` documents every choice and its limits.

### Results on the real data (21 Oct 2026)

- 9.37 M rows read; 704,000 duplicates removed (7.5 %)
- 3 recordings per highway detected, as expected
- **11,982 car-following events, 162.6 vehicle-hours**
- Train / val / test: 6,778 / 1,167 / 4,037 events; test = `i-80_r3`, `us-101_r3`
- Median gap 12.1 m, median speed 27 km/h, median time headway 1.8 s

### What the data taught us

- **Mostly congested traffic** (median 27 km/h, stop-and-go). ACC usually runs faster, so highD (German highways) matters.
- **Constant-speed artefacts:** 29 % of samples have acceleration exactly 0, often at speeds that are multiples of 5 ft/s. This is in the published data itself, and a model could learn it. To be decided before training.

## 5. What comes next

| Phase | In one sentence |
|---|---|
| 2 | Train a small neural network (GRU) that predicts human acceleration from the last few seconds; train 5 copies to measure uncertainty |
| 3 | Put it in the simulator, find where it fails, test whether uncertainty warned early enough |
| 4 | Search for failures that are also realistic; export as OpenSCENARIO, replay in esmini |
| 5 | highD cross-dataset check and technical report |

## 6. Try it yourself

- Change `decel: 4.0` to `8.0` in `configs/demo.yaml`, run `lacc demo`, compare the plot.
- Change the IDM `T` (desired time headway) from 1.5 to 0.8 and see how minimum TTC changes.
- Open `reports/data_ngsim/data_report.png` and explain each panel in your own words.
