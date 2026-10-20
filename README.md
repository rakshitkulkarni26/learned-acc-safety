# When does a learned ACC fail, and does it know?

![CI](https://github.com/rakshitkulkarni26/learned-acc-safety/actions/workflows/ci.yml/badge.svg)

Learned components are entering driving functions such as Adaptive Cruise Control, but they are hard to validate: the scenario space is huge and a learned controller can fail in situations nobody thought to test.

This project trains a car-following policy on real traffic data and asks:

> **When does the policy's own uncertainty give actionable early warning of safety failures, and does acting on that warning improve closed-loop safety beyond a conventional safety monitor?**

A second question: can those failures be found efficiently **without** relying on unrealistic scenario parameters?

Definitions, baselines and metrics are fixed in advance in [`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md). New here? Start with [`docs/PROJECT_GUIDE.md`](docs/PROJECT_GUIDE.md).

## Pipeline

```
NGSIM / highD ─► data pipeline ─► learned policy (GRU deep ensemble) ─┐
                      │                                                ▼
                      └─► scenario distributions ─► closed-loop simulator + safety shield
                                                                       │
                                     scenario search (MC / grid / BO) ◄┘
                                                                       │
                              KPIs, uncertainty evaluation, OpenSCENARIO export → esmini
```

## Status

| Phase | Content | Status |
|---|---|---|
| 0 | Repo, CI, deterministic simulator, IDM baseline, KPIs, experiment protocol | ✅ |
| 1 | NGSIM pipeline: recording detection, smoothing, event extraction, leakage-safe splits — **11,982 events, 163 vehicle-hours** | ✅ |
| 2 | Learned policy: GRU deep ensemble, closed-loop check | ⏳ |
| 3 | Failure labels, uncertainty vs TTC monitor, shield comparison | ⏳ |
| 4 | Plausibility-constrained scenario search, OpenSCENARIO + esmini replay | ⏳ |
| 5 | Bayesian optimisation, highD cross-dataset, technical report | ⏳ |

## Quickstart (no dataset needed)

```bash
git clone https://github.com/rakshitkulkarni26/learned-acc-safety.git
cd learned-acc-safety
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"

pytest                      # unit + scenario regression tests
lacc demo                   # IDM baseline, lead vehicle braking
```

`lacc demo` writes `reports/demo_kpis.json` and `reports/demo_trace.png`.

## Real data (NGSIM)

```bash
lacc data download          # ~1.5 GB public US DOT trajectory data
lacc data prepare           # car-following events + data quality report
```

Processing choices and their validation: [`docs/DATA.md`](docs/DATA.md).

## Repository layout

```
src/lacc/        simulator, controllers, scenarios, metrics, CLI
src/lacc/data/   NGSIM download, cleaning, smoothing, event extraction, splits
tests/           unit tests, safety invariants, scenario regression, synthetic-data pipeline tests
configs/         versioned scenario / controller / vehicle / data parameters
docs/            experiment protocol (pre-registered definitions), data processing
data/            local only, never committed
reports/         generated outputs
```

## Scope and honesty notes

- The learned model imitates human drivers (behaviour cloning). It is the **system under test**, not a proposed safe ACC.
- The safety shield is inspired by RSS, with its assumptions documented; it is not a full RSS implementation.
- This is SOTIF-oriented validation work, not an ISO 21448 compliance claim.
- Uncertainty-aware control, safety shields and scenario-based testing are established research areas. The contribution here is a controlled comparison and a reproducible open pipeline.

## Data

See [`data/README.md`](data/README.md). No dataset files are stored in this repository.

## License

MIT, see [LICENSE](LICENSE).
