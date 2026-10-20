# Data processing (Phase 1: NGSIM)

Every choice below is set in `configs/data_ngsim.yaml`. This file explains why.

## Source

US DOT, *Next Generation Simulation (NGSIM) Vehicle Trajectories and Supporting Data*, data.transportation.gov (public). Video-based trajectories at 10 Hz. Only the two freeway sites are used: **US-101** (Los Angeles) and **I-80** (Emeryville). Raw positions are in feet and are converted to metres.

## Pipeline

```
raw CSV ─► select columns + freeway sites ─► recording key ─► drop duplicates ─► SI units
        ─► Savitzky-Golay smoothing per continuous segment ─► car-following events
        ─► leakage-safe splits ─► events.parquet + data report
```

### 1. Recording key

Vehicle IDs are only unique within one 15-minute recording, and some recordings follow each other directly in time. Within a recording, `Global_Time − Frame_ID · 0.1 s` is constant (the recording start time), so clustering this value per site separates recordings even when they are back-to-back.

**Check after the first real run:** the report should list **3 recordings per site**. The pipeline prints a warning if it does not.

### 2. Duplicates

Rows with the same (recording, vehicle, frame) are dropped, keeping the first.

### 3. Smoothing

Positions come from video tracking and are noisy; differentiating twice amplifies the noise. NGSIM's own `v_Vel` / `v_Acc` columns are kept only for comparison and are not used.

- Position and speed: Savitzky-Golay filter, 1.0 s window, polynomial order 2
- Acceleration: second derivative of a Savitzky-Golay fit, 2.0 s window, order 3
- Each continuous segment (no frame gaps) is filtered separately
- Filter output is biased where the window is one-sided, so **1.0 s is trimmed from both ends of every segment**
- Segments shorter than 4.2 s (two acceleration windows) are dropped

Smoothing is common practice for NGSIM; see Thiemann, Treiber & Kesting (2008) and Coifman & Li (2017) on the data's error characteristics.

**Validation on synthetic trajectories with known acceleration** (`tests/test_data_pipeline.py`):

| Position noise (std) | Accel RMSE, smoothed | Accel RMSE, naive double difference |
|---|---|---|
| 0.1 m | 0.12 m/s² | 6.0 m/s² |
| 0.3 m | 0.36 m/s² | 18 m/s² |
| 0.5 m | 0.59 m/s² | 30 m/s² |

The 2 s acceleration window trades responsiveness for noise suppression: brief acceleration peaks are flattened. This is acceptable for car-following behaviour, but it is a limitation to keep in mind when interpreting jerk.

### 4. Car-following events

An event is a maximal run of consecutive frames where:

- the ego is a car (`v_Class = 2`) in a main lane (lanes 1–5; ramps and auxiliary lanes excluded),
- the ego keeps the same lane and the same `Preceding` vehicle,
- both vehicles have valid smoothed kinematics,
- it lasts at least **15 s**.

Gap = leader front position − leader length − ego front position (NGSIM `Local_Y` is the front of the vehicle).

A whole event is rejected if at any sample the gap is below 0.5 m, |acceleration| exceeds 8 m/s² for either vehicle, or a speed is negative. These indicate tracking errors rather than driving behaviour.

### 5. Splits

Following `docs/EXPERIMENT_PROTOCOL.md`:

1. The **last recording of each site** is held out entirely as the test set.
2. In the remaining recordings, ego vehicles are assigned to train or val (15 %) by a seeded hash of (recording, ego ID).
3. An ego vehicle is never in more than one split. Sliding windows created later never cross events, and events never cross splits.

## Outputs (local only, never committed)

| File | Content |
|---|---|
| `data/raw/ngsim_trajectories.csv` | Full download (~1.5 GB) |
| `data/interim/ngsim_highway.parquet` | Freeway rows, needed columns only (cache) |
| `data/processed/ngsim/events.parquet` | One row per event sample: `event_id, location, recording, ego_id, leader_id, frame_id, t, lane, gap, v_ego, v_lead, a_ego, a_lead, class_lead, split` |
| `reports/data_ngsim/data_report.json` | Counts per step, recordings found, split sizes, signal quantiles |
| `reports/data_ngsim/data_report.png` | Distributions, raw vs smoothed acceleration, hours per split |

## Results of the first real run (21 Oct 2026)

| Quantity | Value |
|---|---|
| Rows read (US-101 + I-80) | 9,369,320 |
| Duplicates removed | 704,000 (7.5 %) |
| Recordings | 3 per site, 15.9–19.5 min, 1,790–2,169 vehicles each |
| Candidate runs / too short / implausible | 20,994 / 8,151 / 861 |
| Events kept | 11,982 (162.6 vehicle-hours) |
| Train / val / test | 6,778 / 1,167 / 4,037 events (85.6 / 15.0 / 62.0 h) |
| Median gap / speed / time headway | 12.1 m / 27 km/h / 1.8 s |
| RMS jerk, NGSIM `v_Acc` vs smoothed | 10.2 vs 1.6 m/s³ |

## Known limitations

- **Constant-speed artefacts in the published data.** 29 % of raw samples have `v_Acc` exactly 0 and 36 % have exactly linear positions (second difference < 1 mm), often at speeds that are multiples of 5 ft/s. NGSIM's own speed column shows the same pattern, so it originates in the data, not in this pipeline. How to handle it (mask, down-weight, or rely more on highD) is decided before Phase 2 training.
- **Mostly congested traffic** (median 27 km/h, frequent standstill), below typical ACC operating speeds.

- NGSIM contains tracking errors that smoothing cannot fully remove; the plausibility filters catch only the gross ones.
- The 2 s acceleration window flattens short peaks.
- `Preceding` is taken from the dataset as given; it is not re-derived from positions.
- Three recordings per site means the test set is one 15-minute period per site. Results will be reported with confidence intervals.

## References

- Thiemann, C., Treiber, M., Kesting, A. (2008). Estimating acceleration and lane-changing dynamics from Next Generation Simulation trajectory data. *Transportation Research Record*, 2088.
- Coifman, B., Li, L. (2017). A critical evaluation of the Next Generation Simulation (NGSIM) vehicle trajectory dataset. *Transportation Research Part B*, 105.
