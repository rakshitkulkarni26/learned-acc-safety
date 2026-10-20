# Data

No dataset files are stored in this repository. Everything in this folder except this README is ignored by git.

| Dataset | Access | Use in this project |
|---|---|---|
| NGSIM (US-101, I-80) | Public, US DOT via data.transportation.gov | Training data for the learned car-following policy |
| highD (German highways) | Free for non-commercial research after registration with the dataset authors | Cross-dataset evaluation |

## NGSIM

```bash
lacc data download    # ~1.5 GB -> data/raw/ngsim_trajectories.csv
lacc data prepare     # -> data/processed/ngsim/events.parquet + reports/data_ngsim/
```

If the download command fails, download the CSV in a browser from
`https://data.transportation.gov/api/views/8ect-6jqj/rows.csv?accessType=DOWNLOAD`
and save it as `data/raw/ngsim_trajectories.csv`.

Processing choices are documented in [`docs/DATA.md`](../docs/DATA.md).

## highD

highD terms prohibit redistributing the data or modified versions of it. Keep raw and processed files in this folder only.
