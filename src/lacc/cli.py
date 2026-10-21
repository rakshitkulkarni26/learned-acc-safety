"""Command-line entry point: ``lacc demo``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .controllers import IDM
from .metrics import kpis, time_to_collision
from .scenarios import LeadBrake
from .sim import VehicleParams, simulate


def _plot(trace, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ttc = time_to_collision(trace.gap, trace.v_ego, trace.v_lead)
    fig, ax = plt.subplots(4, 1, figsize=(8, 8), sharex=True)
    ax[0].plot(trace.t, trace.v_ego * 3.6, label="ego")
    ax[0].plot(trace.t, trace.v_lead * 3.6, "--", label="lead")
    ax[0].set_ylabel("speed [km/h]")
    ax[0].legend(loc="lower left")
    ax[1].plot(trace.t, trace.gap)
    ax[1].set_ylabel("gap [m]")
    ax[2].plot(trace.t, trace.a_ego, label="actual")
    ax[2].plot(trace.t, trace.a_cmd, ":", label="command")
    ax[2].set_ylabel("accel [m/s²]")
    ax[2].legend(loc="lower left")
    ax[3].plot(trace.t, ttc)
    ax[3].set_ylim(0, 15)
    ax[3].set_ylabel("TTC [s]")
    ax[3].set_xlabel("time [s]")
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run_demo(config_path: Path, out_dir: Path) -> dict:
    cfg = yaml.safe_load(config_path.read_text())
    scenario = LeadBrake(**cfg.get("scenario", {}))
    controller = IDM(**cfg.get("idm", {}))
    params = VehicleParams(**cfg.get("vehicle", {}))

    trace = simulate(
        controller,
        scenario.lead_profile(),
        v_ego0=scenario.v_ego0,
        v_lead0=scenario.v_lead0,
        gap0=scenario.gap0,
        duration=scenario.duration,
        dt=cfg.get("dt", 0.1),
        params=params,
    )
    result = {"scenario": scenario.as_dict(), "kpis": kpis(trace)}

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "demo_kpis.json").write_text(json.dumps(result, indent=2))
    _plot(trace, out_dir / "demo_trace.png", "IDM baseline: lead-vehicle braking")
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="lacc")
    sub = parser.add_subparsers(dest="cmd", required=True)
    demo = sub.add_parser("demo", help="run the baseline demo scenario (no dataset needed)")
    demo.add_argument("--config", type=Path, default=Path("configs/demo.yaml"))
    demo.add_argument("--out", type=Path, default=Path("reports"))

    data = sub.add_parser("data", help="dataset download and preparation")
    data_sub = data.add_subparsers(dest="data_cmd", required=True)
    dl = data_sub.add_parser("download", help="download the NGSIM trajectory CSV (~1.5 GB)")
    dl.add_argument("--out", type=Path, default=Path("data/raw/ngsim_trajectories.csv"))
    prep = data_sub.add_parser("prepare", help="clean, smooth, extract events, split")
    prep.add_argument("--config", type=Path, default=Path("configs/data_ngsim.yaml"))

    pol = sub.add_parser("policy", help="learned car-following policy (Phase 2)")
    pol_sub = pol.add_subparsers(dest="policy_cmd", required=True)
    win = pol_sub.add_parser("windows", help="slice events into training windows")
    win.add_argument("--config", type=Path, default=Path("configs/policy_gru.yaml"))
    tr = pol_sub.add_parser("train", help="train the GRU ensemble and evaluate it")
    tr.add_argument("--config", type=Path, default=Path("configs/policy_gru.yaml"))
    tr.add_argument("--members", type=int, help="override ensemble size (e.g. 1 for a quick test)")
    tr.add_argument(
        "--max-train-windows", type=int, help="subsample training data for a quick test"
    )
    rp = pol_sub.add_parser("replay", help="closed-loop replay of test events (Phase 2b)")
    rp.add_argument("--config", type=Path, default=Path("configs/policy_gru.yaml"))
    rp.add_argument("--max-events", type=int, help="replay a random subset for a quick test")

    args = parser.parse_args(argv)

    if args.cmd == "demo":
        result = run_demo(args.config, args.out)
        print("KPIs:")
        for k, v in result["kpis"].items():
            print(f"  {k:<20} {v:.3f}" if isinstance(v, float) else f"  {k:<20} {v}")
        print(f"Wrote {args.out / 'demo_kpis.json'} and {args.out / 'demo_trace.png'}")

    elif args.cmd == "data" and args.data_cmd == "download":
        from .data.ngsim import download

        print(f"Downloading NGSIM to {args.out} ...")
        download(args.out)

    elif args.cmd == "data" and args.data_cmd == "prepare":
        from .data.pipeline import prepare

        report = prepare(args.config)
        print("\nRecordings found:")
        for rec, info in report["clean"]["recordings"].items():
            print(f"  {rec:<12} {info['vehicles']:>5} vehicles  {info['duration_min']:>5} min")
        print(f"\nEvents: {report['events']['events']}  ({report['events']['event_hours']} h)")
        print(f"Per split (events): {report['splits']['events_per_split']}")
        print(f"Test recordings: {report['splits']['test_recordings']}")
        for w in report["warnings"]:
            print(f"WARNING: {w}")
        print("\nReport: reports/data_ngsim/data_report.json and data_report.png")

    elif args.cmd == "policy" and args.policy_cmd == "windows":
        from .learn.train import load_config, make_windows

        cfg = load_config(args.config)
        print(f"Building windows from {cfg['events']} ...")
        for split, info in make_windows(cfg).items():
            print(
                f"  {split:<5} {info['windows']:>9,} windows  {info['events']:>6,} events"
                f"  ({info['artefact_targets']:,} artefact targets)"
            )
        print(f"Saved {cfg['windows_file']}")

    elif args.cmd == "policy" and args.policy_cmd == "train":
        from .learn.train import load_config, train_ensemble

        cfg = load_config(args.config)
        if args.members:
            cfg["train"]["members"] = args.members
        if args.max_train_windows:
            cfg["train"]["max_train_windows"] = args.max_train_windows
        m = train_ensemble(cfg)
        print("\nOpen-loop RMSE of acceleration [m/s^2]       val     test")
        for key, label in [
            ("rmse_zero", "predict 0"),
            ("rmse_idm_default", "IDM, textbook parameters"),
            ("rmse_idm_calibrated", "IDM, calibrated on train"),
            ("rmse_gru_ensemble", "GRU ensemble"),
        ]:
            print(f"  {label:<38} {m['val'][key]:6.3f}   {m['test'][key]:6.3f}")
        print(
            f"\nMean epistemic std  val {m['val']['mean_epistemic_std']:.3f}"
            f"  test {m['test']['mean_epistemic_std']:.3f} m/s^2"
        )
        print(
            "Spearman(epistemic std, |error|)  "
            + "  ".join(
                f"{s} {m[s]['spearman_epistemic_vs_abs_error']:.2f}"
                if m[s]["spearman_epistemic_vs_abs_error"] is not None
                else f"{s} n/a (needs 2+ members)"
                for s in ("val", "test")
            )
        )
        print(f"\nModels: {cfg['out_dir']}   Report: {cfg['report_dir']}/policy_report.png")

    elif args.cmd == "policy" and args.policy_cmd == "replay":
        from .learn.replay import run
        from .learn.train import load_config

        cfg = load_config(args.config)
        if args.max_events:
            cfg["replay"]["max_events"] = args.max_events
        out = run(cfg)
        r = out["results"]
        names = [
            ("idm_accel_calibrated", "IDM, calibrated on acceleration"),
            ("idm_spacing_calibrated", "IDM, calibrated on spacing"),
            ("gru_ensemble", "GRU ensemble"),
        ]
        print(
            "\nClosed-loop replay                 spacing RMSE [m]  speed RMSE  collisions"
            "  events w/ violation"
        )
        print("                                   mean   median     [m/s]")
        for key, label in names:
            x = r[key]
            print(
                f"  {label:<32} {x['spacing_rmse_mean']:5.2f}  {x['spacing_rmse_median']:6.2f}"
                f"   {x['speed_rmse_mean']:6.2f}     {x['collisions']:>5}     "
                f"{x['events_with_violation']:>5} / {x['events']}"
            )
        h = r["human"]
        print(
            f"  {'Human (recorded)':<32}    -       -         -          -      "
            f"{h['events_with_violation']:>5} / {h['events']}"
        )
        print(f"\nReport: {cfg['replay']['report_dir']}/replay_report.png")


if __name__ == "__main__":
    main()
