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
    args = parser.parse_args(argv)

    if args.cmd == "demo":
        result = run_demo(args.config, args.out)
        print("KPIs:")
        for k, v in result["kpis"].items():
            print(f"  {k:<20} {v:.3f}" if isinstance(v, float) else f"  {k:<20} {v}")
        print(f"Wrote {args.out / 'demo_kpis.json'} and {args.out / 'demo_trace.png'}")


if __name__ == "__main__":
    main()
