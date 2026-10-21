"""Train the GRU deep ensemble and evaluate it open-loop against baselines."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from .baselines import IDM_DEFAULT, calibrate_idm, idm_predict
from .model import Ensemble, GRUPolicy, Normaliser, gaussian_nll
from .windows import build_windows, load_windows, save_windows, select, summary


def load_config(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


# --------------------------------------------------------------------------- windows
def make_windows(cfg: dict) -> dict:
    events = pd.read_parquet(cfg["events"])
    w = build_windows(events, dt=cfg.get("dt", 0.1), **cfg["windows"])
    save_windows(Path(cfg["windows_file"]), w)
    return summary(w)


# --------------------------------------------------------------------------- training
def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@torch.no_grad()
def _eval_nll(model, X: torch.Tensor, y: torch.Tensor, device, batch: int = 8192) -> float:
    model.eval()
    total = 0.0
    for i in range(0, len(X), batch):
        mu, lv = model(X[i : i + batch].to(device))
        total += float(gaussian_nll(mu, lv, y[i : i + batch].to(device))) * len(mu)
    return total / len(X)


def train_member(Xtr, ytr, Xva, yva, model_cfg: dict, tcfg: dict, seed: int, log=print):
    torch.manual_seed(seed)
    device = _device(tcfg.get("device", "auto"))
    model = GRUPolicy(**model_cfg).to(device)
    opt = torch.optim.Adam(
        model.parameters(), lr=tcfg["lr"], weight_decay=tcfg.get("weight_decay", 0)
    )
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
        batch_size=tcfg["batch_size"],
        shuffle=True,
        generator=gen,
    )
    Xva_t, yva_t = torch.from_numpy(Xva), torch.from_numpy(yva)

    best, best_state, bad, history = np.inf, None, 0, []
    for epoch in range(1, tcfg["epochs"] + 1):
        t0 = time.time()
        model.train()
        run = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            mu, lv = model(xb)
            loss = gaussian_nll(mu, lv, yb)
            opt.zero_grad()
            loss.backward()
            nn_utils_clip(model)
            opt.step()
            run += loss.item() * len(xb)
        train_nll = run / len(Xtr)
        val_nll = _eval_nll(model, Xva_t, yva_t, device)
        history.append({"epoch": epoch, "train_nll": train_nll, "val_nll": val_nll})
        log(
            f"    epoch {epoch:>2}  train NLL {train_nll:7.4f}  val NLL {val_nll:7.4f}"
            f"  ({time.time() - t0:.0f} s)"
        )
        if val_nll < best - 1e-4:
            best, bad = val_nll, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= tcfg.get("patience", 3):
                log(f"    early stop (best val NLL {best:.4f})")
                break
    model.load_state_dict(best_state)
    return model.cpu(), history


def nn_utils_clip(model, max_norm: float = 1.0) -> None:
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


# --------------------------------------------------------------------------- evaluation
def evaluate(ens: Ensemble, data: dict, idm_cal: np.ndarray) -> dict:
    X, y = data["X"], data["y"]
    p = ens.predict(X)
    total_var = p["aleatoric_std"] ** 2 + p["epistemic_std"] ** 2
    err = p["mean"] - y
    abs_err = np.abs(err)
    # does higher epistemic uncertainty go with larger errors? (Spearman rank correlation)
    rank = lambda a: pd.Series(a).rank().to_numpy()  # noqa: E731
    if len(ens.members) > 1:
        spearman = float(np.corrcoef(rank(p["epistemic_std"]), rank(abs_err))[0, 1])
    else:
        spearman = None  # one member: no disagreement, no epistemic uncertainty
    return {
        "windows": int(len(y)),
        "rmse_zero": float(np.sqrt(np.mean(y**2))),
        "rmse_idm_default": float(np.sqrt(np.mean((idm_predict(IDM_DEFAULT, X) - y) ** 2))),
        "rmse_idm_calibrated": float(np.sqrt(np.mean((idm_predict(idm_cal, X) - y) ** 2))),
        "rmse_gru_ensemble": float(np.sqrt(np.mean(err**2))),
        "mae_gru_ensemble": float(abs_err.mean()),
        "nll_gru_ensemble": float(
            np.mean(0.5 * (np.log(2 * np.pi * total_var) + err**2 / total_var))
        ),
        "mean_epistemic_std": float(p["epistemic_std"].mean()),
        "mean_aleatoric_std": float(p["aleatoric_std"].mean()),
        "spearman_epistemic_vs_abs_error": spearman,
    }


def _plot(histories, metrics, ens, test, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for k, h in enumerate(histories):
        ax[0].plot(
            [e["epoch"] for e in h],
            [e["val_nll"] for e in h],
            marker="o",
            ms=3,
            label=f"member {k}",
        )
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("validation NLL (normalised)")
    ax[0].set_title("Training curves")
    ax[0].legend(fontsize=7)

    names = ["rmse_zero", "rmse_idm_default", "rmse_idm_calibrated", "rmse_gru_ensemble"]
    labels = ["predict 0", "IDM default", "IDM calibrated", "GRU ensemble"]
    x = np.arange(len(names))
    for j, split in enumerate(["val", "test"]):
        ax[1].bar(x + (j - 0.5) * 0.38, [metrics[split][n] for n in names], 0.38, label=split)
    ax[1].set_xticks(x, labels, fontsize=8)
    ax[1].set_ylabel("RMSE acceleration [m/s²]")
    ax[1].set_title("Open-loop accuracy")
    ax[1].legend()

    eid = pd.Series(test["event_id"]).value_counts().index[0]
    m = test["event_id"] == eid
    p = ens.predict(test["X"][m])
    t = test["t"][m]
    ax[2].plot(t, test["y"][m], "k", lw=1.5, label="human (smoothed)")
    ax[2].plot(t, p["mean"], color="#3b6ea5", label="ensemble mean")
    ax[2].fill_between(
        t,
        p["mean"] - 2 * p["aleatoric_std"],
        p["mean"] + 2 * p["aleatoric_std"],
        color="#3b6ea5",
        alpha=0.15,
        label="±2σ aleatoric",
    )
    ax[2].fill_between(
        t,
        p["mean"] - 2 * p["epistemic_std"],
        p["mean"] + 2 * p["epistemic_std"],
        color="#d1495b",
        alpha=0.35,
        label="±2σ epistemic",
    )
    ax[2].set_xlabel("time in event [s]")
    ax[2].set_ylabel("acceleration [m/s²]")
    ax[2].set_title("Longest test event")
    ax[2].legend(fontsize=7)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def train_ensemble(cfg: dict, log=print) -> dict:
    tcfg, mcfg = cfg["train"], cfg["model"]
    excl = cfg.get("exclude_artefacts", True)
    w = load_windows(Path(cfg["windows_file"]))
    tr, va, te = (select(w, s, excl) for s in ("train", "val", "test"))

    if tcfg.get("max_train_windows"):
        rng = np.random.default_rng(0)
        idx = rng.choice(
            len(tr["y"]), size=min(tcfg["max_train_windows"], len(tr["y"])), replace=False
        )
        tr = {k: (v[idx] if k != "meta" else v) for k, v in tr.items()}

    log(
        f"Windows: train {len(tr['y']):,}  val {len(va['y']):,}  test {len(te['y']):,}"
        f"  (artefact targets excluded: {excl})"
    )
    norm = Normaliser.fit(tr["X"], tr["y"])
    Xtr, ytr = norm.x(tr["X"]), norm.y(tr["y"])
    Xva, yva = norm.x(va["X"]), norm.y(va["y"])

    members, histories = [], []
    for k in range(tcfg["members"]):
        seed = tcfg.get("seed", 0) + k
        log(f"  member {k + 1}/{tcfg['members']} (seed {seed})")
        model, hist = train_member(Xtr, ytr, Xva, yva, mcfg, tcfg, seed, log)
        members.append(model)
        histories.append(hist)

    ens = Ensemble(members, norm)
    out_dir = Path(cfg["out_dir"])
    ens.save(out_dir, mcfg)

    log("Calibrating IDM baseline on the training split ...")
    idm = calibrate_idm(tr["X"], tr["y"])
    idm_vec = np.array(list(idm["params"].values()))
    metrics = {
        "idm_calibrated": idm,
        "val": evaluate(ens, va, idm_vec),
        "test": evaluate(ens, te, idm_vec),
        "histories": histories,
        "config": cfg,
    }
    rep = Path(cfg["report_dir"])
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "policy_metrics.json").write_text(json.dumps(metrics, indent=2))
    _plot(histories, metrics, ens, te, rep / "policy_report.png")
    return metrics
