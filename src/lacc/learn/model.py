"""GRU car-following policy with a Gaussian output, and a deep ensemble of it.

Each network predicts a mean acceleration and a variance. The variance captures
*aleatoric* uncertainty (human drivers are not perfectly predictable). Training
several networks from different random seeds and measuring how much their means
disagree captures *epistemic* uncertainty (the models have not seen enough
similar data). RQ1-3 are about the epistemic part.

Reference: Lakshminarayanan, Pritzel & Blundell (2017), deep ensembles.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

LOG_VAR_MIN, LOG_VAR_MAX = -8.0, 4.0


class GRUPolicy(nn.Module):
    def __init__(self, n_features: int = 3, hidden: int = 64, layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out, _ = self.gru(x)
        mean, log_var = self.head(out[:, -1]).unbind(-1)
        return mean, log_var.clamp(LOG_VAR_MIN, LOG_VAR_MAX)


def gaussian_nll(mean: torch.Tensor, log_var: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood of y under N(mean, exp(log_var)), constant dropped."""
    return 0.5 * (log_var + (y - mean) ** 2 / log_var.exp()).mean()


class Normaliser:
    """Standardises inputs and targets with training-set statistics."""

    def __init__(self, x_mean, x_std, y_mean, y_std):
        self.x_mean = np.asarray(x_mean, dtype=np.float32)
        self.x_std = np.asarray(x_std, dtype=np.float32)
        self.y_mean = float(y_mean)
        self.y_std = float(y_std)

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray) -> Normaliser:
        flat = X.reshape(-1, X.shape[-1])
        return cls(flat.mean(0), flat.std(0) + 1e-6, y.mean(), y.std() + 1e-6)

    def x(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.x_mean) / self.x_std).astype(np.float32)

    def y(self, y: np.ndarray) -> np.ndarray:
        return ((y - self.y_mean) / self.y_std).astype(np.float32)

    def to_dict(self) -> dict:
        return {
            "x_mean": self.x_mean.tolist(),
            "x_std": self.x_std.tolist(),
            "y_mean": self.y_mean,
            "y_std": self.y_std,
        }


class Ensemble:
    """Several trained members plus the shared normaliser."""

    def __init__(self, members: list[GRUPolicy], norm: Normaliser):
        self.members = [m.eval() for m in members]
        self.norm = norm

    @torch.no_grad()
    def predict(self, X: np.ndarray, batch: int = 8192) -> dict[str, np.ndarray]:
        """Ensemble prediction in physical units (m/s^2).

        Returns mean, epistemic_std (spread of member means) and
        aleatoric_std (average predicted noise level).
        """
        means, variances = [], []
        Xn = self.norm.x(X)
        for m in self.members:
            mu, var = [], []
            for i in range(0, len(Xn), batch):
                a, lv = m(torch.from_numpy(Xn[i : i + batch]))
                mu.append(a.numpy())
                var.append(lv.exp().numpy())
            means.append(np.concatenate(mu) * self.norm.y_std + self.norm.y_mean)
            variances.append(np.concatenate(var) * self.norm.y_std**2)
        means, variances = np.stack(means), np.stack(variances)
        return {
            "mean": means.mean(0),
            "epistemic_std": means.std(0),
            "aleatoric_std": np.sqrt(variances.mean(0)),
            "member_means": means,
        }

    def save(self, out_dir: Path, model_cfg: dict) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for k, m in enumerate(self.members):
            torch.save(m.state_dict(), out_dir / f"member_{k}.pt")
        meta = {"model": model_cfg, "norm": self.norm.to_dict(), "members": len(self.members)}
        (out_dir / "ensemble.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, out_dir: Path) -> Ensemble:
        meta = json.loads((out_dir / "ensemble.json").read_text())
        members = []
        for k in range(meta["members"]):
            m = GRUPolicy(**meta["model"])
            m.load_state_dict(torch.load(out_dir / f"member_{k}.pt", weights_only=True))
            members.append(m)
        return cls(members, Normaliser(**meta["norm"]))
