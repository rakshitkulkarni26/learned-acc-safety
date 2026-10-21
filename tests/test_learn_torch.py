import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_ensemble_trains_saves_and_reloads(tmp_path):
    from lacc.learn.model import Ensemble, Normaliser
    from lacc.learn.train import train_member

    rng = np.random.default_rng(3)
    X = rng.normal(size=(3000, 10, 3)).astype(np.float32)
    y = (0.8 * X[:, -1, 2] - 0.3 * X[:, -1, 0] + rng.normal(0, 0.1, 3000)).astype(np.float32)
    norm = Normaliser.fit(X, y)
    Xn, yn = norm.x(X), norm.y(y)
    cfg_m = {"n_features": 3, "hidden": 16, "layers": 1}
    cfg_t = {"lr": 0.01, "batch_size": 256, "epochs": 6, "patience": 3, "device": "cpu"}

    members, hists = [], []
    for seed in (0, 1):
        m, h = train_member(
            Xn[:2500], yn[:2500], Xn[2500:], yn[2500:], cfg_m, cfg_t, seed, log=lambda *_: None
        )
        members.append(m)
        hists.append(h)
    assert hists[0][-1]["val_nll"] < hists[0][0]["val_nll"]  # it learns

    ens = Ensemble(members, norm)
    p = ens.predict(X[2500:])
    rmse = np.sqrt(np.mean((p["mean"] - y[2500:]) ** 2))
    assert rmse < 0.5 * y.std()
    assert (p["epistemic_std"] >= 0).all() and (p["aleatoric_std"] > 0).all()

    ens.save(tmp_path, cfg_m)
    p2 = Ensemble.load(tmp_path).predict(X[2500:])
    np.testing.assert_allclose(p["mean"], p2["mean"], rtol=1e-5, atol=1e-6)


def test_members_with_different_seeds_disagree():
    from lacc.learn.train import train_member

    rng = np.random.default_rng(4)
    X = rng.normal(size=(800, 5, 3)).astype(np.float32)
    y = X[:, -1, 1].astype(np.float32)
    cfg_m = {"n_features": 3, "hidden": 8, "layers": 1}
    cfg_t = {"lr": 0.01, "batch_size": 128, "epochs": 2, "patience": 5, "device": "cpu"}
    a, _ = train_member(X, y, X, y, cfg_m, cfg_t, 0, log=lambda *_: None)
    b, _ = train_member(X, y, X, y, cfg_m, cfg_t, 1, log=lambda *_: None)
    with torch.no_grad():
        xa = torch.from_numpy(X[:50])
        assert not torch.allclose(a(xa)[0], b(xa)[0])
