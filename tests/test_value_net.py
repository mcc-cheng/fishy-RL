"""The value network: shapes, range, saving and loading, and that it can learn."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fish.cards import N_SETS  # noqa: E402
from fish.features import N_FEATURES  # noqa: E402
from fish.value_net import ValueNet, load, predict, save  # noqa: E402


def test_output_shape_and_range():
    torch.manual_seed(0)
    net = ValueNet(hidden=32, layers=2)
    x = np.random.default_rng(0).random((N_SETS, N_FEATURES), dtype=np.float32)
    one = predict(net, x)  # one position: a value per set
    assert one.shape == (N_SETS,) and (np.abs(one) < 1.0).all()
    batch = predict(net, np.stack([x, x, np.zeros_like(x)]))
    assert batch.shape == (3, N_SETS)
    assert np.allclose(batch[0], batch[1]) and np.allclose(batch[0], one)
    # every set is scored by the same network: equal rows get equal values
    assert np.allclose(batch[2], batch[2, 0])
    assert predict(net, np.zeros((0, N_SETS, N_FEATURES), dtype=np.float32)).shape == (0, N_SETS)
    assert net.training  # predict leaves the network in training mode


def test_save_and_load_round_trip(tmp_path):
    torch.manual_seed(1)
    net = ValueNet(hidden=16, layers=1)
    path = str(tmp_path / "ckpt" / "net.pt")
    save(net, path, games=42, note="test")
    back, info = load(path)
    assert info == {"games": 42, "note": "test"}
    assert back.hidden == 16 and back.layers == 1
    x = np.random.default_rng(0).random((5, N_SETS, N_FEATURES), dtype=np.float32)
    assert np.allclose(predict(net, x), predict(back, x))


def test_a_checkpoint_made_for_other_inputs_is_refused(tmp_path):
    path = str(tmp_path / "net.pt")
    save(ValueNet(hidden=8, layers=1), path)
    data = torch.load(path, map_location="cpu", weights_only=True)
    data["features"] = data["features"][:-1]
    torch.save(data, path)
    with pytest.raises(ValueError):
        load(path)


def test_it_can_fit_a_simple_target():
    """The first feature decides the answer; a few hundred steps should learn that."""
    torch.manual_seed(2)
    rng = np.random.default_rng(2)
    x = rng.random((256, N_FEATURES), dtype=np.float32)
    y = np.where(x[:, 0] > 0.5, 0.8, -0.8).astype(np.float32)
    net = ValueNet(hidden=64, layers=1)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    xt, yt = torch.as_tensor(x), torch.as_tensor(y)

    def current_loss() -> float:
        with torch.no_grad():
            return float(torch.mean((net(xt) - yt) ** 2))

    first = current_loss()
    for _ in range(500):
        loss = torch.mean((net(xt) - yt) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert current_loss() < 0.5 * first
