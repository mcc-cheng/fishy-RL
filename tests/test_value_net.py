"""The value network: shapes, range, saving and loading, and that it can learn."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fish.env import FishEnv  # noqa: E402
from fish.features import FEATURE_SIZE, encode  # noqa: E402
from fish.value_net import ValueNet, load, predict, save  # noqa: E402


def test_output_shape_and_range():
    torch.manual_seed(0)
    net = ValueNet(hidden=32, layers=2)
    env = FishEnv()
    view = env.reset(seed=0)
    x = encode(view.know, view.player)
    one = predict(net, x)
    assert np.ndim(one) == 0 and -1.0 < float(one) < 1.0
    batch = predict(net, np.stack([x, x, np.zeros_like(x)]))
    assert batch.shape == (3,) and np.isclose(batch[0], batch[1]) and np.isclose(batch[0], one)
    assert predict(net, np.zeros((0, FEATURE_SIZE), dtype=np.float32)).shape == (0,)
    assert net.training  # predict leaves the network in training mode


def test_save_and_load_round_trip(tmp_path):
    torch.manual_seed(1)
    net = ValueNet(hidden=16, layers=1)
    path = str(tmp_path / "ckpt" / "net.pt")
    save(net, path, games=42, note="test")
    back, info = load(path)
    assert info == {"games": 42, "note": "test"}
    assert back.hidden == 16 and back.layers == 1
    x = np.random.default_rng(0).random((5, FEATURE_SIZE), dtype=np.float32)
    assert np.allclose(predict(net, x), predict(back, x))


def test_it_can_fit_a_simple_target():
    """The first feature decides the answer; a few hundred steps should learn that."""
    torch.manual_seed(2)
    rng = np.random.default_rng(2)
    x = rng.random((256, FEATURE_SIZE), dtype=np.float32)
    y = np.where(x[:, 0] > 0.5, 0.8, -0.8).astype(np.float32)
    net = ValueNet(hidden=32, layers=1)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    xt, yt = torch.as_tensor(x), torch.as_tensor(y)
    first = None
    for _ in range(300):
        loss = torch.mean((net(xt) - yt) ** 2)
        first = float(loss) if first is None else first
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss) < 0.1 * first
