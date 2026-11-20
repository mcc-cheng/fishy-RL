"""The value network: how good is a position for the player's team?

Input: the feature vector from :mod:`fish.features`.
Output: one number between -1 and +1, the predicted final set margin for the
player's team divided by 9 (+1 = win all nine sets, -1 = lose all nine).

Needs PyTorch (``uv pip install torch``).
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
from torch import nn

from .cards import N_SETS
from .features import FEATURE_SIZE

#: predictions and training targets are set margins divided by this
MARGIN_SCALE = float(N_SETS)


class ValueNet(nn.Module):
    """A small fully connected network.

    ``hidden`` is the width of each hidden layer and ``layers`` how many there are.
    """

    def __init__(self, hidden: int = 256, layers: int = 2) -> None:
        super().__init__()
        self.hidden = hidden
        self.layers = layers
        blocks: list[nn.Module] = []
        width = FEATURE_SIZE
        for _ in range(layers):
            blocks += [nn.Linear(width, hidden), nn.ReLU()]
            width = hidden
        blocks.append(nn.Linear(width, 1))
        self.body = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: (batch, FEATURE_SIZE) -> (batch,) values in (-1, 1)."""
        return torch.tanh(self.body(x)).squeeze(-1)


def predict(net: ValueNet, features: np.ndarray) -> np.ndarray:
    """Values for one feature vector or a batch of them, as a NumPy array.

    No gradients are tracked; use ``net(x)`` directly when training.
    """
    x = torch.as_tensor(np.asarray(features, dtype=np.float32))
    single = x.ndim == 1
    if single:
        x = x.unsqueeze(0)
    if x.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)
    was_training = net.training
    net.eval()
    with torch.no_grad():
        out = net(x).numpy()
    net.train(was_training)
    return out[0] if single else out


def save(net: ValueNet, path: str, **info: Any) -> None:
    """Write the network to ``path``.

    ``info`` is stored alongside it (e.g. ``games=12000``); keep it to plain
    numbers, strings, lists and dicts.  The file is written under a temporary
    name first so that an interrupted save never leaves a broken checkpoint.
    """
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    tmp = path + ".tmp"
    torch.save({
        "feature_size": FEATURE_SIZE,
        "hidden": net.hidden,
        "layers": net.layers,
        "weights": net.state_dict(),
        "info": info,
    }, tmp)
    os.replace(tmp, path)


def load(path: str) -> tuple[ValueNet, dict[str, Any]]:
    """Read a network written by :func:`save`; returns it with its ``info``."""
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data["feature_size"] != FEATURE_SIZE:
        raise ValueError(
            f"checkpoint expects {data['feature_size']} features, the code produces {FEATURE_SIZE}")
    net = ValueNet(data["hidden"], data["layers"])
    net.load_state_dict(data["weights"])
    return net, dict(data["info"])
