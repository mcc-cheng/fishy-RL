"""The value network: who will end up with each set?

Input: one row of numbers describing one set (see :mod:`fish.features`).
Output: one number between -1 and +1 for that set: +1 means the player's team
is sure to win it, -1 that the opponents are, 0 that it could go either way.

The same small network is used for every set.  The value of a whole position
is the sets already scored plus these predictions for the sets still in play.

Needs PyTorch (``uv pip install torch``).
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
from torch import nn

from .cards import N_SETS
from .features import FEATURE_NAMES, N_FEATURES

#: move values are reported as expected set margins divided by this
MARGIN_SCALE = float(N_SETS)


class ValueNet(nn.Module):
    """A small fully connected network.

    ``hidden`` is the width of each hidden layer and ``layers`` how many there are.
    """

    def __init__(self, hidden: int = 64, layers: int = 2) -> None:
        super().__init__()
        self.hidden = hidden
        self.layers = layers
        blocks: list[nn.Module] = []
        width = N_FEATURES
        for _ in range(layers):
            blocks += [nn.Linear(width, hidden), nn.ReLU()]
            width = hidden
        blocks.append(nn.Linear(width, 1))
        self.body = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: (..., N_FEATURES) -> (...) values in (-1, 1), one per row."""
        return torch.tanh(self.body(x)).squeeze(-1)


def predict(net: ValueNet, features: np.ndarray) -> np.ndarray:
    """Values for rows of features, as a NumPy array: one value per row, so
    input of shape ``(positions, sets, N_FEATURES)`` gives ``(positions, sets)``.

    No gradients are tracked; use ``net(x)`` directly when training.
    """
    x = torch.as_tensor(np.asarray(features, dtype=np.float32))
    if x.numel() == 0:
        return np.zeros(tuple(x.shape[:-1]), dtype=np.float32)
    was_training = net.training
    net.eval()
    with torch.no_grad():
        out = net(x).numpy()
    net.train(was_training)
    return out


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
        "features": list(FEATURE_NAMES),
        "hidden": net.hidden,
        "layers": net.layers,
        "weights": net.state_dict(),
        "info": info,
    }, tmp)
    os.replace(tmp, path)


def load(path: str) -> tuple[ValueNet, dict[str, Any]]:
    """Read a network written by :func:`save`; returns it with its ``info``."""
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data.get("features") != list(FEATURE_NAMES):
        raise ValueError("this checkpoint was trained with different inputs than the code now uses")
    net = ValueNet(data["hidden"], data["layers"])
    net.load_state_dict(data["weights"])
    return net, dict(data["info"])
