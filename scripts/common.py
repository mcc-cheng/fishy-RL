"""Shared helpers for the experiment scripts: agent specs, job running, plot style."""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fish.arena import Spec, run_jobs  # noqa: E402

SPECS: dict[str, Spec] = {
    "random": ("random", {}),
    "rule": ("rule", {}),
    "greedy": ("greedy", {}),
    "rollout": ("rollout", {}),
    "ismcts": ("ismcts", {}),
}

# Categorical palette (fixed order), ink and surface colours.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
DIVERGING = ["#eb6834", "#f0efec", "#2a78d6"]  # negative, neutral, positive


def base_parser(desc: str, games: int) -> argparse.ArgumentParser:
    """Argument parser with the options every experiment shares."""
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--games", type=int, default=games, help="games per match-up")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--out", default="results", help="output directory")
    return p


def run(jobs: list[tuple], workers: int) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    """Run arena jobs with a progress line on stderr."""
    t0 = time.time()

    def progress(done: int, total: int) -> None:
        if done % max(1, total // 50) == 0 or done == total:
            print(f"\r  {done}/{total} games  {time.time() - t0:5.0f}s", end="", file=sys.stderr)

    res = run_jobs(jobs, workers, progress)
    print(file=sys.stderr)
    return res


def new_axes(ncols: int = 1, width: float = 6.4, height: float = 4.2):
    """Figure + axes with the shared recessive style."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, ncols, figsize=(width * ncols, height), facecolor=SURFACE)
    axes = [axes] if ncols == 1 else list(axes)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=9, length=0)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    return fig, axes


def label(ax, title: str, xlabel: str, ylabel: str) -> None:
    """Title and axis labels in text ink."""
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def legend(ax, **kw) -> None:
    """Frameless legend in secondary ink."""
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, **kw)


def save(fig, path: str) -> None:
    """Save a figure and report where."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    print(f"wrote {path}")
