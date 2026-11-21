"""The training script, run for a handful of tiny games from start to finish."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]

# small and fast: short games, few imagined deals, one checkpoint at the end
TINY = ["--particles", "16", "--max-asks", "3", "--heuristic-particles", "20",
        "--max-actions", "30", "--batch", "16", "--updates", "2", "--save-every", "3"]


def train(*options: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "scripts/train.py", *TINY, *options],
                          cwd=ROOT, capture_output=True, text=True, timeout=900)


def test_training_runs_saves_and_resumes(tmp_path):
    out = tmp_path / "net.pt"
    log = tmp_path / "net_log.csv"
    # one warm-start game, then two more with the heuristic as opponent half the time,
    # and a check against both baselines at the end
    run = train("--games", "3", "--warmup", "1", "--heuristic-share", "0.5",
                "--eval-games", "2", "--out", str(out))
    assert run.returncode == 0, run.stderr
    assert out.exists() and "3 games" in run.stdout
    rows = log.read_text().strip().splitlines()
    assert len(rows) == 2 and rows[0].startswith("games,loss") and rows[1].startswith("3,")
    # continuing adds to the same log
    run = train("--games", "4", "--warmup", "1", "--eval-games", "0", "--out", str(out), "--resume")
    assert run.returncode == 0, run.stderr
    assert "resumed" in run.stdout
    rows = log.read_text().strip().splitlines()
    assert len(rows) == 3 and rows[2].startswith("4,")
    # nothing left to do
    run = train("--games", "4", "--out", str(out), "--resume")
    assert run.returncode == 0 and "already at 4 games" in run.stdout
