"""Stage 5: arena."""
from __future__ import annotations

import math

import numpy as np

from fish.arena import (
    RESULT_FIELDS,
    TURN_FIELDS,
    play_game,
    play_match,
    play_match_game,
    read_csv,
    summarize,
    write_csv,
)
from fish.agents import make_agent

DET = [f for f in RESULT_FIELDS if not f.startswith("time_")]


def test_rows_have_all_fields_and_are_consistent():
    turn_rows: list[dict] = []
    row = play_match_game(("greedy", {"n_particles": 80}), ("rule", {}), 0, seed=5,
                          match="m", turn_rows=turn_rows)
    assert set(row) == set(RESULT_FIELDS)
    assert row["sets_x"] + row["sets_y"] == 9 and row["diff"] == row["sets_x"] - row["sets_y"]
    assert row["decl_x"] + row["decl_y"] == 9
    assert row["decl_correct_x"] <= row["decl_x"] and row["decl_correct_y"] <= row["decl_y"]
    assert row["length"] == len(turn_rows) and not row["truncated"]
    assert 0 <= row["brier_x"] <= 2 and math.isnan(row["brier_y"])
    assert row["time_max_x"] >= row["time_mean_x"] > 0
    assert set(turn_rows[0]) == set(TURN_FIELDS)
    assert sum(r["action"] == "declare" for r in turn_rows) == 9
    assert all(r["match"] == "m" and r["game"] == 0 for r in turn_rows)
    # sets won by X = its correct declarations + the opponents' wrong ones
    assert row["sets_x"] == row["decl_correct_x"] + row["decl_y"] - row["decl_correct_y"]


def test_parallel_results_equal_serial_and_are_seeded(tmp_path):
    x, y = ("greedy", {"n_particles": 60}), ("rule", {})
    serial, _ = play_match(x, y, 6, seed=2, workers=1)
    par, turns = play_match(x, y, 6, seed=2, workers=2, log_turns=True)
    other, _ = play_match(x, y, 6, seed=3, workers=1)
    strip = lambda rows: [{k: (r[k] if r[k] == r[k] else None) for k in DET} for r in rows]
    assert strip(serial) == strip(par)
    assert strip(serial) != strip(other)
    assert [r["game"] for r in par] == list(range(6))
    assert [r["x_team"] for r in par] == [0, 1] * 3 and [r["deal"] for r in par] == [0, 0, 1, 1, 2, 2]
    assert len(turns) == sum(r["length"] for r in par)
    path = tmp_path / "out" / "games.csv"
    write_csv(par, str(path), RESULT_FIELDS)
    back = read_csv(str(path))
    assert len(back) == 6 and int(back[3]["sets_x"]) == par[3]["sets_x"]
    assert summarize(back)["mean_diff"] == summarize(par)["mean_diff"]


def test_summarize():
    rows = [{"deal": d, "diff": v, "length": 100, "truncated": 0, "decl_x": 4, "decl_correct_x": 3,
             "decl_y": 5, "decl_correct_y": 5, "brier_x": 0.5, "brier_y": math.nan,
             "time_mean_x": 0.1, "time_mean_y": 0.2, "time_max_x": 0.3, "time_max_y": 0.4}
            for d, v in [(0, 3), (0, 1), (1, -1), (1, 1), (2, 5), (2, -3)]]
    s = summarize(rows)
    assert s["games"] == 6 and np.isclose(s["mean_diff"], 1.0)
    per_deal = np.array([2.0, 0.0, 1.0])
    assert np.isclose(s["ci95"], 1.96 * per_deal.std(ddof=1) / np.sqrt(3))
    assert np.isclose(s["win_rate"], 4 / 6) and np.isclose(s["decl_acc_x"], 0.75)
    assert s["decl_acc_y"] == 1.0 and math.isnan(s["brier_y"]) and s["time_max_y"] == 0.4


def test_truncation_and_turn_hook():
    seen = []
    agents = [make_agent("random") for _ in range(6)]
    res = play_game(agents, np.random.default_rng(0), np.random.default_rng(0), max_actions=5,
                    on_turn=lambda t, state, view, agent, dt: seen.append((t, view.player, state.turn)))
    assert res["truncated"] and res["length"] == 5 and len(seen) == 5
    assert all(p == turn for _, p, turn in seen)


def test_mirror_match_is_roughly_fair():
    rows, _ = play_match(("rule", {}), ("rule", {"name": "rule2"}), 60, seed=1, workers=1)
    s = summarize(rows)
    assert abs(s["mean_diff"]) < 2.5 and s["truncated"] == 0
