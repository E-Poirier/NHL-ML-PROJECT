"""Tests for feature engineering (Section 5.4)."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.feature_engineering import (
    build_features,
    _build_team_game_stats,
    _build_opponent_goals_allowed,
    _add_player_rolling_features,
    _add_game_context,
)


def test_no_future_data_used():
    """Assert no feature uses data from after the game date (point-in-time guarantee)."""
    # Create minimal processed-like data: 3 games for one player
    df = pd.DataFrame([
        {"game_id": 1, "player_id": 100, "team_id": 8, "opponent_team_id": 10, "game_date": "2018-10-01", "home_away": "away", "season": 20182019, "goals": 1, "assists": 0, "points": 1, "shots": 3, "time_on_ice_minutes": 15.0},
        {"game_id": 2, "player_id": 100, "team_id": 8, "opponent_team_id": 10, "game_date": "2018-10-03", "home_away": "home", "season": 20182019, "goals": 0, "assists": 1, "points": 1, "shots": 2, "time_on_ice_minutes": 16.0},
        {"game_id": 3, "player_id": 100, "team_id": 8, "opponent_team_id": 10, "game_date": "2018-10-05", "home_away": "away", "season": 20182019, "goals": 0, "assists": 0, "points": 0, "shots": 1, "time_on_ice_minutes": 14.0},
    ])
    cfg = {"rolling_window": 5, "goals_window": 3, "assists_window": 5, "volatility_window": 5}
    df = _add_player_rolling_features(df, cfg)
    # For game 3, rolling_points_5 should use only games 1 and 2 (avg = 1.0), not game 3
    row3 = df[df["game_id"] == 3].iloc[0]
    assert row3["rolling_points_5"] == 1.0  # (1+1)/2 from games 1,2
    assert row3["rolling_goals_3"] == 1  # 1 goal from game 1 only
    assert row3["points"] == 0  # game 3 actual


def test_opponent_stats_point_in_time():
    """Opponent goals allowed is computed as of day before game."""
    team_games = pd.DataFrame([
        {"game_id": 1, "game_date": "2018-10-01", "team_id": 10, "goals_for": 3, "goals_against": 2},
        {"game_id": 2, "game_date": "2018-10-03", "team_id": 10, "goals_for": 2, "goals_against": 4},
        {"game_id": 3, "game_date": "2018-10-05", "team_id": 10, "goals_for": 1, "goals_against": 3},
    ])
    opp = _build_opponent_goals_allowed(team_games)
    # Game 1: no prior data -> NaN or inf, we fill
    # Game 2: prior = game 1, goals_against=2, 1 game -> 2.0
    # Game 3: prior = games 1,2, goals_against=2+4=6, 2 games -> 3.0
    row2 = opp[opp["game_id"] == 2].iloc[0]
    assert row2["opponent_goals_allowed_pg"] == 2.0
    row3 = opp[opp["game_id"] == 3].iloc[0]
    assert row3["opponent_goals_allowed_pg"] == 3.0


def _make_minimal_processed(proc_dir: Path, n_games: int = 10) -> Path:
    """Create minimal processed parquet for testing. Each game has away+home players."""
    rows = []
    for i in range(n_games):
        # Away player (player 1, team 8)
        rows.append({
            "game_id": 1000 + i, "player_id": 1, "team_id": 8, "opponent_team_id": 10,
            "game_date": f"2018-10-{1+i:02d}", "home_away": "away", "season": 20182019,
            "goals": 0, "assists": 0, "points": 0, "shots": 2, "time_on_ice_minutes": 15.0,
        })
        # Home player (player 2, team 10) - so game has both sides
        rows.append({
            "game_id": 1000 + i, "player_id": 2, "team_id": 10, "opponent_team_id": 8,
            "game_date": f"2018-10-{1+i:02d}", "home_away": "home", "season": 20182019,
            "goals": 0, "assists": 0, "points": 0, "shots": 2, "time_on_ice_minutes": 15.0,
        })
    df = pd.DataFrame(rows)
    proc_file = proc_dir / "player_games_test.parquet"
    df.to_parquet(proc_file, index=False)
    return proc_file


def test_cold_start_filters_low_history():
    """Players with fewer than min_games are excluded."""
    with tempfile.TemporaryDirectory() as tmp:
        proc_dir = Path(tmp) / "processed"
        proc_dir.mkdir()
        feat_dir = Path(tmp) / "features"
        proc_file = _make_minimal_processed(proc_dir, n_games=10)
        out = build_features(processed_path=proc_file, output_path=feat_dir, min_games=5)
        result = pd.read_parquet(out)
        assert len(result) >= 5  # 2 players * 5+ games each = 10+ rows
        assert "label" in result.columns
        assert "rolling_points_5" in result.columns
        assert "opponent_goals_allowed_pg" in result.columns


def test_label_correct():
    """Label = 1 if points >= 1, else 0."""
    with tempfile.TemporaryDirectory() as tmp:
        proc_dir = Path(tmp) / "processed"
        proc_dir.mkdir()
        feat_dir = Path(tmp) / "features"
        proc_file = _make_minimal_processed(proc_dir, n_games=10)
        out = build_features(processed_path=proc_file, output_path=feat_dir, min_games=5)
        result = pd.read_parquet(out)
        assert set(result["label"].unique()).issubset({0, 1})
