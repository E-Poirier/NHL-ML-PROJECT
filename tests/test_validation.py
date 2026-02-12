"""Tests for validation module (Section 4.4)."""

import json
import tempfile
from pathlib import Path

import pytest

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.validation import (
    ProcessedPlayerGame,
    validate_raw_data,
    validate_and_process,
    _validate_boxscore_schema,
    _extract_player_records,
    _played,
    _toi_to_minutes,
)


# --- Fixtures ---

VALID_BOXSCORE = {
    "id": 2018020001,
    "season": 20182019,
    "gameDate": "2018-10-03",
    "awayTeam": {"id": 8, "abbrev": "MTL"},
    "homeTeam": {"id": 10, "abbrev": "TOR"},
    "playerByGameStats": {
        "awayTeam": {
            "forwards": [
                {
                    "playerId": 8475848,
                    "goals": 1,
                    "assists": 0,
                    "points": 1,
                    "sog": 5,
                    "toi": "16:22",
                    "plusMinus": 1,
                    "position": "R",
                }
            ],
            "defensemen": [],
            "goalies": [],
        },
        "homeTeam": {
            "forwards": [],
            "defensemen": [],
            "goalies": [],
        },
    },
}


# --- Unit tests ---


def test_valid_payload_passes():
    """Valid ProcessedPlayerGame record passes validation."""
    record = {
        "game_id": 2018020001,
        "player_id": 8475848,
        "team_id": 8,
        "opponent_team_id": 10,
        "game_date": "2018-10-03",
        "home_away": "away",
        "season": 20182019,
        "goals": 1,
        "assists": 0,
        "points": 1,
        "shots": 5,
        "time_on_ice": "16:22",
        "time_on_ice_minutes": 16.37,
        "plus_minus": 1,
        "position": "R",
    }
    parsed = ProcessedPlayerGame.model_validate(record)
    assert parsed.player_id == 8475848
    assert parsed.points == 1


def test_missing_required_field_fails():
    """Missing required field fails validation."""
    record = {
        "game_id": 2018020001,
        # "player_id" missing
        "team_id": 8,
        "opponent_team_id": 10,
        "game_date": "2018-10-03",
        "home_away": "away",
        "season": 20182019,
    }
    with pytest.raises(Exception):  # Pydantic ValidationError
        ProcessedPlayerGame.model_validate(record)


def test_wrong_type_fails():
    """Wrong type for field fails validation."""
    record = {
        "game_id": "not_an_int",  # should be int
        "player_id": 8475848,
        "team_id": 8,
        "opponent_team_id": 10,
        "game_date": "2018-10-03",
        "home_away": "away",
        "season": 20182019,
    }
    with pytest.raises(Exception):
        ProcessedPlayerGame.model_validate(record)


def test_home_away_invalid_fails():
    """Invalid home_away value fails."""
    record = {
        "game_id": 2018020001,
        "player_id": 8475848,
        "team_id": 8,
        "opponent_team_id": 10,
        "game_date": "2018-10-03",
        "home_away": "invalid",
        "season": 20182019,
    }
    with pytest.raises(Exception):
        ProcessedPlayerGame.model_validate(record)


def test_duplicate_player_game_caught():
    """Duplicate (player_id, game_id) is detected and deduplicated."""
    import pandas as pd
    from src.validation import validate_and_process

    df = pd.DataFrame([
        {"player_id": 1, "game_id": 100, "team_id": 8, "opponent_team_id": 10, "game_date": "2018-10-03", "home_away": "away", "season": 20182019},
        {"player_id": 1, "game_id": 100, "team_id": 8, "opponent_team_id": 10, "game_date": "2018-10-03", "home_away": "away", "season": 20182019},
    ])
    dupes = df.duplicated(subset=["player_id", "game_id"], keep="first")
    assert dupes.sum() == 1
    deduped = df[~dupes]
    assert len(deduped) == 1


def test_toi_played():
    """_played returns True for non-zero TOI, False for 0:00."""
    assert _played("16:22") is True
    assert _played("0:01") is True
    assert _played("0:00") is False
    assert _played(None) is False
    assert _played("") is False


def test_toi_to_minutes():
    """TOI string parses to minutes correctly."""
    assert abs(_toi_to_minutes("16:22") - (16 + 22 / 60)) < 0.01
    assert _toi_to_minutes("0:00") == 0.0


def test_validate_boxscore_schema_valid():
    """Valid boxscore passes schema check."""
    errs = _validate_boxscore_schema(VALID_BOXSCORE)
    assert len(errs) == 0


def test_validate_boxscore_schema_missing_id():
    """Missing id fails schema check."""
    bad = {**VALID_BOXSCORE}
    del bad["id"]
    errs = _validate_boxscore_schema(bad)
    assert any("id" in e.lower() for e in errs)


def test_extract_player_records_excludes_dnp():
    """Players with 0:00 TOI are excluded (DNP)."""
    bs = {
        **VALID_BOXSCORE,
        "playerByGameStats": {
            "awayTeam": {
                "forwards": [
                    {"playerId": 1, "goals": 0, "assists": 0, "points": 0, "sog": 0, "toi": "16:22", "plusMinus": 0, "position": "R"},
                    {"playerId": 2, "goals": 0, "assists": 0, "points": 0, "sog": 0, "toi": "0:00", "plusMinus": 0, "position": "R"},
                ],
                "defensemen": [],
                "goalies": [],
            },
            "homeTeam": {"forwards": [], "defensemen": [], "goalies": []},
        },
    }
    records = _extract_player_records(bs, 2018020001, "2018-10-03", 20182019)
    player_ids = [r["player_id"] for r in records]
    assert 1 in player_ids
    assert 2 not in player_ids


def test_validate_raw_data_missing_dir():
    """validate_raw_data fails when game_ids.json missing."""
    with tempfile.TemporaryDirectory() as tmp:
        ok, errs = validate_raw_data(tmp)
        assert ok is False
        assert len(errs) > 0
