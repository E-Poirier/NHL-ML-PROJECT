"""
Data validation layer (Section 4).

Schema checks, duplicate detection, DNP logic.
Validated data goes to data/processed/.
"""

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from pydantic import BaseModel, Field, field_validator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Schemas ---


class ProcessedPlayerGame(BaseModel):
    """Validated record: one player-game where the player played (DNP excluded)."""

    game_id: int
    player_id: int
    team_id: int
    opponent_team_id: int
    game_date: str
    home_away: str
    season: int
    goals: int = 0
    assists: int = 0
    points: int = 0
    shots: int = 0
    time_on_ice: str = ""
    time_on_ice_minutes: float = 0.0
    plus_minus: int = 0
    position: str = ""

    @field_validator("home_away")
    @classmethod
    def home_away_valid(cls, v: str) -> str:
        if v not in ("home", "away"):
            raise ValueError("home_away must be 'home' or 'away'")
        return v

    @field_validator("game_date")
    @classmethod
    def date_format(cls, v: str) -> str:
        if not v or len(v) != 10 or v[4] != "-" or v[7] != "-":
            raise ValueError("game_date must be YYYY-MM-DD")
        return v


def _toi_to_minutes(toi: str) -> float:
    """Parse TOI string 'MM:SS' to float minutes."""
    if not toi:
        return 0.0
    parts = toi.strip().split(":")
    if len(parts) != 2:
        return 0.0
    try:
        m, s = int(parts[0]), int(parts[1])
        return m + s / 60.0
    except ValueError:
        return 0.0


def _played(toi: str | None) -> bool:
    """Player played if TOI present and not '0:00'."""
    if not toi:
        return False
    return _toi_to_minutes(toi) > 0


def _extract_player_records(
    boxscore: dict[str, Any], game_id: int, game_date: str, season: int
) -> list[dict[str, Any]]:
    """Extract (player_id, game_id) records for players who played. DNP excluded."""
    records: list[dict[str, Any]] = []
    away_team_id = boxscore.get("awayTeam", {}).get("id")
    home_team_id = boxscore.get("homeTeam", {}).get("id")

    if not away_team_id or not home_team_id:
        return records

    pbg = boxscore.get("playerByGameStats", {})

    def add_players(team_players: list, team_id: int, opponent_id: int, home_away: str) -> None:
        for p in team_players or []:
            toi = p.get("toi")
            if not _played(toi):
                continue
            goals = p.get("goals", 0) or 0
            assists = p.get("assists", 0) or 0
            points = p.get("points", goals + assists) or (goals + assists)
            records.append({
                "game_id": game_id,
                "player_id": p.get("playerId"),
                "team_id": team_id,
                "opponent_team_id": opponent_id,
                "game_date": game_date,
                "home_away": home_away,
                "season": season,
                "goals": goals,
                "assists": assists,
                "points": points,
                "shots": p.get("sog", 0) or 0,
                "time_on_ice": toi,
                "time_on_ice_minutes": _toi_to_minutes(toi),
                "plus_minus": p.get("plusMinus", 0) or 0,
                "position": p.get("position", ""),
            })

    away_pbg = pbg.get("awayTeam", {})
    home_pbg = pbg.get("homeTeam", {})

    for plist in (away_pbg.get("forwards", []), away_pbg.get("defensemen", []), away_pbg.get("goalies", [])):
        add_players(plist, away_team_id, home_team_id, "away")
    for plist in (home_pbg.get("forwards", []), home_pbg.get("defensemen", []), home_pbg.get("goalies", [])):
        add_players(plist, home_team_id, away_team_id, "home")

    return records


def _validate_boxscore_schema(boxscore: dict[str, Any]) -> list[str]:
    """Validate boxscore has required fields. Returns list of error messages."""
    errors: list[str] = []
    if "id" not in boxscore:
        errors.append("Missing 'id' (game_id)")
    if "gameDate" not in boxscore:
        errors.append("Missing 'gameDate'")
    if "playerByGameStats" not in boxscore:
        errors.append("Missing 'playerByGameStats'")
    at = boxscore.get("awayTeam", {})
    ht = boxscore.get("homeTeam", {})
    if not isinstance(at.get("id"), (int, float)):
        errors.append("Missing or invalid awayTeam.id")
    if not isinstance(ht.get("id"), (int, float)):
        errors.append("Missing or invalid homeTeam.id")
    return errors


def _load_config() -> dict:
    """Load config from project root."""
    root = Path(__file__).resolve().parent.parent
    config_path = root / "config" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def validate_raw_data(raw_path: str | Path) -> tuple[bool, list[str]]:
    """
    Validate raw ingestion output.
    Returns (success, list of error/warning messages).
    """
    raw_path = Path(raw_path)
    errors: list[str] = []

    game_ids_file = raw_path / "game_ids.json"
    if not game_ids_file.exists():
        return False, ["game_ids.json not found"]

    with open(game_ids_file) as f:
        meta = json.load(f)
    game_ids = meta.get("game_ids", [])
    if not game_ids:
        return False, ["No game_ids in manifest"]

    box_dir = raw_path / "boxscores"
    if not box_dir.exists():
        return False, ["boxscores/ directory not found"]

    schema_failures = 0
    for i, gid in enumerate(game_ids[:100]):  # Sample first 100 for schema check
        box_path = box_dir / f"{gid}.json"
        if not box_path.exists():
            errors.append(f"Missing boxscore {gid}.json")
            schema_failures += 1
            continue
        try:
            with open(box_path) as f:
                bs = json.load(f)
            errs = _validate_boxscore_schema(bs)
            if errs:
                errors.extend([f"{gid}: {e}" for e in errs])
                schema_failures += 1
        except (json.JSONDecodeError, TypeError) as e:
            errors.append(f"{gid}: Parse error - {e}")
            schema_failures += 1

    if schema_failures > 0:
        return False, errors[:20]  # Cap errors reported

    return True, []


def validate_and_process(raw_path: str | Path | None = None, output_path: str | Path | None = None) -> bool:
    """
    Validate raw data and write to processed if valid.
    On failure: log error, do not write to data/processed/.
    Returns True if successful.
    """
    config = _load_config()
    root = Path(__file__).resolve().parent.parent

    if raw_path is None:
        raw_base = root / config["paths"]["raw_data"]
        runs = sorted([d for d in raw_base.iterdir() if d.is_dir() and d.name != ".gitkeep"], reverse=True)
        if not runs:
            logger.error("No ingestion runs found in %s", raw_base)
            return False
        raw_path = runs[0]
    else:
        raw_path = Path(raw_path)

    if output_path is None:
        output_path = root / config["paths"]["processed_data"]
    else:
        output_path = Path(output_path)

    # Quick validation
    ok, errs = validate_raw_data(raw_path)
    if not ok:
        logger.error("Validation failed: %s", errs)
        return False

    logger.info("Processing %s -> %s", raw_path, output_path)

    game_ids_file = raw_path / "game_ids.json"
    with open(game_ids_file) as f:
        meta = json.load(f)
    game_ids = meta["game_ids"]
    box_dir = raw_path / "boxscores"

    all_records: list[dict[str, Any]] = []
    failed_games: list[str] = []

    for gid in game_ids:
        box_path = box_dir / f"{gid}.json"
        if not box_path.exists():
            failed_games.append(gid)
            continue
        try:
            with open(box_path) as f:
                bs = json.load(f)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning("Failed to load %s: %s", gid, e)
            failed_games.append(gid)
            continue

        errs = _validate_boxscore_schema(bs)
        if errs:
            logger.warning("Schema error %s: %s", gid, errs)
            failed_games.append(gid)
            continue

        game_id = bs.get("id")
        if isinstance(game_id, float):
            game_id = int(game_id)
        game_date = bs.get("gameDate", "")
        season = bs.get("season", 0)
        if isinstance(season, float):
            season = int(season)

        records = _extract_player_records(bs, game_id, game_date, season)
        for r in records:
            if r.get("player_id") is None:
                continue
            try:
                ProcessedPlayerGame.model_validate(r)
                all_records.append(r)
            except Exception as e:
                logger.debug("Invalid record for game %s: %s", gid, e)

    if not all_records:
        logger.error("No valid records extracted")
        return False

    df = pd.DataFrame(all_records)

    # Duplicate check: (player_id, game_id)
    dupes = df.duplicated(subset=["player_id", "game_id"], keep="first")
    n_dupes = dupes.sum()
    if n_dupes > 0:
        logger.warning("Found %d duplicate (player_id, game_id) - deduplicating (keep first)", n_dupes)
        df = df[~dupes]

    output_path.mkdir(parents=True, exist_ok=True)
    run_id = raw_path.name
    out_file = output_path / f"player_games_{run_id}.parquet"
    df.to_parquet(out_file, index=False)

    logger.info("Wrote %d records to %s", len(df), out_file)
    if failed_games:
        failed_file = raw_path.parent / "failed" / "validation_failed_games.json"
        failed_file.parent.mkdir(parents=True, exist_ok=True)
        with open(failed_file, "w") as f:
            json.dump({"failed": failed_games[:500]}, f, indent=2)
        logger.info("Logged %d failed games to %s", min(len(failed_games), 500), failed_file)

    return True


if __name__ == "__main__":
    success = validate_and_process()
    exit(0 if success else 1)
