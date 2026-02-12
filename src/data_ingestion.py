"""
NHL data ingestion (Section 3).

Pulls game-level and player-level stats from NHL API via nhl-api-py.
Stores raw JSON responses in data/raw/YYYY_MM_DD/.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path

import yaml
from nhlpy import NHLClient  # type: ignore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Game type: 2 = regular season
GAME_TYPE_REGULAR = 2


def _load_config() -> dict:
    """Load config from project root."""
    root = Path(__file__).resolve().parent.parent
    config_path = root / "config" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def _get_output_dir(config: dict) -> Path:
    """Get output directory: data/raw/YYYY_MM_DD/."""
    root = Path(__file__).resolve().parent.parent
    raw_path = root / config["paths"]["raw_data"]
    run_date = datetime.now().strftime("%Y_%m_%d")
    out_dir = raw_path / run_date
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _fetch_with_retry(client: NHLClient, fetch_fn, *args, max_attempts: int = 3, backoff: float = 2.0):
    """Call fetch_fn with retries and exponential backoff."""
    last_err = None
    for attempt in range(max_attempts):
        try:
            return fetch_fn(*args)
        except Exception as e:
            last_err = e
            if attempt < max_attempts - 1:
                wait = backoff ** attempt
                logger.warning("Request failed (attempt %d/%d): %s. Retrying in %.1fs", attempt + 1, max_attempts, e, wait)
                time.sleep(wait)
    raise last_err


def run_ingestion(output_dir: str | None = None, seasons: list[str] | None = None) -> None:
    """
    Pull NHL data and store raw JSON responses.

    Args:
        output_dir: Override output directory. Default: data/raw/YYYY_MM_DD/
        seasons: Override seasons to ingest. Default: from config (train + val + test).
    """
    config = _load_config()
    ing_config = config.get("ingestion", {})
    rate_delay = ing_config.get("rate_limit_delay_sec", 0.5)
    max_retries = ing_config.get("retry_attempts", 3)
    backoff = ing_config.get("retry_backoff_factor", 2.0)

    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = _get_output_dir(config)

    if seasons is None:
        train_seasons = config.get("training", {}).get("train_seasons", [])
        val_season = config.get("training", {}).get("val_season", [])
        test_season = config.get("training", {}).get("test_season", [])
        seasons = list(train_seasons) + [val_season] + [test_season]
        seasons = list(dict.fromkeys(seasons))  # dedupe, preserve order

    logger.info("Ingesting seasons %s → %s", seasons, out_dir)

    client = NHLClient(timeout=30)

    # 1. Get game IDs per season (regular season only)
    all_game_ids: set[str] = set()
    for season in seasons:
        time.sleep(rate_delay)
        game_ids = _fetch_with_retry(
            client,
            lambda s: client.helpers.game_ids_by_season(s, game_types=[GAME_TYPE_REGULAR], api_sleep_rate=rate_delay),
            season,
            max_attempts=max_retries,
            backoff=backoff,
        )
        all_game_ids.update(str(gid) for gid in game_ids)
        logger.info("Season %s: %d games", season, len(game_ids))

    all_game_ids = sorted(all_game_ids)
    logger.info("Total unique games: %d", len(all_game_ids))

    # Save game ID list for reference
    with open(out_dir / "game_ids.json", "w") as f:
        json.dump({"game_ids": all_game_ids, "seasons": seasons}, f, indent=2)

    # 2. Fetch boxscore for each game
    boxscores_dir = out_dir / "boxscores"
    boxscores_dir.mkdir(exist_ok=True)
    fetched = 0
    failed: list[str] = []

    for i, game_id in enumerate(all_game_ids):
        boxscore_path = boxscores_dir / f"{game_id}.json"
        if boxscore_path.exists():
            logger.debug("Skipping existing %s", game_id)
            fetched += 1
            continue

        time.sleep(rate_delay)
        try:
            data = _fetch_with_retry(
                client,
                client.game_center.boxscore,
                game_id,
                max_attempts=max_retries,
                backoff=backoff,
            )
            with open(boxscore_path, "w") as f:
                json.dump(data, f, indent=2)
            fetched += 1
            if (i + 1) % 100 == 0:
                logger.info("Progress: %d/%d boxscores", i + 1, len(all_game_ids))
        except Exception as e:
            logger.error("Failed to fetch boxscore %s: %s", game_id, e)
            failed.append(game_id)

    logger.info("Ingestion complete. Fetched: %d, Failed: %d", fetched, len(failed))
    if failed:
        with open(out_dir / "failed_game_ids.json", "w") as f:
            json.dump({"failed": failed}, f, indent=2)


if __name__ == "__main__":
    run_ingestion()
