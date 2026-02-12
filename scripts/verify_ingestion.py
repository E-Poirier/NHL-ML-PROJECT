"""
Quick verification that ingestion completed successfully.
Run after: python -m src.data_ingestion
"""

import json
from pathlib import Path


def verify(raw_path: Path) -> bool:
    """Verify ingestion output. Returns True if all checks pass."""
    ok = True

    # 1. game_ids.json exists
    game_ids_file = raw_path / "game_ids.json"
    if not game_ids_file.exists():
        print("FAIL: game_ids.json not found")
        return False

    with open(game_ids_file) as f:
        meta = json.load(f)

    game_ids = meta.get("game_ids", [])
    seasons = meta.get("seasons", [])
    print(f"Seasons: {seasons}")
    print(f"Games in manifest: {len(game_ids)}")

    # 2. Boxscores directory exists
    box_dir = raw_path / "boxscores"
    if not box_dir.exists():
        print("FAIL: boxscores/ directory not found")
        return False

    count = sum(1 for _ in box_dir.glob("*.json"))
    print(f"Boxscore files: {count}")

    if count != len(game_ids):
        print(f"WARN: Mismatch (expected {len(game_ids)}, found {count})")
        ok = False
    else:
        print("OK: Game count matches boxscore count")

    # 3. Sample boxscore has required fields
    sample_id = game_ids[0] if game_ids else None
    if sample_id:
        sample_path = box_dir / f"{sample_id}.json"
        if sample_path.exists():
            with open(sample_path) as f:
                bs = json.load(f)
            required = ["id", "gameDate", "awayTeam", "homeTeam", "playerByGameStats"]
            missing = [k for k in required if k not in bs]
            if missing:
                print(f"FAIL: Sample boxscore missing keys: {missing}")
                ok = False
            else:
                # Check player stats
                pbg = bs.get("playerByGameStats", {})
                away_fwds = pbg.get("awayTeam", {}).get("forwards", [])
                if away_fwds:
                    p = away_fwds[0]
                    has_player_fields = all(
                        k in p for k in ["playerId", "goals", "assists", "points", "toi"]
                    )
                    if has_player_fields:
                        print("OK: Sample boxscore has player stats (playerId, goals, assists, toi)")
                    else:
                        print("WARN: Sample player missing some fields")
                else:
                    print("WARN: No forwards in sample boxscore")
        else:
            print(f"FAIL: Sample boxscore {sample_id}.json not found")
            ok = False

    return ok


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    raw_base = root / "data" / "raw"

    # Find most recent ingestion run
    runs = sorted([d for d in raw_base.iterdir() if d.is_dir() and d.name != ".gitkeep"], reverse=True)
    if not runs:
        print("No ingestion runs found in data/raw/")
        exit(1)

    latest = runs[0]
    print(f"\nVerifying: data/raw/{latest.name}/\n")
    success = verify(latest)
    print("\n" + ("All checks passed." if success else "Some checks failed."))
    exit(0 if success else 1)
