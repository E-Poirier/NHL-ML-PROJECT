"""
Demo script: call /predict endpoint.

Usage:
  uvicorn api.main:app --reload  # in one terminal
  python scripts/demo_predict.py  # in another
"""

import requests

API_URL = "http://127.0.0.1:8000"


def main():
    # Use a (player_id, game_id) from the feature table (2018-2024 seasons)
    resp = requests.post(
        f"{API_URL}/predict",
        json={"player_id": 8464989, "game_id": 2018020090},
    )
    print(resp.status_code, resp.json())
    if resp.status_code == 404:
        print("(Player-game not in feature table - try another from 2018-2024 seasons)")


if __name__ == "__main__":
    main()
