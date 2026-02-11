"""
Demo script: call /predict endpoint.

Usage:
  uvicorn api.main:app --reload  # in one terminal
  python scripts/demo_predict.py  # in another
"""

import requests

API_URL = "http://127.0.0.1:8000"


def main():
    # Example: Connor McDavid, game 2023020123
    resp = requests.post(
        f"{API_URL}/predict",
        json={"player_id": 8478402, "game_id": 2023020123},
    )
    print(resp.status_code, resp.json())


if __name__ == "__main__":
    main()
