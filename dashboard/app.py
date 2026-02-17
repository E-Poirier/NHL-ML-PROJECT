"""
Simple dashboard for NHL Point Prediction API.
Calls /predict, /health, and /sample endpoints.
"""

import os
import streamlit as st
import requests

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")


def main():
    st.set_page_config(page_title="NHL Point Prediction", page_icon="🏒", layout="centered")
    st.title("🏒 NHL Point Prediction")
    st.caption("Predict probability a player records ≥1 point in a game (given they play).")

    # Model version from health
    try:
        health = requests.get(f"{API_URL}/health", timeout=3)
        if health.status_code == 200:
            ver = health.json().get("model_version", "?")
            st.success(f"API connected · Model: **{ver}**")
        else:
            st.warning("API returned non-200. Check if the server is running.")
    except requests.exceptions.RequestException:
        st.error("Cannot reach API. Start it with: `uvicorn api.main:app --reload` or `docker compose up`.")
        return

    # Browse: fetch sample and allow "Use" to populate form
    default_player = st.session_state.get("player_id", 8464989)
    default_game = st.session_state.get("game_id", 2018020090)

    with st.expander("Browse sample player-games", expanded=False):
        try:
            resp = requests.get(f"{API_URL}/sample?limit=30", timeout=5)
            if resp.status_code == 200:
                samples = resp.json()
                if samples:
                    cols = st.columns([2, 2, 1])
                    cols[0].write("**Player ID**")
                    cols[1].write("**Game ID**")
                    cols[2].write("")
                    for row in samples:
                        c0, c1, c2 = st.columns([2, 2, 1])
                        c0.write(row["player_id"])
                        c1.write(row["game_id"])
                        if c2.button("Use", key=f"use_{row['player_id']}_{row['game_id']}"):
                            st.session_state["player_id"] = row["player_id"]
                            st.session_state["game_id"] = row["game_id"]
                            st.rerun()
                else:
                    st.caption("No samples available.")
            else:
                st.caption("Could not load samples.")
        except requests.exceptions.RequestException:
            st.caption("Could not reach /sample endpoint.")

    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            player_id = st.number_input("Player ID", min_value=1, value=default_player, step=1)
        with col2:
            game_id = st.number_input("Game ID", min_value=1, value=default_game, step=1)
        submitted = st.form_submit_button("Predict")

    if submitted:
        with st.spinner("Predicting..."):
            try:
                resp = requests.post(
                    f"{API_URL}/predict",
                    json={"player_id": int(player_id), "game_id": int(game_id)},
                    timeout=5,
                )
                data = resp.json()
            except requests.exceptions.RequestException as e:
                st.error(f"Request failed: {e}")
                return

        if resp.status_code == 200:
            prob = data["prediction_probability"]
            binary = data["binary_prediction"]
            st.metric("Probability (≥1 point)", f"{prob:.1%}")
            st.write(f"Binary prediction: **{'Yes' if binary else 'No'}** (threshold applied)")
        elif resp.status_code == 404:
            st.warning("Player-game not found. Player may have insufficient history or the game is not in the feature table.")
        else:
            st.error(f"Error {resp.status_code}: {data.get('detail', 'Unknown')}")

    st.divider()
    st.caption("IDs from 2018–24 seasons. Start API: `uvicorn api.main:app` or `docker compose up`.")


if __name__ == "__main__":
    main()
