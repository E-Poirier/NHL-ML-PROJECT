"""
Prediction API (Section 10).

FastAPI endpoint: POST /predict
Returns P(≥1 point | player plays) for given player_id, game_id.
"""

from fastapi import FastAPI
from pydantic import BaseModel


class PredictRequest(BaseModel):
    player_id: int
    game_id: int


app = FastAPI(
    title="NHL Point Prediction API",
    description="Predict probability a player records ≥1 point in a game (given they play).",
)


@app.get("/health")
def health():
    """Health check."""
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictRequest):
    """
    Predict P(≥1 point | player plays).
    TODO: Load model, resolve features, return probability.
    """
    # TODO: Implement
    return {
        "player_id": request.player_id,
        "game_id": request.game_id,
        "prediction_probability": None,
        "model_version": None,
        "binary_prediction": None,
    }
