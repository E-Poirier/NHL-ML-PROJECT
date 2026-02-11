"""
Feature engineering pipeline (Section 5).

Point-in-time features only. Opponent and team stats computed as of
end of day before game (no leakage).
"""


def build_features(processed_path: str, output_path: str) -> None:
    """Build features for all player-games in processed data."""
    # TODO: Implement rolling stats, opponent features, game context
    raise NotImplementedError("Feature engineering not yet implemented.")
