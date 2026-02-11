"""
NHL data ingestion (Section 3).

Pulls game-level and player-level stats from NHL API.
Stores raw JSON responses in data/raw/YYYY_MM_DD/.
"""


def run_ingestion(output_dir: str | None = None) -> None:
    """Pull NHL data and store raw responses."""
    # TODO: Implement
    raise NotImplementedError("Data ingestion not yet implemented.")


if __name__ == "__main__":
    run_ingestion()
