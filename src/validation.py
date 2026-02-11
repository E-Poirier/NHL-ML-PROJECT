"""
Data validation layer (Section 4).

Schema checks, duplicate detection, DNP logic.
Validated data goes to data/processed/.
"""


def validate_raw_data(raw_path: str) -> bool:
    """Validate raw ingestion output. Returns True if valid."""
    # TODO: Implement Pydantic schemas, duplicate checks
    raise NotImplementedError("Validation not yet implemented.")


def validate_and_process(raw_path: str, output_path: str) -> bool:
    """Validate raw data and write to processed if valid."""
    # TODO: Implement
    raise NotImplementedError("Validation not yet implemented.")
