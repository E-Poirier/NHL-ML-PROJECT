#!/bin/bash
# Run data ingestion (Section 3)

set -e
cd "$(dirname "$0")/.."
python -m src.data_ingestion
