#!/bin/bash
# Run feature engineering (Section 5)

set -e
cd "$(dirname "$0")/.."
python -m src.feature_engineering
