#!/bin/bash
# Run retraining pipeline (Section 12)

set -e
cd "$(dirname "$0")/.."
python -m src.retrain
