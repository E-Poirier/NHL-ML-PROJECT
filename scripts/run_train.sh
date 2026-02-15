#!/bin/bash
# Run model training (Section 7)

set -e
cd "$(dirname "$0")/.."
python -m src.train
