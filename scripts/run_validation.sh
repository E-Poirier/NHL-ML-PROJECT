#!/bin/bash
# Run data validation (Section 4)

set -e
cd "$(dirname "$0")/.."
python -m src.validation
