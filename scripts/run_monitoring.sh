#!/bin/bash
# Run monitoring (Section 11): prediction dist, feature drift, daily evaluation

set -e
cd "$(dirname "$0")/.."
python -m monitoring.monitor
