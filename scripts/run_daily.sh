#!/bin/bash
# Daily pipeline: ingestion → validation → features → monitoring
# Schedule with cron (e.g. 6 AM daily after games finish)

set -e
cd "$(dirname "$0")/.."

echo "[$(date)] Starting daily pipeline..."
python -m src.data_ingestion
python -m src.validation
python -m src.feature_engineering
python -m monitoring.monitor
echo "[$(date)] Daily pipeline complete."
