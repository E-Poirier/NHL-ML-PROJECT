#!/bin/bash
# Run Streamlit dashboard (API must be running on 8000, or set API_URL)
set -e
cd "$(dirname "$0")/.."
streamlit run dashboard/app.py
