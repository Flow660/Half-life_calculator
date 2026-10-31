#!/usr/bin/env bash
# Launch the SILK Half-life Calculator (macOS / Linux).
set -e
cd "$(dirname "$0")"

# create a local virtual environment on first run (optional but recommended)
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

exec streamlit run app.py
