#!/bin/bash
# Launch the Census Explorer web app.
#   ./run_explorer.sh    then open http://127.0.0.1:5000
# Requires: python3 with flask (pip install flask). Reads census_explorer.sqlite in this folder.
set -euo pipefail
cd "$(dirname "$0")/../../.."   # repo root (where census_explorer.py lives)
exec python3 census_explorer.py
