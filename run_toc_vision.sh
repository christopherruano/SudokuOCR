#!/usr/bin/env bash
# Gemini TOC vision read. Uses ONE Harvard key (KI selects which; default last) so it can run
# alongside the OCR without grabbing a busy chunk's key. Gemini-only.
set -uo pipefail
cd "$(dirname "$0")"
set -a; source .env 2>/dev/null; set +a
unset OPENAI_API_KEY; unset ANTHROPIC_API_KEY
IFS=',' read -ra KEYS <<< "$HARVARD_GEMINI_API_KEYS"
KI="${KI:-$(( ${#KEYS[@]} - 1 ))}"
export HARVARD_GEMINI_API_KEY="${KEYS[$KI]}"
export GEMINI_MIN_INTERVAL="${GEMINI_MIN_INTERVAL:-1.2}"
export PYTHONUNBUFFERED=1
python3 toc_vision.py "$@"
