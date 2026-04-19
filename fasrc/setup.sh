#!/usr/bin/env bash
# One-time setup for SudokuOCR on Harvard FASRC.
#
# Usage:
#   ssh <fasrc_username>@login.rc.fas.harvard.edu
#   cd /n/holyscratch01/<lab_name>/<username>/SudokuOCR
#   bash fasrc/setup.sh
#
# Prerequisites:
#   - Transfer the repo + image data to FASRC (see bottom of this file)

set -euo pipefail

echo "=== SudokuOCR FASRC Setup ==="

# 1. Load system Python (FASRC module system)
module load python/3.12.5-fasrc01 2>/dev/null || module load python/3.10.13-fasrc01 || {
    echo "ERROR: Could not load a Python module. Run 'module avail python' to see options."
    exit 1
}

# 2. Create virtual environment
VENV_DIR="$HOME/.envs/sudokuocr"
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at $VENV_DIR"
else
    echo "Creating virtual environment at $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# 3. Install dependencies
pip install --upgrade pip
pip install -r fasrc/requirements.txt

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Create .env with your API keys (see .env.example)"
echo "  2. Test with:  source $VENV_DIR/bin/activate && python3 batch_production.py age_tables/ --dry-run"
echo "  3. Submit jobs: sbatch fasrc/submit.sh"
echo ""
echo "--- Data transfer (if not done yet) ---"
echo "From your local machine:"
echo "  rsync -avP --progress age_tables/ <username>@login.rc.fas.harvard.edu:/n/holyscratch01/<lab>//<username>/SudokuOCR/age_tables/"
echo "Or use Globus for large transfers: https://docs.rc.fas.harvard.edu/kb/globus-file-transfer/"
