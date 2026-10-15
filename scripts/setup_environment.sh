#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
CONDA_EXE="${CONDA_EXE:-/usr/local/anaconda3/bin/conda}"
if [[ ! -x .conda/bin/python ]]; then
  "$CONDA_EXE" create --prefix "$PWD/.conda" python=3.11 pip -y
fi
.conda/bin/python -m pip install torch==2.8.0+cpu --index-url https://download.pytorch.org/whl/cpu
.conda/bin/python -m pip install -e '.[dev,ticvla]'
.conda/bin/python -m pip check
mkdir -p docs/locks
.conda/bin/python -m pip freeze --exclude-editable > docs/locks/python-linux-cpu.txt
"$CONDA_EXE" list --prefix "$PWD/.conda" --explicit > docs/locks/conda-linux-64.txt
