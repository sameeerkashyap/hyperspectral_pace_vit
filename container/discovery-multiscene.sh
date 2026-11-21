#!/bin/sh
set -eu
mkdir -p /app/src /app/experiments
cp /source/*.py /app/src/
cp /source/pyproject.toml /app/
cp /source/discovery.json /app/experiments/
cd /app
test -f /results/discovery-prepared-v1/ready
mkdir -p /runtime/data
cp /results/discovery-prepared-v1/*.npz /runtime/data/
export DATA_DIR=/runtime/data
python -m pip install --no-cache-dir uv==0.9.9
uv venv /runtime/venv
export VIRTUAL_ENV=/runtime/venv
export PATH="$VIRTUAL_ENV/bin:$PATH"
uv pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
uv pip install '/app[discovery]' --constraint /source/constraints.txt
python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(), flush=True)'
exec python -m src.runner --plan /app/experiments/discovery.json
