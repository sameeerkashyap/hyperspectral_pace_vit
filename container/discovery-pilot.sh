#!/bin/sh
set -eu
mkdir -p /app/src /results/data
cp /source/*.py /app/src/
cp /source/pyproject.toml /app/
cd /app
python -m pip install --no-cache-dir uv==0.9.9
uv venv /runtime/venv
export VIRTUAL_ENV=/runtime/venv
export PATH="$VIRTUAL_ENV/bin:$PATH"
uv pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
uv pip install '/app[discovery]' --constraint /source/constraints.txt
echo 'Waiting for the existing Earthdata file to be copied to the results volume.'
while [ ! -f /results/data/pilot-input.ready ]; do sleep 2; done
python -c 'from pathlib import Path; from src.runner import sha256; p=Path("/results/data/PACE_OCI.20240305T000858.L2.OC_AOP.V3_2.nc"); assert sha256(p)=="b8013c3b16d73f1cbf8121745dc30c2360d5c095124928e520df6d0470c10d1e", "Input checksum mismatch"'
python -c 'import torch; assert torch.cuda.is_available(), "This pilot requires a GPU"; print(torch.cuda.get_device_name(), flush=True)'
python -m src.runner --plan /source/discovery-pilot.json
python -m src.discovery_compare /results/discovery-pilot-v1/*/discovery --output /results/discovery-pilot-v1/comparison.json
