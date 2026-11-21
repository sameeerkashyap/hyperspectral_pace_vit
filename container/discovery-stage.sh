#!/bin/sh
set -eu
apt-get update -qq
apt-get install -y -qq --no-install-recommends curl ca-certificates
python -m pip install --no-cache-dir uv==0.9.9
uv venv /runtime/venv
export VIRTUAL_ENV=/runtime/venv
export PATH="$VIRTUAL_ENV/bin:$PATH"
uv pip install 'numpy>=1.26' 'h5netcdf>=1.3' 'h5py>=3.10'
mkdir -p /app/src
cp /source/*.py /app/src/
cd /app
exec python -m src.discovery_stage --candidates /source/discovery-candidates-v2.json \
  --data-dir /results/data/discovery-v2 --output /results/discovery-corpus-v2 \
  --netrc /credentials/.netrc
