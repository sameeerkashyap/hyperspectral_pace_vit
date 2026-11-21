#!/bin/sh
set -eu
python -m pip install --no-cache-dir 'numpy>=1.26' 'h5netcdf>=1.3' 'h5py>=3.10'
mkdir -p /app/src
cp /source/*.py /app/src/
cd /app
exec python -m src.discovery_prepare --scenes /source/discovery-scenes-v1.json \
  --data-dir /results/data/discovery-v2 --output /results/discovery-prepared-v1
