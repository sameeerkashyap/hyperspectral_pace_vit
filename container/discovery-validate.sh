#!/bin/sh
set -eu
python -m pip install --no-cache-dir 'numpy>=1.26' 'h5netcdf>=1.3' 'h5py>=3.10'
mkdir -p /app/src /runtime/data
cp /source/*.py /app/src/
cp /results/data/discovery-v2/*.nc /runtime/data/
cd /app
python -c 'import json; from pathlib import Path; from src.discovery_data import load_aop_scenes; p=Path("/results/discovery-corpus-v3"); p.mkdir(exist_ok=False); manifest=p/"scenes.json"; manifest.write_text(Path("/source/discovery-scenes-v1.json").read_text()); scenes,grid=load_aop_scenes(manifest,"/runtime/data"); inventory={"bands":grid.tolist(),"scenes":[{k:v for k,v in s.items() if k not in ("x","uncertainty","pixels","locations","audit")} | {"chips":len(s["x"])} for s in scenes]}; (p/"inventory.json").write_text(json.dumps(inventory,indent=2)+"\n"); (p/"ready").touch(); print(json.dumps({"validation":"complete","scenes":len(scenes)}),flush=True)'
