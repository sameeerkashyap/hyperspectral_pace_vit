"""Download public teaching samples. No Earthdata credentials are needed."""

import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

import h5netcdf


BASE = "https://github.com/opengeos/datasets/releases/download/hypercoast/"
SAMPLES = ["PACE_OCI.20240615T182549.L2.OC_BGC.V1_0_0.NRT.nc",
           "PACE_OCI.20240730T181157.L2.OC_AOP.V2_0.NRT.nc"]


def main():
    destination = Path("data/raw")
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for name in SAMPLES:
        path = destination / name
        if not path.exists():
            temporary = path.with_suffix(".nc.part")
            try:
                with urllib.request.urlopen(BASE + name, timeout=60) as response, temporary.open("wb") as out:
                    shutil.copyfileobj(response, out)
                with h5netcdf.File(temporary):
                    pass
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        with h5netcdf.File(path):
            pass
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        records.append({"file": name, "url": BASE + name, "bytes": path.stat().st_size,
                        "sha256": digest})
    Path("data/manifest.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
