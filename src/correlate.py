"""Per-band Rrs correlation with log10 concentration in a matched scene."""

import argparse
import json
from pathlib import Path

import h5netcdf
import numpy as np

from src.data import check_pair, correlation, quality_mask, read_valid


def analyze(aop_path, bgc_path, stride=4):
    if stride < 1:
        raise ValueError("Stride must be positive.")
    selection = (slice(None, None, stride), slice(None, None, stride))
    with h5netcdf.File(aop_path) as aop, h5netcdf.File(bgc_path) as bgc:
        check_pair(aop, bgc, selection)
        a, b = aop.groups["geophysical_data"].variables, bgc.groups["geophysical_data"].variables
        good = quality_mask(a["l2_flags"], selection) & quality_mask(b["l2_flags"], selection)
        targets = {n: np.where(good, read_valid(b[n], selection), np.nan)
                   for n in ("chlor_a", "carbon_phyto")}
        rrs = a["Rrs"]
        # shortcut: last-axis wavelength layout only, extend after inspecting a different schema.
        if len(rrs.shape) != 3 or rrs.shape[:2] != b["chlor_a"].shape:
            raise ValueError("Expected Rrs in scan, pixel, wavelength order.")
        spectral = aop.groups["sensor_band_parameters"].variables
        wavelengths = read_valid(spectral["wavelength_3d"])
        if wavelengths.shape != (rrs.shape[-1],) or not np.isfinite(wavelengths).all():
            raise ValueError("Rrs wavelength metadata does not match the spectral axis.")
        rows = []
        for index, wavelength in enumerate(wavelengths):
            reflectance = read_valid(rrs, selection + (index,))
            rows.append({"wavelength_nm": float(wavelength),
                         **{n: correlation(reflectance, y) for n, y in targets.items()}})
    return {"aop": str(aop_path), "bgc": str(bgc_path), "stride": stride,
            "note": "Rrs versus log10 satellite-derived concentration; not independent validation.",
            "bands": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aop")
    parser.add_argument("bgc")
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--output", default="data/correlations.json")
    args = parser.parse_args()
    result = analyze(args.aop, args.bgc, args.stride)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
