"""Small matched-scene loader with explicit geographic and temporal splits."""

from datetime import datetime
import json
from pathlib import Path

import h5netcdf
import numpy as np

from src.data import check_pair, quality_mask, read_valid
from src.runner import sha256


def validate_scene_manifest(records):
    if not records or {r["split"] for r in records} != {"train", "validation", "test"}:
        raise ValueError("Supply train, validation and test scenes explicitly.")
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("Scene IDs must be unique.")
    for record in records:
        box = record["bbox"]
        if (len(box) != 4 or not np.isfinite(box).all()
                or not -180 <= box[0] < box[2] <= 180 or not -90 <= box[1] < box[3] <= 90):
            raise ValueError("Use bbox [west, south, east, north]; dateline-crossing boxes are unsupported.")
    for i, left in enumerate(records):
        for right in records[i + 1:]:
            a, b = left["bbox"], right["bbox"]
            if (left["split"] != right["split"] and a[0] < b[2] and b[0] < a[2]
                    and a[1] < b[3] and b[1] < a[3]):
                raise ValueError("Geographic boxes must not overlap across splits.")


def load_matched_scenes(manifest, data_dir, chip_size=8, limit=64, band_count=0, stride=1):
    if min(chip_size, limit, stride) < 1 or band_count < 0:
        raise ValueError("Use positive chip size, sample limit and stride, and nonnegative band count.")
    records = json.loads(Path(manifest).read_text())["scenes"]
    validate_scene_manifest(records)
    dates, scenes, reference_wavelengths, reference_version = {}, [], None, None
    selection = (slice(None, None, stride),) * 2
    for record in records:
        aop_path, bgc_path = (Path(data_dir) / record[key] for key in ("aop", "bgc"))
        with h5netcdf.File(aop_path) as aop, h5netcdf.File(bgc_path) as bgc:
            check_pair(aop, bgc, ...)
            version = str(aop.attrs["processing_version"])
            if reference_version is not None and reference_version != version:
                raise ValueError("Use the same processing version across the benchmark.")
            reference_version = version
            for key in ("time_coverage_start", "time_coverage_end"):
                day = datetime.fromisoformat(aop.attrs[key].replace("Z", "+00:00")).date().isoformat()
                if day in dates and dates[day] != record["split"]:
                    raise ValueError("Acquisition dates must not be shared across splits.")
                dates[day] = record["split"]
            spectral = aop.groups["sensor_band_parameters"].variables
            all_wavelengths = read_valid(spectral["wavelength_3d"])
            bands = np.flatnonzero((all_wavelengths >= 400) & (all_wavelengths <= 700))
            if band_count:
                if len(bands) < band_count:
                    raise ValueError("Not enough bands for the requested subset.")
                bands = bands[np.linspace(0, len(bands) - 1, band_count).astype(int)]
            wavelengths = all_wavelengths[bands].astype("float32")
            if len(wavelengths) < 2 or (np.diff(wavelengths) <= 0).any():
                raise ValueError("Need at least two strictly increasing visible wavelengths.")
            if reference_wavelengths is not None and not np.array_equal(reference_wavelengths, wavelengths):
                raise ValueError("All scenes must have the same physical wavelength grid.")
            reference_wavelengths = wavelengths
            geo, targets = aop.groups["geophysical_data"].variables, bgc.groups["geophysical_data"].variables
            if geo["Rrs"].attrs.get("units") != "sr^-1" or targets["chlor_a"].attrs.get("units") != "mg m^-3":
                raise ValueError("Expected Rrs in sr^-1 and chlor_a in mg m^-3.")
            rrs = geo["Rrs"]
            if rrs.shape != (*targets["chlor_a"].shape, len(all_wavelengths)):
                raise ValueError("Expected aligned scan, pixel, wavelength Rrs and 2D chlor_a.")
            nav = aop.groups["navigation_data"].variables
            lat, lon = (read_valid(nav[name], selection) for name in ("latitude", "longitude"))
            west, south, east, north = record["bbox"]
            good = quality_mask(geo["l2_flags"], selection)
            good &= np.isfinite(lat) & np.isfinite(lon)
            good &= (lon >= west) & (lon < east) & (lat >= south) & (lat < north)
            chlorophyll = read_valid(targets["chlor_a"], selection)
            valid = quality_mask(targets["l2_flags"], selection)
            valid &= np.isfinite(chlorophyll) & (chlorophyll > 0)
            chips, labels, masks, coordinates = [], [], [], []
            for row in range(0, good.shape[0] - chip_size + 1, chip_size):
                if len(chips) >= limit:
                    break
                for col in range(0, good.shape[1] - chip_size + 1, chip_size):
                    window = (slice(row, row + chip_size), slice(col, col + chip_size))
                    if not good[window].all() or not valid[window].any():
                        continue
                    raw_window = (slice(row * stride, (row + chip_size) * stride, stride),
                                  slice(col * stride, (col + chip_size) * stride, stride),
                                  slice(int(bands[0]), int(bands[-1]) + 1))
                    spectrum = read_valid(rrs, raw_window)[..., bands - bands[0]]
                    if not np.isfinite(spectrum).all():
                        continue
                    chips.append(spectrum.transpose(2, 0, 1))
                    labels.append(chlorophyll[window][None])
                    masks.append(valid[window][None])
                    coordinates.append([row * stride, col * stride])
                    if len(chips) >= limit:
                        break
            if not chips:
                raise ValueError(f"No usable matched chips for scene {record['id']}")
            scene = {**record, "x": np.asarray(chips, dtype="float32"),
                     "target": np.asarray(labels, dtype="float32"), "valid": np.asarray(masks),
                     "pixels": coordinates, "wavelengths_nm": wavelengths,
                     "acquisition_start": str(aop.attrs["time_coverage_start"]),
                     "processing_version": str(aop.attrs["processing_version"]),
                     "sha256": {"aop": sha256(aop_path), "bgc": sha256(bgc_path)}}
            scenes.append(scene)
    return scenes, reference_wavelengths
