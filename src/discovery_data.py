"""AOP-only, native-resolution chips with explicit scene/date/region splits."""

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re

import h5netcdf
import numpy as np

from src.data import EXCLUDE_FLAGS, quality_mask, read_valid
from src.matched_data import validate_scene_manifest
from src.runner import sha256


# Chlorophyll algorithm warnings must not select an unlabeled optical training set.
AOP_EXCLUDE_FLAGS = tuple(f for f in EXCLUDE_FLAGS if f not in ("CHLFAIL", "CHLWARN"))
AUDIT_FIELDS = ("aot_865", "angstrom", "avw", "nflh")


def pilot_partition(row, col, chip_size, block_size):
    """Leave one chip of unused pixels on each side of every spatial block."""
    if block_size < 4 * chip_size or block_size % chip_size:
        raise ValueError("Pilot blocks must be chip-aligned and at least four chips wide.")
    if any(p % block_size < chip_size or p % block_size + chip_size > block_size - chip_size
           for p in (row, col)):
        return None
    block = f'{row // block_size},{col // block_size}'
    bucket = int(hashlib.sha256(block.encode()).hexdigest()[:8], 16) % 10
    return 'train' if bucket < 6 else 'validation' if bucket < 8 else 'test'


def load_aop_scenes(manifest, data_dir, chip_size=16, limit=128):
    if min(chip_size, limit) < 1:
        raise ValueError("Chip size and scene limit must be positive.")
    specification = json.loads(Path(manifest).read_text())
    if 'prepared' in specification:
        return load_prepared_scenes(specification, data_dir, chip_size, limit)
    records = specification["scenes"]
    pilot = specification.get('scope') == 'single-scene-spatial-pilot'
    block_size = specification.get('block_size', 128)
    if pilot:
        if len(records) != 1 or records[0]['split'] != 'train' or records[0]['bbox'] != [-180, -90, 180, 90]:
            raise ValueError("A single-scene pilot needs one full-scene train source.")
        pilot_partition(0, 0, chip_size, block_size)
    else:
        if specification.get('scope') not in (None, 'multi-scene'):
            raise ValueError("Unknown discovery scope.")
        validate_scene_manifest(records)
    scenes, dates, reference, version = [], {}, None, None
    root_dir = Path(data_dir).resolve()
    for record in records:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", record["id"]):
            raise ValueError("Scene IDs must be plain names.")
        path = (root_dir / record["aop"]).resolve()
        if not path.is_relative_to(root_dir):
            raise ValueError("AOP files must be inside data-dir.")
        digest = sha256(path)
        if record.get('sha256', digest) != digest:
            raise ValueError(f"AOP checksum mismatch for {record['id']}.")
        # Keep neighboring compressed chunks cached during shuffled chip reads on CephFS.
        with h5netcdf.File(path, rdcc_nbytes=256 * 1024 ** 2, rdcc_nslots=10007) as root:
            if root.attrs.get("platform") != "PACE" or root.attrs.get("instrument") != "OCI":
                raise ValueError("Expected PACE OCI scenes.")
            current_version = str(root.attrs["processing_version"])
            if version is not None and current_version != version:
                raise ValueError("Use one processing version across scenes.")
            version = current_version
            start, end = (datetime.fromisoformat(str(root.attrs[key]).replace("Z", "+00:00"))
                          for key in ("time_coverage_start", "time_coverage_end"))
            if start.tzinfo is None or end.tzinfo is None or end < start:
                raise ValueError("Use valid UTC acquisition times.")
            start, end = start.astimezone(timezone.utc), end.astimezone(timezone.utc)
            day = start.date()
            while day <= end.date():
                if day in dates and dates[day] != record["split"]:
                    raise ValueError("Acquisition dates must not be shared across splits.")
                dates[day] = record["split"]
                day += timedelta(days=1)
            grid = read_valid(root.groups["sensor_band_parameters"].variables["wavelength_3d"])
            if grid.ndim != 1 or not np.isfinite(grid).all() or (np.diff(grid) <= 0).any():
                raise ValueError("Need a finite, increasing wavelength grid.")
            bands = np.flatnonzero((grid >= 400) & (grid <= 700))
            selected = grid[bands].astype("float32")
            if len(selected) < 2 or (reference is not None and not np.array_equal(reference, selected)):
                raise ValueError("All scenes need the same visible wavelength grid.")
            reference = selected
            geo, nav = root.groups["geophysical_data"].variables, root.groups["navigation_data"].variables
            rrs = geo["Rrs"]
            lat, lon = (read_valid(nav[name]) for name in ("latitude", "longitude"))
            if rrs.shape != (*lat.shape, len(grid)) or lat.shape != lon.shape or rrs.attrs.get("units") != "sr^-1":
                raise ValueError("Expected aligned Rrs in sr^-1, navigation, and wavelengths.")
            flags = geo["l2_flags"]
            if flags.shape != lat.shape or geo["Rrs_unc"].shape != rrs.shape:
                raise ValueError("Quality and uncertainty shapes must match reflectance.")
            good = quality_mask(flags, exclude_flags=AOP_EXCLUDE_FLAGS)
            full_flags = np.asarray(flags[:]).astype(np.uint32)
            west, south, east, north = record["bbox"]
            good &= np.isfinite(lat) & np.isfinite(lon)
            good &= (lon >= west) & (lon < east) & (lat >= south) & (lat < north)
            candidates = [(r, c) for r in range(0, good.shape[0] - chip_size + 1, chip_size)
                          for c in range(0, good.shape[1] - chip_size + 1, chip_size)
                          if good[r:r + chip_size, c:c + chip_size].all()]
            rng = np.random.default_rng(0)
            chips, uncertainty, coordinates, locations, audit = [], [], [], [], []
            available_audit = {name: read_valid(geo[name]) for name in AUDIT_FIELDS if name in geo}
            for index in rng.permutation(len(candidates)):
                row, col = candidates[index]
                window = (slice(row, row + chip_size), slice(col, col + chip_size))
                selection = window + (slice(int(bands[0]), int(bands[-1]) + 1),)
                x = read_valid(rrs, selection)[..., bands - bands[0]]
                sigma = read_valid(geo["Rrs_unc"], selection)[..., bands - bands[0]]
                if not np.isfinite(x).all():
                    continue
                chips.append(x)
                uncertainty.append(sigma)
                coordinates.append([row, col])
                locations.append([float(lat[window].mean()), float(lon[window].mean())])
                audit.append([float(np.nanmean(available_audit[name][window]))
                              if name in available_audit and np.isfinite(available_audit[name][window]).any()
                              else float("nan") for name in AUDIT_FIELDS])
                if not pilot and len(chips) == limit:
                    break
            if not chips:
                raise ValueError(f"No usable AOP chips for {record['id']}")
            flag_rates = {name: float(((full_flags & np.uint32(mask)) != 0).mean())
                          for name, mask in zip(flags.attrs["flag_meanings"].split(),
                                                flags.attrs["flag_masks"]) if name != "SPARE"}
            scenes.append({**record, "x": np.asarray(chips, dtype="float32"),
                           "uncertainty": np.asarray(uncertainty, dtype="float32"),
                           "pixels": np.asarray(coordinates), "locations": np.asarray(locations),
                           "audit": np.asarray(audit, dtype="float32"),
                           "shape": list(good.shape), "flag_rates": flag_rates,
                           "acquisition_start": start.isoformat(), "acquisition_end": end.isoformat(),
                           "processing_version": version, "sha256": digest,
                           "candidate_clear_chips": len(candidates)})
            print(json.dumps({'loaded_scene': record['id'], 'chips': len(chips)}), flush=True)
    if pilot:
        source = scenes[0]
        assignments = np.asarray([pilot_partition(r, c, chip_size, block_size) for r, c in source['pixels']])
        scenes = []
        for split in ('train', 'validation', 'test'):
            indices = np.flatnonzero(assignments == split)[:limit]
            if not len(indices):
                raise ValueError(f"No usable buffered pilot chips for {split}.")
            scene = {**source, 'id': f"{source['id']}-{split}", 'split': split,
                     'scope': 'single-scene-spatial-pilot', 'block_size': block_size,
                     'guard_pixels_per_block_edge': chip_size,
                     'candidate_clear_chips': int((assignments == split).sum())}
            for key in ('x', 'uncertainty', 'pixels', 'locations', 'audit'):
                scene[key] = source[key][indices]
            scenes.append(scene)
    return scenes, reference


def load_prepared_scenes(specification, data_dir, chip_size, limit):
    """Read checksum-pinned chips made by the raw loader, preserving its sample order."""
    records, prepared = specification['scenes'], specification['prepared']
    validate_scene_manifest(records)
    if (prepared['chip_size'] != chip_size or prepared['limit'] < limit
            or prepared['excluded_flags'] != list(AOP_EXCLUDE_FLAGS)):
        raise ValueError('Prepared chip settings differ from this experiment.')
    grid = np.asarray(prepared['wavelengths_nm'], dtype='float32')
    if (grid.ndim != 1 or len(grid) < 2 or not np.isfinite(grid).all() or (np.diff(grid) <= 0).any()
            or (grid < 400).any() or (grid > 700).any()):
        raise ValueError('Invalid prepared wavelength grid.')
    if len({s['processing_version'] for s in records}) != 1:
        raise ValueError('Prepared processing versions differ.')
    dates, scenes = {}, []
    root = Path(data_dir).resolve()
    for record in records:
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', record['id']):
            raise ValueError('Scene IDs must be plain names.')
        start, end = (datetime.fromisoformat(record[k]) for k in ('acquisition_start', 'acquisition_end'))
        if start.tzinfo is None or end.tzinfo is None or end < start:
            raise ValueError('Invalid prepared acquisition times.')
        day, final = start.astimezone(timezone.utc).date(), end.astimezone(timezone.utc).date()
        while day <= final:
            if day in dates and dates[day] != record['split']:
                raise ValueError('Acquisition dates must not be shared across splits.')
            dates[day] = record['split']
            day += timedelta(days=1)
        path = (root / record['cache']).resolve()
        if not path.is_relative_to(root) or sha256(path) != record['cache_sha256']:
            raise ValueError('Prepared chip path or checksum mismatch.')
        with np.load(path, allow_pickle=False) as data:
            arrays = {k: data[k][:limit] for k in ('x', 'uncertainty', 'pixels', 'locations', 'audit')}
        x, pixels = arrays['x'], arrays['pixels']
        if (x.ndim != 4 or x.dtype != np.float32 or arrays['uncertainty'].dtype != np.float32
                or not len(x) or x.shape[1:] != (chip_size, chip_size, len(grid))
                or arrays['uncertainty'].shape != x.shape or not np.isfinite(x).all()
                or pixels.shape != (len(x), 2) or arrays['locations'].shape != (len(x), 2)
                or arrays['audit'].shape != (len(x), len(AUDIT_FIELDS))
                or not np.isfinite(arrays['locations']).all() or not np.issubdtype(pixels.dtype, np.integer)
                or (pixels < 0).any() or (pixels % chip_size).any()
                or (pixels + chip_size > np.asarray(record['shape'])).any()):
            raise ValueError('Invalid prepared chip arrays.')
        scenes.append({**record, **arrays, 'chips': len(x)})
    return scenes, grid
