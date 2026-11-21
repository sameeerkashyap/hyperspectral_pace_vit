"""Download frozen NASA candidates with curl and screen AOP chips before training."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import h5netcdf
import numpy as np

from src.data import quality_mask, read_valid
from src.discovery_catalog import candidate
from src.discovery_data import AOP_EXCLUDE_FLAGS, load_aop_scenes
from src.runner import sha256


def screen(path, record, chip_size, minimum):
    with h5netcdf.File(path) as root:
        if (root.attrs.get('platform') != 'PACE' or root.attrs.get('instrument') != 'OCI'
                or str(root.attrs.get('processing_version')) != '3.2'):
            raise ValueError('Expected PACE OCI v3.2.')
        start = datetime.fromisoformat(str(root.attrs['time_coverage_start']).replace('Z', '+00:00'))
        if start.strftime('PACE_OCI.%Y%m%dT%H%M%S') != record['aop'].split('.L2.')[0]:
            raise ValueError('Acquisition metadata does not match the candidate filename.')
        geo, nav = root.groups['geophysical_data'].variables, root.groups['navigation_data'].variables
        grid = read_valid(root.groups['sensor_band_parameters'].variables['wavelength_3d'])
        bands = np.flatnonzero((grid >= 400) & (grid <= 700))
        if len(bands) != 136 or not np.isfinite(grid).all() or (np.diff(grid) <= 0).any():
            raise ValueError('Expected the 136-band native visible grid.')
        lat, lon = (read_valid(nav[name]) for name in ('latitude', 'longitude'))
        good = quality_mask(geo['l2_flags'], exclude_flags=AOP_EXCLUDE_FLAGS)
        west, south, east, north = record['bbox']
        good &= np.isfinite(lat) & np.isfinite(lon)
        good &= (lon >= west) & (lon < east) & (lat >= south) & (lat < north)
        if geo['Rrs'].shape != (*lat.shape, len(grid)) or geo['Rrs'].attrs.get('units') != 'sr^-1':
            raise ValueError('Invalid reflectance geometry or units.')
        count = 0
        for row in range(0, good.shape[0] - chip_size + 1, chip_size):
            for col in range(0, good.shape[1] - chip_size + 1, chip_size):
                window = (slice(row, row + chip_size), slice(col, col + chip_size))
                if good[window].all():
                    values = read_valid(geo['Rrs'], window + (slice(int(bands[0]), int(bands[-1]) + 1),))
                    if np.isfinite(values[..., bands - bands[0]]).all():
                        count += 1
                        if count >= minimum:
                            return {'confirmed_clear_chips_at_least': count,
                                    'quality_pixel_fraction_in_full_swath': float(good.mean())}
        raise ValueError(f'Only {count} usable chips; require {minimum}.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--netrc', type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.candidates.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    records, attempts = [], []
    with tempfile.TemporaryDirectory() as private:
        cookies = Path(private) / 'cookies'
        cookies.touch(mode=0o600)
        for slot in spec['slots']:
            accepted = None
            for option in slot['candidates']:
                # Recheck the frozen filename and download host before sending credentials.
                checked = candidate({'producer_granule_id': option['aop'], 'id': option['cmr_id'],
                                     'time_start': option['acquisition_start'], 'time_end': option['acquisition_end'],
                                     'granule_size': option['estimated_mib'],
                                     'links': [{'rel': '/data#', 'href': option['url']}]})
                record = {key: slot[key] for key in ('id', 'split', 'bbox')}
                record.update(aop=checked['aop'])
                target = args.data_dir / record['aop']
                if target.exists():
                    raise FileExistsError(f'Refusing to replace {target.name}.')
                required = int(option['estimated_mib'] * 1024 ** 2 * 1.3) + 4 * 1024 ** 3
                if shutil.disk_usage(args.data_dir).free < required:
                    raise ValueError('Insufficient free storage for download plus a 4 GiB results reserve.')
                with tempfile.NamedTemporaryFile(dir=args.data_dir, suffix='.part', delete=False) as file:
                    partial = Path(file.name)
                attempt = {'id': slot['id'], 'aop': record['aop'], 'cmr_id': option['cmr_id']}
                try:
                    print(json.dumps({'downloading': record['aop'], 'slot': slot['id']}), flush=True)
                    subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error',
                                    '--retry', '3', '--netrc-file', str(args.netrc),
                                    '--cookie', str(cookies), '--cookie-jar', str(cookies),
                                    '--output', str(partial), checked['url']], check=True)
                    screening = screen(partial, record, spec['chip_size'], spec['minimum_chips'])
                    accepted = {**record, 'url': checked['url'], 'cmr_id': checked['cmr_id'],
                                'sha256': sha256(partial), 'bytes': partial.stat().st_size, **screening}
                    partial.replace(target)
                    attempt['status'] = 'accepted'
                    records.append(accepted)
                except subprocess.CalledProcessError:
                    # Authentication and transport errors must not silently alter the chosen dataset.
                    raise
                except (ValueError, OSError, KeyError) as error:
                    attempt.update(status='rejected', reason=str(error))
                finally:
                    partial.unlink(missing_ok=True)
                    attempts.append(attempt)
                    (args.output / 'attempts.json').write_text(json.dumps(attempts, indent=2) + '\n')
                print(json.dumps(attempt), flush=True)
                if accepted:
                    break
            if accepted is None:
                raise ValueError(f'No candidate passed the fixed quality rule for {slot["id"]}.')
            (args.output / 'selected-progress.json').write_text(json.dumps(records, indent=2) + '\n')
    manifest = args.output / 'scenes.json'
    manifest.write_text(json.dumps({'scope': 'multi-scene', 'scenes': records}, indent=2) + '\n')
    scenes, grid = load_aop_scenes(manifest, args.data_dir)
    inventory = {'bands': grid.tolist(), 'scenes': [{k: v for k, v in s.items()
                 if k not in ('x', 'uncertainty', 'pixels', 'locations', 'audit')} | {'chips': len(s['x'])}
                 for s in scenes]}
    (args.output / 'inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
    (args.output / 'ready').touch()
    print(json.dumps({'staging': 'complete', 'scenes': len(scenes)}), flush=True)


if __name__ == '__main__':
    main()
