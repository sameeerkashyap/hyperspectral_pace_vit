"""Freeze geographically and temporally separate AOP candidates from NASA CMR."""

import argparse
from datetime import date, timedelta
import json
from pathlib import Path
import re
import urllib.parse
import urllib.request

import numpy as np

from src.matched_data import validate_scene_manifest


COLLECTION = 'C4124887023-OB_CLOUD'
REGIONS = (
    ('california', 'train', [-135, 24, -117, 43]),
    ('north-pacific', 'train', [-160, 10, -140, 30]),
    ('north-atlantic', 'train', [-55, 25, -30, 50]),
    ('east-atlantic', 'validation', [-25, 25, -10, 45]),
    ('south-indian', 'validation', [45, -35, 75, -10]),
    ('benguela', 'test', [0, -35, 20, -10]),
    ('south-pacific', 'test', [-115, -35, -85, -10]),
)


def candidate(entry):
    name = entry['producer_granule_id']
    if not re.fullmatch(r'PACE_OCI\.\d{8}T\d{6}\.L2\.OC_AOP\.V3_2\.nc', name):
        raise ValueError('Expected an AOP v3.2 granule filename.')
    urls = [link['href'] for link in entry.get('links', [])
            if not link.get('inherited') and link.get('rel', '').endswith('/data#')
            and link['href'] == 'https://obdaac-tea.earthdatacloud.nasa.gov/ob-cumulus-prod-public/' + name]
    if not urls:
        raise ValueError('Missing the expected NASA HTTPS download link.')
    return {'aop': name, 'url': urls[0], 'cmr_id': entry['id'],
            'acquisition_start': entry['time_start'], 'acquisition_end': entry['time_end'],
            'cloud_cover_percent': float(entry.get('cloud_cover', 100)),
            'estimated_mib': float(entry['granule_size'])}


def coverage(entry, bbox):
    """Approximate overlap for ranking only; the pixel screen makes the final choice."""
    west, south, east, north = bbox
    overlap = 0.
    for group in entry.get('polygons', []):
        for polygon in group:
            points = np.asarray(list(map(float, polygon.split()))).reshape(-1, 2)
            lon = points[:, 1]
            if np.ptp(lon) > 180:
                lon = np.where(lon > 0, lon - 360, lon)
            height = max(0., min(north, points[:, 0].max()) - max(south, points[:, 0].min()))
            for shift in (-360, 0, 360):
                width = max(0., min(east, lon.max() + shift) - max(west, lon.min() + shift))
                overlap = max(overlap, width * height)
    return overlap / ((east - west) * (north - south))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    slots = []
    for region, split, bbox in REGIONS:
        month, days = (6, (1, 5, 9, 13, 17, 21)) if split == 'train' else (
            (7, (1, 9, 17)) if split == 'validation' else (8, (1, 9, 17)))
        window_days = 4 if split == 'train' else 8
        for day in days:
            start = date(2024, month, day)
            parameters = {'collection_concept_id': COLLECTION, 'bounding_box': ','.join(map(str, bbox)),
                          'temporal': f'{start}T00:00:00Z,{start + timedelta(days=window_days)}T00:00:00Z',
                          'page_size': '200'}
            url = 'https://cmr.earthdata.nasa.gov/search/granules.json?' + urllib.parse.urlencode(parameters)
            with urllib.request.urlopen(url, timeout=120) as response:
                entries = json.load(response)['feed']['entry']
            if len(entries) >= 200:
                raise ValueError('CMR page is full; paginate before selecting candidates.')
            entries = [e for e in entries
                       if str(start) <= e['time_start'][:10] < str(start + timedelta(days=window_days))]
            options = sorted(({**candidate(e), 'approximate_region_overlap': coverage(e, bbox)} for e in entries),
                             key=lambda e: (-e['approximate_region_overlap'] * (1 - e['cloud_cover_percent'] / 100),
                                            e['aop']))[:5]
            if not options:
                raise ValueError(f'No candidates for {region} on {start}.')
            slot = {'id': f'{region}-{start}', 'split': split, 'bbox': bbox,
                    'window_start': str(start), 'window_days': window_days, 'query_url': url, 'candidates': options}
            slots.append(slot)
            print(json.dumps({'id': slot['id'], 'candidates': len(options)}), flush=True)
    validate_scene_manifest(slots)
    result = {'collection': COLLECTION, 'scope': 'multi-scene', 'minimum_chips': 16,
              'chip_size': 16, 'slots': slots,
              'selection': 'Fixed regions and disjoint date windows. Rank up to five candidates by approximate regional overlap times whole-granule clear fraction; accept the first with at least 16 finite, quality-screened chips. No model or biological scores select scenes.'}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
