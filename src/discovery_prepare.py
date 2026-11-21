"""Save the raw loader's verified samples once for repeated Nautilus experiments."""

import argparse
import json
from pathlib import Path

import numpy as np

from src.discovery_data import AOP_EXCLUDE_FLAGS, load_aop_scenes
from src.runner import sha256


def prepare(manifest, data_dir, output, chip_size=16, limit=128):
    output.mkdir(parents=True, exist_ok=False)
    scenes, grid = load_aop_scenes(manifest, data_dir, chip_size, limit)
    if any(s.get('scope') == 'single-scene-spatial-pilot' for s in scenes):
        raise ValueError('Preparation requires independent multi-scene splits.')
    records = []
    keys = ('x', 'uncertainty', 'pixels', 'locations', 'audit')
    for scene in scenes:
        path = output / f"{scene['id']}.npz"
        np.savez_compressed(path, **{key: scene[key] for key in keys})
        records.append({k: v for k, v in scene.items() if k not in keys}
                       | {'cache': path.name, 'cache_sha256': sha256(path), 'chips': len(scene['x'])})
    result = {'scope': 'multi-scene', 'scenes': records,
              'prepared': {'chip_size': chip_size, 'limit': limit,
                           'excluded_flags': list(AOP_EXCLUDE_FLAGS), 'wavelengths_nm': grid.tolist()}}
    (output / 'scenes.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    loaded, restored_grid = load_aop_scenes(output / 'scenes.json', output, chip_size, limit)
    np.testing.assert_array_equal(grid, restored_grid)
    for original, restored in zip(scenes, loaded):
        for key in keys:
            np.testing.assert_array_equal(original[key], restored[key])
    (output / 'ready').touch()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenes', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.scenes, args.data_dir, args.output)
    print(json.dumps({'preparation': 'complete', 'scenes': len(result['scenes'])}), flush=True)


if __name__ == '__main__':
    main()
