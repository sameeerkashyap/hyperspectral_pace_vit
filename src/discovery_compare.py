"""Compare completed discovery runs and align optical groups across seeds."""

import argparse
from itertools import combinations
import json
from pathlib import Path
import time

import numpy as np


def wait_for_runs(root, count):
    while True:
        files = list(root.glob('*/run.json'))
        if len(files) > count:
            raise ValueError('More experiments than the expected comparison count.')
        try:
            statuses = [json.loads(path.read_text())['status'] for path in files]
        except (FileNotFoundError, json.JSONDecodeError):
            statuses = []
        if 'failed' in statuses:
            raise RuntimeError('An experiment failed; inspect its run.json and console.log.')
        if len(statuses) == count and all(status == 'complete' for status in statuses):
            return
        print(json.dumps({'waiting_for_experiments': count,
                          'completed': statuses.count('complete')}), flush=True)
        time.sleep(30)


def adjusted_rand(left, right):
    if len(left) != len(right) or len(left) < 2:
        raise ValueError('Need matching cluster assignments for at least two patches.')
    _, a = np.unique(left, return_inverse=True)
    _, b = np.unique(right, return_inverse=True)
    table = np.zeros((a.max() + 1, b.max() + 1), dtype='int64')
    np.add.at(table, (a, b), 1)
    pairs = lambda values: float((values * (values - 1) / 2).sum())
    total = len(left) * (len(left) - 1) / 2
    expected = pairs(table.sum(0)) * pairs(table.sum(1)) / total
    maximum = (pairs(table.sum(0)) + pairs(table.sum(1))) / 2
    return (pairs(table) - expected) / (maximum - expected) if maximum > expected else 1.


def compare(paths):
    runs, summaries = [], []
    reference = None
    for path in paths:
        report = json.loads((path / 'metrics.json').read_text())
        provenance = [(s['id'], s['split'], s['sha256'], s['bbox'], s.get('cache_sha256'))
                      for s in report['scene_manifest']]
        if reference is not None and reference != provenance:
            raise ValueError('Compare only runs with identical scene inputs and splits.')
        reference = provenance
        with np.load(path / 'trained/features.npz', allow_pickle=False) as data:
            features = {key: data[key] for key in ('scene_ids', 'scene_index', 'positions', 'cluster',
                                                   'retrieval_bank', 'retrieval_queries', 'neighbors')}
        label = f"{report['config']['method']}-{report['config']['masking']}-seed-{report['config']['seed']}"
        rows = report['test']['scenes']
        summaries.append({'run': label, 'path': str(path),
                          'test_scene_mean_rmse_sr_inv': report['test']['scene_mean_rmse_sr_inv'],
                          'spectral_gap_rmse_sr_inv': float(np.mean([r['rmse_sr_inv'] for r in rows if r['mask'] == 'spectral'])),
                          'spatial_hole_rmse_sr_inv': float(np.mean([r['rmse_sr_inv'] for r in rows if r['mask'] == 'spatial'])),
                          'noise_cluster_agreement_test': report['discovery']['noise_cluster_agreement_test'],
                          'noise_neighbor_overlap_at_5': report['discovery']['noise_neighbor_overlap_at_5']})
        test_scenes = [i for i, s in enumerate(report['scene_manifest']) if s['split'] == 'test']
        runs.append((label, features, np.isin(features['scene_index'], test_scenes)))
    agreements = []
    for (a_name, a, test), (b_name, b, other_test) in combinations(runs, 2):
        for key in ('scene_ids', 'scene_index', 'positions', 'retrieval_bank', 'retrieval_queries'):
            if not np.array_equal(a[key], b[key]):
                raise ValueError(f'Patch or retrieval sampling differs: {key}')
        if not np.array_equal(test, other_test):
            raise ValueError('Test patches differ.')
        agreements.append({'left': a_name, 'right': b_name,
                           'adjusted_rand_test': adjusted_rand(a['cluster'][test], b['cluster'][test]),
                           'neighbor_overlap_at_5': float(np.mean([len(set(x) & set(y)) / len(x)
                                                                  for x, y in zip(a['neighbors'], b['neighbors'])]))})
    return {'runs': summaries, 'agreements': agreements,
            'scope': 'Descriptive comparison; seed agreement and reconstruction do not establish biological identity.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path, nargs='+', help='Completed discovery/ directories')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
