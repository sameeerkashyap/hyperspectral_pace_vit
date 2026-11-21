import argparse
import importlib.util
import json
import subprocess
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5netcdf
import numpy as np
import torch

from src.data import EXCLUDE_FLAGS, quality_mask
from src.discovery import SpectralPCA, kmeans, run
from src.discovery_compare import adjusted_rand, compare, wait_for_runs
from src.discovery_data import AOP_EXCLUDE_FLAGS, load_aop_scenes, pilot_partition
from src.discovery_catalog import candidate, coverage, REGIONS
from src.discovery_stage import screen
from src.discovery_prepare import prepare
from src.discovery_worker import run_worker
from src.matched_data import validate_scene_manifest
from src.grouped_model import GroupedSpectralMAE, SpectralAutoencoder, token_mask


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_grouped_roundtrip_structured_masks_and_hidden_value_leakage(self):
        grid = torch.linspace(400, 699, 16)
        model = GroupedSpectralMAE(grid, group_size=2, patch_size=2, embed_dim=24).eval()
        x = torch.rand(2, 8, 8, 16)
        torch.testing.assert_close(model.unpatchify(model.patchify(x), x.shape), x)
        for mode in ('spectral', 'spatial', 'mixed', 'random'):
            mask = token_mask(2, 4, 4, 8, mode, seed=2)
            self.assertTrue(torch.equal(mask, token_mask(2, 4, 4, 8, mode, seed=2)))
            grouped = mask.reshape(2, 4, 4, 8)
            if mode in ('mixed', 'spectral'):
                self.assertTrue(grouped.all((1, 2)).sum(1).ge(2).all())
            if mode in ('mixed', 'spatial'):
                self.assertTrue(grouped.all(-1).sum((1, 2)).ge(4).all())
            if mode in ('mixed', 'random'):
                self.assertTrue((mask.sum(1) == 96).all())
            hidden = model.unpatchify(mask[..., None].expand(-1, -1, 8), x.shape)
            altered = x.masked_fill(hidden, 999)
            a, _, target = model(x, grid, mask)
            b, _, _ = model(altered, grid, mask)
            torch.testing.assert_close(a, b)
            loss = model.masked_loss(a, mask, target)
            model.zero_grad()
            loss.backward()
            self.assertTrue(torch.isfinite(model.patch_embed.weight.grad).all())
        self.assertEqual(model.spatial_features(x, grid).shape, (2, 4, 4, 24))
        spectral_ae = SpectralAutoencoder(16, 24)
        torch.testing.assert_close(spectral_ae(x, hidden), spectral_ae(altered, hidden))
        with self.assertRaisesRegex(ValueError, 'grid'):
            model(x, grid + 1)
        for invalid in (torch.ones(2, 128, dtype=torch.bool), torch.zeros(2, 128, dtype=torch.bool)):
            with self.assertRaises(ValueError):
                model(x, grid, invalid)
        large = GroupedSpectralMAE(torch.linspace(400, 699, 136))
        self.assertEqual(large.patchify(torch.zeros(1, 16, 16, 136)).shape, (1, 1088, 32))

    def make_scenes(self, root, size=16, bands=8):
        records = []
        for index, split in enumerate(('train', 'validation', 'test')):
            record = {'id': split, 'split': split, 'aop': f'{split}.nc',
                      'bbox': [index * 20, 0, index * 20 + 10, 10]}
            records.append(record)
            with h5netcdf.File(root / record['aop'], 'w') as f:
                f.dimensions = {'scan': size, 'pixel': size, 'band': bands}
                f.attrs.update(platform='PACE', instrument='OCI', processing_version='3.2',
                               time_coverage_start=f'2024-07-0{index + 1}T12:00:00Z',
                               time_coverage_end=f'2024-07-0{index + 1}T12:05:00Z')
                spectral = f.create_group('sensor_band_parameters')
                spectral.create_variable('wavelength_3d', ('band',), dtype='f4')[:] = np.linspace(400, 699, bands)
                nav = f.create_group('navigation_data')
                nav.create_variable('latitude', ('scan', 'pixel'), dtype='f4')[:] = 5
                nav.create_variable('longitude', ('scan', 'pixel'), dtype='f4')[:] = index * 20 + 5
                geo = f.create_group('geophysical_data')
                flags = geo.create_variable('l2_flags', ('scan', 'pixel'), dtype='i4')
                flags.attrs.update(flag_meanings=' '.join(EXCLUDE_FLAGS),
                                   flag_masks=2 ** np.arange(len(EXCLUDE_FLAGS), dtype='int32'))
                flags[:] = 0
                flags[0, 0] = 1
                rng = np.random.default_rng(index)
                rrs = geo.create_variable('Rrs', ('scan', 'pixel', 'band'), dtype='i2', fillvalue=-32767)
                rrs.attrs.update(units='sr^-1', scale_factor=1e-6, add_offset=0., valid_min=-1000, valid_max=30000)
                rrs[:] = rng.integers(1000, 10000, (size, size, bands), dtype='int16')
                unc = geo.create_variable('Rrs_unc', ('scan', 'pixel', 'band'), dtype='f4', fillvalue=-32767)
                unc[:] = .0001
                geo.create_variable('aot_865', ('scan', 'pixel'), dtype='f4')[:] = .05 + index * .02
        manifest = root / 'scenes.json'
        manifest.write_text(json.dumps({'scenes': records}))
        return manifest, records

    def test_single_scene_pilot_has_buffered_disjoint_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, records = self.make_scenes(root, size=128)
            records[0]['bbox'] = [-180, -90, 180, 90]
            manifest.write_text(json.dumps({'scope': 'single-scene-spatial-pilot',
                                           'block_size': 32, 'scenes': records[:1]}))
            scenes, _ = load_aop_scenes(manifest, root, 8, 100)
            blocks = []
            for scene in scenes:
                self.assertGreater(len(scene['x']), 0)
                blocks.append({tuple(p // 32) for p in scene['pixels']})
                for row, col in scene['pixels']:
                    self.assertEqual(pilot_partition(row, col, 8, 32), scene['split'])
                    self.assertTrue(all(8 <= p % 32 <= 16 for p in (row, col)))
            self.assertFalse(blocks[0] & blocks[1] or blocks[0] & blocks[2] or blocks[1] & blocks[2])
            self.assertEqual(len({s['sha256'] for s in scenes}), 1)
            repeated, _ = load_aop_scenes(manifest, root, 8, 100)
            np.testing.assert_array_equal(scenes[0]['pixels'], repeated[0]['pixels'])
            manifest.write_text(json.dumps({'scenes': records[:1]}))
            with self.assertRaises(ValueError):
                load_aop_scenes(manifest, root, 8, 100)

    def test_aop_loader_flags_validity_and_split_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, records = self.make_scenes(root)
            scenes, _ = load_aop_scenes(manifest, root, chip_size=8, limit=4)
            self.assertEqual(len(scenes[0]['x']), 3)
            self.assertNotIn([0, 0], scenes[0]['pixels'].tolist())
            records[0]['sha256'] = '0' * 64
            manifest.write_text(json.dumps({'scenes': records}))
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                load_aop_scenes(manifest, root, 8, 4)
            del records[0]['sha256']
            manifest.write_text(json.dumps({'scenes': records}))
            with h5netcdf.File(root / records[0]['aop'], 'a') as f:
                flags = f.groups['geophysical_data'].variables['l2_flags']
                flags[0, 0] = 2 ** EXCLUDE_FLAGS.index('CHLWARN')
                self.assertFalse(quality_mask(flags)[0, 0])
                self.assertTrue(quality_mask(flags, exclude_flags=AOP_EXCLUDE_FLAGS)[0, 0])
                f.groups['geophysical_data'].variables['Rrs'][0, 0, 0] = -32767
            self.assertEqual(len(load_aop_scenes(manifest, root, 8, 4)[0][0]['x']), 3)
            altered = json.loads(manifest.read_text())
            altered['scenes'][1]['bbox'] = records[0]['bbox']
            manifest.write_text(json.dumps(altered))
            with self.assertRaisesRegex(ValueError, 'overlap'):
                load_aop_scenes(manifest, root)
            manifest.write_text(json.dumps({'scenes': records}))
            with h5netcdf.File(root / records[1]['aop'], 'a') as f:
                f.attrs['processing_version'] = 'other'
            with self.assertRaisesRegex(ValueError, 'version'):
                load_aop_scenes(manifest, root, 8, 4)
            with h5netcdf.File(root / records[1]['aop'], 'a') as f:
                f.attrs['processing_version'] = '3.2'
                f.groups['sensor_band_parameters'].variables['wavelength_3d'][1] += 1
            with self.assertRaisesRegex(ValueError, 'grid'):
                load_aop_scenes(manifest, root, 8, 4)
            with h5netcdf.File(root / records[1]['aop'], 'a') as f:
                f.groups['sensor_band_parameters'].variables['wavelength_3d'][1] -= 1
            with h5netcdf.File(root / records[1]['aop'], 'a') as f:
                f.attrs['time_coverage_start'] = '2024-07-01T12:00:00Z'
            with self.assertRaisesRegex(ValueError, 'dates'):
                load_aop_scenes(manifest, root, 8, 4)

    def test_acquisition_host_split_and_quality_guards(self):
        name = 'PACE_OCI.20240701T120000.L2.OC_AOP.V3_2.nc'
        entry = {'producer_granule_id': name, 'id': 'G1', 'time_start': '2024-07-01T12:00:00Z',
                 'time_end': '2024-07-01T12:05:00Z', 'granule_size': '1',
                 'links': [{'rel': '/data#', 'href': 'https://example.com/' + name}]}
        with self.assertRaisesRegex(ValueError, 'NASA HTTPS'):
            candidate(entry)
        entry['links'][0]['href'] = 'https://obdaac-tea.earthdatacloud.nasa.gov/ob-cumulus-prod-public/' + name
        self.assertEqual(candidate(entry)['aop'], name)
        self.assertEqual(coverage({'polygons': [['0 170 10 170 10 -150 0 -150 0 170']]},
                                  [-160, 0, -140, 10]), .5)
        validate_scene_manifest([{'id': r, 'split': s, 'bbox': b} for r, s, b in REGIONS])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, records = self.make_scenes(root)
            record = {**records[0], 'aop': name}
            # A valid shape is not enough: staging rejects an incompatible native grid.
            with self.assertRaisesRegex(ValueError, '136-band'):
                screen(root / 'train.nc', record, 8, 3)
            with h5netcdf.File(root / 'train.nc', 'a') as f:
                f.attrs['processing_version'] = 'bad'
            with self.assertRaisesRegex(ValueError, 'v3.2'):
                screen(root / 'train.nc', record, 8, 3)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, records = self.make_scenes(root, bands=136)
            record = {**records[0], 'aop': name}
            self.assertEqual(screen(root / 'train.nc', record, 8, 3)['confirmed_clear_chips_at_least'], 3)
            with self.assertRaisesRegex(ValueError, 'Only 3'):
                screen(root / 'train.nc', record, 8, 4)

    def test_pca_uses_visible_entries_and_clustering(self):
        rng = np.random.default_rng(0)
        pixels = rng.normal(size=(3, 4, 4, 8)).astype('float32')
        pca = SpectralPCA().fit([{'x': pixels}], 0., 1.)
        hidden = np.zeros_like(pixels, dtype=bool)
        hidden[..., 2:4] = True
        altered = np.where(hidden, 1000, pixels)
        np.testing.assert_allclose(pca.reconstruct(pixels, hidden), pca.reconstruct(altered, hidden))
        pca.rank = 2
        pca.mean = np.zeros(3, dtype='float32')
        pca.components = np.array([[1, 0, 1], [0, 1, 1]], dtype='float32')
        low_rank = np.array([[[[2., 3., 5.]]]], dtype='float32')
        np.testing.assert_allclose(pca.reconstruct(low_rank, np.array([[[[False, False, True]]]])), low_rank)
        x = np.array([[0., 0.], [.01, 0.], [10., 10.], [10.01, 10.]])
        centers = kmeans(x, 2)
        self.assertEqual(centers.shape, (2, 2))
        self.assertEqual(adjusted_rand([0, 0, 1, 1], [5, 5, 3, 3]), 1.)

    def test_comparison_waits_for_completion_and_reports_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / 'run' / 'run.json'
            state.parent.mkdir()
            state.write_text('{')
            with patch('src.discovery_compare.time.sleep',
                       side_effect=lambda _: state.write_text('{"status":"complete"}')) as sleep:
                wait_for_runs(root, 1)
                sleep.assert_called_once_with(30)
            state.write_text('{"status":"failed"}')
            with self.assertRaisesRegex(RuntimeError, 'failed'):
                wait_for_runs(root, 1)

    def test_worker_publishes_complete_attempts_without_overwriting(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / 'discovery.json').write_text('{"experiments":[{"name":"example"}]}')
            root = source / 'results'
            def finish(*args, **kwargs):
                path = Path(kwargs['env']['OUTPUT_ROOT']) / 'example'
                path.mkdir(parents=True)
                (path / 'run.json').write_text('{"status":"complete"}')
                (path / 'weights').write_bytes(b'saved')
            with patch('src.discovery_worker.subprocess.run', side_effect=finish) as command:
                run_worker(source, root, 0, 'pod-1')
                self.assertEqual((root / 'example/weights').read_bytes(), b'saved')
                run_worker(source, root, 0, 'pod-2')
                command.assert_called_once()
            (root / 'example/run.json').write_text('{"status":"running"}')
            with self.assertRaises(FileExistsError):
                run_worker(source, root, 0, 'pod-3')
            with self.assertRaises(ValueError):
                run_worker(source, root, -1, 'pod-3')
            retry_root = source / 'retry'
            def interrupt(*args, **kwargs):
                path = Path(kwargs['env']['OUTPUT_ROOT']) / 'example'
                path.mkdir(parents=True)
                (path / 'run.json').write_text('{"status":"failed"}')
                raise subprocess.CalledProcessError(137, args[0])
            with patch('src.discovery_worker.subprocess.run', side_effect=interrupt):
                with self.assertRaises(subprocess.CalledProcessError):
                    run_worker(source, retry_root, 0, 'interrupted')
            self.assertFalse((retry_root / 'example').exists())
            with patch('src.discovery_worker.subprocess.run', side_effect=finish):
                run_worker(source, retry_root, 0, 'replacement')
            self.assertTrue((source / 'retry-attempts/interrupted/example/run.json').exists())
            self.assertEqual((retry_root / 'example/weights').read_bytes(), b'saved')

    def test_prepared_chips_preserve_values_order_and_split_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _ = self.make_scenes(root)
            output = root / 'prepared'
            result = prepare(manifest, root, output, chip_size=8, limit=3)
            self.assertTrue((output / 'ready').exists())
            raw, grid = load_aop_scenes(manifest, root, 8, 3)
            cached, cached_grid = load_aop_scenes(output / 'scenes.json', output, 8, 3)
            np.testing.assert_array_equal(grid, cached_grid)
            for a, b in zip(raw, cached):
                for key in ('x', 'uncertainty', 'pixels', 'locations', 'audit'):
                    np.testing.assert_array_equal(a[key], b[key])
            with self.assertRaisesRegex(ValueError, 'settings'):
                load_aop_scenes(output / 'scenes.json', output, 8, 4)
            original = (output / 'scenes.json').read_text()
            result['scenes'][1]['acquisition_start'] = result['scenes'][0]['acquisition_start']
            (output / 'scenes.json').write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, 'dates'):
                load_aop_scenes(output / 'scenes.json', output, 8, 3)
            (output / 'scenes.json').write_text(original)
            with (output / result['scenes'][0]['cache']).open('ab') as stream:
                stream.write(b'corruption')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                load_aop_scenes(output / 'scenes.json', output, 8, 3)

    @unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'Install the discovery extra')
    def test_end_to_end_all_methods_without_bgc(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _ = self.make_scenes(root)
            args = argparse.Namespace(scenes=manifest, data_dir=root, output=root / 'mae', method='mae',
                                      masking='mixed', seed=0, steps=2, chips=2, chip_size=8, patch_size=2,
                                      group_size=2, embed_dim=24, clusters=2, batch_size=2, threads=1)
            paths = []
            for method in ('mae', 'spectral-ae', 'pca'):
                args.method, args.output = method, root / method
                report = run(args)
                self.assertTrue(np.isfinite(report['test']['scene_mean_rmse_sr_inv']))
                expected = load_aop_scenes(manifest, root, 8, 2)[0][0]['x'].mean(dtype='float64')
                self.assertAlmostEqual(report['normalization']['mean'], expected)
                self.assertNotIn('CHLWARN', report['excluded_flags'])
                self.assertTrue((args.output / 'discovery/trained/map-test.png').exists())
                with np.load(args.output / 'discovery/trained/features.npz', allow_pickle=False) as data:
                    ids = data['scene_index']
                    self.assertTrue((ids[data['neighbors']] == 0).all())
                    self.assertTrue((ids[data['retrieval_queries']] == 2).all())
                    np.testing.assert_allclose(data['feature_mean'], data['features'][data['retrieval_bank']].mean(0))
                paths.append(args.output / 'discovery')
            self.assertEqual(len(compare(paths)['agreements']), 3)
            with self.assertRaises(FileExistsError):
                run(args)


if __name__ == '__main__':
    unittest.main()
