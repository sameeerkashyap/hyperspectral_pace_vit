import argparse
import json
from pathlib import Path
import tempfile
import unittest

import h5netcdf
import numpy as np
import torch

from src.benchmark import ARMS, run
from src.chlorophyll import ChlorophyllRegressor, chlorophyll_loss
from src.data import EXCLUDE_FLAGS
from src.matched_data import load_matched_scenes
from src.model import WavelengthAwareMAE
from src.spatial_model import SpatialSpectralMAE


class ChlorophyllTests(unittest.TestCase):
    def test_full_spectrum_tokens_roundtrip_and_dense_output(self):
        torch.set_num_threads(1)
        wavelengths = torch.linspace(400, 700, 172)
        model = SpatialSpectralMAE(wavelengths, embed_dim=96, encoder_blocks=1, decoder_blocks=1)
        x = torch.rand(1, 172, 128, 128)
        self.assertEqual(model.patch_embed(x).shape, (1, 1024, 96))
        self.assertEqual(model.patchify(x).shape, (1, 1024, 2752))
        torch.testing.assert_close(model.unpatchify(model.patchify(x), x.shape), x)
        small = x[:, :, :8, :8]
        prediction, mask, target = model(small, wavelengths)
        self.assertEqual(prediction.shape, (1, 4, 2752))
        self.assertEqual(int(mask.sum()), 3)
        model.masked_loss(prediction, mask, target).backward()
        self.assertTrue(torch.isfinite(model.patch_embed.proj.weight.grad).all())
        regressor = ChlorophyllRegressor(model)
        self.assertEqual(regressor(small, wavelengths).shape, (1, 1, 8, 8))
        with self.assertRaisesRegex(ValueError, "grid"):
            regressor(small, wavelengths + 1)

    def test_valid_pixel_log_loss_and_mixed_mask(self):
        prediction = torch.tensor([[[[1., 999.]]]], requires_grad=True)
        target = torch.tensor([[[[100., float("nan")]]]])
        valid = torch.tensor([[[[True, False]]]])
        loss = chlorophyll_loss(prediction, target, valid)
        self.assertEqual(loss.item(), 1.)
        loss.backward()
        self.assertEqual(prediction.grad[0, 0, 0, 1].item(), 0.)
        with self.assertRaises(ValueError):
            chlorophyll_loss(prediction, target, ~valid)
        with self.assertRaises(ValueError):
            chlorophyll_loss(prediction, target, torch.zeros_like(valid))
        model = WavelengthAwareMAE(embed_dim=24, patch_size=4, encoder_blocks=1, decoder_blocks=1)
        x = torch.rand(2, 8, 8, 8)
        wavelengths = torch.linspace(400, 700, 8)
        pred, mask, patches = model(x, wavelengths, masking="mixed")
        grouped = mask.reshape(2, 4, 8)
        self.assertTrue((mask.sum(1) == 24).all())
        for sample in grouped:
            self.assertTrue(sample.all(dim=1).any())
            spectral = sample.all(dim=0)
            self.assertTrue((spectral[:-1] & spectral[1:]).any())
        model.masked_loss(pred, mask, patches).backward()
        self.assertEqual(ChlorophyllRegressor(model)(x.permute(0, 3, 1, 2), wavelengths).shape,
                         (2, 1, 8, 8))

    def make_scenes(self, root):
        records = []
        for number, split in enumerate(("train", "validation", "test")):
            west = number * 20
            record = {"id": split, "split": split, "bbox": [west, 0, west + 10, 10],
                      "aop": f"{split}-aop.nc", "bgc": f"{split}-bgc.nc"}
            records.append(record)
            for kind in ("aop", "bgc"):
                with h5netcdf.File(root / record[kind], "w") as f:
                    f.dimensions = {"scan": 16, "pixel": 16, "bands": 4}
                    f.attrs.update(time_coverage_start=f"2024-07-0{number + 1}T12:00:00Z",
                                   time_coverage_end=f"2024-07-0{number + 1}T12:05:00Z",
                                   processing_version="test", platform="PACE", instrument="OCI")
                    nav = f.create_group("navigation_data")
                    nav.create_variable("latitude", ("scan", "pixel"), dtype="f4")[:] = 5
                    nav.create_variable("longitude", ("scan", "pixel"), dtype="f4")[:] = west + 5
                    geo = f.create_group("geophysical_data")
                    flags = geo.create_variable("l2_flags", ("scan", "pixel"), dtype="i4")
                    flags[:] = 0
                    flags.attrs.update(flag_meanings=" ".join(EXCLUDE_FLAGS),
                                       flag_masks=2 ** np.arange(len(EXCLUDE_FLAGS), dtype="int32"))
                    if kind == "aop":
                        spectral = f.create_group("sensor_band_parameters")
                        spectral.create_variable("wavelength_3d", ("bands",), dtype="i4")[:] = [410, 443, 550, 665]
                        rrs = geo.create_variable("Rrs", ("scan", "pixel", "bands"), dtype="f4", fillvalue=-32767)
                        rrs.attrs["units"] = "sr^-1"
                        rrs[:] = np.random.default_rng(number).uniform(.001, .01, (16, 16, 4))
                    else:
                        chlorophyll = geo.create_variable("chlor_a", ("scan", "pixel"), dtype="f4", fillvalue=-32767)
                        chlorophyll.attrs["units"] = "mg m^-3"
                        chlorophyll[:] = 1 + number
                        chlorophyll[0, 0] = -32767
        manifest = root / "scenes.json"
        manifest.write_text(json.dumps({"scenes": records}))
        return manifest, records

    def test_all_benchmark_arms_and_split_rejections(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, records = self.make_scenes(root)
            scenes, _ = load_matched_scenes(manifest, root, limit=2)
            self.assertFalse(scenes[0]["valid"][0, 0, 0, 0])
            args = argparse.Namespace(scenes=manifest, data_dir=root, output=root / "run",
                                      arm="spatial", steps=2, pretrain_steps=2, threads=1,
                                      batch_size=2, chip_size=8, chips=2, bands=4, stride=1,
                                      embed_dim=24, seed=0)
            for arm in ARMS:
                args.arm, args.output = arm, root / arm
                report = run(args)
                self.assertTrue(np.isfinite(report["test"]["pixel_log10_rmse"]))
                self.assertTrue((args.output / "best.pt").exists())
                self.assertEqual((args.output / "pretrained.pt").exists(), arm not in ("mlp", "spatial-scratch"))
            altered = json.loads(manifest.read_text())
            altered["scenes"][1]["bbox"] = altered["scenes"][0]["bbox"]
            manifest.write_text(json.dumps(altered))
            with self.assertRaisesRegex(ValueError, "overlap"):
                load_matched_scenes(manifest, root)
            manifest.write_text(json.dumps({"scenes": records}))
            with h5netcdf.File(root / records[1]["aop"], "a") as f:
                f.groups["sensor_band_parameters"].variables["wavelength_3d"][:] = [411, 444, 551, 666]
            with self.assertRaisesRegex(ValueError, "wavelength grid"):
                load_matched_scenes(manifest, root)
            with h5netcdf.File(root / records[1]["aop"], "a") as f:
                f.groups["sensor_band_parameters"].variables["wavelength_3d"][:] = [410, 443, 550, 665]
            for kind in ("aop", "bgc"):
                with h5netcdf.File(root / records[1][kind], "a") as f:
                    f.attrs["time_coverage_start"] = "2024-07-01T13:00:00Z"
                    f.attrs["time_coverage_end"] = "2024-07-01T13:05:00Z"
            with self.assertRaisesRegex(ValueError, "dates"):
                load_matched_scenes(manifest, root)


if __name__ == "__main__":
    unittest.main()
