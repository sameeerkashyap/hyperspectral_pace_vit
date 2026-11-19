import tempfile
import unittest
from pathlib import Path

import h5netcdf
import numpy as np

from src.correlate import analyze
from src.data import EXCLUDE_FLAGS, correlation, read_valid


class BootstrapTests(unittest.TestCase):
    def test_pair_quality_packing_and_correlation(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ("aop.nc", "bgc.nc")]
            for path in paths:
                with h5netcdf.File(path, "w") as root:
                    root.dimensions = {"scan": 2, "pixel": 3, "band": 2}
                    root.attrs.update(time_coverage_start="start", time_coverage_end="end",
                                      processing_version="test", platform="PACE", instrument="OCI")
                    nav = root.create_group("navigation_data")
                    for name in ("latitude", "longitude"):
                        var = nav.create_variable(name, ("scan", "pixel"), dtype="f4")
                        var[:] = np.arange(6).reshape(2, 3)
                    geo = root.create_group("geophysical_data")
                    flags = geo.create_variable("l2_flags", ("scan", "pixel"), dtype="i4")
                    flags.attrs.update(flag_meanings=" ".join(EXCLUDE_FLAGS),
                                       flag_masks=2 ** np.arange(len(EXCLUDE_FLAGS), dtype="int32"))
                    flags[:] = [[1, 0, 0], [0, 0, 0]]
                    if path == paths[0]:
                        spectral = root.create_group("sensor_band_parameters")
                        spectral.create_variable("wavelength_3d", ("band",), dtype="f4")[:] = [443, 550]
                        rrs = geo.create_variable("Rrs", ("scan", "pixel", "band"), dtype="i2", fillvalue=-32767)
                        rrs.attrs.update(scale_factor=.001, add_offset=.01, valid_min=0, valid_max=100)
                        rrs[:] = np.stack([np.arange(6).reshape(2, 3)] * 2, axis=-1)
                        rrs[0, 1, 0] = -32767
                    else:
                        for name in ("chlor_a", "carbon_phyto"):
                            geo.create_variable(name, ("scan", "pixel"), dtype="f8")[:] = 10. ** np.arange(6).reshape(2, 3)
            result = analyze(*paths, stride=1)
            self.assertEqual(result["bands"][0]["chlor_a"]["n"], 4)
            self.assertAlmostEqual(result["bands"][0]["chlor_a"]["pearson_r_log10_target"], 1.)
            with h5netcdf.File(paths[0]) as root:
                decoded = read_valid(root.groups["geophysical_data"].variables["Rrs"])
                self.assertTrue(np.isnan(decoded[0, 1, 0]))
                self.assertAlmostEqual(decoded[1, 2, 0], .015)
            with h5netcdf.File(paths[1], "a") as root:
                root.attrs["time_coverage_start"] = "other scene"
            with self.assertRaisesRegex(ValueError, "time_coverage_start"):
                analyze(*paths)

    def test_empty_or_constant_correlation(self):
        self.assertIsNone(correlation(np.ones(3), np.arange(1, 4))["pearson_r_log10_target"])
        self.assertEqual(correlation(np.array([1., 2.]), np.array([0., np.nan]))["n"], 0)


if __name__ == "__main__":
    unittest.main()
