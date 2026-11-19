# Sample inspection

Two public PACE OCI L2 teaching files were downloaded and opened as NetCDF4.
The full names, source URLs, sizes and SHA-256 hashes are in
[the manifest](../data/manifest.json). All fields observed in each group are in
[the generated inventory](fields.md). The files are local under `data/raw/`.

| Property | BGC sample | AOP sample |
| --- | --- | --- |
| Acquisition | 2024-06-15 18:25:49 UTC | 2024-07-30 18:11:57 UTC |
| Processing version | V1.0.0 | 2.0 |
| Grid | 1710 scans × 1272 pixels | 1710 scans × 1272 pixels |
| Main fields | `chlor_a`, `carbon_phyto`, `poc` | `Rrs`, `Rrs_unc`, `aot_865`, `angstrom`, `avw` |
| Target uncertainty | `chlor_a_unc`, `carbon_phyto_unc` | Not present |
| Shared fields | Latitude, longitude, scan time, `l2_flags` | Latitude, longitude, scan time, `l2_flags` |
| Spectral coordinates | `wavelength`: 286 sensor bands | `wavelength`: 286; `wavelength_3d`: 184 Rrs bands |

`Rrs` has shape `(1710, 1272, 184)`, with wavelength last. Its wavelength
coordinate is `sensor_band_parameters/wavelength_3d`, spanning 339–719 nm.
Do not use the 286-element sensor coordinate as its spectral axis.
The concentration maps have shape `(1710, 1272)` and units mg m⁻³.
Swath pixels have 2D coordinates; they are not a uniform latitude/longitude grid.

`Rrs` is packed as int16 with scale 0.000002 and offset 0.05 sr⁻¹. BGC `poc`
is packed with scale 0.2 and offset 6400 mg m⁻³. Fill values and valid limits
apply to the stored values before scaling. `read_valid` follows this order.
Quality flags are bit fields, not a continuous score. The code uses their
names and masks from each file instead of assuming a fixed bit number.

The BGC sample contains 182,287 valid chlorophyll values, 171,149 valid carbon
values, and 170,715 valid POC values before the selected quality screen.
The quality screen leaves 98,448 pixels. These counts are for the whole swath.

An exploratory check in this single BGC scene gives **Pearson r = 0.6550**
between log10 chlorophyll-a and log10 phytoplankton carbon, with **98,448** valid
screened pairs. This compares two derived biological estimates. It does not
measure the requested reflectance–concentration relation or validate a model.
Reproduce the check from the repository root:

```python
import h5netcdf
import numpy as np
from src.data import correlation, quality_mask, read_valid

with h5netcdf.File("data/raw/PACE_OCI.20240615T182549.L2.OC_BGC.V1_0_0.NRT.nc") as f:
    variables = f.groups["geophysical_data"].variables
    chlorophyll = read_valid(variables["chlor_a"])
    carbon = read_valid(variables["carbon_phyto"])
    valid = quality_mask(variables["l2_flags"]) & (chlorophyll > 0)
    log_chlorophyll = np.full(chlorophyll.shape, np.nan)
    log_chlorophyll[valid] = np.log10(chlorophyll[valid])
    print(correlation(log_chlorophyll, carbon))
```

## Remaining data requirement

The public release supplies these two different scenes. A same-scene counterpart
was not available at the tested public teaching URLs. A NASA download attempt
returned an HTML login page rather than a NetCDF file. An authenticated Earthdata
download is needed to get a matched AOP/BGC pair. Use the current reprocessed
collection and match acquisition and processing version; do not rely on these
old NRT files remaining available from NASA.

No reflectance–concentration result or trained accuracy is reported. The
correlation command rejects these two local samples because their times differ.
Its numerical path is tested with a small generated, aligned NetCDF pair,
including packed data, missing values and rejected quality flags.

Sources: [HyperCoast AOP sample](https://hypercoast.org/examples/pace/) and
[HyperCoast BGC sample](https://hypercoast.org/examples/pace_oci_l2/).

## Bootstrap validation

The data checks cover packed NetCDF values, flags, known correlation, and
mismatched acquisition rejection. Model checks cover patch round trips, physical
wavelength positions, band reordering, changed chip dimensions and band counts,
random masking, reconstruction loss, gradients, and invalid inputs. Both downloaded
samples were opened and inventoried. The real unmatched pair is rejected.

The project keeps its source directly under `src/` and was installed locally in
editable mode. All code belongs to this repository; there is no OETHER link or
external source dependency. A short real-data reconstruction run is documented
in [the pilot report](experiment_design.md). Independent biological accuracy has
not been tested. A clean dependency sync previously failed on package-host DNS;
the installed environment is sufficient for the local checks.
