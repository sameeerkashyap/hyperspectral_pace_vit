# Transfer between EMIT and PACE

This note describes the original wavelength-token model. The newer
[full-spectrum spatial model](chlorophyll_benchmark.md) uses a fixed input grid
and does not automatically retain the variable-band transfer property.

OETHER's `hyperspectral_encoder/model_definition.py` uses `arange(num_c)` for
spectral position. It groups 15 adjacent bands and fixes the input at 285 bands.
Changing sensors moves physical wavelengths between those groups. Changing only
the positional encoding does not fix the fixed input projection or decoder.

This repository rewrites that design in `src/model.py`. All source is
local; OETHER is only a design reference. The new model retains random masking,
transformer encoding and decoding, patch reconstruction, and masked MSE loss.

Each token covers a spatial patch at one wavelength. A shared projection accepts
any band count. Row, column and physical wavelength encodings occupy separate
parts of the embedding. The wavelength encoding uses fixed nm units, so 550 nm
has the same position in either sensor. Chip dimensions come from the input,
rather than fixed EMIT constants. Tests cover patch round trips, wavelength
alignment, changed band count, masking, reconstruction loss and gradients.

This single-band token design changes the reference's projection and decoder
shapes. Its checkpoints cannot be loaded directly. Compare it against grouped
spectral tokens before claiming a transfer benefit. Spectral response width and
cross-sensor radiometric differences remain unresolved.

PACE OCI and EMIT overlap in visible wavelengths. The downloaded PACE AOP file
has 184 `Rrs` bands from 339 to 719 nm. Do not equate the full OCI sensor range
with the ocean-color `Rrs` range. Read EMIT's actual band centers and valid-band
mask before computing the overlap. EMIT's commonly cited range is about
380–2500 nm; its atmospheric absorption bands need care.

Absolute wavelength is one part of transfer. Band width and the sensor spectral
response also matter. A narrow and broad band with the same center can measure
different values. EMIT surface reflectance and PACE water-leaving remote-sensing
reflectance describe different physical quantities. They also have different
spatial resolution, viewing geometry, noise and atmospheric correction. Do not
concatenate their raw values into one training set as equivalent measurements.
Do not normalize each scene's wavelength list to its own min/max: that would
give the same wavelength different positions across sensors.

## Experiments

1. Establish per-band correlations and a simple band-ratio regression baseline
   on matched, screened PACE scenes. Predict log10 chlorophyll or carbon.
2. Compare band-index position with absolute-nm position on identical splits.
   Test held-out bands, changed sampling, and held-out regions and dates.
3. Compare common-grid interpolation with native-band encoding. Use measured
   spectral response functions when available. Do not extrapolate outside overlap.
4. Pretrain on separate sensor datasets, then compare frozen encoder, small
   adaptation, and full fine-tuning. Report the data and training cost for each.
5. Test band width or response-aware embedding only after the center-wavelength
   baseline works. Compare uncertainty and errors against independent matchups.

A possible research question is whether response-aware spectral position and
mixed-sensor pretraining improve ocean-color transfer under shifted band grids.
Wavelength-aware transformers already exist; absolute wavelength alone is not
a new contribution. Transfer without full retraining is a hypothesis to measure.

## Sources

- [NASA OCI file structure](https://nasa.github.io/oceandata-notebooks/notebooks/oci/oci_file_structure.html)
- [NASA OCI data access](https://nasa.github.io/oceandata-notebooks/notebooks/oci/oci_data_access.html)
- [PACE product table](https://pace.oceansciences.org/data_table.htm)
- [DOFA: Neural Plasticity-Inspired Multimodal Foundation Model for Earth Observation](https://arxiv.org/abs/2403.15356)
  uses a wavelength-conditioned dynamic network to construct embedding weights.
- [HyperFree, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Li_HyperFree_A_Channel-adaptive_and_Tuning-free_Foundation_Model_for_Hyperspectral_Remote_CVPR_2025_paper.html)
  uses a wavelength weight dictionary to handle different spectral channels.
