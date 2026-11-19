# Full-spectrum patches and a biological prediction benchmark

The first supervised milestone tests agreement with PACE's satellite-derived
chlorophyll product. It does not establish independent phytoplankton detection.
The original reconstruction pilot remains available and its historical results
are unchanged.

## Full-spectrum spatial model

```text
Rrs input: B × C × H × W
              |
              v
Conv2d: C → D channels, 4 × 4 kernel, stride 4
Each token sees all C wavelengths in a spatial patch
              |
              v
B × ((H/4) × (W/4)) × D tokens
Add row and column position encodings
              |
       +------+---------------------------+
       |                                  |
PRETRAINING                          CHLOROPHYLL REGRESSION
Hide 75% of spatial tokens           Encode every spatial token
       |                                  |
Transformer encoder                  Same transformer encoder
       |                                  |
Restore sequence with mask tokens    Linear head: D → 16 pixel predictions
Add spatial positions again               |
       |                             Rearrange to B × 1 × H × W
Transformer decoder                       |
       |                             Predict log10 chlorophyll-a
Linear head: D → 16 × C values             |
       |                             MSE on valid log10 BGC targets
Hidden-patch reconstruction MSE
```

With C=172 and D=96, each token projects 2752 reflectance measurements into
96 features. A 128 × 128 chip produces 1024 tokens instead of 176,128 in the
wavelength-token model. The reduction is exactly 172-fold; full attention's
cost grows with the square of token count. Pretraining sends only about 25%
of tokens to the encoder, but its decoder still processes all spatial tokens.

Source: [spatial_model.py](../src/spatial_model.py). The `Conv2d` projection
learns spectral relationships immediately. This is a meaningful compression
bottleneck. The main transformer plan tests D=96, and a separate plan repeats the spatial
pretrained/scratch comparison at D=192. A small spectral encoder before
compression is deferred until these baselines are measured.

This model has **fixed channel meanings**. Its constructor records the expected
physical wavelengths, and each call rejects another grid or order. The
wavelength list is a check, not a spectral positional embedding. This version
does not inherit the old model's variable-grid capability. Align sensor grids
explicitly before transfer; do not load a shifted grid into the same projection.
The loader reads the actual band centers rather than assuming PACE has 172 bands.

## Comparison arms

| CLI arm | Pretraining | Chlorophyll predictor |
| --- | --- | --- |
| `mlp` | None | Two-hidden-layer per-pixel spectral MLP |
| `spectral` | Original wavelength-specific random masking | Original encoder, pool wavelength features at each spatial patch, dense head |
| `spatial` | Full-spectrum spatial tokens, 75% spatial masking | New spatial encoder and dense head |
| `mixed` | Explicit wavelength tokens; contiguous spectral span plus whole spatial patches | Same encoder/head as `spectral` |
| `spatial-scratch` | None | Exactly the same encoder/head initialization and configuration as `spatial` |

`mixed` retains separate wavelength tokens. It uses the original joint-attention
encoder, not a new factorized-attention implementation. It first chooses a
contiguous span covering about one-quarter of bands and about one-quarter of
whole spatial patches. The union is hidden; additional random tokens bring the
total to 75%. Mixed masking requires at least two bands, two spatial patches,
and a masking ratio of at least 75%. It is not band masking after spectral collapse.

The main plan runs five arms × three seeds = 15 experiments. All arms use the
same scenes, band subset, chips, normalization, downstream batch sampling, and
supervised budget. The `spatial` and `spatial-scratch` pair isolates the effect
of pretraining. The MLP and different token designs do not have equal parameter
counts; reports include both total and supervised parameter counts. Do not
attribute a cross-architecture gain solely to pretraining.

The main plan gives the MLP 576 hidden features to bring its capacity closer to
the spatial model when using roughly 150–172 bands. The exact count depends on
the selected grid. All transformer arms use 96 features. These settings are
fixed before evaluation; use the reported counts to assess remaining differences.

## Prediction and loss

[chlorophyll.py](../src/chlorophyll.py) outputs **log10 concentration** directly,
with shape B × 1 × H × W. Convert with `10 ** prediction` when concentration
in mg m⁻³ is needed. Direct log prediction avoids negative concentration outputs
and a logarithm of an unconstrained prediction during training.

```text
loss = mean((predicted_log10_chlor_a - log10(target_chlor_a))²)
                          over valid target pixels only
```

This is the base-10 form of the proposed log-space objective. Invalid, missing,
zero and flagged labels are excluded before taking the logarithm. Input chips
must have finite reflectance and pass AOP quality screening. Partial target
coverage is allowed. Empty-target chips are rejected. There is no uncertainty
head, inverse optical-parameter distribution, or physics reconstruction loss.

## Required scene manifest

Copy [the example manifest](../experiments/chlorophyll-scenes.example.json) and
replace all filenames and boxes with your matched, current-version AOP/BGC
scenes. Each bounding box is `[west, south, east, north]` in degrees. It defines
the area from which chips may be read, not just a label. Dateline-crossing boxes
are unsupported in this first loader.

The loader checks:

- Each pair has equal acquisition times, version, instrument and full navigation grid.
- Every split is present. Scene IDs are unique, and geographic boxes do not
  overlap across train, validation and test.
- Acquisition dates do not occur in more than one split. This intentionally
  enforces both temporal and geographic separation rather than accepting random chips.
- All scenes share one processing version and the same selected wavelength grid.
- Quality flags, packed-value validity and positive chlorophyll targets are applied.

Default inputs use **all native 400–700 nm bands**. `--bands 25` is an explicit
smaller spectral test, shared across all arms. `--stride 1` keeps adjacent pixels;
a larger stride makes chips spatially subsampled and is recorded in the report.
The loader streams small spectral chips instead of loading a full hyperspectral
swath into RAM. Up to 64 non-overlapping chips are retained per scene, in scan
order. For publication, increase scene coverage and assess sampling bias.

The two teaching downloads in this workspace are not a matched pair. They are
not valid inputs for this supervised benchmark. No synthetic label is substituted
for real data when running the benchmark.

## Run

```sh
uv run python -m src.benchmark --scenes /path/to/scenes.json \
  --data-dir /path/to/netcdf --arm spatial --output outputs/chlorophyll-spatial

SCENE_MANIFEST=/path/to/scenes.json uv run python -m src.runner \
  --plan experiments/chlorophyll.json --data-dir /path/to/netcdf \
  --output-root outputs/chlorophyll-v1
```

Use the existing lightweight Docker image and
[Nautilus chlorophyll Job](../nautilus/chlorophyll-job.yaml). Mount the real
manifest at `/results/chlorophyll-scenes.json`. Its files are expected under
`/results/data`. The plan's `datasets` list is empty because no valid matched
download URLs/checksums have been supplied. Add the actual public HTTPS URLs,
filenames and SHA-256 values there for automatic runtime download. Authenticated
Earthdata inputs need a custom download script and mounted credentials, as
described in [the runtime guide](running_experiments.md). Neither datasets nor
credentials are baked into the image.

The main plan uses 200 pretraining steps and 200 supervised steps, with 2 encoder
blocks, 1 decoder block, 4 heads, 4 × 4 patches, and 8 × 8 input chips. These are
small pilot settings, not optimized training. The wavelength-token arms can
still be expensive at large chip sizes. The optional
[wide plan](../experiments/chlorophyll-wide.json) repeats the spatial pair with
192 features across three seeds. Set Nautilus `completions: 6` for that plan.
Use namespace `sea-meets-the-stars`. Change the image tag and output root before submitting another run.

Only training inputs fit the scalar reflectance normalization. Only training
scenes are used for pretraining and supervised optimization. Validation
scene-average log10 RMSE selects the best supervised checkpoint. The test set
is evaluated once after selection. The decoder is unused and frozen during
supervised fine-tuning. All encoder weights and the dense head are fine-tuned.

Outputs include `pretrained.pt` for pretrained arms, `best.pt`, and `metrics.json`.
The generic runner also saves logs, installed package versions and execution
status. Reports record wavelengths, normalization, input checksums, boxes, dates,
chip coordinates, valid-pixel counts, budgets, model size, and per-scene log10
RMSE, bias and absolute error. Seeds are paired across arms; pretraining costs
extra compute and should be reported separately from the supervised budget.

## Interpretation and validation status

The implementation is tested with small generated, matched NetCDF fixtures.
All five arms run pretraining where applicable, dense supervised updates,
validation selection and held-out evaluation in the test. Tests also verify
the 1024-token/172-band example, mixed masks, finite gradients, target masking,
and rejection of leakage and mismatched grids.

No real matched-scene biological benchmark has run yet. Reconstruction results
in the earlier pilot are not chlorophyll accuracy measurements. Evidence of
usefulness would be paired improvement over `spatial-scratch`, consistency
across held-out scenes, and then gains on independent in-situ observations.
Independent uncertainty calibration remains a later experiment. BING-style
distributional inversion is deferred as requested; none of these point predictors
resolves the physical ambiguity described in the paper.
