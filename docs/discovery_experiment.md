# Run the AOP-only discovery experiment

This experiment learns optical patterns from reflectance cubes. It has no
chlorophyll head, BGC training target, or biological label in checkpoint
selection. The [research assessment](self_supervised_research.md) explains the
method choices and the limits of interpreting optical groups.

## Inputs and splits

Copy [the scene example](../experiments/discovery-scenes.example.json) and
replace its filenames, IDs, and bounding boxes with real AOP scenes. This is
an example of the schema, not a ready scientific dataset. Every record needs:

```json
{
  "id": "offshore-july-01",
  "split": "train",
  "aop": "PACE_OCI.YOUR_ACQUISITION.L2.OC_AOP.V3_2.nc",
  "bbox": [-150, 30, -140, 40]
}
```

The manifest wraps records in `{"scenes": [...]}`. Supply `train`, `validation`,
and `test` explicitly. File paths are relative to `--data-dir` and must stay
inside it. The loader rejects shared acquisition dates, overlapping geographic
boxes across splits, changed processing versions or wavelength grids, invalid
units, and invalid scene geometry. Bounding boxes use
`[west, south, east, north]`. Dateline-crossing boxes are unsupported; select a
box on one side of the date line for the inspected North Pacific granule.

Use multiple dates and optical settings, balanced across scenes. One granule
cannot supply independent acquisition splits for this benchmark. It can run
the separate spatial pilot below.
About 30 scenes is the proposed pilot scale, not a required or validated minimum.

The loader selects every native 400–700 nm band and adjacent spatial pixels.
It samples up to 128 non-overlapping 16 × 16 chips per scene, in a fixed shuffled
order shared by all runs. Every input pixel must pass the AOP quality screen and
have finite reflectance. Invalid uncertainty is retained as missing diagnostic
data, never used as a target or inverse loss weight. Packed values are decoded.
The scene sample order does not change with the model seed.

The AOP screen keeps the existing atmospheric, cloud, glint, navigation, and
other exclusions, but omits `CHLFAIL` and `CHLWARN`. Those flags describe a
chlorophyll algorithm, not an independent validity label for discovery. Reports
record every scene's flag rates and the exact exclusions. This screen remains
conservative; review its domain bias before making scientific claims.

## Models and evaluation

| Method | Training and feature extraction |
| --- | --- |
| `mae` | Grouped wavelength-aware MAE. Each token contains 2 × 2 pixels × 8 adjacent bands. Two encoder blocks, one decoder block, 96 features, four attention heads. |
| `spectral-ae` | Per-pixel denoising autoencoder. It receives visible reflectance plus observed-band indicators. It has no spatial context. Features are pooled over the same 2 × 2 support as the MAE. |
| `pca` | Training-scene covariance and linear spectral components. Rank is chosen from 4, 8, 16, and 32, capped below the band count, using validation reconstruction. Features use the same spatial pooling. |

The model grid must contain a multiple of `--group-size` bands and at least two
groups. The inspected file has 136 visible bands, so group size 8 gives
17 spectral groups and 1088 tokens per 16 × 16 chip. Other grids are rejected
if incompatible; bands are never silently dropped or invented.

Neural models use training-only scalar normalization, AdamW, gradient clipping,
and masked MSE. Mini-batches sample scenes uniformly, then chips within scenes.
Normalization gives each training scene equal weight. Default training is 1000
updates, batch size 4. These are pilot settings, not optimized values.

Mixed training hides a contiguous quarter of the band groups everywhere in the
chip and a spatial rectangle across all bands, then fills the remaining mask
budget at random to hide 75% of tokens. The random-mask ablation hides 75%
without mandatory spectral or spatial blocks. Targets never enter the encoder
through a mixed-spectrum token or a spatial convolution.

Validation and test use two separate tasks: contiguous hidden spectral intervals
and full-spectrum spatial holes. Masks are fixed per chip across methods and
seeds. Validation scene-average RMSE selects neural checkpoints and PCA rank;
the final model's test reconstruction is scored after selection. Reported
controls are the training-mean spectrum and interpolation from visible data.
Spatial holes use nearby visible pixels; spectral gaps use visible wavelengths.
PCA estimates coefficients using only observed entries. A fully hidden pixel
falls back to PCA's training mean because it has no spatial context.

Scores include per-scene and per-band RMSE and bias, separate blue/green/red
and 670–690 nm windows, masked spectral angle, and diagonal uncertainty-scaled
residuals. Uncertainty-scaled residuals are not calibrated confidence intervals.
Plain MSE can underweight weak red features; use the band/window report to
detect this before changing the loss.

## First run with the existing download

[The pilot manifest](../experiments/discovery-pilot-scenes.json) explicitly sets
`scope: single-scene-spatial-pilot`. This mode accepts exactly one full-scene
source. It keeps the multi-scene benchmark's date and geographic checks intact.
The source is `PACE_OCI.20240305T000858.L2.OC_AOP.V3_2.nc` from Earthdata.
Its SHA-256 is
`b8013c3b16d73f1cbf8121745dc30c2360d5c095124928e520df6d0470c10d1e`.

The pilot divides native scan/pixel coordinates into 128 × 128 blocks. A fixed
SHA-256 hash assigns each block to train, validation, or test with probabilities
60%, 20%, and 20%. One 16-pixel chip is left unused along every block edge.
Adjacent selected blocks therefore have at least 32 unused pixels between them.
Chips never overlap. Quality coverage makes the final split counts unequal:
72 training chips, 116 validation chips, and 47 test chips, each 16 × 16 × 136.
These counts were checked against the existing file.

[The three-run plan](../experiments/discovery-pilot.json) fits PCA and trains
the spectral autoencoder and mixed-mask MAE at seed 0. Neural training uses
1000 updates. It also saves the untrained MAE features. Retrieval compares
separate spatial blocks of the same acquisition. This run does not test new
dates or regions, identify phytoplankton species, or establish seed stability.

[The pilot Job](../nautilus/discovery-pilot-job.yaml) uses one GPU in the required
namespace and a 20 GiB results PVC. The original run used a source ConfigMap;
that snapshot is now archived at `/results/code/discovery-pilot-source.tar.gz`
on the PVC. The updated launch template verifies and unpacks that archive.
No Git credentials or Earthdata keys enter the pod.
The original local NetCDF is copied with `kubectl cp`; a checksum check gates
training. The archived source checksum is pinned in the launch template.
The base image and dependency resolution are not fully locked; the runner
records installed package versions for each method.

The archived source and input must be on the PVC before submission. Use new
Job and output names for another run; the historical names already exist.

```sh
rtk kubectl --context nautilus -n sea-meets-the-stars apply -f nautilus/pvc.yaml
rtk kubectl --context nautilus -n sea-meets-the-stars apply -f nautilus/discovery-pilot-job.yaml
```

After the pod starts, copy the file to `/results/data/pilot-input.part` with
`kubectl cp`. In the pod, rename it to the original filename and create
`/results/data/pilot-input.ready`. The startup script verifies the checksum,
then runs all three methods. Results go to `/results/discovery-pilot-v1`.
Use new Job, ConfigMap, marker, and output names for another run.

The first submission failed before container startup because the assigned node
reported a lost GPU. The retry excludes that node. No training workload was
stopped. There are no workload deadlines or automatic timed deletions.

All three pilot methods completed on 2026-11-20. See the
[results and next step](discovery_pilot_results.md).

## Launch on Nautilus

Install plotting support when preparing or testing the environment:

```sh
uv sync --extra discovery
uv run python -m unittest discover -s tests -v
```

The [multi-scene campaign](discovery_multiscene_run.md) now supplies frozen NASA
candidates and a curl staging Job. Its selected manifest is
`/results/discovery-prepared-v1/scenes.json`; raw files are under
`/results/data/discovery-v2/` on the results PVC. Preparation saves the selected
chips once; workers copy the prepared arrays to scratch storage. The plan's download list stays
empty because authenticated downloads and quality screening happen in staging.
Only the staging Job mounts the private `earthdata-netrc` Secret.

[discovery-job.yaml](../nautilus/discovery-job.yaml) verifies and unpacks a
versioned source bundle from the PVC and uses a public Python image. It installs
the `discovery` plotting extra at startup. It requests one GPU, four CPUs, and
16 GiB RAM per worker, with two workers at a time. It leaves priority unset,
requires amd64, and prefers recent GPUs. Check quota and storage.
Do not submit training until preparation completes and writes its `ready`
marker. The generic image-build path remains in [the runtime guide](running_experiments.md).

```sh
rtk kubectl --context nautilus -n sea-meets-the-stars apply -f nautilus/pvc.yaml
rtk kubectl --context nautilus -n sea-meets-the-stars apply -f nautilus/discovery-job.yaml
rtk kubectl --context nautilus -n sea-meets-the-stars get pods -l job-name=pace-optical-discovery-v3
```

The [ten-run plan](../experiments/discovery.json) runs PCA once, and the spectral
autoencoder, mixed-mask MAE, and random-mask MAE at seeds 0, 1, and 2. Each MAE
run also exports its untrained encoder as a control. There are no elapsed-time
deadlines, workload timeouts, or automatic timed deletion. Let workloads finish.
Use a new Job name and output root for subsequent campaigns.

For one entry inside the configured Nautilus environment:

```sh
SCENE_MANIFEST=/results/discovery-prepared-v1/scenes.json DATA_DIR=/results/data/discovery-v2 \
  OUTPUT_ROOT=/results/discovery-v1 \
  python -m src.runner --plan experiments/discovery.json --index 0
```

The underlying command is `python -m src.discovery --scenes MANIFEST
--data-dir DATA --method mae --output OUTPUT`. Artifacts go into a new
`OUTPUT/discovery/` directory. Existing results are never overwritten.

## Outputs and comparisons

Each `discovery/` directory contains:

- `metrics.json`: configuration, normalization, scene checksums, quality rates,
  history, validation, test, and discovery summaries.
- `best.pt` for neural models, or `model.npz` for PCA. Neural checkpoint files
  include the model configuration, wavelength grid, and reflectance scaling.
- `trained/features.npz`: patch features, optical group IDs, distance to the
  nearest group, patch coordinates, scene IDs, pooled raw spectra, audit fields,
  retrieval bank/query indices, and nearest-neighbor indices.
- `trained/groups.json`: training-group representative spectra, group counts,
  test nuisance-field summaries, and added-noise stability checks.
- `trained/prototype-spectra.png` and `trained/map-SCENE.png`: representative
  spectra and held-out maps in scan/pixel coordinates. A dot represents a
  spatial patch; blank regions are unselected or screened out.
- For MAEs, `untrained.pt` and `untrained/` provide the matching untrained control.

Feature scaling and k-means centers use a balanced bank from training scenes
only, with at most 128 patches per scene. The default eight
groups are a fixed analysis setting; change it before the final evaluation if
the validation study supports another choice. Group IDs are arbitrary. Distance
to a center is an unusual-pattern score, not biological uncertainty. Inspect
the spectra before interpreting any group.

Retrieval uses that same training bank and up to 32 queries
per test scene. It reports five neighbors from other scenes, never neighboring
test pixels. All runs use the same bank and query coordinates. Noise tests add
reported-uncertainty-scaled Gaussian noise with an assumed spectral correlation
of 0.8. They measure sensitivity to that assumption, not measured OCI covariance.

After copying completed outputs from Nautilus:

```sh
python -m src.discovery_compare outputs/discovery-v1/*/discovery \
  --output outputs/discovery-v1/comparison.json
```

The comparison checks identical inputs, splits, patch coordinates, and retrieval
sampling. It reports reconstruction scores, noise stability, adjusted Rand
agreement between cluster assignments, and nearest-neighbor overlap. Cluster
agreement accounts for arbitrary group numbering. Compare MAE seeds and the
random-mask and untrained controls before claiming robust optical patterns.

## Verification and limits

Generated AOP fixtures test all three methods without BGC files. Checks cover
packed values, quality flags, split rejection, token round trips, structured
masks, hidden-value leakage, finite gradients, training-only scaling, visible-only
PCA recovery, cross-scene retrieval, plots, and comparison artifacts.

The 30-scene, ten-run campaign completed. See its
[results](discovery_multiscene_results.md) and [protocol](discovery_multiscene_run.md).
Spatial-offset robustness, cloud-edge
distance analysis, and independent field-matchup validation remain later checks.
Audit fields are chip-level means, not extra model inputs. Only optical patterns
are claimed; group IDs are not phytoplankton species or concentration estimates.
