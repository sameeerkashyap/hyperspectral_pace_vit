# Multi-scene AOP discovery campaign

This campaign tests the existing small models on separate acquisitions and
geographic regions. It extends the [single-scene pilot](discovery_pilot_results.md).
It keeps reflectance-only training, the native 136-band visible grid, 16-pixel
chips, model sizes, losses, and 1000-update neural training unchanged.

## Frozen data selection

[The NASA candidate list](../experiments/discovery-candidates-v2.json) fixes
30 scene slots before any model is trained on this corpus:

| Split | Regions | Acquisition windows in 2024 | Scene slots |
| --- | --- | --- | ---: |
| Training | California, North Pacific, North Atlantic | Six separate four-day windows, June 1–24 | 18 |
| Validation | East Atlantic, South Indian Ocean | Three separate eight-day windows, July 1–24 | 6 |
| Test | Benguela region, South Pacific | Three separate eight-day windows, August 1–24 | 6 |

Geographic boxes do not overlap across splits. Whole acquisition dates remain
separate across splits. Each slot has up to five candidates ranked by approximate
regional coverage times the whole-granule clear fraction reported by CMR.
Coverage uses a bounding rectangle, so it is only a ranking estimate. Actual
navigation coordinates and quality flags determine whether a scene is accepted.
The first candidate with at least 16 finite, fully screened chips is accepted.
No biological label, model score, or test reconstruction result selects scenes.
The training loader then selects up to 128 chips per accepted scene.

The first attempt used three candidates on each exact date. Its first California
slot had no usable chips, so staging failed before training. The second attempt
uses the wider windows above. It retains the original quality screen and the
same split regions and months. Failed attempts are recorded separately.

This selection favors clear observations and is not a representative global
ocean survey. Date and region both change between splits; this campaign does
not separate those two effects. The original one-scene pilot file is excluded
from the new corpus.

## Access and provenance

NASA CMR collection: `C4124887023-OB_CLOUD`, AOP v3.2. Metadata searches use
[NASA's CMR API](https://cmr.earthdata.nasa.gov/search/site/docs/search/api.html).
Each candidate records its query URL, CMR granule ID, acquisition times,
download URL, reported cloud cover, and estimated file size.

The existing `earthdata-netrc` Secret in `sea-meets-the-stars` passed an
authenticated byte-range check: HTTP 206 and the eight-byte HDF5 header.
Only the staging Job mounts it. Credentials are not copied into the repository,
source bundle, training pods, or recorded outputs. Downloads use curl.
Any later transfer to NRP S3 must use AWS CLI and a private destination prefix.

Accepted downloads are written to a temporary file, opened as NetCDF, checked
against acquisition metadata and quality rules, hashed, and then renamed.
The stage records `attempts.json`, `scenes.json`, and `inventory.json`. The full
training loader checks dates, regions, units, wavelength grids, and processing
versions before it writes a `ready` marker. Existing files are never overwritten.

## Nautilus resources and artifacts

Context: `nautilus`. Namespace: `sea-meets-the-stars`.

- Staging Job: `pace-discovery-stage-v2`.
- Training code: `/results/code/discovery-v5.tar.gz` on the PVC.
- Source SHA-256: `42306420d54b540cb03dce35bf30bc83bbe269c9a3af29e7b0965e2b21322402`.
- Code archive SHA-256: `d364ab0a1a406479d4430821a48f724b3118fc5fbfa49f99a58fbe6548cb6095`.
- Storage: PVC `pace-experiment-results`, 20 GiB.
- Raw inputs: `/results/data/discovery-v2/`.
- Raw selected manifest: `experiments/discovery-scenes-v1.json` and `/results/discovery-corpus-v2/scenes.json`.
- Prepared chips and training manifest: `/results/discovery-prepared-v1/`.
- Preparation Job: `pace-discovery-prepare-v1`.
- Training Job: `pace-optical-discovery-v3`, ten indexed completions, two at once.
- Automatic comparison Job: `pace-discovery-compare-v1`.
- Training outputs: `/results/discovery-v1/`.
- Per-pod attempts: `/results/discovery-v1-attempts/`.

The first-choice downloads are estimated at 9.0 GiB. Before each download,
staging checks free storage and leaves a 4 GiB reserve for results. GPU training
starts only after preparation passes. The source snapshot uses a public Python
3.11 image. Dependencies are installed at startup; each training run records
installed package versions. The base image and all dependencies are not fully
locked. The known failed GPU node from the first pilot is excluded.

The raw download contains 30 distinct granules and totals 9.1 GiB. The verified
sample contains 2206 training chips, 657 validation chips, and 768 test chips.
Prepared arrays occupy about 516 MiB. Reading
shuffled compressed tiles directly from CephFS was slow. Preparation therefore
uses the same raw loader once, saves its exact selected arrays as compressed
NumPy files, and records hashes for both raw files and prepared files. It checks
a complete save/load round trip before writing `ready`. No values or sample
order change. Each training worker copies these smaller prepared files to its
scratch disk and verifies their hashes before loading. Split dates, geographic
boxes, native wavelengths, quality exclusions, and chip settings remain checked.

Both raw verification Jobs and preparation completed. Preparation used source
snapshot v4 (`158445760602b3108bd0d5ff11c3213f42200c328ba9fcd1eb68a5e918ab7caf`)
on the download node to reuse its file cache. Training uses the v5 bundle above.
Stage and preparation code archives are also stored under `/results/code/`.
New launch templates read code from the PVC, following
[NRP's storage policy](https://nrp.ai/documentation/userdocs/start/policies/).

The first GPU Job was preempted before producing experiment outputs. The retry
leaves priority unset, which is the normal setting for these GPUs under
[NRP's priority rules](https://nrp.ai/documentation/userdocs/running/priority-classes/).
Cluster disruptions do not count as failed experiments. Other infrastructure
failures can retry up to the count in the Job spec. Program exits 1 and 2 fail
the Job. Other exit codes use the retry budget. No elapsed-time cutoff is used.
The v2 Job ended after a faulty node later reported exit 137 for an earlier
admission failure. The v3 retry excludes that node and reuses its five completed
results. No completed experiment is overwritten or retrained.

Each pod writes to a separate attempt directory. Only a completed run is moved
to the published results directory. A retry skips an already completed result
and keeps interrupted attempts. The comparison waits for ten complete published
runs, then checks matching source hashes, prepared hashes, splits, and sample
coordinates before comparing methods and seeds.

The [ten-run plan](../experiments/discovery.json) fits PCA once and trains
spectral autoencoders, mixed-mask MAEs, and random-mask MAEs at seeds 0, 1, and 2.
Each MAE also exports an untrained encoder control. There are no elapsed-time
deadlines, workload timeouts, or automatic timed deletions.

## Interpretation

Compare per-scene spectral-gap and spatial-hole reconstruction against PCA,
the training mean, and visible-data interpolation. Compare group and neighbor
stability across seeds and noise checks. Review scan position and aerosol
associations before giving an optical group a physical interpretation.
This campaign can test transfer to new acquisitions. It cannot establish
phytoplankton species or calibrated biological concentrations.

The final comparison and completion state must come from completed run artifacts.
Field matchups, spatial-offset robustness, and separate time-only and region-only
holdouts are outside this campaign.

All ten experiments and 45 pairwise comparisons completed on 2026-11-20.
See the [measured results](discovery_multiscene_results.md) and
[machine-readable launch record](discovery_multiscene_run.json).
