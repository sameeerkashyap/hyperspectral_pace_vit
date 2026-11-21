# Single-scene discovery pilot results

The existing Earthdata download supports training. All three methods completed
on Nautilus on 2026-11-20. This pilot does not show a clear MAE advantage over
the strongest simple controls. Add data from other dates and regions before
increasing model size.

## Data and run

Input: `PACE_OCI.20240305T000858.L2.OC_AOP.V3_2.nc`. Training used all 136 native
visible reflectance bands. No BGC file or biological training label was used.
Buffered 128-pixel blocks supplied 72 training, 116 validation, and 47 test
chips. Each chip contains 16 × 16 adjacent pixels. The test data comes from
the same acquisition as training.

The run used one NVIDIA GeForce GTX 1080 Ti, PyTorch 2.5.1+cu124, seed 0,
and 1000 updates for each neural model. Validation selected PCA rank 4,
MAE step 950, and spectral autoencoder step 1000. The Job
`pace-discovery-pilot-v1-retry1` completed successfully in
`nautilus / sea-meets-the-stars`. Its first submission failed before startup
because the assigned node reported a lost GPU.

[The launch record](discovery_pilot_run.json) includes the source and input
checksums. [The experiment guide](discovery_experiment.md) describes the split,
masking, and evaluation rules.

## Held-out reconstruction

RMSE is the root mean squared reflectance error in sr⁻¹. Lower is better.
The average gives equal weight to the two masking tasks. It is not a biological
discovery score.

| Method | Hidden spectral interval | Full-spectrum spatial hole | Average |
| --- | ---: | ---: | ---: |
| Training-mean spectrum | 0.00033922 | 0.00042073 | 0.00037997 |
| Visible-data interpolation | 0.00031465 | **0.00018189** | **0.00024827** |
| PCA | **0.00021971** | 0.00042073 | 0.00032022 |
| Spectral autoencoder | 0.00025468 | 0.00039691 | 0.00032579 |
| Grouped MAE | 0.00033134 | 0.00029460 | 0.00031297 |

The MAE's average error is 2.3% below PCA. PCA is better at reconstructing
missing spectral intervals. Nearby-pixel interpolation is better at filling
spatial holes than any learned method. The MAE has not beaten those two
strong controls on their respective tasks.

## Feature checks

| Representation | Test group agreement after added noise | Neighbor overlap at 5 |
| --- | ---: | ---: |
| PCA | 84.1% | 80.0% |
| Spectral autoencoder | 87.8% | 56.9% |
| Trained MAE | 80.2% | 71.9% |
| Untrained MAE | 79.8% | 60.6% |

Noise uses reported reflectance uncertainty and assumed spectral correlation
0.8. It does not reproduce measured OCI error covariance. Retrieval uses
training blocks and separate test blocks of this same scene, not other dates.
The trained MAE improves neighbor stability over its untrained control, but
its group stability changes little. These descriptive checks do not establish
more useful optical groups than PCA.

The representative spectra mostly show differences in brightness and spectral
shape. One training-only MAE group has a narrow spike near 462 nm. Review its
source pixels and quality fields before attaching a physical meaning to it.
Group numbers do not identify phytoplankton species.

## Artifacts and next step

The full copied artifacts are in `outputs/discovery-pilot-v1/` and on PVC
`pace-experiment-results` under `/results/discovery-pilot-v1/`. They include
checkpoints, installed package versions, complete metrics, embeddings, optical
group maps, representative spectra, and `comparison.json`. The raw file is on
that PVC under `/results/data/`; the original local download remains intact.
Outputs and NetCDF files are excluded from Git.

Next, assemble diverse AOP v3.2 scenes across dates and regions, then use the
strict multi-scene manifest and existing ten-run plan. About 30 granules is a
planning target. Keep the current model and controls first. Check whether groups
repeat on unseen scenes and follow water spectra rather than acquisition
geometry, aerosol fields, or isolated band spikes. Biological interpretation
still needs independent observations.

Checks: 18 unit tests passed; one Jupyter execution test was skipped because
the local sandbox blocks kernel ports. All three real-data methods and the
comparison completed on the GPU. Other seeds, random-mask ablation, unseen
acquisitions, spatial-offset tests, and field matchups were not run.
