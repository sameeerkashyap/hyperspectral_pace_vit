# What this experiment can establish

The first test asks whether the model learns reflectance structure on a small
PACE sample and handles a changed wavelength grid. Biological retrieval needs
a separate test with independent observations.

## Paper assessment

[Prochaska and Frouin (2025)](https://bg.copernicus.org/articles/22/4705/2025/)
use Bayesian inversion, MCMC and model comparisons on 3320 modeled optical
spectra. Their simulated PACE analysis uses 400–700 nm, 5 nm sampling and an
uncertainty estimate from one L2 granule. Under their assumptions, multispectral
data constrain about three optical parameters; PACE-like data support about
four, sometimes five. These are study-specific results, not universal limits
on every observation.

The governing reflectance approximation depends on the ratio of total
backscattering to absorption. Distinct optical properties can give the same
spectrum. CDOM/detritus can also mimic phytoplankton absorption. More wavelengths
do not guarantee a unique retrieval. The authors assumed a perfect forward
model and uncorrelated errors in key simulations. They recommend better priors,
independent observations and uncertainty treatment. Machine learning inherits
its training distribution as an implicit prior; it cannot recover distinctions
absent from its input. See sections 2.2–2.4, 3.2–3.3 and 4–5.

**Design conclusion:** test spectral representation first. Reconstruction is not
evidence of phytoplankton identification. Test the biological task later with
independent labels and explicit uncertainty.

## Executable pilot

Entry point: [src/experiment.py](../src/experiment.py).
Plan: [experiments/sample.json](../experiments/sample.json).

| Choice | Implementation |
| --- | --- |
| Source | Public 2024-07-30 AOP granule, verified by SHA-256 at runtime |
| Bands | 25 native band centers spaced across 400–700 nm; no invented EMIT data |
| Train/test wavelength grids | Train on 13 alternating centers; test also on the 12 interior centers never used in training |
| Spatial split | Deterministic hash of 32 × 32 sampled-pixel blocks; 60/20/20 assignment to train/validation/test |
| Sampling | Every fourth scan and pixel; 4 × 4 sampled-pixel chips, up to 64 per split |
| Guard | A one-chip margin around each block; chips cannot cross block boundaries |
| Quality | Flags, finite Rrs and positive finite Rrs_unc at every selected chip pixel |
| Scaling | One mean and standard deviation fitted only on training chips and training bands |
| Model | Same MAE in both arms: dimension 48, 2 encoder blocks, 1 decoder block, 4 heads, 2 × 2 spatial patches |
| Arms | Physical wavelength values in nm versus positive band indices; all other settings identical |
| Optimization | AdamW, learning rate 0.001, 500 steps, batch size 8, 75% token masking |
| Replicates | Seeds 0, 1, 2 for each arm; six independent jobs |
| Model selection | Lowest validation RMSE on the training wavelength grid; fixed validation masks |
| Test scoring | Fixed masks, identical across arms for each grid; test data never used for checkpoint selection |

These chips contain spaced samples, not contiguous full-resolution pixels.
Blocks are defined in swath coordinates, not equal-distance map coordinates.
The small single-scene split limits leakage but cannot establish geographic,
temporal, or sensor generalization. Split membership and source coordinates are
saved. Inspect how many distinct blocks remain after quality filtering.

### Metrics and controls

- Report hidden-value RMSE in sr⁻¹ on the seen and unseen grids. Compare arms
  seed by seed; do not select the seed with the best test score.
- Score a training-mean spectrum baseline on the same masks. Interpolate the
  training mean to unseen wavelengths.
- Score per-pixel linear interpolation from visible test bands on the same mask.
  Edge values use nearest endpoints. If a spatial patch has no visible bands,
  fall back to the training mean. This measures how far smooth spectra alone go.
- Repeat scoring after adding Rrs_unc-scaled Gaussian noise with spectral
  correlation 0.8. This is an added-noise sensitivity test; actual observations
  already contain noise. It is not an estimate of OCI's full error covariance.
- Report uncertainty-normalized residual RMSE. The MAE is not a probabilistic
  model, so this is not confidence-interval coverage or a calibrated chi-squared.
- An algebraic negative control scales total absorption and backscattering
  together by three. The paper's forward equations give unchanged reflectance.
  This demonstrates the information limit; it is not an ocean constituent model
  or a reproduction of BING/L23.

### Decision after the six-run pilot

Check finite losses and gradients, disjoint block IDs, sufficient valid chips,
and stable validation behavior. For a provisional go/no-go, require each arm
to beat the training-mean baseline on seen-grid test data. A candidate wavelength
benefit should improve unseen-grid RMSE against the index arm in all three paired
seeds, without a greater than 10% seen-grid regression. These are engineering
screening thresholds, not statistical significance criteria or mission requirements.
Also compare both arms with interpolation. If a simple baseline wins, keep it.

Failure is informative. Increase training steps only if optimization is still
improving; add scenes before making scientific claims. Do not tune against this
test set. A native-band split changes sampling within one sensor, not its spectral
response or atmospheric correction, so it cannot validate EMIT-to-PACE transfer.

## Local pipeline check already run

CPU, seed 0, 30 steps, at most 16 chips per split. This checks the executable
path with real sample data; it is shorter than the six-run experiment above.

| Arm | Seen-grid RMSE | Unseen-grid RMSE |
| --- | ---: | ---: |
| Wavelength | 0.001931 | 0.002049 |
| Band index | 0.002042 | 0.002119 |
| Training mean | 0.002340 | 0.002189 |
| Visible-band interpolation | 0.003291 | 0.003510 |

Units: sr⁻¹. Full results and split coordinates are in
[pilot_results.json](pilot_results.json). Both arms beat the mean in this short
run. The wavelength arm's unseen-grid error is about 3.3% lower than the index
arm's. One seed and one scene do not establish a reliable advantage.
The large uncertainty-normalized residuals also show that these predictions
are far from reconstructing within the reported measurement uncertainty.

## Biological validation after the architecture test

1. Collect matched current AOP/BGC granules across dates and water types.
   Freeze held-out dates and regions. Satellite BGC targets test agreement with
   the processing algorithm, not independent biological truth.
2. Obtain independent chlorophyll/optical matchups from
   [SeaBASS](https://seabass.gsfc.nasa.gov/). Carbon validation needs suitable
   independent carbon measurements; chlorophyll labels cannot substitute for them.
3. Compare a band-ratio baseline, a low-rank spectral regression baseline,
   a frozen encoder with a small head, and full fine-tuning. Report log-space
   RMSE, bias, errors by concentration/water type, and bootstrap intervals by scene.
4. Test uncertainty coverage with held-out observations and measurement errors.
   Report upper limits or unresolved cases when observations do not distinguish
   biological hypotheses. Test CDOM-rich and low-chlorophyll regimes separately.
5. Compare 400–700 nm with an explicitly justified near-685 nm fluorescence
   feature, then external location/season constraints. Extra information must
   be isolated in ablations. Do not attribute a gain from priors to wavelengths.

The runner can host each stage as a custom script or notebook. This document
covers the original reconstruction pilot. The later
[chlorophyll benchmark](chlorophyll_benchmark.md) implements dense supervision
and pretraining comparisons; it still needs real matched-scene inputs.
