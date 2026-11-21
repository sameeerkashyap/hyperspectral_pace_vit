# Self-supervised discovery from PACE OCI spectra

Research date: 2026-11-19. Scope: discover phytoplankton-related spectral
patterns and optical water types from L2 OC_AOP cubes without fitting
chlorophyll regressors. No model was trained
and no job was launched for this memo. Published findings below are separate
from proposed tests and recommendations.

Implementation update: the first experiment is now available in the
[discovery guide](discovery_experiment.md). Generated-fixture checks exercise
the pipeline. The [30-scene campaign](discovery_multiscene_results.md) has now
completed. PCA wins spectral-gap reconstruction and interpolation wins spatial
holes on that corpus. The literature assessment below remains separate from
these measured results; biological interpretations still require field evidence.

## Recommended first approach

**Recommendation:** start with one small wavelength-aware masked autoencoder
(MAE), with narrow contiguous band groups as tokens. Train it to recover hidden
spectral intervals and spatial patches from
valid OCI reflectance. Use its frozen embeddings for nearest-neighbor searches
and optical water-type discovery. An embedding is a short vector that describes
a spectrum or patch. Compare it with PCA and a small spectral autoencoder before
adding a teacher model or other objectives.

This is a testable starting choice, not a proved best method. It fits the data,
the current code, and a direct check of retained spectral information. The
literature supports spectral positions and masks across both dimensions. It
does not establish that an OCI MAE discovers phytoplankton identities better
than DINO, JEPA, or simple spectral methods.

```text
quality-screened OCI Rrs cubes + physical wavelengths
                        |
               hide bands and patches
                        |
                  small MAE encoder
                    /          \
        recover hidden Rrs      frozen embeddings
                    |                  |
      independent test scenes    neighbors + optical groups
                                       |
                       nuisance checks and field associations
```

The immediate products are an optical similarity map, representative spectra,
and a search tool for repeated or unusual patterns. Species names and measured
constituent concentrations require further evidence.

## What the observations can support

NASA states that apparent optical properties depend on optically active
materials and on illumination, viewing angle, and the atmosphere. Its Rrs
product is obtained after quality control and atmospheric correction. Therefore,
a coherent learned feature can reflect biology, dissolved material, particles,
or an observation artifact. [NASA AOP](https://pace.oceansciences.org/pace_aop.htm),
[NASA Rrs validation](https://pace.oceansciences.org/pace_rrs.htm).

There is direct evidence that unsupervised optical groups can relate to
phytoplankton assemblages. Uitz et al. used field pigment measurements and
hyperspectral absorption and Rrs. Their optical clusters agreed with pigment
groups, with stronger agreement for phytoplankton absorption than for Rrs.
Their EOF analysis linked optical variation to pigment composition, cell size,
and intracellular pigment concentration. This supports testing field
associations. It does not turn arbitrary satellite clusters into species.
[Uitz et al. (2015), original study](https://doi.org/10.1016/j.rse.2015.09.027).

An earlier study applied unsupervised hierarchical clustering to pigments,
absorption, and Rrs in open-ocean conditions. It provides a useful classical
comparison before deep learning. [Torrecilla et al. (2011)](https://doi.org/10.1016/j.rse.2011.05.014).

Jia et al. produced 15 fuzzy optical water types from a **synthetic** PACE
hyperspectral dataset. Fuzzy membership allows a spectrum to belong partly to
more than one optical class. This is relevant to mixed waters and gradual
transitions. The number 15 is their fitted scheme, not a required number of
natural classes or observed OCI taxa. [Jia et al. (2021)](https://www.mdpi.com/2072-4292/13/19/4018).

**Inference:** describe new groups first by their observed properties, such as
spectral shape, amplitude, fluorescence-region contrast, and spatial setting.
Use names such as “optical group 4” until independent evidence supports a more
specific interpretation. A good reconstruction score alone does not resolve
the non-unique optical inverse problem discussed in the existing
[experiment assessment](experiment_design.md).

Locatello et al. show that unsupervised recovery of separate explanatory
factors needs assumptions about models and data. This is a general result,
not an OCI information bound. **Inference:** a latent axis cannot be called
“CDOM” or “diatoms” merely because it varies smoothly or has an attractive map.
[Original paper](https://proceedings.mlr.press/v97/locatello19a.html).

## Evidence for model choices

| Method | Primary-source result | Decision for OCI discovery |
| --- | --- | --- |
| MAE / SatMAE | SatMAE uses band groups, spectral position encodings, and temporal masking. Its evidence is mainly supervised transfer on satellite benchmarks. [Paper and official project](https://sustainlab-group.github.io/SatMAE/) | Reuse the idea of physical spectral positions. Test masks that prevent easy copying from nearby wavelengths. Temporal fusion can wait until repeated, aligned scenes are available. |
| SpectralGPT | Uses 3D spatial-spectral tokens and reconstruction. Official pretraining uses fMoW-Sentinel and BigEarthNet; transfer includes scene classes, segmentation, and change detection. [Paper](https://arxiv.org/abs/2311.07113), [official code](https://github.com/danfenghong/IEEE_TPAMI_SpectralGPT) | Useful design evidence. Its published checkpoints are not demonstrated OCI water encoders. Do not assume a model called “spectral” was trained on hundreds of ocean bands. |
| General Purpose Spectral Foundation Model | Uses single-band spatial tokens, wavelength and bandwidth encoding, and both spatial tube and whole-channel masking. It studies 25–100 sampled channels. The manuscript still lists a future code URL. [Laprade et al. (2025), full paper](https://arxiv.org/html/2503.01628v1) | A closer architectural match to the repository's wavelength model. The authors acknowledge unequal weight-update counts in the channel-count comparison. Its best 25-channel result does not prove that OCI should discard bands. |
| HyperspectralMAE | The preprint reports dual spatial/band masking, wavelength Fourier positions, and MSE plus spectral-angle reconstruction loss. Evaluation transfers to Indian Pines land-cover classes. [Jeong et al. (2025)](https://arxiv.org/abs/2505.05710) | Supports a dual-mask hypothesis. This memo did not reproduce its results or verify a released implementation. Treat it as design evidence, not an established ocean benchmark. |
| OceanMAE | The preprint uses 11 Sentinel-2 bands and external descriptors, including bathymetry, chlorophyll level, and Secchi depth, during reconstruction. It evaluates marine segmentation and bathymetry. [Stamer et al. (2026), §III-A2](https://arxiv.org/html/2604.08171v1) | Relevant evidence for training on ocean data. Keep descriptor fusion out of the first AOP-only discovery experiment. Injected descriptors would change the question and could make a later “discovery” association circular. This is not a flaw in its stated transfer task. |
| DOFA | A wavelength-conditioned dynamic network accepts different sensor inputs. The paper evaluates transfer across Earth-observation modalities. [Paper](https://arxiv.org/abs/2403.15356), [official code linked by authors](https://github.com/zhu-xlab/DOFA) | Useful if cross-sensor transfer becomes a goal. It adds machinery that one fixed OCI spectral grid does not yet need. Wavelength support alone does not validate transfer of water optics. |

### Why defer DINO, iBOT, JEPA, and contrastive learning

These are valid alternatives, with different assumptions. None of the cited
results establishes a winner for OCI AOP discovery.

| Alternative | What the original work establishes | OCI-specific concern and next use |
| --- | --- | --- |
| DINO | Teacher/student self-distillation across image views yields useful visual features and emergent image segmentation. [Original paper](https://arxiv.org/abs/2104.14294), [official code](https://github.com/facebookresearch/dino) | Recommendation: test later if MAE embeddings retain nuisance detail or give poor retrieval. RGB color jitter can remove the very spectral differences of interest. Distant crops of one ocean image can also contain different waters. Define valid matching views first. |
| iBOT | Adds masked patch-token self-distillation through an online teacher tokenizer, together with class-token self-distillation. [Original paper](https://arxiv.org/abs/2111.07832), [official code](https://github.com/bytedance/ibot) | Recommendation: the most direct later teacher-based comparison for local optical structure. It adds teacher updates and view choices. Do not add it to MAE merely because both are available. |
| I-JEPA | Predicts target-block representations from context. Its original image results depend on large target blocks and informative distributed context. [Original paper](https://arxiv.org/abs/2301.08243), [official code](https://github.com/facebookresearch/ijepa) | Inference: latent prediction may suppress measurement detail, which can help robustness but could discard weak pigment features. It needs a spectral-information test independent of latent loss. A later comparison is reasonable once data and tests are fixed. |
| Contrastive learning, such as SimCLR | Learns agreement between augmented views. The original study shows that augmentation composition strongly affects results. [Original paper](https://proceedings.mlr.press/v119/chen20j.html), [official code](https://github.com/google-research/simclr) | Inference: same-location dates are not guaranteed to have the same biology. Different water pixels can have similar optics, so treating all other samples as negatives can be wrong. Defer until matching views have a physical justification. |

No paper result licenses arbitrary intensity changes, independent band jitter,
or treating every crop of a granule as the same optical object. For the initial
MAE, use masking and conservative spatial transforms. Test their effect on
spectra before calling them harmless augmentations.

### Keep simple baselines

**Recommendation:** PCA is the first baseline. Fit its mean and basis only on
training scenes. Compare reconstructed spectra and neighbors at several small
ranks selected on validation data. This tests whether the useful structure is
mostly linear. The EOF analysis in Uitz et al. is a direct ocean precedent for
examining dominant optical modes.

NMF learns nonnegative additive factors. [Lee and Seung's original
paper](https://www.nature.com/articles/44565) establishes that representation,
not a unique physical unmixing of water constituents. **Recommendation:** use
NMF only as an optional interpretation baseline if the chosen valid input is
nonnegative. Do not silently clip negative Rrs to make the solver run. Report
the excluded domain. A factor is not automatically a pigment spectrum or its
coefficient a constituent concentration.

A small per-pixel autoencoder is the next baseline. It tests nonlinear spectral
compression without spatial context. Neural bottlenecks are an established
way to reduce dimension. [Hinton and Salakhutdinov (2006), author publication
record](https://www.cs.toronto.edu/~rsalakhu/bibtex/HinSal06.html).
**Recommendation:** give it the same training data and embedding width as the
first MAE comparison. Do not launch a broad ensemble. If PCA gives equally
useful, stable groups and neighbors, keep PCA.

## Data and experiment plan

The current task's inspected granule is
`PACE_OCI.20240305T000858.L2.OC_AOP.V3_2.nc`: 1709 × 1272 pixels,
172 Rrs bands from 346–719 nm, including 136 native centers from 400–700 nm.
Inspection during this task found about 12% of pixels pass the current strict
flag screen. These are measurements of the local file, not literature results.
They differ from the older teaching sample
in [sample_findings.md](sample_findings.md).

**Phase 1: freeze the data and tests.** Assemble a modest multi-scene dataset
with clear offshore water, productive water, and coastal or optically complex
water across dates and regions. Count retained pixels and usable contiguous
chips by scene. Hold out whole geographic areas and dates, with a guard against
overlapping chips and neighboring swaths. One large granule cannot establish
generalization. Keep a separate final test set that does not select masks,
cluster counts, normalization, or checkpoints.

An initial planning target is roughly 30 diverse AOP granules, divided by
geography and date into training, validation, and test groups before chip
extraction. The count is a pilot budget, not a statistically justified minimum.
Balance retained chips across scenes and optical settings so clear offshore
water cannot dominate solely through pixel count. Add scenes when usable
coverage is insufficient. BGC files are not required for pretraining, embedding
extraction, or clustering; optional BGC comparisons happen only after the
discovery pipeline is frozen.

Read [data access](data_access.md) before changing ingestion and
[runtime instructions](running_experiments.md) before experiments. Use one
reprocessing version and read native wavelength coordinates and flags from each
file. Do not count interpolated channels as new measurements. Preserve raw Rrs
and validity masks. Fit all scaling on training data. Choose 400–700 nm as a
controlled starting domain, then test the extra UV bands separately when their
quality supports it; this is a proposal, not a finding that UV lacks value.

**Phase 2: establish baselines and one MAE.** Compare training-mean spectra,
visible-band interpolation, PCA, and a small spectral autoencoder. Train one
small wavelength-aware MAE using the existing code where possible. A full-spectrum
token is convenient for spatial learning; a per-band or narrow-band-group token
allows explicit spectral hiding. Test that target values cannot enter the
encoder through another input path. Invalid observations must not become
reconstruction targets or be filled from held-out data.

Start from `WavelengthAwareMAE` in [src/model.py](../src/model.py). Its single-band
tokens need a band-group extension. `SpatialSpectralMAE` in
[src/spatial_model.py](../src/spatial_model.py) projects the full spectrum into
each spatial token; hiding wavelengths after this projection would leak them.
Both encoders already expose features, so discovery needs no regression head.
An initial engineering setting is a 16 × 16 native-pixel chip, 2 × 2 spatial
patches, and 8 adjacent native bands per token. For 136 bands this gives
8 × 8 × 17 = 1088 tokens before masking. Try a small encoder width such as 96;
these values are feasible starting points, not optimized choices.

Keep each group's true wavelength coordinates and check the selected grid across
scenes. The 136-band count describes this inspected file, not every PACE product.
The current single-band model would use 8704 tokens at this chip and patch size;
eight-band grouping reduces that to 1088. This changes the input and output
projections. The implementation and mask-leakage test are now in
[grouped_model.py](../src/grouped_model.py) and
[test_discovery.py](../tests/test_discovery.py).

The main task's new quality-mask inspection found 407 nonoverlapping clear
16 × 16 tiles in this granule, compared with 64 at 32 × 32 and 7 at 64 × 64.
Extraction must still check finite data. This favors small chips over a large
land-image crop. It does not make this one scene a sufficient training corpus.

Use the existing training-only scalar mean/standard-deviation scaling and
masked MSE first. Preserve spectral amplitude; do not normalize each pixel to
unit length and discard its magnitude. MSE can favor strong blue signals over
weak red features. Report errors by band and window, including the fluorescence
region, so a good global score cannot hide this failure. Defer spectral-angle,
derivative, and contrastive loss terms. A later single ablation can test
band-balanced scaling with a floor if weak bands are demonstrably neglected.
Do not start with inverse-uncertainty loss weights: spuriously small reported
uncertainty can give an unreliable sample extreme weight.

Start with a fixed mixed spectral/spatial mask recipe and a small validation
comparison with random token masking. Hide whole contiguous band groups across
the chip and spatial tubes that hide all bands at a location. Assert that the
requested blocks are fully hidden. A checkerboard of missing
single bands mainly tests spectral interpolation. Use several seeds once the
pipeline works. Keep model size and training examples comparable. The exact
mask ratio and patch size remain empirical choices; imported 90–99% masking
settings are not OCI requirements.

**Phase 3: freeze the encoder and inspect discovery.** Export local embeddings,
nearest neighbors, representative spectra, and a small set of cluster maps.
Start with one clustering method and a few validation-selected cluster counts.
Allow weak membership or an unresolved group rather than forcing a taxonomic
label onto every pixel. Use PCA or a two-dimensional display only to view
results; a visually separated plot is not proof of valid classes.

Use one feature vector per spatial patch first, formed by averaging that
patch's encoded spectral-group tokens. Keep the full tokens for spectral
inspection. Fit any embedding scaling and cluster centers on training scenes
only, choose cluster settings on validation scenes without biological labels,
and assign the untouched test scenes without refitting. Pool pixel-level
baselines over the same spatial support for feature comparisons. Include an
untrained encoder control to detect patterns caused by the architecture or
positional encodings alone. A 2 × 2 patch map is not a pixel-resolution map.

**Phase 4: add one alternative only for a measured failure.** If MAE beats
simple reconstruction baselines but its neighbors are unstable or its clusters
follow scene artifacts, compare a conservative iBOT-style teacher method using
the same data and evaluation. If neither improves robust discovery over PCA,
retain PCA and improve data coverage before scaling the model.

All future research workloads must use context `nautilus` and namespace
`sea-meets-the-stars` explicitly. Use epoch counts or validation-based early
stopping. Do not add workload timeouts, elapsed-time deadlines, or timed deletion.

## Evaluate discovery without training regressors

The following are proposed checks. They establish different kinds of evidence;
no single score proves biological discovery.

| Question | Test and report |
| --- | --- |
| Does the model retain useful spectral information? | On untouched scenes, score hidden contiguous band intervals and full-spectrum spatial holes separately. Report per-band bias, RMSE in sr⁻¹, and spectral angle against the same masks for all models. Handle near-zero spectral norms explicitly. Report both amplitude and shape; angle alone discards amplitude. |
| Does it beat easy smoothness? | Compare visible-band interpolation, training-mean spectra, and PCA fit using observed values only. For PCA masked recovery, estimate coefficients from visible entries; never compute them from the complete target spectrum. Score gap widths separately. |
| Are embeddings stable? | Repeat seeds, chip offsets, missing-band patterns, and justified small noise perturbations. Report neighbor overlap and cluster agreement across runs. Report scene-level uncertainty across held-out granules, not millions of pixels as independent samples. |
| Are local features useful? | Search for similar spectra across different dates and regions. Show neighbors beside raw spectra, quality, and geometry. Compare with direct spectral-angle and PCA neighbors. Restrict searches to other scenes to prevent adjacent-pixel copies from dominating. |
| Are groups optical patterns or artifacts? | Compare group membership with scan position, sun/view geometry, aerosol fields, quality flags, distance to cloud/land edges, missing-band patterns, and processing version. Repeat within comparable geometry and water regimes. Stratified associations identify warnings; correlation alone cannot prove a cause. |
| Is structure real and repeatable? | Check whether group spectra and memberships persist on unseen scenes. Spatial smoothness alone is weak evidence because observation artifacts are also smooth. Compare full spectra with shape-normalized spectra as a controlled analysis, since brightness can be signal or nuisance. |
| Is there biological evidence? | After choices are frozen, compare groups with independent field pigments, microscopy or other community observations, and field optics using valid spatial/time matchups. Report distributions and unresolved mixtures. Satellite BGC agreement is an algorithm comparison; it is not independent truth. |

NASA lists atmospheric-correction failure, glint, cloud/ice, high sensor zenith,
and probable stray light among L2 flags. Its matchup practice also applies
quality screens. Use these as concrete nuisance controls rather than learning
from all finite pixels. [NASA flag definitions](https://oceandata.sci.gsfc.nasa.gov/ocl2flags/),
[SeaBASS matchup screen](https://seabass.gsfc.nasa.gov/wiki/time_series_tool_description).

Audit the current shared mask before treating it as an AOP-only selection rule.
For example, NASA defines `CHLFAIL` as chlorophyll-algorithm failure and
`CHLWARN` as chlorophyll out of bounds. These are not independent field labels,
but selecting on them can bias the spectra retained for discovery. Record their
effect and justify the fixed AOP screen before training. Keep atmospheric,
cloud, glint, navigation, and fill-value safeguards; do not loosen them merely
to obtain more distinct clusters. Likewise, a water-type flag is not a reason
to remove that water type from a discovery study.

Use reported Rrs uncertainty as a diagnostic where valid. A diagonal
uncertainty-normalized residual does not model the full spectral error
covariance and does not make a deterministic MAE probabilistically calibrated.
Noise experiments should state their assumed covariance. Do not claim measured
OCI robustness from an arbitrary independent Gaussian noise test.

## Decision rule and remaining limits

**Proposed decision:** advance the small MAE only if it improves blocked-gap
and spatial-hole reconstruction over strong simple baselines on unseen scenes,
preserves reproducible optical neighborhoods, and passes the nuisance review.
Set quantitative acceptance margins after a baseline variance check and before
the final test. A better decoder with unchanged or worse embeddings is not a
discovery win. Do not trade away weak spectral features merely to make clusters
look cleaner.

The strongest near-term scientific claim is “repeatable optical patterns in
quality-screened OCI Rrs.” A phytoplankton interpretation requires independent
associations and a stated domain. Verified species identity or absolute
constituent retrieval requires stronger evidence than this unlabeled design.

Not checked: reproduced paper results, pretrained checkpoint compatibility,
new multi-scene data, field matchups, or empirical model ranking. Main risks:
spectral redundancy, mixed and non-unique optical signals, biased quality
selection, and features driven by atmospheric correction or scene geometry.
