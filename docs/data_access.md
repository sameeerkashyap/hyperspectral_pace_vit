# PACE fields and NRP data access

## Fields used by the code

| Field | Source | Current use |
| --- | --- | --- |
| `geophysical_data/Rrs` | L2 AOP | Reflectance input in sr⁻¹; reconstruction target during pretraining |
| `sensor_band_parameters/wavelength_3d` | L2 AOP | Select 400–700 nm bands; wavelength positions in the original model; fixed-grid validation in the spatial model |
| `geophysical_data/Rrs_unc` | L2 AOP | Reconstruction-pilot screening, uncertainty-scaled noise tests, and residual scoring; not a model input or part of the supervised loss |
| `geophysical_data/chlor_a` | L2 BGC | Positive, valid labels for dense log10 chlorophyll regression; also a correlation target |
| `geophysical_data/carbon_phyto` | L2 BGC | Correlation target only; no carbon prediction model is trained |
| `geophysical_data/l2_flags` | Both bundles | Reject land, clouds, atmospheric failures, glint, navigation failures, and other excluded quality conditions |
| `navigation_data/latitude`, `longitude` | Both bundles | Check pixel alignment, select geographic areas, reject invalid locations, and record pilot chip locations; not predictor channels |
| Acquisition start/end, processing version, platform, instrument | File attributes | Reject unmatched pairs; keep processing versions consistent and dates separate across benchmark splits |
| Fill values, valid ranges, scale factor, offset, units | Variable attributes | Decode packed values and validate inputs before training |

`avw`, `aot_865`, `angstrom`, `poc`, and target uncertainty may appear in the
field inventory. They are not current training inputs or loss terms.
Use `wavelength_3d`, not the sensor-wide `wavelength` array, for the `Rrs` axis.

## How the data is used

1. Read NetCDF values, decode their scale and offset, and reject fill or invalid values.
2. Apply quality flags. Match AOP and BGC acquisition metadata and pixel locations.
3. Select visible wavelengths and small spatial chips. The reconstruction pilot
   uses 25 bands and every fourth pixel; the chlorophyll benchmark defaults to
   all native 400–700 nm bands and adjacent pixels.
4. Separate training, validation, and test data. The reconstruction pilot uses
   separate spatial blocks in one scene. The chlorophyll benchmark requires
   separate acquisition dates and non-overlapping geographic boxes.
5. Fit reflectance mean and standard deviation on training data only.
6. Pretrain by hiding tokens and reconstructing reflectance with squared error.
7. Fine-tune a dense head on matched BGC labels, using squared error in log10
   chlorophyll over valid positive target pixels. Select weights on validation
   scenes and score the held-out test scenes once.

The downloaded teaching AOP and BGC scenes have different dates and versions.
They support inspection and AOP reconstruction, but cannot support paired
chlorophyll training. Satellite BGC labels test agreement with the satellite
algorithm. Independent field observations are needed for biological validation.

## AWS CLI on macOS or Unix

Use the [official AWS CLI installation guide](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
if `aws --version` fails. AWS CLI v2 is already installed on the current Mac.
Use a separate `nrp` profile to keep existing AWS settings intact:

```sh
mkdir -p "$HOME/.aws"
chmod 700 "$HOME/.aws"
aws configure --profile nrp
```

Enter the NRP access key and secret key locally, without quotes. Set the signing
region to `us-east-1` and output to `json`. The endpoint below selects NRP Ceph
storage; this is not Amazon S3. Credentials go in `~/.aws/credentials`, and
region/output settings go in `~/.aws/config`. Keep both files outside Git.

```sh
chmod 600 "$HOME/.aws/config" "$HOME/.aws/credentials"
aws --profile nrp --endpoint-url https://s3-west.nrp-nautilus.io s3 ls s3://llc
```

The [NRP Ceph S3 guide](https://nrp.ai/documentation/userdocs/storage/ceph-s3/)
documents this endpoint and the AWS CLI approach. Passing the endpoint explicitly
avoids an extra plugin. No Python S3 client is needed.

## Stage data for training

List the bucket before choosing a prefix. Bucket access alone does not establish
that it contains PACE data. Download only the chosen files or prefix:

```sh
aws --profile nrp --endpoint-url https://s3-west.nrp-nautilus.io \
  s3 ls s3://llc/CHOSEN_PREFIX/
mkdir -p data/raw
aws --profile nrp --endpoint-url https://s3-west.nrp-nautilus.io \
  s3 cp s3://llc/CHOSEN_PREFIX/ data/raw/ --recursive
```

Replace `CHOSEN_PREFIX` after inspecting the bucket. Avoid downloading the whole
bucket. Inspect the NetCDF fields and match AOP/BGC files before creating the
[scene manifest](../experiments/chlorophyll-scenes.example.json).

The current loader reads local NetCDF paths, not `s3://` URLs. On Nautilus, stage
inputs onto mounted storage before running training. Mount credentials through
a Kubernetes Secret or a private credential file. Do not put keys in manifests,
images, notebooks, logs, or run-plan JSON. The current runtime image does not
install AWS CLI; S3 staging needs an AWS CLI environment before training.

All project experiments must use context `nautilus` and namespace
`sea-meets-the-stars`. Follow the [runtime guide](running_experiments.md).
Do not add elapsed-time limits or automatic timed deletion to running workloads.
