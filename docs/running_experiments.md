# Run experiments locally or on Nautilus

For the current AOP-only research goal, use the
[self-supervised discovery guide](discovery_experiment.md) and
`nautilus/discovery-job.yaml`. It requires no BGC training labels.

The [completed 30-scene campaign](discovery_multiscene_results.md) uses versioned
code archives on the results PVC. The launch templates verify archive hashes
before unpacking code. New campaigns need new source versions, Job names, and
output directories. The [protocol](discovery_multiscene_run.md) records the
actual paths and hashes. The image-build route below remains available for
other plans.

For the new full-spectrum spatial MAE and dense supervised comparison, use
[the chlorophyll benchmark guide](chlorophyll_benchmark.md) and
`nautilus/chlorophyll-job.yaml`. That 15-run plan requires real matched scenes;
the six-run reconstruction plan below remains a separate sample experiment.

The image contains Python, uv, project source, and small experiment plans. It
does not contain PyTorch, CUDA libraries, datasets, checkpoints, or the README
banner. Dependencies and data are downloaded when the container starts.
This reduces image transfer size but moves substantial downloads to startup.
The final image size has not been measured because the local Docker daemon is
not running. A slim base does not imply a small runtime disk footprint.

## Local sample

Use local commands below for inspection or quick checks. Run research experiments
on Nautilus in `sea-meets-the-stars`.

```sh
uv sync
uv run python -m src.runner --plan experiments/sample.json
```

The plan has six experiments: wavelength/index position × seeds 0/1/2. A local
run executes them sequentially and stops on failure. `--index 0` selects one.
The data file is downloaded once, checked against its manifest SHA-256, and
reused. Temporary downloads are unique and atomically renamed only after
verification. Bad or interrupted files cannot become valid cache entries.

For a shorter local test:

```sh
uv run python -m src.experiment --aop data/raw/PACE_OCI.20240730T181157.L2.OC_AOP.V2_0.NRT.nc \
  --steps 30 --chips 16 --threads 1 --output outputs/quick-check
```

This direct command needs an existing sample. Use the runner to download inputs.
The experiment design and actual short-run results are in
[experiment_design.md](experiment_design.md).

## Build a small runtime image

Replace the image name with a registry where you can push. Build on an amd64
builder or use Docker's amd64 emulation when building from an ARM laptop.

```sh
IMAGE=ghcr.io/YOUR_ORG/pace-experiments:sample-v1
docker build --platform linux/amd64 -t "$IMAGE" .
docker image inspect "$IMAGE" --format '{{.Size}}'
docker push "$IMAGE"
```

The `.dockerignore` allowlist excludes local data, outputs, Git, virtual
environments, documents, and images. No data is downloaded during build.
Use a unique image tag or digest when recording a research run.

Startup creates a private venv and installs CUDA 12.4 PyTorch 2.5.1, then the
project dependencies. The NVIDIA host driver must support that wheel.
The GPU runtime itself is supplied by the cluster. Other package versions are
resolved at startup and recorded in `packages.json`; they are not fully locked.
For byte-for-byte environment reproduction, freeze those versions in a later
constraints file. Test this pinned torch version in the actual image before
starting the six-job sweep; the local pilot used the installed torch version
recorded in `docs/pilot_results.json`.

For CPU-only local Docker execution:

```sh
docker run --rm -e TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu \
  -v "$PWD/data/raw:/data" -v "$PWD/outputs:/outputs" \
  -e DATA_DIR=/data -e OUTPUT_ROOT=/outputs/run-1 "$IMAGE" \
  python -m src.runner --plan experiments/sample.json --index 0
```

## Nautilus indexed Job

Files: [Job](../nautilus/job.yaml), [results PVC](../nautilus/pvc.yaml).
Use context `nautilus` and namespace `sea-meets-the-stars` for this repository.
The manifests set that namespace explicitly. The example PVC uses `rook-cephfs`
and requires ReadWriteMany access so multiple workers can write separate result
directories. Confirm that storage class and your GPU/resource quota first.
Replace the Job image with the pushed tag. Registry and image tags remain placeholders.
Do not add elapsed-time limits or automatic timed deletion to Nautilus workloads.

```sh
kubectl --context nautilus get storageclass
kubectl --context nautilus apply -n sea-meets-the-stars -f nautilus/pvc.yaml
kubectl --context nautilus apply -n sea-meets-the-stars -f nautilus/job.yaml
kubectl --context nautilus get pods -n sea-meets-the-stars -l job-name=pace-spectral-validation
kubectl --context nautilus logs -n sea-meets-the-stars -f JOB_POD_NAME
```

Six completions select six JSON-plan entries using `JOB_COMPLETION_INDEX`.
At most two workers run together. Set `completions` to the plan length and
`parallelism` to your quota. Each worker requests one GPU, four CPUs and 8 GiB
RAM, with private runtime dependency storage. Downloaded data and output files
live on the shared PVC. The training Jobs have no elapsed-time deadline.

`backoffLimit: 0` makes the Job fail fast. It does not retry failed experiments.
Inspect `run.json` and the logs; use a new Job name and `OUTPUT_ROOT` for a rerun.
The runner refuses to overwrite an existing experiment directory. A completed
Job can be deleted without deleting its PVC. Do not delete the PVC before
copying the results.

### Retrieve completed results

Create a read-only PVC reader after the Job finishes:

```sh
kubectl --context nautilus apply -n sea-meets-the-stars -f nautilus/results.yaml
kubectl --context nautilus wait -n sea-meets-the-stars --for=condition=Ready pod/pace-results --timeout=120s
kubectl --context nautilus cp -n sea-meets-the-stars pace-results:/results/sample-v1 ./outputs/sample-v1
kubectl --context nautilus delete pod -n sea-meets-the-stars pace-results
```

Delete the reader after copying results; it has no timed exit. Each experiment directory has
`metrics.json`, `best.pt`, `console.log`, `packages.json`, and `run.json`.
The metrics include both wavelength grids, split blocks and coordinates,
training normalization, seed, source version, and noise-test results. The
checkpoint is weights only; use the model dimensions in the design document
and the saved normalization/wavelength metadata when loading it.

## Custom scripts and notebooks

The runner accepts trusted local JSON plans. Commands are argument lists, not
shell strings. A plan can download any checksum-pinned HTTPS inputs, or have
an empty `datasets` list when the custom script fetches data itself. Mounted
scripts can run outside the packaged project.

```json
{
  "datasets": [],
  "experiments": [
    {
      "name": "custom-training",
      "command": ["python", "/results/scripts/train.py", "--epochs", "5"],
      "parameters": {"learning_rate": 0.0003}
    },
    {
      "name": "custom-notebook",
      "notebook": "/results/scripts/train.ipynb",
      "parameters": {"epochs": 5}
    }
  ]
}
```

Save this plan on the mounted PVC and change Job args to
`python -m src.runner --plan /results/custom.json`; set `completions: 2`.
Custom files must be accessible at the given paths inside each container.
Paths are resolved from `/app`, the container working directory, not the plan's
directory. Scripts get `DATA_DIR`, `OUTPUT_DIR`, and JSON
`EXPERIMENT_PARAMETERS` environment variables. Write artifacts to `OUTPUT_DIR`.
Each entry may also supply an `env` map for non-secret settings.

For notebooks, add `PROJECT_EXTRAS: "[notebooks]"` to the Job environment.
An executable training example is supplied in
[experiments/train.ipynb](../experiments/train.ipynb), with
[its one-entry plan](../experiments/notebook.json). Execute it locally with:

```sh
uv sync --extra notebooks
uv run python -m src.runner --plan experiments/notebook.json
```

Notebook cells have a one-hour default timeout. The executed notebook is saved
even if a cell fails, and its error
causes a nonzero process exit and a failed experiment status.

Set `EXTRA_REQUIREMENTS` to a mounted requirements-file path to install custom
dependencies at startup. Keep credentials in Kubernetes Secrets or mounted
credential files, not JSON plans, image layers or command arguments. The public
sample needs no credentials. For authenticated Earthdata inputs, use a custom
download/training script with `earthaccess` and mounted credentials; the generic
public URL downloader does not implement Earthdata login. No full dataset or
secret is built into this image.

## Verification limits

The local real-data pilot and runner tests were executed. Nautilus access to
`sea-meets-the-stars` and permission to create Jobs were verified on 2026-11-19.
No project image has been built or pushed. The AOP-only single-scene pilot was
submitted on 2026-11-20 using a public Python image and a mounted source snapshot;
see [its setup and limits](discovery_experiment.md#first-run-with-the-existing-download).
Always select `--context nautilus` explicitly.
Storage class, quota, GPU driver support, registry access and final image size
must be checked in the target environment.

The notebook artifact-on-error path was tested, but actual kernel execution is
blocked here by the sandbox's restriction on local network ports. The notebook
integration test skips with that reason in this environment. Run it in the
container before scheduling notebook training.
