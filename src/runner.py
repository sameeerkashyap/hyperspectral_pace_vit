"""Run a JSON experiment plan locally or one index per Kubernetes Job pod."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import urllib.request


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(dataset, destination):
    name, expected = dataset["file"], dataset["sha256"]
    if Path(name).name != name or name in ("", ".", ".."):
        raise ValueError("Dataset file must be a plain filename.")
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("Each dataset needs a SHA-256 checksum.")
    if not dataset["url"].startswith("https://"):
        raise ValueError("Dataset downloads must use HTTPS.")
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / name
    if target.exists() and sha256(target) == expected:
        return target
    # Unique temporary names also make concurrent indexed downloads safe.
    with tempfile.NamedTemporaryFile(dir=destination, suffix=".part", delete=False) as temporary:
        partial = Path(temporary.name)
    try:
        for attempt in range(3):
            try:
                with urllib.request.urlopen(dataset["url"], timeout=120) as response, partial.open("wb") as out:
                    while chunk := response.read(1024 * 1024):
                        out.write(chunk)
                if sha256(partial) != expected:
                    raise ValueError(f"Checksum mismatch for {name}")
                partial.replace(target)
                return target
            except (OSError, ValueError):
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
    finally:
        partial.unlink(missing_ok=True)


def run_experiment(experiment, datasets, data_dir, output_root):
    name = experiment["name"]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError("Use a plain experiment name.")
    has_command, has_notebook = "command" in experiment, "notebook" in experiment
    if has_command == has_notebook:
        raise ValueError("Specify exactly one command or notebook per experiment.")
    if has_command:
        command = experiment["command"]
        if not isinstance(command, list) or not command or not all(isinstance(a, str) for a in command):
            raise ValueError("Command must be a nonempty list of arguments.")
        if command[0] in ("python", "python3"):
            command = [sys.executable, *command[1:]]
    # Do not overwrite earlier results, including partial or failed runs.
    output = output_root / name
    output.mkdir(parents=True, exist_ok=False)
    state = {"experiment": experiment, "datasets": datasets, "status": "preparing",
             "started_unix": time.time()}
    (output / "packages.json").write_text(json.dumps(
        {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}, indent=2) + "\n")
    status_file = output / "run.json"
    status_file.write_text(json.dumps(state, indent=2) + "\n")
    try:
        for dataset in datasets:
            download(dataset, data_dir)
        if has_notebook:
            command = [sys.executable, "-m", "src.notebook", experiment["notebook"],
                       "--output", str(output / "executed.ipynb")]
        env = {**os.environ, **{k: str(v) for k, v in experiment.get("env", {}).items()},
               "DATA_DIR": str(data_dir), "OUTPUT_DIR": str(output),
               "EXPERIMENT_PARAMETERS": json.dumps(experiment.get("parameters", {}))}
        state["status"] = "running"
        status_file.write_text(json.dumps(state, indent=2) + "\n")
        with (output / "console.log").open("w") as log:
            with subprocess.Popen(command, env=env, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True) as process:
                for line in process.stdout:
                    print(f"[{name}] {line}", end="", flush=True)
                    log.write(line)
                    log.flush()
                returncode = process.wait()
        state["returncode"] = returncode
        if returncode:
            raise subprocess.CalledProcessError(returncode, command)
        state["status"] = "complete"
    except BaseException as error:
        state.update(status="failed", error=str(error))
        raise
    finally:
        state["finished_unix"] = time.time()
        status_file.write_text(json.dumps(state, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--index", type=int, default=os.environ.get("JOB_COMPLETION_INDEX"))
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("DATA_DIR", "data/raw")))
    parser.add_argument("--output-root", type=Path, default=Path(os.environ.get("OUTPUT_ROOT", "outputs")))
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    experiments = plan["experiments"]
    names = [experiment["name"] for experiment in experiments]
    if not experiments or len(set(names)) != len(names):
        parser.error("The plan must have experiments with unique names.")
    if args.index is not None:
        if not 0 <= args.index < len(experiments):
            parser.error("Experiment index is outside the plan.")
        experiments = [experiments[args.index]]
    for experiment in experiments:
        run_experiment(experiment, plan.get("datasets", []),
                       args.data_dir.resolve(), args.output_root.resolve())


if __name__ == "__main__":
    main()
