import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import io
import importlib.util
import os
import socket

from src.runner import download, run_experiment
from src.experiment import degeneracy_check


class RunnerTests(unittest.TestCase):
    def test_download_integrity_and_cache(self):
        body = b"a small runtime dataset"
        dataset = {"file": "sample.bin", "url": "https://example.org/sample.bin",
                   "sha256": hashlib.sha256(body).hexdigest()}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("src.runner.urllib.request.urlopen", return_value=io.BytesIO(body)) as request:
                self.assertEqual(download(dataset, root).read_bytes(), body)
                download(dataset, root)
                self.assertEqual(request.call_count, 1)
            # An invalid replacement must not overwrite the previous cached file.
            changed = {**dataset, "sha256": "0" * 64}
            with patch("src.runner.urllib.request.urlopen", side_effect=lambda *a, **k: io.BytesIO(b"bad")), patch("src.runner.time.sleep"):
                with self.assertRaises(ValueError):
                    download(changed, root)
            self.assertEqual((root / "sample.bin").read_bytes(), body)
            self.assertEqual(list(root.glob("*.part")), [])
            with self.assertRaises(ValueError):
                download({**dataset, "file": "../escape"}, root)

    def test_custom_script_outputs_failure_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command = [sys.executable, "-c", "import os,pathlib; p=pathlib.Path(os.environ['OUTPUT_DIR']); (p/'result.txt').write_text(os.environ['EXPERIMENT_PARAMETERS']); print('finished')"]
            experiment = {"name": "custom", "command": command, "parameters": {"learning_rate": .01}}
            run_experiment(experiment, [], root / "data", root / "outputs")
            result = root / "outputs/custom"
            self.assertEqual(json.loads((result / "result.txt").read_text()), {"learning_rate": .01})
            self.assertEqual(json.loads((result / "run.json").read_text())["status"], "complete")
            self.assertIn("finished", (result / "console.log").read_text())
            with self.assertRaises(FileExistsError):
                run_experiment(experiment, [], root / "data", root / "outputs")
            with self.assertRaises(subprocess.CalledProcessError):
                run_experiment({"name": "failure", "command": [sys.executable, "-c", "raise SystemExit(7)"]}, [], root / "data", root / "outputs")
            status = json.loads((root / "outputs/failure/run.json").read_text())
            self.assertEqual(status["returncode"], 7)
            self.assertEqual(status["status"], "failed")

    def test_physical_degeneracy_control(self):
        self.assertLess(degeneracy_check()["maximum_rrs_difference"], 1e-12)

    @unittest.skipUnless(importlib.util.find_spec("nbclient"), "Install the notebooks extra")
    def test_notebook_execution_and_failed_cell_artifact(self):
        import nbformat

        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", 0))
            except PermissionError:
                self.skipTest("Sandbox blocks the local ports required by a Jupyter kernel")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            notebook = root / "train.ipynb"
            env = {"JUPYTER_RUNTIME_DIR": str(root / "runtime"), "IPYTHONDIR": str(root / "ipython")}
            for fail in (False, True):
                code = "import os, pathlib; print(os.environ['EXPERIMENT_PARAMETERS'])"
                if fail:
                    code += "\nraise ValueError('expected notebook failure')"
                nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(code)]), notebook)
                experiment = {"name": "notebook-fail" if fail else "notebook-pass",
                              "notebook": str(notebook), "parameters": {"epochs": 2}}
                with patch.dict(os.environ, env):
                    if fail:
                        with self.assertRaises(subprocess.CalledProcessError):
                            run_experiment(experiment, [], root / "data", root / "outputs")
                    else:
                        run_experiment(experiment, [], root / "data", root / "outputs")
                output = root / "outputs" / experiment["name"]
                executed = nbformat.read(output / "executed.ipynb", as_version=4)
                outputs = executed.cells[0].outputs
                self.assertTrue(any(o.output_type == "stream" for o in outputs))
                self.assertEqual(any(o.output_type == "error" for o in outputs), fail)

    @unittest.skipUnless(importlib.util.find_spec("nbclient"), "Install the notebooks extra")
    def test_notebook_saved_when_execution_fails(self):
        import nbformat
        from src.notebook import main

        with tempfile.TemporaryDirectory() as directory:
            notebook, output = Path(directory) / "input.ipynb", Path(directory) / "executed.ipynb"
            nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("1 / 0")]), notebook)
            with patch.object(sys, "argv", ["notebook", str(notebook), "--output", str(output)]), patch("nbclient.NotebookClient.execute", side_effect=RuntimeError("cell failed")):
                with self.assertRaisesRegex(RuntimeError, "cell failed"):
                    main()
            self.assertEqual(nbformat.read(output, as_version=4).cells[0].source, "1 / 0")


if __name__ == "__main__":
    unittest.main()
