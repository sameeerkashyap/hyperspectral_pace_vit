"""Execute a notebook and preserve outputs even when a cell fails."""

import argparse
from pathlib import Path


def main():
    import nbformat
    from nbclient import NotebookClient

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout", type=int, default=3600, help="Seconds per cell")
    args = parser.parse_args()
    notebook = nbformat.read(args.notebook, as_version=4)
    try:
        NotebookClient(notebook, timeout=args.timeout, kernel_name="python3").execute()
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(notebook, args.output)


if __name__ == "__main__":
    main()
