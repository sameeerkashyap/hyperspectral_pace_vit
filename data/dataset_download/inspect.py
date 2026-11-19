"""Write the complete field inventory from actual NetCDF metadata."""

import argparse
from pathlib import Path

import h5netcdf


def inventory(path):
    lines = [f"## {Path(path).name}", ""]
    with h5netcdf.File(path) as root:
        for key in ("title", "processing_version", "time_coverage_start", "time_coverage_end"):
            lines.append(f"- {key}: {root.attrs.get(key, 'absent')}")
        lines += ["", "| Field | Shape | Storage | Units | Meaning |",
                  "| --- | --- | --- | --- | --- |"]

        def visit(group, prefix=""):
            for name, var in group.variables.items():
                meaning = str(var.attrs.get("long_name", "")).replace("|", "/")
                lines.append(f"| `{prefix}{name}` | {var.shape} | {var.dtype} | "
                             f"{var.attrs.get('units', '—')} | {meaning} |")
            for name, child in group.groups.items():
                visit(child, prefix + name + "/")

        visit(root)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+")
    parser.add_argument("--output", default="docs/fields.md")
    args = parser.parse_args()
    Path(args.output).write_text("# Observed sample fields\n\nGenerated from local files. "
                               "Fields can change with product version.\n\n" +
                               "\n\n".join(inventory(p) for p in args.files) + "\n")


if __name__ == "__main__":
    main()
