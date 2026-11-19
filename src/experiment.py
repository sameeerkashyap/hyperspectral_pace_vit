"""Small PACE reconstruction experiment; no biological retrieval claim."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform

import h5netcdf
import numpy as np
import torch

from src.data import quality_mask, read_valid
from src.model import WavelengthAwareMAE


def load_chips(path, limit=64, band_count=25, stride=4, chip_size=4):
    """Use separated 128-pixel blocks with a one-chip boundary guard."""
    if min(limit, stride, chip_size) < 1 or not 4 <= band_count <= 60:
        raise ValueError("Use positive sampling dimensions and 4–60 bands.")
    selection = (slice(None, None, stride),) * 2
    with h5netcdf.File(path) as root:
        geo = root.groups["geophysical_data"].variables
        wavelengths = read_valid(root.groups["sensor_band_parameters"].variables["wavelength_3d"])
        visible = np.flatnonzero((wavelengths >= 400) & (wavelengths <= 700))
        if len(visible) < band_count or geo["Rrs"].shape[-1] != len(wavelengths):
            raise ValueError("Sample does not have the required visible Rrs bands.")
        bands = visible[np.linspace(0, len(visible) - 1, band_count).astype(int)]
        spectra = np.stack([read_valid(geo["Rrs"], selection + (int(b),)) for b in bands], -1)
        uncertainty = np.stack([read_valid(geo["Rrs_unc"], selection + (int(b),)) for b in bands], -1)
        good = quality_mask(geo["l2_flags"], selection)
        good &= np.isfinite(spectra).all(-1) & np.isfinite(uncertainty).all(-1)
        good &= (uncertainty > 0).all(-1)
        nav = root.groups["navigation_data"].variables
        latitude = read_valid(nav["latitude"], selection)
        longitude = read_valid(nav["longitude"], selection)
        good &= np.isfinite(latitude) & np.isfinite(longitude)
        source = {key: str(root.attrs[key]) for key in
                  ("product_name", "processing_version", "time_coverage_start")}
    groups = {name: [] for name in ("train", "validation", "test")}
    block_size = 32
    for row in range(0, spectra.shape[0] - chip_size + 1, chip_size):
        for column in range(0, spectra.shape[1] - chip_size + 1, chip_size):
            # No chip may straddle a block or touch its boundary.
            if (row % block_size < chip_size or column % block_size < chip_size
                    or row % block_size + chip_size > block_size - chip_size
                    or column % block_size + chip_size > block_size - chip_size):
                continue
            block = (row // block_size, column // block_size)
            fold = int(hashlib.sha256(f"{block}".encode()).hexdigest()[:8], 16) % 10
            split = "train" if fold < 6 else "validation" if fold < 8 else "test"
            if len(groups[split]) >= limit:
                continue
            window = (slice(row, row + chip_size), slice(column, column + chip_size))
            if good[window].all():
                groups[split].append((spectra[window], uncertainty[window],
                                      (row * stride, column * stride), block,
                                      (float(latitude[row, column]), float(longitude[row, column]))))
    if any(len(chips) < 4 for chips in groups.values()):
        raise ValueError(f"Too few quality-screened chips by split: {[(k, len(v)) for k, v in groups.items()]}")
    return {name: {"spectra": np.array([c[0] for c in chips], dtype="float32"),
                   "uncertainty": np.array([c[1] for c in chips], dtype="float32"),
                   "pixels": [c[2] for c in chips], "blocks": [c[3] for c in chips],
                   "lat_lon": [c[4] for c in chips]}
            for name, chips in groups.items()}, wavelengths[bands].astype("float32"), source


def positions(wavelengths, mode, device):
    values = wavelengths if mode == "wavelength" else np.arange(1, len(wavelengths) + 1)
    return torch.as_tensor(values, device=device, dtype=torch.float32)


def score(model, values, wavelengths, mode, mean, scale, seed, baseline, uncertainty, device):
    model.eval()
    errors, base_errors, interpolation_errors, normalized_errors = [], [], [], []
    with torch.no_grad(), torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        torch.manual_seed(seed)
        for start in range(0, len(values), 8):
            x = torch.as_tensor((values[start:start + 8] - mean) / scale, device=device)
            pred, mask, target = model(x, positions(wavelengths, mode, device))
            residual = (pred - target).cpu().numpy() * scale
            hidden = mask.cpu().numpy()
            errors.append(residual[hidden].reshape(-1))
            base = np.broadcast_to(baseline, values[start:start + 8].shape)
            base_patches = model.patchify(torch.as_tensor(base.copy(), device=device)).cpu().numpy()
            original = model.patchify(torch.as_tensor(values[start:start + 8], device=device)).cpu().numpy()
            base_errors.append((base_patches - original)[hidden].reshape(-1))
            # Fit no parameters on test data; interpolate its visible bands at each pixel.
            grouped = original.reshape(len(x), -1, len(wavelengths), model.patch_size ** 2)
            grouped_mask = hidden.reshape(len(x), -1, len(wavelengths))
            interpolated = np.broadcast_to(baseline[None, None, :, None], grouped.shape).copy()
            for batch in range(len(x)):
                for spatial in range(grouped.shape[1]):
                    visible = ~grouped_mask[batch, spatial]
                    if visible.any():
                        for pixel in range(grouped.shape[-1]):
                            interpolated[batch, spatial, :, pixel] = np.interp(
                                wavelengths, wavelengths[visible], grouped[batch, spatial, visible, pixel])
            interpolation_errors.append((interpolated.reshape(original.shape) - original)[hidden].reshape(-1))
            sigma = model.patchify(torch.as_tensor(uncertainty[start:start + 8], device=device)).cpu().numpy()
            normalized_errors.append((residual / np.maximum(sigma, 1e-12))[hidden].reshape(-1))
    return {"rmse_sr_inv": float(np.sqrt(np.mean(np.concatenate(errors) ** 2))),
            "mean_spectrum_rmse_sr_inv": float(np.sqrt(np.mean(np.concatenate(base_errors) ** 2))),
            "visible_band_interpolation_rmse_sr_inv": float(np.sqrt(np.mean(np.concatenate(interpolation_errors) ** 2))),
            "uncertainty_normalized_rmse": float(np.sqrt(np.mean(np.concatenate(normalized_errors) ** 2))),
            "masked_values": int(sum(len(e) for e in errors))}


def degeneracy_check():
    """Paper equations 8–10: scaling total a and bb leaves Rrs unchanged."""
    a = np.array([0.03, 0.08, 0.4])
    bb = np.array([0.004, 0.003, 0.002])

    def reflectance(absorption, backscatter):
        u = backscatter / (absorption + backscatter)
        subsurface = 0.0949 * u + 0.0794 * u ** 2
        return 0.52 * subsurface / (1 - 1.17 * subsurface)

    difference = float(np.max(np.abs(reflectance(a, bb) - reflectance(3 * a, 3 * bb))))
    if difference > 1e-12:
        raise AssertionError("Reflectance degeneracy check failed")
    return {"total_iop_scale": 3, "maximum_rrs_difference": difference,
            "note": "Algebraic control only; not a constituent model or paper reproduction."}


def train(args):
    if args.steps < 1 or args.threads < 1:
        raise ValueError("Steps and thread count must be positive.")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    groups, wavelengths, source = load_chips(args.aop, limit=args.chips)
    train_bands = np.arange(0, len(wavelengths), 2)
    unseen_bands = np.arange(1, len(wavelengths), 2)
    training = groups["train"]["spectra"][..., train_bands]
    mean, scale = float(training.mean()), max(float(training.std()), 1e-6)
    mean_spectrum = training.mean(axis=(0, 1, 2))
    model = WavelengthAwareMAE(embed_dim=48, patch_size=2, encoder_blocks=2,
                               decoder_blocks=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    x = torch.as_tensor((training - mean) / scale, device=device)
    train_positions = positions(wavelengths[train_bands], args.position, device)
    args.output.mkdir(parents=True, exist_ok=True)
    history, best = [], float("inf")
    checkpoint = args.output / "best.pt"
    for step in range(args.steps):
        model.train()
        indices = torch.randint(len(x), (min(8, len(x)),), device=device)
        optimizer.zero_grad()
        pred, mask, target = model(x[indices], train_positions)
        loss = model.masked_loss(pred, mask, target)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if step == 0 or (step + 1) % 25 == 0 or step == args.steps - 1:
            group = groups["validation"]
            validation = score(model, group["spectra"][..., train_bands], wavelengths[train_bands],
                               args.position, mean, scale, 1000, mean_spectrum,
                               group["uncertainty"][..., train_bands], device)
            history.append({"step": step + 1, "training_loss": float(loss.detach()),
                            "validation": validation})
            print(json.dumps(history[-1]), flush=True)
            if validation["rmse_sr_inv"] < best:
                best = validation["rmse_sr_inv"]
                torch.save(model.state_dict(), checkpoint)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    results = {}
    test = groups["test"]
    for label, bands in (("seen_grid", train_bands), ("unseen_grid", unseen_bands)):
        values, sigma = test["spectra"][..., bands], test["uncertainty"][..., bands]
        baseline = np.interp(wavelengths[bands], wavelengths[train_bands], mean_spectrum).astype("float32")
        for noise in (False, True):
            rng = np.random.default_rng(2000)
            perturbation = (np.sqrt(0.8) * rng.normal(size=values.shape[:-1] + (1,))
                            + np.sqrt(0.2) * rng.normal(size=values.shape)) * sigma
            noisy_values = (values + perturbation).astype("float32") if noise else values
            results[label + ("_correlated_noise" if noise else "")] = score(
                model, noisy_values, wavelengths[bands], args.position, mean, scale,
                1001, baseline, sigma, device)
    report = {"position": args.position, "seed": args.seed, "steps": args.steps,
              "device": str(device), "versions": {"python": platform.python_version(),
              "torch": torch.__version__, "numpy": np.__version__}, "source": source,
              "mean": mean, "scale": scale, "train_wavelengths_nm": wavelengths[train_bands].tolist(),
              "unseen_wavelengths_nm": wavelengths[unseen_bands].tolist(),
              "split": {name: {k: v for k, v in group.items() if k not in ("spectra", "uncertainty")}
                        for name, group in groups.items()},
              "history": history, "test": results, "degeneracy_control": degeneracy_check(),
              "scope": "Single-scene architecture pilot; no independent biological validation.",
              "noise": "Additional Rrs_unc-scaled Gaussian noise, spectral rho=0.8; sensitivity only."}
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(results, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aop", type=Path, default=Path(os.environ.get("DATA_DIR", "data/raw")) /
                        "PACE_OCI.20240730T181157.L2.OC_AOP.V2_0.NRT.nc")
    parser.add_argument("--output", type=Path, default=Path(os.environ.get("OUTPUT_DIR", "outputs/sample")))
    parser.add_argument("--position", choices=("wavelength", "index"), default="wavelength")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--chips", type=int, default=64, help="Maximum chips in each spatial split")
    parser.add_argument("--threads", type=int, default=4)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
