"""Matched-scene chlorophyll benchmark: spectral MLP and MAE ablations."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from src.chlorophyll import ChlorophyllRegressor, SpectralMLP, chlorophyll_loss
from src.matched_data import load_matched_scenes
from src.model import WavelengthAwareMAE
from src.spatial_model import SpatialSpectralMAE


ARMS = ("mlp", "spectral", "spatial", "mixed", "spatial-scratch")


def evaluate(model, scenes, wavelengths, mean, scale, device):
    model.eval()
    rows = []
    with torch.no_grad():
        for scene in scenes:
            residuals = []
            for start in range(0, len(scene["x"]), 4):
                x = torch.as_tensor((scene["x"][start:start + 4] - mean) / scale, device=device)
                prediction = model(x, wavelengths).cpu().numpy()
                target = scene["target"][start:start + 4]
                valid = scene["valid"][start:start + 4]
                if valid.any():
                    residuals.append(prediction[valid] - np.log10(target[valid]))
            residual = np.concatenate(residuals)
            if not np.isfinite(residual).all():
                raise ValueError("Nonfinite evaluation residual")
            rows.append({"id": scene["id"], "pixels": len(residual),
                         "log10_mse": float(np.mean(residual ** 2)),
                         "log10_bias": float(residual.mean()),
                         "log10_mae": float(np.abs(residual).mean())})
    return {"scene_mean_log10_rmse": float(np.mean([r["log10_mse"] ** .5 for r in rows])),
            "pixel_log10_rmse": float((sum(r["pixels"] * r["log10_mse"] for r in rows)
                                       / sum(r["pixels"] for r in rows)) ** .5), "scenes": rows}


def run(args):
    if (args.arm not in ARMS or min(args.steps, args.threads, args.batch_size) < 1
            or args.pretrain_steps < 0 or args.chip_size < 8 or args.chip_size % 4 or args.embed_dim % 12
            or args.embed_dim < 12):
        raise ValueError("Use chip size >=8 in multiples of 4 and embedding dimension in multiples of 12.")
    if args.arm not in ("mlp", "spatial-scratch") and args.pretrain_steps == 0:
        raise ValueError("Pretrained arms require positive pretraining steps.")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    scenes, grid = load_matched_scenes(args.scenes, args.data_dir, chip_size=args.chip_size,
                                      limit=args.chips, band_count=args.bands, stride=args.stride)
    train = [s for s in scenes if s["split"] == "train"]
    validation = [s for s in scenes if s["split"] == "validation"]
    test = [s for s in scenes if s["split"] == "test"]
    training = np.concatenate([s["x"] for s in train])
    mean, scale = float(training.mean()), max(float(training.std()), 1e-6)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wavelengths = torch.as_tensor(grid, device=device)
    if args.arm == "mlp":
        model = SpectralMLP(len(grid), args.embed_dim)
    else:
        options = dict(embed_dim=args.embed_dim, patch_size=4, encoder_blocks=2, decoder_blocks=1)
        backbone = (SpatialSpectralMAE(grid, **options) if args.arm.startswith("spatial")
                    else WavelengthAwareMAE(**options))
        model = ChlorophyllRegressor(backbone)
    model.to(device)
    args.output.mkdir(parents=True, exist_ok=True)
    x = torch.as_tensor((training - mean) / scale, device=device)
    y = torch.as_tensor(np.concatenate([s["target"] for s in train]), device=device)
    valid = torch.as_tensor(np.concatenate([s["valid"] for s in train]), device=device)
    pretraining = []
    if args.arm not in ("mlp", "spatial-scratch"):
        torch.manual_seed(args.seed + 2)
        optimizer = torch.optim.AdamW(model.backbone.parameters(), lr=1e-3, weight_decay=.01)
        for step in range(args.pretrain_steps):
            model.train()
            batch = x[torch.randint(len(x), (min(args.batch_size, len(x)),), device=device)]
            if isinstance(model.backbone, SpatialSpectralMAE):
                prediction, mask, target = model.backbone(batch, wavelengths)
            else:
                prediction, mask, target = model.backbone(batch.permute(0, 2, 3, 1), wavelengths,
                                                         masking="mixed" if args.arm == "mixed" else "random")
            loss = model.backbone.masked_loss(prediction, mask, target)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite pretraining loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.backbone.parameters(), 1.)
            optimizer.step()
            if step == 0 or step == args.pretrain_steps - 1:
                pretraining.append({"step": step + 1, "loss": float(loss.detach())})
        torch.save(model.backbone.state_dict(), args.output / "pretrained.pt")
    if args.arm != "mlp":
        for module in (model.backbone.decoder, model.backbone.pred_head):
            module.requires_grad_(False)
        model.backbone.mask_token.requires_grad_(False)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=3e-4, weight_decay=.01)
    torch.manual_seed(args.seed + 1)
    best, history = float("inf"), []
    for step in range(args.steps):
        model.train()
        indices = torch.randint(len(x), (min(args.batch_size, len(x)),), device=device)
        prediction = model(x[indices], wavelengths)
        loss = chlorophyll_loss(prediction, y[indices], valid[indices])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.)
        optimizer.step()
        if step == 0 or (step + 1) % 25 == 0 or step == args.steps - 1:
            scores = evaluate(model, validation, wavelengths, mean, scale, device)
            history.append({"step": step + 1, "training_log10_mse": float(loss.detach()),
                            "validation": scores})
            print(json.dumps(history[-1]), flush=True)
            if scores["scene_mean_log10_rmse"] < best:
                best = scores["scene_mean_log10_rmse"]
                torch.save(model.state_dict(), args.output / "best.pt")
    model.load_state_dict(torch.load(args.output / "best.pt", map_location=device, weights_only=True))
    results = evaluate(model, test, wavelengths, mean, scale, device)
    metadata = [{k: v for k, v in s.items() if k not in ("x", "target", "valid", "wavelengths_nm")}
                for s in scenes]
    report = {"arm": args.arm, "seed": args.seed, "steps": args.steps,
              "pretrain_steps": args.pretrain_steps if pretraining else 0,
              "embed_dim": args.embed_dim, "chip_size": args.chip_size, "patch_size": 4,
              "stride": args.stride, "requested_bands": args.bands,
              "encoder_blocks": 2, "decoder_blocks": 1, "batch_size": args.batch_size,
              "device": str(device), "torch_version": torch.__version__,
              "total_parameters": sum(p.numel() for p in model.parameters()),
              "supervised_parameters": sum(p.numel() for p in parameters),
              "wavelengths_nm": grid.tolist(), "normalization": {"mean": mean, "scale": scale},
              "scene_manifest": metadata, "pretraining": pretraining, "history": history,
              "test": results, "target": "log10 satellite-derived chlor_a in mg m^-3",
              "scope": "Algorithm-agreement benchmark; not independent phytoplankton validation."}
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(results, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", type=Path, default=os.environ.get("SCENE_MANIFEST"))
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("DATA_DIR", "data/raw")))
    parser.add_argument("--output", type=Path, default=Path(os.environ.get("OUTPUT_DIR", "outputs/chlorophyll")))
    parser.add_argument("--arm", choices=ARMS, default="spatial")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--embed-dim", type=int, default=96)
    parser.add_argument("--pretrain-steps", type=int, default=200)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--chip-size", type=int, default=8)
    parser.add_argument("--chips", type=int, default=64, help="Maximum chips per scene")
    parser.add_argument("--bands", type=int, default=0, help="0 selects all native 400–700 nm bands")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.scenes is None:
        parser.error("Supply --scenes or the SCENE_MANIFEST environment variable.")
    run(args)


if __name__ == "__main__":
    main()
