"""AOP-only representation experiment: grouped MAE, PCA, and spectral autoencoder."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from src.discovery_data import AOP_EXCLUDE_FLAGS, AUDIT_FIELDS, load_aop_scenes
from src.grouped_model import GroupedSpectralMAE, SpectralAutoencoder, token_mask


def distances(values, centers):
    return np.maximum((values ** 2).sum(1)[:, None] + (centers ** 2).sum(1)
                      - 2 * values @ centers.T, 0)


def kmeans(values, count, seed=0):
    if count < 2 or len(values) < count or not np.isfinite(values).all():
        raise ValueError("Clustering needs finite features and at least one sample per group.")
    rng = np.random.default_rng(seed)
    centers = [values[rng.integers(len(values))]]
    for _ in range(1, count):
        weight = distances(values, np.asarray(centers)).min(1).astype('float64')
        if weight.sum() <= 0:
            raise ValueError("Too few distinct feature vectors for the requested groups.")
        centers.append(values[rng.choice(len(values), p=weight / weight.sum())])
    centers = np.asarray(centers).copy()
    for _ in range(40):
        d = distances(values, centers)
        labels = d.argmin(1)
        updated = np.asarray([values[labels == i].mean(0) if (labels == i).any()
                              else values[d.min(1).argmax()] for i in range(count)])
        if np.allclose(updated, centers, rtol=0, atol=1e-5):
            break
        centers = updated
    return centers


class SpectralPCA:
    def fit(self, scenes, mean, scale):
        self.mean = np.mean([((s['x'] - mean) / scale).mean((0, 1, 2)) for s in scenes], 0)
        bands = len(self.mean)
        covariance = np.zeros((bands, bands), dtype='float64')
        for scene in scenes:
            pixels = scene['x'].reshape(-1, bands)
            for start in range(0, len(pixels), 4096):
                x = ((pixels[start:start + 4096] - mean) / scale - self.mean).astype('float64')
                covariance += x.T @ x / (len(pixels) * len(scenes))
        _, vectors = np.linalg.eigh(covariance)
        self.components = vectors[:, ::-1].T.astype('float32')
        self.rank = min(16, bands - 1)
        return self

    def encode(self, values):
        return (values - self.mean) @ self.components[:self.rank].T

    def reconstruct(self, values, hidden):
        flat, masks = values.reshape(-1, values.shape[-1]), hidden.reshape(-1, hidden.shape[-1])
        result = np.broadcast_to(self.mean, flat.shape).copy()
        patterns, membership = np.unique(masks, axis=0, return_inverse=True)
        basis = self.components[:self.rank]
        for i, mask in enumerate(patterns):
            observed = ~mask
            if observed.any():
                rows = membership == i
                # Estimate coefficients from visible entries, never from target bands.
                coefficients = np.linalg.lstsq(basis[:, observed].T,
                                               (flat[rows][:, observed] - self.mean[observed]).T,
                                               rcond=1e-5)[0].T
                result[rows] = self.mean + coefficients @ basis
        return result.reshape(values.shape)


def interpolate_visible(values, hidden, grid, fallback):
    result = np.broadcast_to(fallback, values.shape).copy()
    for item, cube in enumerate(values):
        flat = cube.reshape(-1, len(grid))
        masks = hidden[item].reshape(flat.shape)
        target = result[item].reshape(flat.shape)
        for i in np.flatnonzero((~masks).any(1)):
            observed = ~masks[i]
            target[i] = np.interp(grid, grid[observed], flat[i, observed])
        visible_pixels = np.flatnonzero((~masks).all(1))
        missing_pixels = np.flatnonzero(masks.all(1))
        if len(visible_pixels) and len(missing_pixels):
            coords = np.stack(np.indices(cube.shape[:2]), -1).reshape(-1, 2)
            d = ((coords[missing_pixels, None] - coords[visible_pixels]) ** 2).sum(-1)
            nearest = np.argsort(d, axis=1)[:, :4]
            target[missing_pixels] = flat[visible_pixels[nearest]].mean(1)
    return result


def mask_for(layout, values, mode, first=0, scene_index=0):
    _, h, w, _ = values.shape
    mask = torch.cat([token_mask(1, h // layout.patch_size, w // layout.patch_size,
                                 layout.groups, mode, 4000 + scene_index * 100000 + first + i)
                      for i in range(len(values))])
    expanded = mask[..., None].expand(-1, -1, layout.patch_size ** 2 * layout.group_size)
    return mask, layout.unpatchify(expanded, values.shape).numpy()


@torch.no_grad()
def predict(model, layout, values, hidden, token_hidden, grid, device):
    if isinstance(model, SpectralPCA):
        return model.reconstruct(values, hidden)
    model.eval()
    x = torch.as_tensor(values, device=device)
    if isinstance(model, GroupedSpectralMAE):
        prediction, _, _ = model(x, grid, token_hidden.to(device))
        result = layout.unpatchify(prediction, x.shape)
    else:
        result = model(x, torch.as_tensor(hidden, device=device))
    return result.cpu().numpy()


def spectral_metrics(prediction, target, hidden, sigma):
    residual = prediction.astype('float64') - target
    if not np.isfinite(residual).all():
        raise ValueError("Nonfinite reconstruction.")
    count = hidden.sum((0, 1, 2))
    squared = np.where(hidden, residual ** 2, 0).sum((0, 1, 2))
    bias = np.where(hidden, residual, 0).sum((0, 1, 2))
    a, b = np.where(hidden, target, 0), np.where(hidden, prediction, 0)
    norm = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    valid = norm > 1e-12
    angles = np.degrees(np.arccos(np.clip((a * b).sum(-1)[valid] / norm[valid], -1, 1)))
    valid_sigma = hidden & np.isfinite(sigma) & (sigma > 0)
    return {'count': count, 'squared': squared, 'bias_sum': bias,
            'angle_sum': float(angles.sum()), 'angle_count': int(valid.sum()),
            'sigma_squared': float(((residual[valid_sigma] / sigma[valid_sigma]) ** 2).sum()),
            'sigma_count': int(valid_sigma.sum())}


def summarize_metric(totals, grid):
    count, squared = totals['count'], totals['squared']
    if not count.sum():
        raise ValueError("Evaluation mask has no hidden values.")
    def window(low, high):
        selected = (grid >= low) & (grid <= high)
        return float(np.sqrt(squared[selected].sum() / count[selected].sum())) if count[selected].sum() else None
    return {'rmse_sr_inv': float(np.sqrt(squared.sum() / count.sum())),
            'per_band_rmse_sr_inv': [float(np.sqrt(v / n)) if n else None for v, n in zip(squared, count)],
            'per_band_bias_sr_inv': [float(v / n) if n else None for v, n in zip(totals['bias_sum'], count)],
            'per_band_hidden_values': count.tolist(),
            'windows_rmse_sr_inv': {'blue_400_500': window(400, 500), 'green_500_600': window(500, 600),
                                   'red_600_700': window(600, 700), 'fluorescence_670_690': window(670, 690)},
            'masked_spectral_angle_degrees': totals['angle_sum'] / totals['angle_count'] if totals['angle_count'] else None,
            'spectral_angle_pixels': totals['angle_count'],
            'diagonal_uncertainty_normalized_rmse': float(np.sqrt(totals['sigma_squared'] / totals['sigma_count']))
            if totals['sigma_count'] else None}


def evaluate(model, scenes, layout, grid, mean, scale, device, batch_size=4, fallback=None):
    results = []
    for scene_index, scene in enumerate(scenes):
        for mode in ('spectral', 'spatial'):
            totals = {}
            control_totals = {}
            for start in range(0, len(scene['x']), batch_size):
                raw = scene['x'][start:start + batch_size]
                values = (raw - mean) / scale
                mask, hidden = mask_for(layout, values, mode, start, scene_index)
                prediction = predict(model, layout, values, hidden, mask, grid, device) * scale + mean
                sigma = scene['uncertainty'][start:start + batch_size]
                measured = spectral_metrics(prediction, raw, hidden, sigma)
                for key, value in measured.items():
                    totals[key] = totals.get(key, 0) + value
                if fallback is not None:
                    controls = {'training_mean': np.broadcast_to(fallback, raw.shape),
                                'visible_interpolation': interpolate_visible(raw, hidden, grid, fallback)}
                    for name, reconstructed in controls.items():
                        store = control_totals.setdefault(name, {})
                        for key, value in spectral_metrics(reconstructed, raw, hidden, sigma).items():
                            store[key] = store.get(key, 0) + value
            results.append({'scene': scene['id'], 'mask': mode, **summarize_metric(totals, grid),
                            'controls': {name: summarize_metric(value, grid) for name, value in control_totals.items()}})
    return {'scene_mean_rmse_sr_inv': float(np.mean([r['rmse_sr_inv'] for r in results])), 'scenes': results}


def pool_pixels(values, patch):
    b, h, w, c = values.shape
    return values.reshape(b, h // patch, patch, w // patch, patch, c).mean((2, 4))


@torch.no_grad()
def extract_features(model, scenes, layout, grid, mean, scale, device, noisy=False):
    chunks = []
    rng = np.random.default_rng(7000)
    for scene in scenes:
        for start in range(0, len(scene['x']), 4):
            raw = scene['x'][start:start + 4]
            if noisy:
                sigma = scene['uncertainty'][start:start + 4]
                sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 0)
                noise = np.sqrt(.8) * rng.normal(size=raw.shape[:-1] + (1,)) + np.sqrt(.2) * rng.normal(size=raw.shape)
                raw = (raw + sigma * noise).astype('float32')
            values = (raw - mean) / scale
            if isinstance(model, SpectralPCA):
                features = pool_pixels(model.encode(values), layout.patch_size)
            else:
                model.eval()
                x = torch.as_tensor(values, device=device)
                if isinstance(model, GroupedSpectralMAE):
                    features = model.spatial_features(x, grid).cpu().numpy()
                else:
                    features = pool_pixels(model.encode(x, torch.zeros_like(x, dtype=torch.bool)).cpu().numpy(),
                                           layout.patch_size)
            chunks.append(features.reshape(-1, features.shape[-1]))
    result = np.concatenate(chunks)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite encoder features.")
    return result


def discovery_outputs(model, scenes, layout, grid, mean, scale, device, count, seed, output):
    output.mkdir()
    features = extract_features(model, scenes, layout, grid, mean, scale, device)
    noisy_features = extract_features(model, scenes, layout, grid, mean, scale, device, noisy=True)
    spectra = np.concatenate([pool_pixels(s['x'], layout.patch_size).reshape(-1, len(grid)) for s in scenes])
    scene_indices, positions, audit = [], [], []
    for i, scene in enumerate(scenes):
        n, h, w, _ = scene['x'].shape
        rows, cols = np.indices((h // layout.patch_size, w // layout.patch_size))
        offsets = np.stack((rows, cols), -1).reshape(-1, 2) * layout.patch_size
        positions.append((scene['pixels'][:, None] + offsets[None]).reshape(-1, 2))
        scene_indices.extend([i] * (n * len(offsets)))
        # Audit fields are chip means, not extra predictor channels.
        audit.append(np.repeat(np.column_stack((scene['locations'], scene['pixels'], scene['audit'])),
                               len(offsets), axis=0))
    scene_indices = np.asarray(scene_indices)
    positions, audit = np.concatenate(positions), np.concatenate(audit)
    train = np.isin(scene_indices, [i for i, s in enumerate(scenes) if s['split'] == 'train'])
    test = np.isin(scene_indices, [i for i, s in enumerate(scenes) if s['split'] == 'test'])
    rng = np.random.default_rng(0)
    training_indices = [np.flatnonzero(scene_indices == i) for i, s in enumerate(scenes) if s['split'] == 'train']
    # shortcut: 128 fit/retrieval patches per scene, enlarge after checking rare-pattern coverage and cost.
    bank_size = min(128, min(map(len, training_indices)))
    bank = np.concatenate([rng.choice(indices, bank_size, replace=False) for indices in training_indices])
    center, spread = features[bank].mean(0), np.maximum(features[bank].std(0), 1e-6)
    z, noisy_z = (features - center) / spread, (noisy_features - center) / spread
    centers = kmeans(z[bank], count, seed)
    d = distances(z, centers)
    labels = d.argmin(1)
    noisy_labels = distances(noisy_z, centers).argmin(1)
    prototypes = []
    for i in range(count):
        selected = train & (labels == i)
        prototypes.append({'group': i, 'training_patches': int(selected.sum()),
                           'mean_rrs': spectra[selected].mean(0).tolist() if selected.any() else None})
    queries = np.concatenate([rng.choice(np.flatnonzero(scene_indices == i),
                                        min(32, int((scene_indices == i).sum())), replace=False)
                              for i, s in enumerate(scenes) if s['split'] == 'test'])
    neighbors = bank[np.argsort(distances(z[queries], z[bank]), axis=1)[:, :5]]
    perturbed_neighbors = bank[np.argsort(distances(noisy_z[queries], z[bank]), axis=1)[:, :5]]
    overlap = np.mean([len(set(a) & set(b)) / len(a) for a, b in zip(neighbors, perturbed_neighbors)])
    nuisance = []
    for i in range(count):
        selected = test & (labels == i)
        fields = {}
        for col, name in enumerate(('latitude', 'longitude', 'scan_line', 'across_track_pixel', *AUDIT_FIELDS)):
            values = audit[selected, col]
            values = values[np.isfinite(values)]
            fields[name] = {'mean': float(values.mean()), 'p10': float(np.percentile(values, 10)),
                            'p90': float(np.percentile(values, 90))} if len(values) else None
        nuisance.append({'group': i, 'test_patches': int(selected.sum()), 'fields': fields})
    np.savez_compressed(output / 'features.npz', features=features, cluster=labels,
                        distance_to_group=d.min(1), feature_mean=center, feature_scale=spread,
                        centers=centers, scene_index=scene_indices, positions=positions,
                        scene_ids=np.asarray([s['id'] for s in scenes]), spectra=spectra, audit=audit,
                        retrieval_bank=bank, retrieval_queries=queries, neighbors=neighbors)
    report = {'embedding_dim': features.shape[1], 'groups': count, 'prototypes': prototypes,
              'training_fit_patches_per_scene': bank_size,
              'noise_cluster_agreement_test': float((labels[test] == noisy_labels[test]).mean()),
              'noise_neighbor_overlap_at_5': float(overlap),
              'noise': 'Added Rrs_unc-scaled Gaussian noise with assumed spectral rho=0.8; not measured covariance.',
              'nuisance_test': nuisance,
              'scene_group_counts': {s['id']: np.bincount(labels[scene_indices == i], minlength=count).tolist()
                                     for i, s in enumerate(scenes)},
              'interpretation': 'Optical groups and distances; no species labels or calibrated probabilities.'}
    (output / 'groups.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    plot_discovery(scenes, scene_indices, positions, labels, prototypes, grid, count, output)
    return report


def plot_discovery(scenes, indices, positions, labels, prototypes, grid, count, output):
    os.environ.setdefault('MPLCONFIGDIR', str(output / 'matplotlib-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    colors = plt.get_cmap('tab20')(np.linspace(0, 1, count))
    cmap = ListedColormap(colors)
    fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
    for prototype in prototypes:
        if prototype['mean_rrs'] is not None:
            ax.plot(grid, prototype['mean_rrs'], color=colors[prototype['group']],
                    label=f"Group {prototype['group']}")
    ax.set(xlabel='Wavelength [nm]', ylabel='Reflectance [sr⁻¹]', title='Training-group representative spectra')
    ax.legend()
    fig.savefig(output / 'prototype-spectra.png', dpi=150)
    plt.close(fig)
    for i, scene in enumerate(scenes):
        if scene['split'] != 'test':
            continue
        selected = indices == i
        fig, ax = plt.subplots(figsize=(7, 7), layout='constrained')
        image = ax.scatter(positions[selected, 1], positions[selected, 0], c=labels[selected],
                           marker='s', s=6, cmap=cmap, vmin=-.5, vmax=count - .5)
        ax.set(xlim=(0, scene['shape'][1]), ylim=(scene['shape'][0], 0),
               xlabel='Across-track pixel', ylabel='Scan line', title=f"{scene['id']} · held-out optical groups")
        fig.colorbar(image, ax=ax, ticks=np.arange(count), label='Group ID')
        fig.savefig(output / f"map-{scene['id']}.png", dpi=150)
        plt.close(fig)


def run(args):
    if (args.method not in ('mae', 'pca', 'spectral-ae')
            or min(args.steps, args.batch_size, args.threads, args.patch_size, args.group_size, args.chips) < 1
            or args.chip_size < 4 or args.chip_size % args.patch_size or args.chip_size // args.patch_size < 2
            or args.clusters < 2 or args.masking not in ('mixed', 'random')):
        raise ValueError("Invalid discovery dimensions, budget, method, or mask.")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    scenes, grid = load_aop_scenes(args.scenes, args.data_dir, args.chip_size, args.chips)
    train = [s for s in scenes if s['split'] == 'train']
    validation = [s for s in scenes if s['split'] == 'validation']
    test = [s for s in scenes if s['split'] == 'test']
    mean = float(np.mean([s['x'].mean(dtype='float64') for s in train]))
    scale = max(float(np.sqrt(np.mean([((s['x'].astype('float64') - mean) ** 2).mean() for s in train]))), 1e-6)
    fallback = np.mean([s['x'].mean((0, 1, 2)) for s in train], 0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    layout = GroupedSpectralMAE(grid, args.group_size, args.patch_size, args.embed_dim).to(device)
    output = args.output / 'discovery'
    output.mkdir(parents=True, exist_ok=False)
    history, random_report = [], None
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    if args.method == 'pca':
        model = SpectralPCA().fit(train, mean, scale)
        candidates = sorted({min(rank, len(grid) - 1) for rank in (4, 8, 16, 32)})
        for rank in candidates:
            model.rank = rank
            scores = evaluate(model, validation, layout, grid, mean, scale, device, args.batch_size)
            history.append({'rank': rank, 'validation_rmse_sr_inv': scores['scene_mean_rmse_sr_inv']})
        model.rank = min(history, key=lambda item: item['validation_rmse_sr_inv'])['rank']
        np.savez(output / 'model.npz', mean=model.mean, components=model.components[:model.rank],
                 normalization_mean=mean, normalization_scale=scale, wavelengths_nm=grid)
    else:
        model = layout if args.method == 'mae' else SpectralAutoencoder(len(grid), args.embed_dim).to(device)
        if args.method == 'mae':
            torch.save({'state_dict': model.state_dict(), 'config': config, 'wavelengths_nm': grid.tolist(),
                        'mean': mean, 'scale': scale}, output / 'untrained.pt')
            random_report = discovery_outputs(model, scenes, layout, grid, mean, scale, device,
                                               args.clusters, args.seed, output / 'untrained')
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=.01)
        best = float('inf')
        for step in range(args.steps):
            model.train()
            selected = [train[rng.integers(len(train))] for _ in range(args.batch_size)]
            raw = np.stack([s['x'][rng.integers(len(s['x']))] for s in selected])
            x = torch.as_tensor((raw - mean) / scale, device=device)
            mask = token_mask(len(x), x.shape[1] // args.patch_size, x.shape[2] // args.patch_size,
                              layout.groups, args.masking, args.seed * 1000000 + step).to(device)
            if args.method == 'mae':
                prediction, _, target = model(x, grid, mask)
                loss = model.masked_loss(prediction, mask, target)
            else:
                hidden = layout.unpatchify(mask[..., None].expand(-1, -1, args.patch_size ** 2 * args.group_size), x.shape)
                loss = (model(x, hidden)[hidden] - x[hidden]).square().mean()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite discovery training loss.')
            optimizer.zero_grad()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(norm):
                raise ValueError('Nonfinite discovery gradients.')
            optimizer.step()
            if step == 0 or (step + 1) % 50 == 0 or step == args.steps - 1:
                scores = evaluate(model, validation, layout, grid, mean, scale, device, args.batch_size)
                history.append({'step': step + 1, 'training_mse': float(loss.detach()),
                                'validation_rmse_sr_inv': scores['scene_mean_rmse_sr_inv']})
                print(json.dumps(history[-1]), flush=True)
                if scores['scene_mean_rmse_sr_inv'] < best:
                    best = scores['scene_mean_rmse_sr_inv']
                    torch.save({'state_dict': model.state_dict(), 'config': config, 'wavelengths_nm': grid.tolist(),
                                'mean': mean, 'scale': scale}, output / 'best.pt')
        model.load_state_dict(torch.load(output / 'best.pt', map_location=device, weights_only=True)['state_dict'])
    validation_scores = evaluate(model, validation, layout, grid, mean, scale, device, args.batch_size, fallback)
    test_scores = evaluate(model, test, layout, grid, mean, scale, device, args.batch_size, fallback)
    groups = discovery_outputs(model, scenes, layout, grid, mean, scale, device,
                               args.clusters, args.seed, output / 'trained')
    metadata = [{key: value for key, value in s.items()
                 if key not in ('x', 'uncertainty', 'pixels', 'locations', 'audit')} for s in scenes]
    report = {'config': config, 'device': str(device), 'torch_version': torch.__version__,
              'wavelengths_nm': grid.tolist(), 'normalization': {'mean': mean, 'scale': scale},
              'excluded_flags': list(AOP_EXCLUDE_FLAGS), 'scene_manifest': metadata,
              'history': history, 'validation': validation_scores, 'test': test_scores,
              'discovery': groups, 'untrained_discovery': random_report,
              'scope': ('Single-scene spatial pilot; retrieval is between blocks of the same acquisition. '
                        'No evidence of transfer across dates or regions.'
                        if scenes[0].get('scope') == 'single-scene-spatial-pilot' else
                        'Unlabeled optical patterns; no concentration regression or biological identity claim.')}
    (output / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'method': args.method, 'test_scene_mean_rmse_sr_inv': test_scores['scene_mean_rmse_sr_inv']}))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenes', type=Path, default=os.environ.get('SCENE_MANIFEST'))
    parser.add_argument('--data-dir', type=Path, default=Path(os.environ.get('DATA_DIR', 'data/raw')))
    parser.add_argument('--output', type=Path, default=Path(os.environ.get('OUTPUT_DIR', 'outputs/discovery')))
    parser.add_argument('--method', choices=('mae', 'pca', 'spectral-ae'), default='mae')
    parser.add_argument('--masking', choices=('mixed', 'random'), default='mixed')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--chips', type=int, default=128)
    parser.add_argument('--chip-size', type=int, default=16)
    parser.add_argument('--patch-size', type=int, default=2)
    parser.add_argument('--group-size', type=int, default=8)
    parser.add_argument('--embed-dim', type=int, default=96)
    parser.add_argument('--clusters', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.scenes is None:
        parser.error('Supply --scenes or SCENE_MANIFEST with real AOP scene splits.')
    run(args)


if __name__ == '__main__':
    main()
