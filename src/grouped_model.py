"""Masked reconstruction of native spatial patches and contiguous band groups."""

import torch
from torch import nn

from src.model import WavelengthAwareMAE, positional_encoding


def token_mask(batch, rows, columns, groups, mode="mixed", seed=0):
    """CPU masks, with whole spectral intervals and contiguous spatial holes."""
    if min(batch, rows, columns, groups) < 1 or rows * columns < 2 or groups < 2:
        raise ValueError("Masking needs multiple spatial patches and band groups.")
    if mode not in ("mixed", "random", "spectral", "spatial"):
        raise ValueError("Unknown masking mode.")
    rng = torch.Generator().manual_seed(seed)
    mask = torch.zeros(batch, rows, columns, groups, dtype=torch.bool)
    for item in range(batch):
        if mode in ("mixed", "spectral"):
            span = max(1, groups // 4)
            start = int(torch.randint(groups - span + 1, (), generator=rng))
            mask[item, :, :, start:start + span] = True
        if mode in ("mixed", "spatial"):
            height, width = max(1, rows // 2), max(1, columns // 2)
            row = int(torch.randint(rows - height + 1, (), generator=rng))
            col = int(torch.randint(columns - width + 1, (), generator=rng))
            mask[item, row:row + height, col:col + width, :] = True
        if mode in ("mixed", "random"):
            flat = mask[item].flatten()
            count = flat.numel() - max(1, int(flat.numel() * .25))
            remaining = (~flat).nonzero().flatten()
            extra = count - int(flat.sum())
            if extra < 0:
                raise ValueError("Structured mask exceeds the 75% masking budget.")
            flat[remaining[torch.randperm(len(remaining), generator=rng)[:extra]]] = True
    return mask.flatten(1)


class GroupedSpectralMAE(WavelengthAwareMAE):
    """BHWC cubes; each token contains p × p pixels × q adjacent bands."""

    def __init__(self, wavelengths_nm, group_size=8, patch_size=2, embed_dim=96):
        super().__init__(embed_dim=embed_dim, patch_size=patch_size,
                         encoder_blocks=2, decoder_blocks=1)
        grid = torch.as_tensor(wavelengths_nm, dtype=torch.float32)
        if (group_size < 1 or grid.ndim != 1 or len(grid) < 2 * group_size
                or len(grid) % group_size or not torch.isfinite(grid).all()
                or (grid <= 0).any() or (grid.diff() <= 0).any()):
            raise ValueError("Need an increasing finite grid divisible into at least two band groups.")
        self.register_buffer("wavelengths_nm", grid.clone())
        self.group_size = group_size
        self.groups = len(grid) // group_size
        self.patch_embed = nn.Linear(patch_size ** 2 * group_size, embed_dim)
        self.pred_head = nn.Linear(embed_dim, patch_size ** 2 * group_size)

    def patchify(self, x):
        p, q = self.patch_size, self.group_size
        if (x.ndim != 4 or min(x.shape) < 1 or x.shape[-1] != len(self.wavelengths_nm)
                or x.shape[1] % p or x.shape[2] % p or not torch.isfinite(x).all()):
            raise ValueError("Use finite BHWC cubes on the model grid and divisible by the spatial patch.")
        b, h, w, _ = x.shape
        return (x.reshape(b, h // p, p, w // p, p, self.groups, q)
                .permute(0, 1, 3, 5, 2, 4, 6).reshape(b, -1, p * p * q))

    def unpatchify(self, patches, shape):
        b, h, w, c = shape
        p, q = self.patch_size, self.group_size
        if (min(shape) < 1 or h % p or w % p or c != len(self.wavelengths_nm)
                or patches.shape != (b, h // p * (w // p) * self.groups, p * p * q)):
            raise ValueError("Grouped patches do not match the requested cube shape.")
        return (patches.reshape(b, h // p, w // p, self.groups, p, p, q)
                .permute(0, 1, 4, 2, 5, 3, 6).reshape(shape))

    def _tokens(self, x, wavelengths_nm):
        grid = torch.as_tensor(wavelengths_nm, dtype=torch.float32, device=x.device)
        if grid.shape != self.wavelengths_nm.shape or not torch.equal(grid, self.wavelengths_nm):
            raise ValueError("Input wavelength grid must match the model exactly.")
        patches = self.patchify(x)
        centers = grid.reshape(self.groups, self.group_size).mean(1)
        positions = positional_encoding(x.shape[1] // self.patch_size,
                                        x.shape[2] // self.patch_size, centers, self.embed_dim)
        tokens = self.patch_embed(patches)
        positions = positions.to(tokens.dtype)
        return tokens + positions, positions, patches

    def forward(self, x, wavelengths_nm, hidden_mask=None):
        tokens, positions, targets = self._tokens(x, wavelengths_nm)
        b, count, dim = tokens.shape
        if hidden_mask is None:
            hidden_mask = token_mask(b, x.shape[1] // self.patch_size,
                                     x.shape[2] // self.patch_size, self.groups).to(x.device)
        if (hidden_mask.dtype != torch.bool or hidden_mask.shape != (b, count)
                or hidden_mask.device != x.device or not hidden_mask.any(1).all()
                or hidden_mask.all(1).any()
                or not (hidden_mask.sum(1) == hidden_mask.sum(1)[0]).all()):
            raise ValueError("Use boolean masks with equal, nonempty visible and hidden token counts.")
        indices = (~hidden_mask).nonzero()[:, 1].reshape(b, -1)
        expanded = indices[..., None].expand(-1, -1, dim)
        encoded = self.encoder(tokens.gather(1, expanded))
        restored = self.mask_token.expand(b, count, dim).clone()
        restored.scatter_(1, expanded, encoded)
        return self.pred_head(self.decoder(restored + positions)), hidden_mask, targets

    def spatial_features(self, x, wavelengths_nm):
        b, h, w, _ = x.shape
        return self.encode(x, wavelengths_nm).reshape(
            b, h // self.patch_size, w // self.patch_size, self.groups, self.embed_dim).mean(3)


class SpectralAutoencoder(nn.Module):
    """Per-pixel denoising baseline; an explicit observed-band mask accompanies zeros."""

    def __init__(self, bands, embed_dim=96):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(2 * bands, 128), nn.GELU(), nn.Linear(128, embed_dim))
        self.decoder = nn.Sequential(nn.Linear(embed_dim, 128), nn.GELU(), nn.Linear(128, bands))

    def encode(self, x, hidden):
        return self.encoder(torch.cat((x.masked_fill(hidden, 0), (~hidden).to(x.dtype)), -1))

    def forward(self, x, hidden):
        return self.decoder(self.encode(x, hidden))
