"""Spatial/spectral masked autoencoder, with OETHER as the design reference."""

import math

import torch
from torch import nn


def sinusoidal_encoding(positions, dim):
    if dim < 2 or dim % 2:
        raise ValueError("Position dimension must be positive and even.")
    frequencies = torch.exp(
        torch.arange(0, dim, 2, device=positions.device, dtype=torch.float32)
        * (-math.log(10000.0) / dim)
    )
    angles = positions.float().unsqueeze(-1) * frequencies
    return torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(-2)


def positional_encoding(rows, columns, wavelengths_nm, dim):
    if dim < 6 or dim % 6:
        raise ValueError("Embedding dimension must be a positive multiple of six.")
    if (wavelengths_nm.ndim != 1 or wavelengths_nm.numel() == 0
            or not torch.isfinite(wavelengths_nm).all() or (wavelengths_nm <= 0).any()):
        raise ValueError("Provide one finite positive wavelength in nm per band.")
    device = wavelengths_nm.device
    bands = wavelengths_nm.numel()
    h = sinusoidal_encoding(torch.arange(rows, device=device), dim // 3)
    w = sinusoidal_encoding(torch.arange(columns, device=device), dim // 3)
    # Fixed nm units give a physical wavelength the same position across sensors.
    c = sinusoidal_encoding(wavelengths_nm, dim // 3)
    return torch.cat((
        h[:, None, None].expand(rows, columns, bands, -1),
        w[None, :, None].expand(rows, columns, bands, -1),
        c[None, None].expand(rows, columns, bands, -1),
    ), dim=-1).reshape(rows * columns * bands, dim)


class WavelengthAwareMAE(nn.Module):
    """BHWC chips → reconstructed patches, hidden-token mask, target patches.

    Each spatial patch at one wavelength is a token. The shared projection
    keeps its weights when the chip size or number of bands changes.
    """

    def __init__(self, embed_dim=96, patch_size=4, num_heads=4,
                 encoder_blocks=4, decoder_blocks=2, masking_ratio=0.75):
        super().__init__()
        if (embed_dim < 6 or embed_dim % 6 or num_heads < 1 or embed_dim % num_heads
                or patch_size < 1 or encoder_blocks < 1 or decoder_blocks < 1):
            raise ValueError("Invalid model dimensions.")
        if not 0 < masking_ratio < 1:
            raise ValueError("Masking ratio must be between zero and one.")
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.masking_ratio = masking_ratio
        self.patch_embed = nn.Linear(patch_size ** 2, embed_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, num_heads, embed_dim * 4,
                                       dropout=0.0, activation="gelu", batch_first=True,
                                       norm_first=True),
            encoder_blocks, enable_nested_tensor=False,
        )
        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, num_heads, embed_dim * 4,
                                       dropout=0.0, activation="gelu", batch_first=True,
                                       norm_first=True),
            decoder_blocks, enable_nested_tensor=False,
        )
        self.mask_token = nn.Parameter(torch.empty(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.pred_head = nn.Linear(embed_dim, patch_size ** 2)

    def patchify(self, x):
        if x.ndim != 4 or not torch.isfinite(x).all():
            raise ValueError("Use a finite BHWC tensor; remove invalid chips first.")
        batch, height, width, bands = x.shape
        p = self.patch_size
        if min(x.shape) < 1 or height % p or width % p:
            raise ValueError("Chip height and width must be positive multiples of patch size.")
        return (x.reshape(batch, height // p, p, width // p, p, bands)
                .permute(0, 1, 3, 5, 2, 4).reshape(batch, -1, p * p))

    def unpatchify(self, patches, shape):
        batch, height, width, bands = shape
        p = self.patch_size
        if (min(shape) < 1 or height % p or width % p
                or patches.shape != (batch, height // p * (width // p) * bands, p * p)):
            raise ValueError("Patch shape does not match the requested BHWC shape.")
        return (patches.reshape(batch, height // p, width // p, bands, p, p)
                .permute(0, 1, 4, 2, 5, 3).reshape(shape))

    def _tokens(self, x, wavelengths_nm):
        patches = self.patchify(x)
        wavelengths = torch.as_tensor(wavelengths_nm, device=x.device)
        if wavelengths.shape != (x.shape[-1],):
            raise ValueError("Wavelength count must match the number of bands.")
        positions = positional_encoding(x.shape[1] // self.patch_size,
                                        x.shape[2] // self.patch_size,
                                        wavelengths, self.embed_dim)
        tokens = self.patch_embed(patches)
        positions = positions.to(tokens.dtype)
        return tokens + positions, positions, patches

    def encode(self, x, wavelengths_nm):
        """Encode every token for downstream concentration regression."""
        tokens, _, _ = self._tokens(x, wavelengths_nm)
        return self.encoder(tokens)

    def forward(self, x, wavelengths_nm, masking="random"):
        tokens, positions, patches = self._tokens(x, wavelengths_nm)
        batch, count, dim = tokens.shape
        if count < 2:
            raise ValueError("Masked reconstruction needs at least two tokens.")
        visible = max(1, min(count - 1, int(count * (1 - self.masking_ratio))))
        priority = torch.rand(batch, count, device=x.device)
        if masking == "mixed":
            if self.masking_ratio < .75:
                raise ValueError("Mixed masking requires a masking ratio of at least 0.75.")
            bands = x.shape[-1]
            spatial = count // bands
            if min(bands, spatial) < 2:
                raise ValueError("Mixed masking needs at least two bands and spatial patches.")
            span = max(1, bands // 4)
            priority = priority.reshape(batch, spatial, bands)
            for item in range(batch):
                start = int(torch.randint(bands - span + 1, (), device=x.device))
                priority[item, :, start:start + span] += 2
                locations = torch.randperm(spatial, device=x.device)[:max(1, spatial // 4)]
                priority[item, locations, :] += 2
            priority = priority.reshape(batch, count)
        elif masking != "random":
            raise ValueError("Masking must be random or mixed.")
        indices = priority.argsort(dim=1)[:, :visible]
        expanded = indices.unsqueeze(-1).expand(-1, -1, dim)
        encoded = self.encoder(tokens.gather(1, expanded))
        decoder_in = self.mask_token.expand(batch, count, dim).clone()
        decoder_in.scatter_(1, expanded, encoded)
        prediction = self.pred_head(self.decoder(decoder_in + positions))
        mask = torch.ones(batch, count, device=x.device, dtype=torch.bool)
        mask.scatter_(1, indices, False)
        return prediction, mask, patches

    @staticmethod
    def masked_loss(prediction, mask, targets):
        if prediction.shape != targets.shape or mask.shape != prediction.shape[:2]:
            raise ValueError("Prediction, target and mask shapes must match.")
        if mask.dtype != torch.bool or not mask.any():
            raise ValueError("Use a boolean mask with at least one hidden token.")
        return (prediction - targets).square().mean(dim=-1)[mask].mean()

    @torch.no_grad()
    def reconstruct(self, x, wavelengths_nm):
        """Reconstruct with a fresh random mask, including in evaluation mode."""
        prediction, _, _ = self(x, wavelengths_nm)
        return self.unpatchify(prediction, x.shape)
