"""Full-spectrum spatial tokens: one transformer token per image patch."""

import torch
from torch import nn

from src.model import WavelengthAwareMAE, sinusoidal_encoding


class SpectralSpatialPatchEmbed(nn.Module):
    def __init__(self, num_bands, embed_dim=96, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(num_bands, embed_dim, patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class SpatialSpectralMAE(nn.Module):
    """BCHW full-spectrum MAE. The input wavelength grid is fixed per model.

    Wavelengths are checked, not embedded: channels are learned jointly by a
    fixed Conv2d projection. A changed sensor grid needs explicit alignment.
    """

    def __init__(self, wavelengths_nm, embed_dim=96, patch_size=4, num_heads=4,
                 encoder_blocks=4, decoder_blocks=2, masking_ratio=0.75):
        super().__init__()
        wavelengths = torch.as_tensor(wavelengths_nm, dtype=torch.float32)
        if (wavelengths.ndim != 1 or wavelengths.numel() < 1
                or not torch.isfinite(wavelengths).all() or (wavelengths <= 0).any()
                or (wavelengths.diff() <= 0).any()):
            raise ValueError("Provide finite positive wavelengths in strictly increasing nm order.")
        if (embed_dim < 4 or embed_dim % 4 or num_heads < 1 or embed_dim % num_heads
                or patch_size < 1 or min(encoder_blocks, decoder_blocks) < 1):
            raise ValueError("Invalid model dimensions.")
        if not 0 < masking_ratio < 1:
            raise ValueError("Masking ratio must be between zero and one.")
        self.register_buffer("wavelengths_nm", wavelengths.clone())
        self.embed_dim, self.patch_size = embed_dim, patch_size
        self.num_bands, self.masking_ratio = len(wavelengths), masking_ratio
        self.patch_embed = SpectralSpatialPatchEmbed(self.num_bands, embed_dim, patch_size)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, num_heads, embed_dim * 4, dropout=0.,
                                       activation="gelu", batch_first=True, norm_first=True),
            encoder_blocks, enable_nested_tensor=False)
        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, num_heads, embed_dim * 4, dropout=0.,
                                       activation="gelu", batch_first=True, norm_first=True),
            decoder_blocks, enable_nested_tensor=False)
        self.mask_token = nn.Parameter(torch.empty(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=.02)
        self.pred_head = nn.Linear(embed_dim, patch_size ** 2 * self.num_bands)

    def patchify(self, x):
        p = self.patch_size
        if (x.ndim != 4 or min(x.shape) < 1 or x.shape[1] != self.num_bands
                or x.shape[2] % p or x.shape[3] % p or not torch.isfinite(x).all()):
            raise ValueError("Use finite BCHW chips on the model grid, divisible by patch size.")
        b, c, h, w = x.shape
        return (x.reshape(b, c, h // p, p, w // p, p).permute(0, 2, 4, 3, 5, 1)
                .reshape(b, -1, p * p * c))

    def unpatchify(self, patches, shape):
        b, c, h, w = shape
        p = self.patch_size
        if (min(shape) < 1 or c != self.num_bands or h % p or w % p
                or patches.shape != (b, h // p * (w // p), p * p * c)):
            raise ValueError("Patch shape does not match BCHW shape.")
        return (patches.reshape(b, h // p, w // p, p, p, c)
                .permute(0, 5, 1, 3, 2, 4).reshape(shape))

    def _tokens(self, x, wavelengths_nm):
        patches = self.patchify(x)
        wavelengths = torch.as_tensor(wavelengths_nm, device=x.device, dtype=torch.float32)
        if wavelengths.shape != self.wavelengths_nm.shape or not torch.equal(wavelengths, self.wavelengths_nm):
            raise ValueError("Input wavelengths must match the fixed model grid exactly.")
        rows, columns = x.shape[2] // self.patch_size, x.shape[3] // self.patch_size
        row = sinusoidal_encoding(torch.arange(rows, device=x.device), self.embed_dim // 2)
        col = sinusoidal_encoding(torch.arange(columns, device=x.device), self.embed_dim // 2)
        position = torch.cat((row[:, None].expand(rows, columns, -1),
                              col[None].expand(rows, columns, -1)), -1).reshape(-1, self.embed_dim)
        tokens = self.patch_embed(x)
        return tokens + position.to(tokens.dtype), position.to(tokens.dtype), patches

    def encode(self, x, wavelengths_nm):
        tokens, _, _ = self._tokens(x, wavelengths_nm)
        return self.encoder(tokens)

    def forward(self, x, wavelengths_nm):
        tokens, position, target = self._tokens(x, wavelengths_nm)
        b, count, dim = tokens.shape
        if count < 2:
            raise ValueError("Spatial masking needs at least two spatial patches.")
        visible = max(1, min(count - 1, int(count * (1 - self.masking_ratio))))
        ids = torch.rand(b, count, device=x.device).argsort(1)[:, :visible]
        expanded = ids.unsqueeze(-1).expand(-1, -1, dim)
        encoded = self.encoder(tokens.gather(1, expanded))
        decoder_in = self.mask_token.expand(b, count, dim).clone()
        decoder_in.scatter_(1, expanded, encoded)
        prediction = self.pred_head(self.decoder(decoder_in + position))
        mask = torch.ones(b, count, dtype=torch.bool, device=x.device)
        mask.scatter_(1, ids, False)
        return prediction, mask, target

    masked_loss = staticmethod(WavelengthAwareMAE.masked_loss)

    @torch.no_grad()
    def reconstruct(self, x, wavelengths_nm):
        prediction, _, _ = self(x, wavelengths_nm)
        return self.unpatchify(prediction, x.shape)
