"""Dense log10 chlorophyll prediction and valid-pixel supervision."""

import torch
from torch import nn

from src.spatial_model import SpatialSpectralMAE


class ChlorophyllRegressor(nn.Module):
    """Use all encoder tokens; output one log10 concentration per input pixel."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.embed_dim, backbone.patch_size ** 2)

    def forward(self, x, wavelengths_nm):
        b, c, h, w = x.shape
        p = self.backbone.patch_size
        if isinstance(self.backbone, SpatialSpectralMAE):
            features = self.backbone.encode(x, wavelengths_nm)
        else:
            features = self.backbone.encode(x.permute(0, 2, 3, 1), wavelengths_nm)
            features = features.reshape(b, h // p, w // p, c, -1).mean(dim=3)
        pixels = self.head(features).reshape(b, h // p, w // p, p, p)
        return pixels.permute(0, 1, 3, 2, 4).reshape(b, 1, h, w)


class SpectralMLP(nn.Module):
    """Shared per-pixel spectral regression, without spatial attention."""

    def __init__(self, num_bands, hidden_dim=96):
        super().__init__()
        self.network = nn.Sequential(nn.Conv2d(num_bands, hidden_dim, 1), nn.GELU(),
                                     nn.Conv2d(hidden_dim, hidden_dim, 1), nn.GELU(),
                                     nn.Conv2d(hidden_dim, 1, 1))

    def forward(self, x, wavelengths_nm):
        return self.network(x)


def chlorophyll_loss(log10_prediction, concentration, valid):
    if log10_prediction.shape != concentration.shape or valid.shape != concentration.shape:
        raise ValueError("Prediction, concentration and valid mask shapes must match.")
    if valid.dtype != torch.bool or not valid.any():
        raise ValueError("Use a boolean mask with at least one valid target pixel.")
    target = concentration[valid]
    prediction = log10_prediction[valid]
    if not torch.isfinite(target).all() or (target <= 0).any() or not torch.isfinite(prediction).all():
        raise ValueError("Valid targets must be finite and positive; predictions must be finite.")
    # Predicting log10 directly avoids taking a logarithm of unconstrained outputs.
    return (prediction - target.log10()).square().mean()
