"""
Lightweight DiT variant tailored for point cloud data.

The design mirrors the point-cloud DiT used in DiT_I23D_PCD_PixelArt_noclip
from GaussianAnything, but distilled to fit the simplified transformer stack
in this repository.  Each point is embedded with a small MLP that mixes raw
coordinates, optional features, and Fourier-style positional encodings before
being processed by ``ContinuousTransformer``.
"""

import math
import typing as tp

import torch
from torch import nn
from einops import rearrange

from .dit import DiffusionTransformer
from utils.torch_common import exists


class FourierPointEmbedding(nn.Module):
    """Projects 3D coordinates to a periodic embedding."""

    def __init__(self, coord_dim: int, embed_dim: int, std: float = 1.0):
        super().__init__()
        assert embed_dim % 2 == 0, "embed_dim must be divisible by 2."
        self.coord_dim = coord_dim
        self.weight = nn.Parameter(torch.randn(coord_dim, embed_dim // 2) * std)
        self.pi = math.pi

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (batch, points, coord_dim) tensor assumed to be in [-1, 1].
        """
        proj = 2 * self.pi * coords @ self.weight
        return torch.cat([proj.sin(), proj.cos()], dim=-1)


class PointCloudEmbedder(nn.Module):
    """Embeds point coordinates and optional features to the transformer width."""

    def __init__(
        self,
        coord_dim: int,
        feature_dim: int,
        embed_dim: int,
        pos_embed_dim: int,
        use_fourier: bool = True,
    ):
        super().__init__()

        self.use_fourier = use_fourier
        if use_fourier:
            self.positional = FourierPointEmbedding(coord_dim, pos_embed_dim)
            in_dim = pos_embed_dim + feature_dim
        else:
            in_dim = coord_dim + feature_dim

        # Shallow MLP keeps the behaviour close to the reference implementation.
        self.to_embed = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        coords: torch.Tensor,
        features: tp.Optional[torch.Tensor] = None,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            coords: (batch, points, coord_dim)
            features: optional (batch, points, feature_dim)
            mask: optional boolean (batch, points) where False denotes padded points
        """
        pieces = []
        if self.use_fourier:
            pieces.append(self.positional(coords))
        else:
            pieces.append(coords)

        if exists(features):
            pieces.append(features)

        x = torch.cat(pieces, dim=-1)
        x = self.to_embed(x)
        x = self.norm(x)

        if exists(mask):
            x = x.masked_fill(~mask[..., None], 0.0)

        return x


class PointCloudDiffusionTransformer(nn.Module):
    """
    Diffusion transformer specialised for point clouds.

    The module embeds unordered point sets, processes them with the existing
    DiT backbone, and predicts per-point velocities (or any target of interest).
    """

    def __init__(
        self,
        coord_dim: int = 3,
        feature_dim: int = 0,
        embed_dim: int = 128,
        model_dim: int = 512,
        depth: int = 12,
        num_heads: int = 8,
        cond_token_dim: tp.Optional[int] = None,
        use_shortcut: bool = True,
        pos_embed_dim: int = 128,
        output_dim: tp.Optional[int] = None,
        use_fourier: bool = True,
        **dit_kwargs,
    ):
        super().__init__()

        self.coord_dim = coord_dim
        self.feature_dim = feature_dim
        self.output_dim = output_dim or coord_dim

        self.embedder = PointCloudEmbedder(
            coord_dim=coord_dim,
            feature_dim=feature_dim,
            embed_dim=embed_dim,
            pos_embed_dim=pos_embed_dim,
            use_fourier=use_fourier,
        )

        self.dit = DiffusionTransformer(
            dim_in=embed_dim,
            dim=model_dim,
            depth=depth,
            num_heads=num_heads,
            cond_token_dim=cond_token_dim,
            use_shortcut=use_shortcut,
            **dit_kwargs,
        )

        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, self.output_dim),
        )

    def forward(
        self,
        coords: torch.Tensor,
        t: torch.Tensor,
        dt: tp.Optional[torch.Tensor] = None,
        *,
        point_features: tp.Optional[torch.Tensor] = None,
        mask: tp.Optional[torch.Tensor] = None,
        cross_attn_cond: tp.Optional[torch.Tensor] = None,
        cfg_dropout_prob: float = 0.0,
        cfg_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Args:
            coords: (batch, points, coord_dim)
            t: (batch,)
            dt: optional (batch,)
            point_features: optional (batch, points, feature_dim)
            mask: optional boolean (batch, points) indicating valid points
            cross_attn_cond: optional conditioning tokens identical to DiffusionTransformer
        Returns:
            Tensor of shape (batch, output_dim, points)
        """
        assert coords.dim() == 3, "coords must be (batch, points, coord_dim)."
        if exists(point_features):
            assert point_features.shape[:2] == coords.shape[:2], "Feature shape mismatch."
        if exists(mask):
            assert mask.shape == coords.shape[:2], "Mask shape mismatch."

        embedded = self.embedder(coords, point_features, mask=mask)
        embedded = rearrange(embedded, "b n c -> b c n")

        tokens = self.dit(
            embedded,
            t,
            dt=dt,
            cross_attn_cond=cross_attn_cond,
            cfg_dropout_prob=cfg_dropout_prob,
            cfg_scale=cfg_scale,
        )

        tokens = rearrange(tokens, "b c n -> b n c")
        outputs = self.out_proj(tokens)

        if exists(mask):
            outputs = outputs.masked_fill(~mask[..., None], 0.0)

        return rearrange(outputs, "b n c -> b c n")
