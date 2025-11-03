"""
Shortcut model wrapper tailored for point cloud data.

The implementation mirrors the flow / shortcut objective used for images but
operates on sets of 3D points with optional per-point features (e.g. RGB).
It relies on ``PointCloudDiffusionTransformer`` to process unordered point sets
and supports the same self-consistency training used in the image shortcut
model.
"""

from __future__ import annotations

import typing as tp

import torch
from torch import nn
from torch.nn import functional as F

from dit import PointCloudDiffusionTransformer
from utils.torch_common import exists


class PointCloudShortcutModel(nn.Module):
    def __init__(
        self,
        coord_dim: int,
        feature_dim: int,
        num_points: int,
        model_config: dict,
        cond_input_dim: tp.Optional[int] = None,
    ):
        """
        Args:
            coord_dim: Number of coordinate dimensions (e.g. 3 for xyz).
            feature_dim: Additional per-point feature dimension (e.g. rgb -> 3).
            num_points: Expected number of points per example.
            cond_input_dim: Dimension of conditioning features provided by dataset.
            model_config: Parameters forwarded to PointCloudDiffusionTransformer.
        """
        super().__init__()

        self.coord_dim = coord_dim
        self.feature_dim = feature_dim
        self.num_points = num_points
        self.total_dim = coord_dim + feature_dim
        self.cond_token_dim = model_config["cond_token_dim"]
        if cond_input_dim is None:
            cond_input_dim = self.cond_token_dim
        self.cond_embedder = nn.Linear(cond_input_dim, self.cond_token_dim)

        expected_output_dim = model_config.get("output_dim", self.total_dim)
        if expected_output_dim != self.total_dim:
            raise ValueError(
                "PointCloudDiffusionTransformer must output coord_dim + feature_dim "
                f"channels (expected {self.total_dim}, got {expected_output_dim})."
            )

        self.model = PointCloudDiffusionTransformer(**model_config)

    # --------------------------------------------------------------------- util
    def _split_state(
        self, state: torch.Tensor
    ) -> tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]:
        """
        Args:
            state: Tensor of shape (batch, total_dim, num_points)
        Returns:
            coords: (batch, num_points, coord_dim)
            features: (batch, num_points, feature_dim) or None
        """
        coords = state[:, : self.coord_dim, :].transpose(1, 2)
        feat = None
        if self.feature_dim > 0:
            feat = state[:, self.coord_dim :, :].transpose(1, 2)
        return coords, feat

    def _model_forward(
        self,
        state: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        conds: tp.Optional[torch.Tensor],
        cfg_dropout_prob: float,
        cfg_scale: float = 1.0,
    ) -> torch.Tensor:
        coords, feats = self._split_state(state)
        output = self.model(
            coords,
            t,
            dt=dt,
            point_features=feats,
            cross_attn_cond=conds,
            cfg_dropout_prob=cfg_dropout_prob,
            cfg_scale=cfg_scale,
        )
        return output

    # ---------------------------------------------------------------- training
    def train_step(
        self,
        points: torch.Tensor,
        conds: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        num_self_consistency: int,
        cfg_dropout_prob: float = 0.1,
    ):
        """
        Args:
            points: (batch, num_points, total_dim) tensor with [coords | features].
            conds: (batch, cond_len, cond_dim) conditioning tokens.
            t: (batch,) current timesteps.
            dt: (batch,) delta timesteps provided by the sampler.
            num_self_consistency: Number of samples assigned to shortcut objective.
        """
        bs, n_points, dim = points.shape
        assert dim == self.total_dim, "Point dimension mismatch."
        assert len(t) == bs and len(dt) == bs
        assert num_self_consistency < bs

        # Arrange as (batch, channels, points) for the DiT
        x1 = points.transpose(1, 2).contiguous()

        x0 = torch.randn_like(x1)
        x_t = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
        v_t = x1 - x0

        cond_tokens = self.cond_embedder(conds)

        if num_self_consistency > 0:
            x_t_sc = x_t[:num_self_consistency]
            t_sc = t[:num_self_consistency]
            dt_half = dt[:num_self_consistency] * 0.5
            conds_sc = cond_tokens[:num_self_consistency]

            with torch.no_grad():
                v1_sc = self._model_forward(
                    x_t_sc, t_sc, dt_half, conds_sc, cfg_dropout_prob=0.0
                )
                v2_sc = self._model_forward(
                    x_t_sc + dt_half[:, None, None] * v1_sc,
                    t_sc + dt_half,
                    dt_half,
                    conds_sc,
                    cfg_dropout_prob=0.0,
                )
            v_t_sc = (v1_sc + v2_sc) / 2.0

        dt = dt.clone()
        dt[num_self_consistency:] = 0.0

        v_out = self._model_forward(
            x_t, t, dt, cond_tokens, cfg_dropout_prob=cfg_dropout_prob
        )

        loss = F.mse_loss(v_t[num_self_consistency:], v_out[num_self_consistency:])
        output: tp.Dict[str, torch.Tensor] = {}
        output["loss_fm"] = loss.detach()

        if num_self_consistency > 0:
            loss_sc = F.mse_loss(v_t_sc, v_out[:num_self_consistency])
            loss += loss_sc
            output["loss_sc"] = loss_sc.detach()

        output["loss"] = loss
        return output

    def forward(
        self,
        state: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        conds: tp.Optional[torch.Tensor],
        cfg_dropout_prob: float = 0.0,
        cfg_scale: float = 1.0,
    ) -> torch.Tensor:
        return self._model_forward(
            state, t, dt, conds, cfg_dropout_prob=cfg_dropout_prob, cfg_scale=cfg_scale
        )

    # ---------------------------------------------------------------- sampling
    @torch.no_grad()
    def sample(
        self,
        conds: torch.Tensor,
        *,
        n_step: tp.Optional[int] = None,
        dt_list: tp.Optional[tp.List[float]] = None,
        disable_shortcut: bool = False,
        num_points: tp.Optional[int] = None,
    ) -> torch.Tensor:
        assert exists(n_step) or exists(dt_list), "Specify n_step or dt_list."
        device = conds.device
        num_sample = conds.shape[0]
        n_points = num_points or self.num_points

        if exists(n_step):
            dt_list = [1.0 / n_step] * n_step

        assert sum(dt_list) <= 1 + 1e-6

        cond_tokens = self.cond_embedder(conds)
        x = torch.randn(num_sample, self.total_dim, n_points, device=device)

        t_cur = torch.zeros(num_sample, device=device)
        for dt_val in dt_list:
            dt = torch.full((num_sample,), dt_val, device=device)
            dt_in = torch.zeros_like(dt) if disable_shortcut else dt

            vel = self._model_forward(
                x, t_cur, dt_in, cond_tokens, cfg_dropout_prob=0.0, cfg_scale=1.0
            )
            x = x + vel * dt[:, None, None]
            t_cur = t_cur + dt

        return x.transpose(1, 2).contiguous()
