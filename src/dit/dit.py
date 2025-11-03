"""
Copyright (C) 2025 Yukara Ikemiya

Adapted from the following repo's code under MIT License.
https://github.com/Stability-AI/stable-audio-tools

-----------------------------------------------------
Diffusion Transformer which supports Shortcut models.
"""

import typing as tp

import torch
from torch import nn
from einops import rearrange

from .transformer import ContinuousTransformer
from utils.torch_common import exists


class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.in_features = in_features
        self.pi = 3.141592653589793
        self.weight = nn.Parameter(torch.randn([in_features, out_features // 2]) * std)

    def forward(self, x: torch.Tensor):
        """
        x: (..., in_features), elements assumes to be between 0 -- 1
        return : (..., out_features)
        """
        assert x.shape[-1] == self.in_features
        f = 2 * self.pi * x @ self.weight
        return torch.cat([f.cos(), f.sin()], dim=-1)


class DiffusionTransformer(nn.Module):
    def __init__(
        self,
        dim_in: int = 32,
        dim: int = 768,
        depth: int = 12,
        num_heads: int = 8,
        cond_token_dim: tp.Optional[int] = None,
        # shortcut models
        use_shortcut: bool = True,
        **kwargs
    ):
        super().__init__()

        self.use_cond = exists(cond_token_dim)
        self.use_shortcut = use_shortcut

        # timestep embeddings
        dim_time_feat = 256
        self.t_embedder = FourierFeatures(1, dim_time_feat)
        if use_shortcut:
            self.dt_embedder = FourierFeatures(1, dim_time_feat)

        self.to_timestep_embed = nn.Sequential(
            nn.Linear(dim_time_feat, dim, bias=True),
            nn.SiLU(),
            nn.Linear(dim, dim, bias=True),
        )

        if self.use_cond:
            # conditioning tokens
            self.cond_token_dim = cond_token_dim
            self.to_cond_embed = nn.Sequential(
                nn.Linear(cond_token_dim, cond_token_dim, bias=False),
                nn.SiLU(),
                nn.Linear(cond_token_dim, cond_token_dim, bias=False)
            )

            self.cfg_emb = nn.Parameter(torch.randn(cond_token_dim))

        # Transformer

        self.transformer = ContinuousTransformer(
            dim=dim,
            depth=depth,
            dim_heads=dim // num_heads,
            dim_in=dim_in,
            dim_out=dim_in,
            cross_attend=self.use_cond,
            cond_token_dim=cond_token_dim,
            **kwargs
        )

        self.preprocess_conv = nn.Conv1d(dim_in, dim_in, 1, bias=False)
        nn.init.zeros_(self.preprocess_conv.weight)
        self.postprocess_conv = nn.Conv1d(dim_in, dim_in, 1, bias=False)
        nn.init.zeros_(self.postprocess_conv.weight)

    def _forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: tp.Optional[torch.Tensor] = None,
        cross_attn_cond: tp.Optional[torch.Tensor] = None,
        **kwargs
    ):

        if self.use_cond:
            # condition embeddings
            cross_attn_cond = self.to_cond_embed(cross_attn_cond)

        # timestep embeddings
        t_embed = self.to_timestep_embed(self.t_embedder(t[:, None, None]))  # (b, 1, embed_dim)

        if self.use_shortcut:
            dt_embed = self.to_timestep_embed(self.dt_embedder(dt[:, None, None]))  # (b, 1, embed_dim)
            t_embed = torch.cat([t_embed, dt_embed], dim=1)  # (b, 2, embed_dim)

        # preprocess
        x = self.preprocess_conv(x) + x
        x = rearrange(x, "b c t -> b t c")

        # main transformer
        output = self.transformer(x, prepend_cond=t_embed, context=cross_attn_cond, **kwargs)

        # postprocess
        output = rearrange(output, "b t c -> b c t")
        output = self.postprocess_conv(output) + output

        return output

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: tp.Optional[torch.Tensor] = None,
        cross_attn_cond: tp.Optional[torch.Tensor] = None,
        # CFG
        cfg_dropout_prob: float = 0.0,
        cfg_scale=1.0,
        **kwargs
    ):
        bs = x.shape[0]
        assert len(t) == bs
        assert self.use_shortcut == (exists(dt) and len(dt) == bs)

        if self.use_cond:
            assert exists(cross_attn_cond)
            bs_cond, l_cond, dim_cond = cross_attn_cond.shape
            assert bs_cond == bs and dim_cond == self.cond_token_dim

        # CFG dropout
        if self.use_cond and self.training and cfg_dropout_prob > 0.0:
            idxs_null = torch.rand(bs) < cfg_dropout_prob
            cross_attn_cond[idxs_null] = self.cfg_emb

        # CFG sampling
        cfg_sampling = self.use_cond and not self.training and cfg_scale != 1.0
        if not cfg_sampling:
            return self._forward(
                x, t, dt=dt, cross_attn_cond=cross_attn_cond, **kwargs)
        else:
            # Classifier-free guidance
            # Concatenate conditioned and unconditioned inputs on the batch dimension
            x = torch.cat([x, x], dim=0)
            t = torch.cat([t, t], dim=0)
            if self.use_shortcut:
                dt = torch.cat([dt, dt], dim=0)

            cfg_embs = torch.zeros_like(cross_attn_cond, device=cross_attn_cond.device) + self.cfg_emb
            cross_attn_cond = torch.cat([cross_attn_cond, cfg_embs], dim=0)

            batch_output = self._forward(
                x, t, dt=dt, cross_attn_cond=cross_attn_cond, **kwargs)

            cond_output, uncond_output = torch.chunk(batch_output, 2, dim=0)
            cfg_output = uncond_output + (cond_output - uncond_output) * cfg_scale

            return cfg_output
