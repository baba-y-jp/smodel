"""
Copyright (C) 2025 Yukara Ikemiya

-----------------------------------------------------
Shortcut Models.
"""

import typing as tp

import torch
from torch import nn
from torch.nn import functional as F

from dit import DiffusionTransformer
from utils.torch_common import exists


class ShortcutModel(nn.Module):
    """
    Shortcut model class for label-conditioned image generation
    """

    def __init__(
        self,
        num_label: int,
        patch_width: int,
        model_config: dict
    ):
        super().__init__()

        self.num_label = num_label
        self.patch_width = patch_width
        self.input_shape = None

        self.label_embedder = nn.Embedding(num_label, model_config['cond_token_dim'])
        self.model = DiffusionTransformer(**model_config)

    def patchify_2d(self, x: torch.Tensor):
        """ Patchify 2-d data (e.g. image)"""
        bs, ch, H, W = x.shape
        assert H % self.patch_width == 0 and W % self.patch_width == 0
        x = F.unfold(x, kernel_size=self.patch_width, stride=self.patch_width)
        # update input info
        self.ch_in, self.H_in, self.W_in = ch, H, W
        return x

    def unpatchify_2d(self, x: torch.Tensor, H_in: int, W_in: int):
        """ Unpatchify 2-d data (e.g. image) """
        bs, ch, L = x.shape
        x = F.fold(x, output_size=(H_in, W_in), kernel_size=self.patch_width, stride=self.patch_width)
        return x

    def train_step(
        self,
        images: torch.Tensor,
        labels: torch.LongTensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        num_self_consistency: int,
        cfg_dropout_prob: float = 0.1
    ):
        """
        x1: ground-truth data (e.g. image)
        """
        bs, ch, H, W = images.shape
        assert len(labels) == len(t) == len(dt) == bs and torch.all(t + dt <= 1.0)
        assert num_self_consistency < bs

        # patchfy
        x1 = self.patchify_2d(images)

        x0 = torch.randn_like(x1)  # noise
        x_t = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1  # eq.(1)
        v_t = x1 - x0
        conds = self.label_embedder(labels).unsqueeze(1)  # (bs, 1, dim_cond)

        if num_self_consistency > 0:
            x_t_sc = x_t[:num_self_consistency]
            t_sc = t[:num_self_consistency]
            dt_half = dt[:num_self_consistency] * 0.5
            conds_sc = conds[:num_self_consistency]
            # calculate targets for self-consistency term (eq.(5))
            with torch.no_grad():
                v1_sc = self.model(x_t_sc, t_sc, dt_half, conds_sc, cfg_dropout_prob=0.0)
                v2_sc = self.model(x_t_sc + dt_half[:, None, None] * v1_sc, t_sc + dt_half, dt_half, conds_sc, cfg_dropout_prob=0.0)

            v_t_sc = (v1_sc + v2_sc) / 2.

        # dt = 0.0 -> naive flow-matching
        dt[num_self_consistency:] = 0.

        # forward
        v_out = self.model(x_t, t, dt, conds, cfg_dropout_prob=cfg_dropout_prob)

        output = {}

        # flow-matching loss (eq.(5))
        loss = F.mse_loss(v_t[num_self_consistency:], v_out[num_self_consistency:])
        output['loss_fm'] = loss.detach()

        # self-consistency loss (eq.(5))
        if num_self_consistency > 0:
            loss_sc = F.mse_loss(v_t_sc, v_out[:num_self_consistency])
            loss += loss_sc
            output['loss_sc'] = loss_sc.detach()

        output['loss'] = loss
        return output

    @torch.no_grad()
    def sample(
        self,
        labels: torch.LongTensor,
        n_step: tp.Optional[int] = None,
        dt_list: tp.Optional[tp.List[int]] = None,
        input_shape: tp.Optional[tp.List[int]] = None,
        disable_shortcut: bool = False
    ):
        assert exists(n_step) or exists(dt_list)
        device = labels.device
        num_sample = len(labels)

        # specify input shape
        if exists(input_shape):
            ch_in, H_in, W_in = input_shape
        else:
            ch_in, H_in, W_in = self.ch_in, self.H_in, self.W_in

        dummy_input = torch.empty(1, ch_in, H_in, W_in, device=device)
        dummy_pat = self.patchify_2d(dummy_input)
        _, dim_in, L_in = dummy_pat.shape

        if exists(n_step):
            dt_list = [1. / n_step] * n_step

        assert sum(dt_list) <= 1 + 1e-6

        # prepare condition
        conds = self.label_embedder(labels).unsqueeze(1)  # (bs, 1, dim_cond)

        # initial noise
        x = torch.randn(num_sample, dim_in, L_in, device=device)

        # sample
        t_cur = torch.zeros(num_sample, device=device)
        for dt_val in dt_list:
            dt = torch.full((num_sample,), dt_val, device=device)

            # predict
            if disable_shortcut:
                dt_in = torch.zeros_like(dt)
            else:
                dt_in = dt

            vel = self.model(x, t_cur, dt_in, conds)

            # update
            x += vel * dt[:, None, None]

            t_cur = t_cur + dt

        # unpatchify
        gen = self.unpatchify_2d(x, H_in, W_in)

        return gen
