"""
Utility for loading frozen DINOv2 encoders.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn


def _dist_barrier():
    if dist.is_available() and dist.is_initialized():
        backend = dist.get_backend()
        if backend == "nccl" and torch.cuda.is_available():
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()


class FrozenDinoV2Encoder(nn.Module):
    def __init__(self, model_name: str = "dinov2_vits14"):
        super().__init__()
        default_hub = Path("/tmp/dinov2_hub")
        hub_root = Path(os.environ.get("DINOV2_HUBDIR", default_hub))
        hub_root.mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(str(hub_root))

        is_dist = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if is_dist else 0

        tmp_root = Path(os.environ.get("DINOV2_TMPDIR", "/tmp/dinov2_cache"))
        rank_tmp = tmp_root / f"rank{rank}"
        rank_tmp.mkdir(parents=True, exist_ok=True)
        os.environ["TMPDIR"] = str(rank_tmp)

        if not is_dist or rank == 0:
            model = torch.hub.load("facebookresearch/dinov2", model_name)
        if is_dist:
            _dist_barrier()
            if rank != 0:
                model = torch.hub.load("facebookresearch/dinov2", model_name)
            _dist_barrier()

        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        self.model = model

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)
