"""
Copyright (C) 2025 Yukara Ikemiya
"""

import sys
sys.dont_write_bytecode = True
import argparse
import math

import hydra
import torch
import torch.distributed as dist
from einops import rearrange
from accelerate import Accelerator
from omegaconf import OmegaConf
from matplotlib import pyplot as plt

from utils.torch_common import get_rank, get_world_size, print_once


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-dir', type=str, help="Checkpoint directory.")
    parser.add_argument('--output-dir', type=str, help="Output directory.")
    parser.add_argument('--step', type=int, default=4, help="Output directory.")
    parser.add_argument('--dt-list', nargs="*", type=float, default=None,
                        help="A list of 'dt' used for sampling. If None, 'step' variable will be used instead.")
    parser.add_argument('--num-per-label', type=int, default=2, help="Number of samples per label.")
    parser.add_argument('--disable-shortcut', action='store_true')
    args = parser.parse_args()

    ckpt_dir: str = args.ckpt_dir
    output_dir: str = args.output_dir
    step: str = args.step
    dt_list: list = args.dt_list
    num_per_label: int = args.num_per_label
    disable_shortcut: bool = args.disable_shortcut

    # Distributed inference
    accel = Accelerator()
    device = accel.device
    rank = get_rank()
    world_size = get_world_size()

    print_once(f"Ckpt directory   : {ckpt_dir}")
    print_once(f"Output directory : {output_dir}")

    # Load pretrained model
    cfg_ckpt = OmegaConf.load(f'{ckpt_dir}/config.yaml')
    model = hydra.utils.instantiate(cfg_ckpt.model)
    model.load_state_dict(torch.load(f"{ckpt_dir}/model.pth", weights_only=False))
    model.to(device)
    model.eval()
    print_once("->-> Successfully loaded a pretrained model from checkpoint.")

    # Prepare labels (MNIST)
    n_label = 10  # 0--9
    labels = torch.tensor(list(range(n_label)) * num_per_label, device=device)
    input_shape = (1, 28, 28)

    # Distributed inference
    n_per_rank = math.ceil(len(labels) / world_size)
    labels = labels[rank * n_per_rank: (rank + 1) * n_per_rank]

    # Sample
    if dt_list is not None:
        step = None

    gen_samples = model.sample(labels, n_step=step, dt_list=dt_list,
                               input_shape=input_shape, disable_shortcut=disable_shortcut)

    # Gather samples
    gen_samples = gen_samples.cpu()
    gathered = [None for _ in range(world_size)]
    dist.gather_object(gen_samples, gathered if rank == 0 else None)
    if rank == 0:
        gathered = torch.cat(gathered, dim=0)

    # save as figure
    gathered = rearrange(gathered, '(v l) c h w -> c (v h) (l w)', v=num_per_label, l=n_label).squeeze(0)

    dpi = 150
    h, w = gathered.shape[:2]
    figsize = (w / dpi, h / dpi)

    model_type = 'flow' if disable_shortcut else 'short'
    step_type = f"step-{step}" if dt_list is None else f"irrstep-{len(dt_list)}"

    plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(gathered, cmap='gray', vmin=0, vmax=1.0)
    plt.axis('off')
    plt.savefig(f"{output_dir}/{model_type}_{step_type}.png", bbox_inches='tight', pad_inches=0)
    plt.clf()


if __name__ == '__main__':
    main()
