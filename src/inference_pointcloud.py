"""
Inference script for point-cloud shortcut models.

Supports two conditioning modes:
  1. Iterate over an Objaverse-style dataset directory (--dataset-dir).
  2. Provide explicit conditioning images via --image path/to.png (repeatable).

Outputs generated point clouds as .ply files and prints per-sample latency.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import torch
from PIL import Image
from omegaconf import OmegaConf
import hydra
from torchvision import transforms

from shortcut_models.pointcloud import PointCloudShortcutModel
from datasets.objaverse_single_image import (
    ObjaverseSingleImagePointCloudDataset,
    default_image_transform,
)
from encoders.dinov2 import FrozenDinoV2Encoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate point clouds from a trained checkpoint.")
    parser.add_argument("--ckpt-dir", type=Path, required=True, help="Checkpoint directory containing model.pth and config.yaml")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Objaverse-style directory (e.g. /workspace/ssd2/dataset/g-bufferobjaverse/Animals/0)",
    )
    parser.add_argument(
        "--image",
        type=Path,
        action="append",
        default=None,
        help="Optional conditioning image (.png). Can be specified multiple times.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to store generated .ply files")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help="Number of dataset samples to generate when using --dataset-dir (ignored for --image).",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=4,
        help="Number of sampling steps (ignored if --dt-list is provided).",
    )
    parser.add_argument(
        "--dt-list",
        type=float,
        nargs="+",
        default=None,
        help="Optional list of dt values (e.g. 0.6 0.4). Overrides --step when provided.",
    )
    parser.add_argument("--disable-shortcut", action="store_true", help="Disable shortcut branch during sampling")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    return parser.parse_args()


def save_ply(path: Path, points: torch.Tensor):
    coords = points[:, :3].cpu().numpy()
    has_color = points.shape[1] >= 6
    if has_color:
        colors = points[:, 3:6].clamp(0.0, 1.0).cpu().numpy()
        colors = (colors * 255).astype("uint8")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {coords.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if has_color:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for idx in range(coords.shape[0]):
            line = [f"{coords[idx,0]:.6f}", f"{coords[idx,1]:.6f}", f"{coords[idx,2]:.6f}"]
            if has_color:
                line.extend([str(int(colors[idx,0])), str(int(colors[idx,1])), str(int(colors[idx,2]))])
            f.write(" ".join(line) + "\n")


def build_transform(cfg) -> transforms.Compose:
    ds_cfg = cfg.trainer.dataset
    size = getattr(ds_cfg, "image_size", 518)
    return default_image_transform(size)


def load_images(image_paths: Iterable[Path], transform: transforms.Compose, device: torch.device):
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)
        yield path, tensor


def load_dataset(cfg, dataset_dir: Path):
    ds_cfg = cfg.trainer.dataset
    dataset = ObjaverseSingleImagePointCloudDataset(
        root_dirs=[str(dataset_dir)],
        view_subdir=ds_cfg.view_subdir,
        image_extension=ds_cfg.image_extension,
        pointcloud_name=ds_cfg.pointcloud_name,
        points_key=ds_cfg.points_key,
        colours_key=ds_cfg.colours_key,
        num_points=ds_cfg.num_points,
        random_view=False,
        transform=build_transform(cfg),
        seed=None,
    )
    return dataset


def load_model(cfg, ckpt_dir: Path, device: torch.device) -> PointCloudShortcutModel:
    model: PointCloudShortcutModel = hydra.utils.instantiate(cfg.model)
    state = torch.load(ckpt_dir / "model.pth", map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def run_inference(
    model: PointCloudShortcutModel,
    cond: torch.Tensor,
    *,
    n_step: Optional[int],
    dt_list: Optional[List[float]],
    disable_shortcut: bool,
) -> Tuple[torch.Tensor, float]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    generated = model.sample(
        cond,
        n_step=n_step,
        dt_list=dt_list,
        disable_shortcut=disable_shortcut,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return generated.squeeze(0).cpu(), elapsed


def main():
    args = parse_args()
    if args.dataset_dir is None and not args.image:
        raise ValueError("Specify either --dataset-dir or at least one --image.")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = OmegaConf.load(args.ckpt_dir / "config.yaml")
    model = load_model(cfg, args.ckpt_dir, device)

    dino_encoder = FrozenDinoV2Encoder(cfg.trainer.image_encoder.model_name)
    dino_encoder.to(device)
    dino_encoder.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    transform = build_transform(cfg)

    total_time = 0.0
    total_samples = 0

    with torch.no_grad():
        if args.image:
            for image_path, tensor in load_images(args.image, transform, device):
                cond = dino_encoder(tensor).unsqueeze(1)
                pointcloud, elapsed = run_inference(
                    model,
                    cond,
                    n_step=None if args.dt_list else args.step,
                    dt_list=args.dt_list,
                    disable_shortcut=args.disable_shortcut,
                )
                ply_path = args.output_dir / f"{image_path.stem}.ply"
                save_ply(ply_path, pointcloud)
                print(f"Saved {ply_path} (elapsed: {elapsed:.4f}s)")
                total_time += elapsed
                total_samples += 1

        if args.dataset_dir is not None and total_samples < args.num_samples:
            dataset = load_dataset(cfg, args.dataset_dir)
            limit = min(args.num_samples - total_samples, len(dataset))
            for idx in range(limit):
                _, image = dataset[idx]
                tensor = image.unsqueeze(0).to(device)
                cond = dino_encoder(tensor).unsqueeze(1)
                pointcloud, elapsed = run_inference(
                    model,
                    cond,
                    n_step=None if args.dt_list else args.step,
                    dt_list=args.dt_list,
                    disable_shortcut=args.disable_shortcut,
                )
                ply_path = args.output_dir / f"dataset_{idx:03d}.ply"
                save_ply(ply_path, pointcloud)
                print(f"Saved {ply_path} (elapsed: {elapsed:.4f}s)")
                total_time += elapsed
                total_samples += 1

    if total_samples > 0:
        print(f"Average inference time: {total_time / total_samples:.4f}s over {total_samples} sample(s)")


if __name__ == "__main__":
    main()
