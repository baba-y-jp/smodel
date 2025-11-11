"""
Utility script to visualize and export point cloud samples saved by `train_pointcloud.py`.

Each `.pt` file produced by the sampler contains a tensor with shape
`(num_samples, num_points, total_dim)` where the first three dimensions are XYZ
coordinates and the remaining dimensions (if present) are treated as RGB
features.  This script renders each sample as a 3D scatter plot, and optionally
writes each point cloud to a `.ply` file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize point cloud samples.")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to a .pt tensor file produced by sampling.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to store rendered PNG images.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Maximum number of samples to render from the batch. Defaults to all samples.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=5.0,
        help="Scatter point size passed to matplotlib (default: 5.0).",
    )
    parser.add_argument(
        "--elev",
        type=float,
        default=20.0,
        help="Camera elevation angle in degrees (default: 20.0).",
    )
    parser.add_argument(
        "--azim",
        type=float,
        default=-60.0,
        help="Camera azimuth angle in degrees (default: -60.0).",
    )
    parser.add_argument(
        "--ply-dir",
        type=Path,
        default=None,
        help="Optional directory to export each sample as a .ply file.",
    )
    return parser.parse_args()


def normalize_colors(rgb: torch.Tensor) -> torch.Tensor:
    if rgb.numel() == 0:
        return rgb
    rgb_min = rgb.amin(dim=0, keepdim=True)
    rgb_max = rgb.amax(dim=0, keepdim=True)
    denom = torch.clamp(rgb_max - rgb_min, min=1e-6)
    return (rgb - rgb_min) / denom


def render_point_cloud(points: torch.Tensor, out_path: Path, point_size: float, elev: float, azim: float):
    xyz = points[:, :3].cpu().numpy()
    colors = None
    if points.shape[1] >= 6:
        rgb = normalize_colors(points[:, 3:6]).clamp(0.0, 1.0)
        colors = rgb.cpu().numpy()

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        c=colors,
        s=point_size,
        marker="o",
        depthshade=False,
    )
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)


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


def main():
    args = parse_args()

    tensor = torch.load(args.input, map_location="cpu")
    if tensor.dim() != 3:
        raise ValueError(f"Expected tensor with shape (batch, points, dim), got {tensor.shape}")

    num_samples = tensor.shape[0]
    limit = min(args.num_samples or num_samples, num_samples)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.ply_dir is not None:
        args.ply_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(limit):
        points = tensor[idx]
        out_img = args.output_dir / f"sample_{idx:03d}.png"
        render_point_cloud(points, out_img, args.point_size, args.elev, args.azim)
        print(f"Saved {out_img}")
        if args.ply_dir is not None:
            out_ply = args.ply_dir / f"sample_{idx:03d}.ply"
            save_ply(out_ply, points)
            print(f"Saved {out_ply}")


if __name__ == "__main__":
    main()
