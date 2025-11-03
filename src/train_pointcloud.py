"""
Distributed training entrypoint for the point-cloud shortcut model using
synthetic data.
"""

import sys

sys.dont_write_bytecode = True

import os
import random
from typing import Optional

import hydra
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs, DataLoaderConfiguration
from accelerate.utils import ProjectConfiguration
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from trainer import TimestepSampler
from utils.torch_common import print_once, set_seed
from datasets import DummyPointCloudDataset, ObjaverseSingleImagePointCloudDataset
from encoders import FrozenDinoV2Encoder
from utils.logging import MetricsLogger


def _its_time(step: int, interval: int) -> bool:
    if interval <= 0:
        return False
    return (step - 1) % interval == 0


def _build_dataset(cfg):
    ds_cfg = cfg.trainer.dataset
    name = ds_cfg.name
    if name == "dummy":
        dataset = DummyPointCloudDataset(
            num_samples=ds_cfg.num_samples,
            num_points=ds_cfg.num_points,
            coord_range=ds_cfg.coord_range,
            color_range=ds_cfg.color_range,
            cond_dim=ds_cfg.cond_dim,
            cond_length=ds_cfg.cond_length,
            noise_std=ds_cfg.noise_std,
            seed=ds_cfg.seed,
        )
    elif name == "objaverse":
        transform = None
        if "image_size" in ds_cfg:
            from datasets.objaverse_single_image import default_image_transform

            transform = default_image_transform(ds_cfg.image_size)
        dataset = ObjaverseSingleImagePointCloudDataset(
            root_dirs=ds_cfg.root_dirs,
            view_subdir=ds_cfg.view_subdir,
            image_extension=ds_cfg.image_extension,
            pointcloud_name=ds_cfg.pointcloud_name,
            points_key=ds_cfg.points_key,
            colours_key=ds_cfg.colours_key,
            num_points=ds_cfg.num_points,
            random_view=ds_cfg.random_view,
            transform=transform,
            seed=ds_cfg.seed,
        )
    else:
        raise ValueError(f"Unsupported dataset name: {name}")
    return dataset


def _prepare_dataloader(cfg, dataset):
    drop_last = True
    try:
        if len(dataset) < cfg.trainer.batch_size:
            drop_last = False
    except TypeError:
        drop_last = False

    return DataLoader(
        dataset,
        batch_size=cfg.trainer.batch_size,
        shuffle=True,
        num_workers=cfg.trainer.num_workers,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=(cfg.trainer.num_workers > 0),
    )


def _save_checkpoint(accel, model, optimizer, scheduler, cfg, out_dir, step):
    accel.wait_for_everyone()
    if not accel.is_main_process:
        return

    ckpt_dir = os.path.join(out_dir, "ckpt", "latest")
    os.makedirs(ckpt_dir, exist_ok=True)

    torch.save(accel.get_state_dict(model), os.path.join(ckpt_dir, "model.pth"))
    torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pth"))
    torch.save(scheduler.state_dict(), os.path.join(ckpt_dir, "scheduler.pth"))

    OmegaConf.save(cfg, os.path.join(ckpt_dir, "config.yaml"))

    states = {"global_step": step}
    torch.save(states, os.path.join(ckpt_dir, "states.pt"))


def _sample_and_save(model, accel, cfg, out_dir, step, dataset, dataset_name, dino_encoder: Optional[FrozenDinoV2Encoder]):
    if not accel.is_main_process:
        return
    model.eval()
    sample_dir = os.path.join(out_dir, "sample")
    os.makedirs(sample_dir, exist_ok=True)

    ds_cfg = cfg.trainer.dataset
    log_cfg = cfg.trainer.logging

    for n_step in log_cfg.steps:
        if dataset_name == "dummy":
            cond_list = []
            for _ in range(log_cfg.n_samples_per_step):
                _, cond = dataset[random.randrange(len(dataset))]
                cond_list.append(cond)
            conds = torch.stack(cond_list, dim=0).to(accel.device)
        else:
            assert dino_encoder is not None
            images = []
            for _ in range(log_cfg.n_samples_per_step):
                _, image = dataset[random.randrange(len(dataset))]
                images.append(image)
            images = torch.stack(images, dim=0).to(accel.device)
            with torch.no_grad():
                cond_feats = dino_encoder(images)
            conds = cond_feats.unsqueeze(1)
        samples = model.sample(conds, n_step=n_step)
        filename = os.path.join(sample_dir, f"points_step{step}_n{n_step}.pt")
        torch.save(samples.cpu(), filename)
    model.train()


@hydra.main(version_base=None, config_path="../configs/", config_name="pointcloud.yaml")
def main(cfg: DictConfig):
    if cfg.trainer.ckpt_dir is not None:
        overrides = HydraConfig.get().overrides.task
        overrides = [e for e in overrides if isinstance(e, str)]
        override_conf = OmegaConf.from_dotlist(overrides)
        cfg_ckpt = OmegaConf.load(f"{cfg.trainer.ckpt_dir}/config.yaml")
        cfg = OmegaConf.merge(cfg_ckpt, override_conf)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    dl_config = DataLoaderConfiguration(split_batches=True)
    p_config = ProjectConfiguration(project_dir=cfg.trainer.output_dir)

    accel = Accelerator(
        mixed_precision=cfg.trainer.amp,
        dataloader_config=dl_config,
        project_config=p_config,
        log_with="wandb",
        kwargs_handlers=[ddp_kwargs],
    )

    set_seed(cfg.trainer.seed)

    if accel.is_main_process:
        os.makedirs(cfg.trainer.output_dir, exist_ok=True)
        OmegaConf.save(cfg, f"{cfg.trainer.output_dir}/config.yaml")
        print_once("->->-> DDP Initialized.")
        print_once(f"->->-> World size (Number of GPUs): {accel.num_processes}")
    logger_cfg = cfg.trainer.logger
    accel.init_trackers(
        logger_cfg.project_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        init_kwargs={"wandb": {"name": logger_cfg.run_name, "dir": cfg.trainer.output_dir}},
    )

    dataset = _build_dataset(cfg)
    dataloader = _prepare_dataloader(cfg, dataset)

    model = hydra.utils.instantiate(cfg.model)
    optimizer = hydra.utils.instantiate(cfg.optimizer.optimizer)(params=model.parameters())
    scheduler = hydra.utils.instantiate(cfg.optimizer.scheduler)(optimizer=optimizer)

    dataloader, model, optimizer, scheduler = accel.prepare(
        dataloader, model, optimizer, scheduler
    )

    model_unwrapped = accel.unwrap_model(model)

    dataset_name = cfg.trainer.dataset.name
    if dataset_name == "objaverse":
        dino_encoder = FrozenDinoV2Encoder(cfg.trainer.image_encoder.model_name)
        dino_encoder.to(accel.device)
        dino_encoder.eval()
    else:
        dino_encoder = None

    sampler = TimestepSampler(**cfg.trainer.timestep_sampler)
    model.train()
    model_unwrapped.train()

    log_cfg = cfg.trainer.logging
    out_dir = cfg.trainer.output_dir
    max_steps = cfg.trainer.max_steps

    logger_wandb = MetricsLogger()
    logger_print = MetricsLogger()

    step = 1
    while step <= max_steps:
        for batch in dataloader:
            if step > max_steps:
                break

            if dataset_name == "dummy":
                points, conds = batch
                points = points.to(accel.device)
                conds = conds.to(accel.device)
            else:
                points, images = batch
                points = points.to(accel.device)
                images = images.to(accel.device)
                with torch.no_grad():
                    cond_feats = dino_encoder(images)
                conds = cond_feats.unsqueeze(1)

            t, dt, num_sc = sampler.sample_t(points.shape[0], device=points.device)

            optimizer.zero_grad()
            output = model_unwrapped.train_step(
                points,
                conds,
                t,
                dt,
                num_self_consistency=num_sc,
                cfg_dropout_prob=cfg.trainer.cfg_dropout_prob,
            )

            logger_wandb.add(output)
            logger_print.add(output)

            loss = output["loss"]
            accel.backward(loss)
            if accel.sync_gradients:
                accel.clip_grad_norm_(model.parameters(), cfg.trainer.max_grad_norm)
            optimizer.step()
            scheduler.step()

            if accel.is_main_process and _its_time(step, log_cfg.n_step_log):
                metrics = logger_wandb.pop()
                metrics_float = {
                    k: v.detach().cpu().item() if torch.is_tensor(v) else float(v)
                    for k, v in metrics.items()
                }
                if metrics_float:
                    accel.log(metrics_float, step=step)

            if accel.is_main_process and _its_time(step, log_cfg.n_step_print):
                metrics = logger_print.pop()
                metrics_float = {
                    k: v.detach().cpu().item() if torch.is_tensor(v) else float(v)
                    for k, v in metrics.items()
                }
                if metrics_float:
                    metrics_str = " / ".join(
                        f"{k}={v:.4e}" for k, v in sorted(metrics_float.items())
                    )
                    print(f"Step {step}: {metrics_str}")

            if _its_time(step, log_cfg.n_step_ckpt):
                _save_checkpoint(accel, model, optimizer, scheduler, cfg, out_dir, step)

            if _its_time(step, log_cfg.n_step_sample):
                _sample_and_save(model_unwrapped, accel, cfg, out_dir, step, dataset, dataset_name, dino_encoder)

            step += 1
            if step > max_steps:
                break

    accel.wait_for_everyone()
    if accel.is_main_process:
        print("Training finished.")
    accel.end_training()


if __name__ == "__main__":
    main()
