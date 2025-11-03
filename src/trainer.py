"""
Copyright (C) 2025 Yukara Ikemiya
"""

import os

import torch
import wandb
from einops import rearrange

from utils.logging import MetricsLogger
from utils.torch_common import sort_dict, print_once


class TimestepSampler:
    def __init__(
        self,
        rate_self_consistency: float = 0.25,
        min_dt: float = 0.0078125  # 1/128
    ):
        """
        rate_self_consistency: Propotion of samples for self-consistency term (default: 0.25)
        min_dt: Minimum value of 'dt' (default: 1/128)
        """
        assert 0 <= rate_self_consistency <= 1.0
        self.rate_sc = rate_self_consistency
        self.min_dt = min_dt

    def sample_t(self, num: int, device):
        num_sc = round(num * self.rate_sc)
        num_fm = num - num_sc

        # t for flow-matching term
        t_fm = torch.rand(num_fm, device=device)  # 0 -- 1
        dt_fm = torch.zeros(num_fm, device=device)

        # t/dt for self-consistency term
        t_sc = torch.rand(num_sc, device=device) * (1 - self.min_dt)  # 0 -- 1-min_dt
        max_dt = 1. - t_sc
        dt_sc = self.min_dt + torch.rand(num_sc, device=device) * (max_dt - self.min_dt)  # min_dt -- 1-t

        t = torch.cat([t_sc, t_fm])
        dt = torch.cat([dt_sc, dt_fm])
        assert len(t) == len(dt) == num

        return t, dt, num_sc


class Trainer:
    def __init__(
        self,
        model,              # model
        optimizer,          # optimizer
        scheduler,          # scheduler
        train_dataloader,
        accel,              # Accelerator object
        cfg,                # Configurations
        ckpt_dir=None
    ):
        self.model = accel.unwrap_model(model)
        self.opt = optimizer
        self.sche = scheduler
        self.train_dataloader = train_dataloader
        self.accel = accel
        self.cfg = cfg
        self.cfg_t = cfg.trainer
        self.EPS = 1e-8

        # timestep sampler
        cfg_ts_sampler = self.cfg_t.timestep_sampler
        self.ts_sampler = TimestepSampler(**cfg_ts_sampler)

        self.logger = MetricsLogger()           # Logger for WandB
        self.logger_print = MetricsLogger()     # Logger for printing

        self.states = {'global_step': 0, 'best_metrics': float('inf'), 'latest_metrics': float('inf')}

        # time measurement
        self.s_event = torch.cuda.Event(enable_timing=True)
        self.e_event = torch.cuda.Event(enable_timing=True)

        # resume training
        if ckpt_dir is not None:
            self.__load_ckpt(ckpt_dir)

    def start_training(self):
        """
        Start training with infinite loops
        """
        self.model.train()
        self.s_event.record()

        print_once("\n[ Started training ]\n")

        while True:
            for batch in self.train_dataloader:
                # Update
                metrics = self.run_step(batch)

                if self.accel.is_main_process:
                    self.logger.add(metrics)
                    self.logger_print.add(metrics)

                    # Log
                    if self.__its_time(self.cfg_t.logging.n_step_log):
                        self.__log_metrics()

                    # Print
                    if self.__its_time(self.cfg_t.logging.n_step_print):
                        self.__print_metrics()

                    # Save checkpoint
                    if self.__its_time(self.cfg_t.logging.n_step_ckpt):
                        self.__save_ckpt()

                    # Sample
                    if self.__its_time(self.cfg_t.logging.n_step_sample):
                        self.__sampling()

                self.states['global_step'] += 1

    def run_step(self, batch, train: bool = True):
        """ One training step """

        images, labels = batch

        # sample timesteps
        t, dt, num_sc = self.ts_sampler.sample_t(images.shape[0], device=images.device)

        # Update

        if train:
            self.opt.zero_grad()

        output = self.model.train_step(
            images, labels, t, dt, num_self_consistency=num_sc,
            cfg_dropout_prob=self.cfg_t.cfg_dropout_prob)

        if train:
            self.accel.backward(output['loss'])
            if self.accel.sync_gradients:
                self.accel.clip_grad_norm_(self.model.parameters(), self.cfg_t.max_grad_norm)
            self.opt.step()
            self.sche.step()

        return {k: v.detach() for k, v in output.items()}

    @torch.no_grad()
    def __sampling(self):
        self.model.eval()

        device = self.accel.device
        steps: list = self.cfg_t.logging.steps
        n_sample: int = self.cfg_t.logging.n_samples_per_step
        n_label: int = self.cfg.model.num_label

        columns = ['labels', 'images']
        table_image = wandb.Table(columns=columns)

        for step in steps:
            columns = ['labels', 'images']
            labels = torch.randint(n_label, size=(n_sample,), device=device)
            labels_str = '-'.join([str(label.item()) for label in labels])
            log_str = f"Step-{step} / {labels_str}"

            # sampling
            gen_sample = self.model.sample(labels, n_step=step)

            # reshape
            gen_sample = rearrange(gen_sample, 'b c h w -> c h (b w)')

            image = wandb.Image(gen_sample.squeeze(0).cpu().numpy())
            data = [log_str, image]

            table_image.add_data(*data)

        self.accel.log({'Samples': table_image}, step=self.states['global_step'])

        self.model.train()

        print("\t->->-> Sampled.")

    def __save_ckpt(self):
        import shutil
        import json
        from omegaconf import OmegaConf

        out_dir = self.cfg_t.output_dir + '/ckpt'

        # save latest ckpt
        latest_dir = out_dir + '/latest'
        os.makedirs(latest_dir, exist_ok=True)
        ckpts = {'model': self.model,
                 'optimizer': self.opt,
                 'scheduler': self.sche}
        for name, m in ckpts.items():
            torch.save(m.state_dict(), f"{latest_dir}/{name}.pth")

        # save states and configuration
        OmegaConf.save(self.cfg, f"{latest_dir}/config.yaml")
        with open(f"{latest_dir}/states.json", mode="wt", encoding="utf-8") as f:
            json.dump(self.states, f, indent=2)

        # save best ckpt
        if self.states['latest_metrics'] == self.states['best_metrics']:
            shutil.copytree(latest_dir, out_dir + '/best', dirs_exist_ok=True)

        print("\t->->-> Saved checkpoints.")

    def __load_ckpt(self, dir: str):
        import json

        print_once(f"\n[Resuming training from the checkpoint directory] -> {dir}")
        ckpts = {'model': self.model,
                 'optimizer': self.opt,
                 'scheduler': self.sche}

        for k, v in ckpts.items():
            v.load_state_dict(torch.load(f"{dir}/{k}.pth", weights_only=False))

        with open(f"{dir}/states.json", mode="rt", encoding="utf-8") as f:
            self.states.update(json.load(f))

    def __log_metrics(self, sort_by_key: bool = True):
        metrics = self.logger.pop()
        # learning rate
        metrics['lr'] = self.sche.get_last_lr()[0]
        if sort_by_key:
            metrics = sort_dict(metrics)

        self.accel.log(metrics, step=self.states['global_step'])

        # update states
        m_for_ckpt = self.cfg_t.logging.metrics_for_best_ckpt
        m_latest = float(sum([metrics[k].detach() for k in m_for_ckpt]))
        self.states['latest_metrics'] = m_latest
        if m_latest < self.states['best_metrics']:
            self.states['best_metrics'] = m_latest

    def __print_metrics(self, sort_by_key: bool = True):
        self.e_event.record()
        torch.cuda.synchronize()
        p_time = self.s_event.elapsed_time(self.e_event) / 1000.  # [sec]

        metrics = self.logger_print.pop()
        # tensor to scalar
        metrics = {k: v.item() for k, v in metrics.items()}
        if sort_by_key:
            metrics = sort_dict(metrics)

        step = self.states['global_step']
        s = f"Step {step} ({p_time:.1e} [sec]): " + ' / '.join([f"[{k}] - {v:.3e}" for k, v in metrics.items()])
        print(s)

        self.s_event.record()

    def __its_time(self, itv: int):
        return (self.states['global_step'] - 1) % itv == 0
