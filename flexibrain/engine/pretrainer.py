from __future__ import annotations

import math
import os

import torch
import torch.optim as optim

from flexibrain.config import RunConfig
from flexibrain.data import build_pretrain_dataloaders
from flexibrain.data.collate import prepare_batch_data
from flexibrain.distributed import cleanup_distributed, setup_distributed
from flexibrain.models import build_pretrain_model
from flexibrain.utils.logging import setup_logger
from flexibrain.utils.seed import set_seed
from flexibrain.utils.training import get_dynamic_momentum, update_ema


class Pretrainer:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.rank = cfg.training.local_rank
        self.world_size = cfg.training.world_size
        if self.world_size > 1:
            setup_distributed(self.rank, self.world_size)
        self.device = torch.device(f"cuda:{self.rank}" if torch.cuda.is_available() else "cpu")
        self.logger = setup_logger("pretrain", cfg.logging.log_dir, rank=self.rank)

    def build(self):
        set_seed(self.cfg.training.seed)
        self.model = build_pretrain_model(self.cfg.model, self.device)
        self.train_loader, self.val_loader = build_pretrain_dataloaders(self.cfg.data, self.cfg.training, rank=self.rank, world_size=self.world_size)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.cfg.training.lr, weight_decay=self.cfg.training.weight_decay)
        total_steps = max(1, len(self.train_loader) * self.cfg.training.epochs)
        warmup_steps = max(1, len(self.train_loader) * self.cfg.training.warmup_epochs)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            cycle_progress = (progress * 4) % 1.0
            if cycle_progress < 0.8:
                return 0.5 * (1 + math.cos(math.pi * cycle_progress / 0.8))
            return 0.0

        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        return self

    def train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        accumulation_steps = self.cfg.training.grad_accumulation_steps
        momentum = get_dynamic_momentum(epoch, self.cfg.training.epochs, self.cfg.model.momentum, self.cfg.model.final_momentum)
        self.optimizer.zero_grad(set_to_none=True)
        for batch_idx, batch in enumerate(self.train_loader):
            x, meta, orig_Ts, affines = prepare_batch_data(batch, self.device)
            loss, _, _, _ = self.model(x, mask_ratio=self.cfg.training.mask_ratio, meta=meta, orig_Ts=orig_Ts, affines=affines)
            (loss / accumulation_steps).backward()
            total_loss += float(loss.item())
            num_batches += 1
            if (batch_idx + 1) % accumulation_steps == 0:
                if self.cfg.training.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.training.grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                update_ema(self.model, momentum)
                self.scheduler.step()
            if self.rank == 0 and (batch_idx + 1) % self.cfg.logging.log_interval == 0:
                self.logger.info("Epoch %d [%d/%d] loss=%.6f avg=%.6f momentum=%.6f", epoch + 1, batch_idx + 1, len(self.train_loader), loss.item(), total_loss / num_batches, momentum)
        return total_loss / max(1, num_batches)

    @torch.no_grad()
    def validate(self, epoch: int) -> float:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        for batch in self.val_loader:
            x, meta, orig_Ts, affines = prepare_batch_data(batch, self.device)
            loss, _, _, _ = self.model(x, mask_ratio=self.cfg.training.mask_ratio, meta=meta, orig_Ts=orig_Ts, affines=affines)
            total_loss += float(loss.item())
            num_batches += 1
        avg = total_loss / max(1, num_batches)
        if self.rank == 0:
            self.logger.info("Epoch %d validation loss=%.6f", epoch + 1, avg)
        return avg

    def save(self, epoch: int, best_loss: float):
        if self.rank != 0:
            return
        os.makedirs(self.cfg.logging.checkpoint_dir, exist_ok=True)
        payload = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_loss": best_loss,
            "config": vars(self.cfg.model),
        }
        torch.save(payload, os.path.join(self.cfg.logging.checkpoint_dir, "checkpoint_latest.pt"))
        torch.save(payload, os.path.join(self.cfg.logging.checkpoint_dir, "checkpoint_best.pt"))

    def fit(self):
        self.build()
        if self.rank == 0:
            self.logger.info("Starting pretrain on %s", self.device)
            self.logger.info("Train size=%d Val size=%d", len(self.train_loader.dataset), len(self.val_loader.dataset))
        best_loss = float("inf")
        for epoch in range(self.cfg.training.epochs):
            if hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(epoch)
            train_loss = self.train_one_epoch(epoch)
            val_loss = self.validate(epoch)
            if val_loss < best_loss:
                best_loss = val_loss
            self.save(epoch, best_loss)
            if self.rank == 0:
                self.logger.info("Epoch %d done train=%.6f val=%.6f best=%.6f", epoch + 1, train_loss, val_loss, best_loss)
        if self.world_size > 1:
            cleanup_distributed()
