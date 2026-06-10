from __future__ import annotations

import os
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from flexibrain.config import RunConfig
from flexibrain.data import build_downstream_dataloaders
from flexibrain.data.classification import prepare_batch_data
from flexibrain.models import build_downstream_model
from flexibrain.utils.logging import setup_logger
from flexibrain.utils.seed import set_seed


class DownstreamTrainer:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.rank = cfg.training.local_rank
        self.device = torch.device(f"cuda:{self.rank}" if torch.cuda.is_available() else "cpu")
        self.logger = setup_logger("downstream", cfg.logging.log_dir, rank=self.rank)

    def build(self):
        set_seed(self.cfg.training.seed)
        self.model = build_downstream_model(
            self.cfg.model,
            self.device,
            logger=self.logger,
            checkpoint_path=self.cfg.pretrain_checkpoint,
            from_scratch=self.cfg.from_scratch,
            use_checkpoint_config=self.cfg.use_checkpoint_config,
        )
        self.train_loader, self.val_loader, self.test_loader = build_downstream_dataloaders(self.cfg.data, self.cfg.training, rank=self.rank, world_size=self.cfg.training.world_size)
        if self.cfg.training.lr_backbone is not None or self.cfg.training.lr_head is not None:
            backbone_params = list(self.model.backbone.parameters())
            head_params = [p for n, p in self.model.named_parameters() if not n.startswith("backbone.")]
            self.optimizer = optim.AdamW([
                {"params": backbone_params, "lr": self.cfg.training.lr_backbone or self.cfg.training.lr},
                {"params": head_params, "lr": self.cfg.training.lr_head or self.cfg.training.lr},
            ], weight_decay=self.cfg.training.weight_decay)
        else:
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.cfg.training.lr, weight_decay=self.cfg.training.weight_decay)
        total_steps = max(1, len(self.train_loader) * self.cfg.training.epochs)
        warmup_steps = max(1, len(self.train_loader) * self.cfg.training.warmup_epochs)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        return self

    def train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        num_batches = 0
        accum = self.cfg.training.grad_accumulation_steps
        self.optimizer.zero_grad(set_to_none=True)
        for batch_idx, batch in enumerate(self.train_loader):
            x, meta, orig_Ts, labels, affines = prepare_batch_data(batch, self.device)
            with torch.cuda.amp.autocast(enabled=self.cfg.training.use_amp):
                logits = self.model(x, meta=meta, orig_Ts=orig_Ts, affines=affines)
                loss = criterion(logits, labels)
            (loss / accum).backward()
            if (batch_idx + 1) % accum == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.item())
            num_batches += 1
            if self.rank == 0 and (batch_idx + 1) % self.cfg.logging.log_interval == 0:
                self.logger.info("Epoch %d [%d/%d] loss=%.6f", epoch + 1, batch_idx + 1, len(self.train_loader), loss.item())
        if num_batches % accum != 0:
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
        return total_loss / max(1, num_batches)

    @torch.no_grad()
    def evaluate(self, loader, split_name: str) -> Dict[str, float]:
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        preds, labels_all = [], []
        total_loss = 0.0
        num_batches = 0
        for batch in loader:
            x, meta, orig_Ts, labels, affines = prepare_batch_data(batch, self.device)
            logits = self.model(x, meta=meta, orig_Ts=orig_Ts, affines=affines)
            loss = criterion(logits, labels)
            total_loss += float(loss.item())
            num_batches += 1
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            labels_all.extend(labels.cpu().numpy())
        metrics = {
            "loss": total_loss / max(1, num_batches),
            "accuracy": accuracy_score(labels_all, preds),
            "precision_macro": precision_score(labels_all, preds, average="macro", zero_division=0),
            "recall_macro": recall_score(labels_all, preds, average="macro", zero_division=0),
            "f1_macro": f1_score(labels_all, preds, average="macro", zero_division=0),
            "f1_weighted": f1_score(labels_all, preds, average="weighted", zero_division=0),
        }
        if self.rank == 0:
            self.logger.info("%s metrics: %s", split_name, metrics)
        return metrics

    def save(self, epoch: int, metrics: dict):
        if self.rank != 0:
            return
        os.makedirs(self.cfg.logging.checkpoint_dir, exist_ok=True)
        torch.save({
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "metrics": metrics,
            "config": vars(self.cfg.model),
        }, os.path.join(self.cfg.logging.checkpoint_dir, f"downstream_epoch_{epoch:03d}.pt"))

    def fit(self):
        self.build()
        if self.rank == 0:
            self.logger.info("Starting downstream on %s", self.device)
            self.logger.info("Train size=%d Val size=%d", len(self.train_loader.dataset), len(self.val_loader.dataset))
        best_f1 = -1.0
        for epoch in range(self.cfg.training.epochs):
            train_loss = self.train_one_epoch(epoch)
            val_metrics = self.evaluate(self.val_loader, "Validation")
            metrics = {"val": val_metrics, "train_loss": train_loss}
            if self.test_loader is not None:
                metrics["test"] = self.evaluate(self.test_loader, "Test")
            if val_metrics["f1_macro"] > best_f1:
                best_f1 = val_metrics["f1_macro"]
            self.save(epoch, metrics)
            if self.rank == 0:
                self.logger.info("Epoch %d done train=%.6f best_f1=%.6f", epoch + 1, train_loss, best_f1)
