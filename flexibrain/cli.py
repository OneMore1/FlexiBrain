from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import torch

from flexibrain.config import load_config
from flexibrain.engine import DownstreamTrainer, Pretrainer
from flexibrain.models import build_downstream_model, build_pretrain_model


def _add_common(parser):
    parser.add_argument("--config", default=None)
    parser.add_argument("--train-list", default=None)
    parser.add_argument("--val-list", default=None)
    parser.add_argument("--test-list", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--t-prime", type=int, default=None)
    parser.add_argument("--tau-seconds", type=float, default=None)
    parser.add_argument("--default-tr", type=float, default=None, help="Fallback TR in seconds when a NIfTI header has no valid TR.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--grad-accumulation-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--local-rank", type=int, default=None)
    parser.add_argument("--world-size", type=int, default=None)
    parser.add_argument("--use-amp", action="store_true", default=None)
    parser.add_argument("--no-use-amp", dest="use_amp", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.set_defaults(use_amp=None)


def _add_model(parser):
    parser.add_argument("--embed-dim", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--predictor-depth", type=int, default=None)
    parser.add_argument("--drop-path-rate", type=float, default=None)
    parser.add_argument("--bimamba-type", default=None)
    parser.add_argument("--if-bimamba", action="store_true", default=None)
    parser.add_argument("--if-devide-out", action="store_true", default=None)
    parser.add_argument("--no-if-devide-out", dest="if_devide_out", action="store_false")
    parser.add_argument("--mixer-type", default=None)
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--final-momentum", type=float, default=None)


def apply_common(cfg, args):
    for key in ["train_list", "val_list", "test_list", "batch_size", "num_workers", "epochs", "lr", "weight_decay", "warmup_epochs", "grad_accumulation_steps", "seed", "local_rank", "world_size"]:
        value = getattr(args, key, None)
        if value is None:
            continue
        target = cfg.data if key in {"train_list", "val_list", "test_list", "batch_size", "num_workers"} else cfg.training
        setattr(target, key, value)
    if args.t_prime is not None:
        cfg.data.T_prime = args.t_prime
    if args.tau_seconds is not None:
        cfg.data.tau_seconds = args.tau_seconds
    if args.default_tr is not None:
        cfg.data.default_tr = args.default_tr
    if args.use_amp is not None:
        cfg.training.use_amp = args.use_amp
    if args.log_interval is not None:
        cfg.logging.log_interval = args.log_interval
    if args.checkpoint_dir is not None:
        cfg.logging.checkpoint_dir = args.checkpoint_dir
    if args.log_dir is not None:
        cfg.logging.log_dir = args.log_dir


def apply_model(cfg, args):
    for key in ["embed_dim", "depth", "predictor_depth", "drop_path_rate", "bimamba_type", "if_bimamba", "if_devide_out", "mixer_type", "momentum", "final_momentum"]:
        value = getattr(args, key, None)
        if value is not None:
            setattr(cfg.model, key, value)


def parse_args():
    parser = argparse.ArgumentParser(prog="flexibrain")
    sub = parser.add_subparsers(dest="command", required=True)
    pretrain = sub.add_parser("pretrain")
    _add_common(pretrain)
    _add_model(pretrain)
    pretrain.add_argument("--mask-ratio", type=float, default=None)
    pretrain.add_argument("--grad-clip", type=float, default=None)

    downstream = sub.add_parser("downstream")
    _add_common(downstream)
    _add_model(downstream)
    downstream.add_argument("--csv", default=None)
    downstream.add_argument("--id-column", default=None)
    downstream.add_argument("--label-column", default=None)
    downstream.add_argument("--label-mode", default=None)
    downstream.add_argument("--path-id-mode", default=None)
    downstream.add_argument("--pretrain-checkpoint", default=None)
    downstream.add_argument("--from-scratch", action="store_true")
    downstream.add_argument("--ignore-checkpoint-config", action="store_true")
    downstream.add_argument("--num-classes", type=int, default=None)
    downstream.add_argument("--head-type", choices=["transformer", "avgpool"], default=None)
    downstream.add_argument("--freeze-backbone", action="store_true")
    downstream.add_argument("--lr-backbone", type=float, default=None)
    downstream.add_argument("--lr-head", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    apply_common(cfg, args)
    apply_model(cfg, args)
    if args.command == "pretrain":
        if args.mask_ratio is not None:
            cfg.training.mask_ratio = args.mask_ratio
        if args.grad_clip is not None:
            cfg.training.grad_clip = args.grad_clip
        if args.dry_run:
            model = build_pretrain_model(cfg.model, torch.device("cpu"))
            print(json.dumps({"config": asdict(cfg), "parameters": sum(p.numel() for p in model.parameters())}, indent=2))
            return
        Pretrainer(cfg).fit()
    elif args.command == "downstream":
        for key in ["csv", "id_column", "label_column", "label_mode", "path_id_mode"]:
            value = getattr(args, key, None)
            if value is not None:
                setattr(cfg.data, key, value)
        for key in ["num_classes", "head_type", "freeze_backbone"]:
            value = getattr(args, key, None)
            if value is not None:
                setattr(cfg.model, key, value)
        if args.lr_backbone is not None:
            cfg.training.lr_backbone = args.lr_backbone
        if args.lr_head is not None:
            cfg.training.lr_head = args.lr_head
        if args.pretrain_checkpoint is not None:
            cfg.pretrain_checkpoint = args.pretrain_checkpoint
        if args.from_scratch:
            cfg.from_scratch = True
        if args.ignore_checkpoint_config:
            cfg.use_checkpoint_config = False
        if args.dry_run:
            model = build_downstream_model(cfg.model, torch.device("cpu"), checkpoint_path=cfg.pretrain_checkpoint, from_scratch=cfg.from_scratch, use_checkpoint_config=cfg.use_checkpoint_config)
            print(json.dumps({"config": asdict(cfg), "parameters": sum(p.numel() for p in model.parameters())}, indent=2))
            return
        DownstreamTrainer(cfg).fit()


if __name__ == "__main__":
    main()
