#!/usr/bin/env python3
"""
Batch transfer learning with per-run output_dir override.

If a run in the JSON plan specifies "output_dir", that directory and its basename
determine the final checkpoint filename and also the names for the log folder and subset txt:
  ckpt:   {output_dir}/{basename(output_dir)}.pt
  log:    logs/transfer/{basename(output_dir)}/transfer.log
  subset: transfer_work/{basename(output_dir)}.subset.txt

Otherwise fallback to auto tag naming:
  tag = {srcStem}__to__{dstStem}__p{ratio%}
  ckpt:   checkpoints/transfer_decoder/{tag}/{tag}.pt
  log:    logs/transfer/{tag}/transfer.log
  subset: transfer_work/{tag}.subset.txt
"""

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import NiftiTxtDataset
from main import custom_collate_fn, prepare_batch_data, build_model


# ------------------------- Utilities ------------------------- #

def setup_logging(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(__name__ + '.transfer')
    logger.setLevel(logging.DEBUG)
    # Clear old handlers to avoid duplicate logs across multiple runs
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fh = logging.FileHandler(os.path.join(log_dir, 'transfer.log'))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def load_source_checkpoint(model: nn.Module, ckpt_path: str, device: torch.device, logger: logging.Logger):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)
    logger.info(f"Loading checkpoint (strict=True): {ckpt_path}")
    try:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model.module.load_state_dict(state, strict=True)
        else:
            model.load_state_dict(state, strict=True)
    except RuntimeError:
        logger.error("Strict load failed. Ensure build_model hyperparams match the checkpoint.")
        raise
    logger.info("Loaded source checkpoint successfully.")
    return ckpt


def freeze_encoder(model: nn.Module, logger: logging.Logger):
    trainable_names = []
    for name, p in model.named_parameters():
        if any(k in name for k in ['predictor_blocks', 'predictor_norm', 'mask_token_ctx']):
            p.requires_grad = True
            trainable_names.append(name)
        else:
            p.requires_grad = False
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Frozen params: {frozen}, Trainable decoder params: {trainable}")
    logger.debug("Decoder trainable names: " + (", ".join(trainable_names) if trainable_names else "(none)"))


def pick_subset(list_file: str, fraction: float, output_subset_txt: str, logger: logging.Logger) -> List[Path]:
    lines = []
    for line in Path(list_file).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        p = Path(line)
        if not p.exists():
            p = (Path(list_file).parent / line).resolve()
        if p.exists():
            lines.append(p)
        else:
            logger.warning(f"Skip missing path: {line}")
    if not lines:
        raise ValueError(f"No valid paths in list file: {list_file}")
    subset_size = max(1, int(len(lines) * fraction))
    subset = sorted(random.sample(lines, subset_size))
    outp = Path(output_subset_txt)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text("\n".join(str(p) for p in subset) + "\n")
    logger.info(f"Selected {len(subset)} / {len(lines)} (fraction={fraction:.2%}) -> {output_subset_txt}")
    return subset


def build_loader(paths: List[Path], args, tag: str) -> DataLoader:
    tmp_txt = Path(args.work_dir) / f'tmp_{tag}.txt'
    tmp_txt.parent.mkdir(parents=True, exist_ok=True)
    tmp_txt.write_text("\n".join(str(p) for p in paths) + "\n")
    ds = NiftiTxtDataset(
        txt_files=str(tmp_txt),
        return_torch=True,
        memory_map=args.memory_map,
        cache_meta=True,
        T_prime=args.T_prime,
        tau_seconds=args.tau_seconds,
    )
    return DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=True, drop_last=False, collate_fn=custom_collate_fn
    )


def train_decoder(model: nn.Module, loader: DataLoader, device: torch.device, args, logger: logging.Logger):
    model.train()
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                            lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(loader) * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))

    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    gstep = 0
    for ep in range(args.epochs):
        ep_loss = 0.0
        for i, batch in enumerate(loader, 1):
            x, meta, orig_Ts, affines = prepare_batch_data(batch, device)
            with torch.cuda.amp.autocast(enabled=args.amp and torch.cuda.is_available()):
                loss, _, _, _ = model(
                    x,
                    mask_ratio=args.mask_ratio,
                    meta=meta,
                    orig_Ts=orig_Ts,
                    affines=affines,
                )
            opt.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad), args.grad_clip
                )
            opt.step()
            sch.step()
            ep_loss += loss.item()
            gstep += 1
            if i % args.log_interval == 0:
                logger.info(f"Epoch {ep+1} Step {i}/{len(loader)} "
                            f"Loss {loss.item():.6f} Avg {(ep_loss/i):.6f}")
        logger.info(f"[Epoch {ep+1}] Avg Loss: {(ep_loss/max(1,len(loader))):.6f}")


def make_tag(src_ckpt: str, train_list: str, fraction: float) -> str:
    src = Path(src_ckpt).stem
    dst = Path(train_list).stem
    frac = int(round(fraction * 100))
    return f"{src}__to__{dst}__p{frac}"


def ensure_model_defaults(a):
    """
    Ensure args has all hyperparams that build_model expects.
    Keep defaults consistent with your main training script.
    """
    safe_defaults = dict(
        # core
        model_type='mamba',
        embed_dim=512,
        depth=24,
        predictor_depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        drop_path_rate=0.1,
        rms_norm=False,
        predictor_dropout=0.0,
        predictor_mlp_ratio=4.0,
        # mamba/bimamba related
        bimamba_type='none',
        if_bimamba=False,
        mixer_type='mamba',
        if_devide_out=True,
        predictor_hidden=None,
        # training-time model opts
        momentum=0.992,
        norm_target=True,
        use_res_cond=False,
        fused_add_norm=True,
        residual_in_fp32=True,
    )
    for k, v in safe_defaults.items():
        if not hasattr(a, k):
            setattr(a, k, v)
    return a


# --------------------- One Run --------------------- #

def run_one(args, device: torch.device, job: Dict[str, Any]):
    subset_fraction = float(job.get("subset_fraction", 0.05))
    source_ckpt = job["source_checkpoint"]
    train_list = job["train_list"]

    tag = make_tag(source_ckpt, train_list, subset_fraction)

    # Decide naming by plan.json first (if output_dir given)
    if "output_dir" in job and str(job["output_dir"]).strip():
        output_dir = Path(job["output_dir"]).resolve()
        base = output_dir.name
        ckpt_path = output_dir / f"{base}.pt"
        log_dir = Path(args.log_dir) / base
        subset_output = Path(args.work_dir) / f"{base}.subset.txt"
        run_name_for_tmp = base
    else:
        output_dir = (Path(args.output_dir) / tag).resolve()
        ckpt_path = output_dir / f"{tag}.pt"
        log_dir = Path(args.log_dir) / tag
        subset_output = Path(args.work_dir) / f"{tag}.subset.txt"
        run_name_for_tmp = tag

    logger = setup_logging(str(log_dir))
    logger.info(f"=== Running transfer task: {run_name_for_tmp} ===")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Checkpoint will be saved to: {ckpt_path}")

    # model
    ensure_model_defaults(args)
    model = build_model(args, device).to(device)
    _ = load_source_checkpoint(model, source_ckpt, device, logger)
    freeze_encoder(model, logger)

    # data
    subset_paths = pick_subset(train_list, subset_fraction, str(subset_output), logger)
    loader = build_loader(subset_paths, args, run_name_for_tmp)
    logger.info(f"Subset dataset size: {len(loader.dataset)}")

    # optional per-run overrides
    orig_epochs, orig_lr, orig_mask_ratio = args.epochs, args.lr, args.mask_ratio
    if "epochs" in job: args.epochs = int(job["epochs"])
    if "lr" in job: args.lr = float(job["lr"])
    if "mask_ratio" in job: args.mask_ratio = float(job["mask_ratio"])

    # train
    train_decoder(model, loader, device, args, logger)

    # restore globals
    args.epochs, args.lr, args.mask_ratio = orig_epochs, orig_lr, orig_mask_ratio

    # save
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": vars(args),
        "tag": run_name_for_tmp
    }, ckpt_path)
    logger.info(f"Saved checkpoint -> {ckpt_path}")
    logger.info(f"Transfer finished: {run_name_for_tmp}")


# ----------------------- Main ---------------------- #

def main():
    parser = argparse.ArgumentParser(description="Batch Transfer Learning Runner")

    # I/O bases (used when a run doesn't specify output_dir)
    parser.add_argument('--batch_plan', type=str, required=True, help='Path to JSON plan file')
    parser.add_argument('--work_dir', type=str, default='./transfer_work')
    parser.add_argument('--output_dir', type=str, default='./checkpoints/transfer_decoder')
    parser.add_argument('--log_dir', type=str, default='./logs/transfer')

    # Data/common
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--memory_map', action='store_true', default=True)
    parser.add_argument('--T_prime', type=int, default=30)
    parser.add_argument('--tau_seconds', type=float, default=6.0)

    # Train knobs (can be overridden per-run in JSON)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--mask_ratio', type=float, default=0.65)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--amp', action='store_true', default=False)
    parser.add_argument('--log_interval', type=int, default=20)

    # ==== Model hyperparams expected by build_model ====
    parser.add_argument('--model_type', type=str, default='mamba', choices=['mamba','vit'])
    parser.add_argument('--embed_dim', type=int, default=512)
    parser.add_argument('--depth', type=int, default=24)
    parser.add_argument('--predictor_depth', type=int, default=6)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--mlp_ratio', type=float, default=4.0)
    parser.add_argument('--drop_path_rate', type=float, default=0.1)
    parser.add_argument('--rms_norm', action='store_true', default=False)
    parser.add_argument('--predictor_dropout', type=float, default=0.0)
    parser.add_argument('--predictor_mlp_ratio', type=float, default=4.0)

    parser.add_argument('--bimamba_type', type=str, default='none')
    parser.add_argument('--if_bimamba', action='store_true', default=False)
    parser.add_argument('--mixer_type', type=str, default='mamba')
    parser.add_argument('--if_devide_out', action='store_true', default=True)
    parser.add_argument('--predictor_hidden', type=int, default=None)

    parser.add_argument('--momentum', type=float, default=0.992)
    parser.add_argument('--norm_target', action='store_true', default=True)
    parser.add_argument('--use_res_cond', action='store_true', default=False)
    parser.add_argument('--fused_add_norm', action='store_true', default=True)
    parser.add_argument('--residual_in_fp32', action='store_true', default=True)

    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # Seeds & device
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load plan
    plan = json.loads(Path(args.batch_plan).read_text())
    runs = plan.get("runs", [])
    if not runs:
        raise ValueError("JSON plan has no 'runs'.")

    # Execute
    for job in runs:
        if "source_checkpoint" not in job or "train_list" not in job or "subset_fraction" not in job:
            raise ValueError(f"Run missing required keys: {job}")
        run_one(args, device, job)


if __name__ == '__main__':
    main()
