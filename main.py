#!/usr/bin/env python3
"""
Pre-training script for Volume JEPA models on NIfTI medical imaging data.

This script implements the main training loop for both VolumeMambaJEPA and VolumeVitJEPA models,
which perform masked autoencoder pre-training on 3D medical volumes with dynamic EMA.
"""

import argparse
import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple, List, Any
import math

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import nibabel as nib

# Import models and dataset
from mamba_mae.models_vim_mae import VolumeMambaJEPA
from mamba_mae.models_vit_jepa import VolumeVitJEPA
from mamba_mae.moe_gradient_monitor import MoEGradientMonitor
from dataset import NiftiTxtDataset, build_train_val_from_lists
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function to handle:
    1. nibabel headers and other non-tensor objects
    2. Variable-length time dimensions (due to different TR values)

    For variable-length data, we pad to the maximum length in the batch.
    """
    # Separate tensor/array fields from non-collatable fields
    tensor_fields = ['data', 'affine']
    scalar_fields = ['tr', 'subject_idx', 'T_selected', 'T_prime', 'tau_seconds']
    tuple_fields = ['voxel']
    object_fields = ['header', 'path']  # These won't be collated

    collated = {}

    # Handle tensor/array fields with padding for variable-length data
    for field in tensor_fields:
        if field in batch[0]:
            values = [item[field] for item in batch]

            if field == 'data':
                # Data has variable time dimension due to different TR values
                # Pad all to the maximum time length
                max_t = max(v.shape[-1] if len(v.shape) >= 4 else 1 for v in values)

                padded_values = []
                for v in values:
                    if len(v.shape) >= 4 and v.shape[-1] < max_t:
                        # Pad in time dimension (last dimension)
                        pad_amount = max_t - v.shape[-1]
                        if isinstance(v, torch.Tensor):
                            v = torch.nn.functional.pad(v, (0, pad_amount), mode='constant', value=0)
                        else:
                            v = np.pad(v, ((0, 0), (0, 0), (0, 0), (0, pad_amount)), mode='constant', value=0)
                    padded_values.append(v)

                # Convert to tensor and stack
                if isinstance(padded_values[0], torch.Tensor):
                    collated[field] = torch.stack(padded_values)
                else:
                    collated[field] = torch.from_numpy(np.stack(padded_values))
            else:
                # Affine matrices should all be the same size (4x4)
                if isinstance(values[0], torch.Tensor):
                    collated[field] = torch.stack(values)
                else:
                    collated[field] = torch.from_numpy(np.stack(values))

    # Handle scalar fields
    for field in scalar_fields:
        if field in batch[0]:
            values = [item[field] for item in batch]
            if isinstance(values[0], (int, float)):
                collated[field] = torch.tensor(values)
            else:
                collated[field] = values

    # Handle tuple fields (like voxel sizes)
    for field in tuple_fields:
        if field in batch[0]:
            collated[field] = [item[field] for item in batch]

    # Handle object fields (keep as lists)
    for field in object_fields:
        if field in batch[0]:
            collated[field] = [item[field] for item in batch]

    return collated


# Setup logging
def setup_logging(log_dir: str, rank: int = 0) -> logging.Logger:
    """Setup logging configuration."""
    if rank == 0:
        os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG if rank == 0 else logging.WARNING)

    if rank == 0:
        # File handler
        fh = logging.FileHandler(os.path.join(log_dir, 'train.log'))
        fh.setLevel(logging.DEBUG)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


def setup_distributed(rank: int, world_size: int) -> None:
    """Setup distributed training."""
    if world_size > 1:
        os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
        dist.init_process_group(
            backend='nccl',
            rank=rank,
            world_size=world_size
        )


def cleanup_distributed() -> None:
    """Cleanup distributed training."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def prepare_batch_data(batch: Dict, device: torch.device) -> Tuple[torch.Tensor, Dict, np.ndarray, Optional[torch.Tensor]]:
    """
    Prepare batch data for model forward pass.

    Returns:
        x: Input tensor (B, 96, 96, 96, T_max)
        meta: Dict {subject_idx: {"voxel": (vx, vy, vz), "tr": float}}
        orig_Ts: Array of original time steps
        affines: Affine matrices or None
    """
    # Move data to device
    x = batch['data'].to(device, dtype=torch.float32)

    # Build meta dict: {batch_index: {"voxel": (vx, vy, vz), "tr": float}}
    subject_idxs = batch['subject_idx'].cpu().numpy()
    voxels = batch['voxel']  # List of tuples or tensor
    trs = batch['tr'].cpu().numpy() if isinstance(batch['tr'], torch.Tensor) else batch['tr']

    meta = {}
    for i, subject_idx in enumerate(subject_idxs):
        # Handle voxel format
        if isinstance(voxels, (list, tuple)):
            voxel = voxels[i]
        else:
            voxel = tuple(voxels[i].cpu().numpy()) if isinstance(voxels[i], torch.Tensor) else voxels[i]

        tr = float(trs[i])
        # Use batch index (i) as key, not subject_idx
        meta[i] = {"voxel": voxel, "tr": tr}

    # Get original time steps (number of frames, not TR)
    # T_selected is the actual number of time frames selected by the dataset
    # Do NOT use 'tr' (time resolution in seconds) as it will cause incorrect T_pad calculation
    if 'T_selected' in batch:
        orig_Ts = batch['T_selected'].cpu().numpy() if isinstance(batch['T_selected'], torch.Tensor) else batch['T_selected']
    else:
        # Fallback: use actual data time dimension if T_selected is not available
        orig_Ts = np.array([x.shape[-1] for x in batch['data']])

    # Get affines if available
    affines = batch['affine'].to(device, dtype=torch.float32) if 'affine' in batch else None

    return x, meta, orig_Ts, affines


def build_model(args, device) -> nn.Module:
    """Build JEPA model (VolumeMambaJEPA or VolumeVitJEPA)."""
    if args.model_type == 'mamba':
        model = VolumeMambaJEPA(
            embed_dim=args.embed_dim,
            depth=args.depth,
            predictor_depth=args.predictor_depth,
            ssm_cfg=None,
            encoder_attn_layer_idx=None,
            attn_cfg=None,
            drop_path_rate=args.drop_path_rate,
            norm_epsilon=1e-5,
            rms_norm=args.rms_norm,
            initializer_cfg=None,
            fused_add_norm=args.fused_add_norm,
            residual_in_fp32=args.residual_in_fp32,
            device=device,
            dtype=torch.float32,
            bimamba_type=args.bimamba_type,
            if_bimamba=args.if_bimamba,
            mixer_type=args.mixer_type,
            if_devide_out=args.if_devide_out,
            predictor_hidden=args.predictor_hidden,
            momentum=args.momentum,
            norm_target=args.norm_target,
        )
    elif args.model_type == 'vit':
        model = VolumeVitJEPA(
            embed_dim=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            mlp_ratio=args.mlp_ratio,
            predictor_depth=args.predictor_depth,
            drop_path_rate=args.drop_path_rate,
            norm_epsilon=1e-5,
            rms_norm=args.rms_norm,
            device=device,
            dtype=torch.float32,
            momentum=args.momentum,
            norm_target=args.norm_target,
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    return model


def build_dataloaders(args, rank: int = 0, world_size: int = 1) -> Tuple[DataLoader, DataLoader]:
    """Build train and validation dataloaders."""
    # Build datasets
    train_set = NiftiTxtDataset(
        txt_files=args.train_list,
        return_torch=True,
        memory_map=args.memory_map,
        cache_meta=True,
        T_prime=args.T_prime,
        tau_seconds=args.tau_seconds,
    )

    val_set = NiftiTxtDataset(
        txt_files=args.val_list,
        return_torch=True,
        memory_map=args.memory_map,
        cache_meta=True,
        T_prime=args.T_prime,
        tau_seconds=args.tau_seconds,
    )

    # Create samplers for distributed training
    train_sampler = DistributedSampler(
        train_set,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed,
    ) if world_size > 1 else None

    val_sampler = DistributedSampler(
        val_set,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        seed=args.seed,
    ) if world_size > 1 else None

    # Create dataloaders
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=custom_collate_fn,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=custom_collate_fn,
    )

    return train_loader, val_loader


def update_ema(model: nn.Module, momentum: float) -> None:
    """Update target encoder with EMA."""
    if hasattr(model, 'update_target_encoder'):
        model.update_target_encoder(m=momentum)
    elif isinstance(model, DDP) and hasattr(model.module, 'update_target_encoder'):
        model.module.update_target_encoder(m=momentum)


def get_dynamic_momentum(epoch: int, total_epochs: int, base_momentum: float = 0.996, final_momentum: float = 0.9999) -> float:
    """
    Calculate dynamic momentum for EMA.

    Momentum increases from base_momentum to final_momentum over training.
    This helps stabilize training in later epochs.
    """
    progress = epoch / total_epochs
    # Cosine annealing: start at base, end at final
    momentum = final_momentum - (final_momentum - base_momentum) * 0.5 * (1 + np.cos(np.pi * progress))
    return momentum


import time
import torch

def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    args,
    logger: logging.Logger,
    rank: int = 0,
    moe_gradient_monitor: Optional[MoEGradientMonitor] = None,
) -> float:
    """Train for one epoch with timing."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    accumulation_steps = args.grad_accumulation_steps

    # timing meters
    data_time_sum = 0.0          # DataLoader 产出 batch 的等待时间
    h2d_time_sum = 0.0           # CPU->GPU 张量搬运/prepare_batch_data 时间
    compute_time_sum = 0.0       # 前向+反向(+step) 计算时间

    dynamic_momentum = get_dynamic_momentum(epoch, total_epochs, args.momentum, args.final_momentum)

    # 用于测量 "等待下一个 batch" 的起点
    end = time.perf_counter()

    # 计算全局步数（用于 MoE 监测）
    global_step = epoch * len(train_loader)

    for batch_idx, batch in enumerate(train_loader):
        # 1) 等待 DataLoader 产出 batch 的时间
        data_time = time.perf_counter() - end
        data_time_sum += data_time

        # 2) 设备搬运/预处理时间（通常你的 prepare_batch_data 会 .to(device)）
        t0 = time.perf_counter()
        x, meta, orig_Ts, affines = prepare_batch_data(batch, device)
        h2d_time = time.perf_counter() - t0
        h2d_time_sum += h2d_time

        # 3) 计算时间（GPU 上需同步保证计时准确）
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_compute = time.perf_counter()

        loss, _, _, _ = model(
            x,
            mask_ratio=args.mask_ratio,
            meta=meta,
            orig_Ts=orig_Ts,
            affines=affines,
        )
        scaled_loss = loss / accumulation_steps
        scaled_loss.backward()

        total_loss += loss.item()
        num_batches += 1

        if (batch_idx + 1) % accumulation_steps == 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # 🔬 MoE 梯度监测（在 optimizer.step() 之前）
            if moe_gradient_monitor is not None and rank == 0:
                current_step = global_step + batch_idx + 1
                # 每 N 步监测一次梯度
                if hasattr(args, 'moe_gradient_log_interval') and current_step % args.moe_gradient_log_interval == 0:
                    moe_gradient_monitor.log_gradient_stats(
                        step=current_step,
                        prefix=f"Epoch {epoch} "
                    )

            optimizer.step()
            optimizer.zero_grad()
            update_ema(model, dynamic_momentum)
            if scheduler is not None:
                scheduler.step()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        compute_time = time.perf_counter() - t_compute
        compute_time_sum += compute_time

        # 下一次循环前的“等待 DataLoader”起点
        end = time.perf_counter()

        if rank == 0 and (batch_idx + 1) % args.log_interval == 0:
            avg_loss = total_loss / num_batches
            logger.info(
                f"Epoch {epoch} [{batch_idx + 1}/{len(train_loader)}] "
                f"Loss: {loss.item():.6f} (Avg: {avg_loss:.6f}) "
                f"Momentum: {dynamic_momentum:.6f} | "
                f"data: {data_time*1000:.1f}ms h2d: {h2d_time*1000:.1f}ms "
                f"compute: {compute_time*1000:.1f}ms"
            )

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    if rank == 0 and num_batches > 0:
        logger.info(
            f"[Epoch {epoch} Summary] "
            f"Avg loss: {avg_loss:.6f} | "
            f"Avg data: {data_time_sum/num_batches*1000:.1f}ms "
            f"Avg h2d: {h2d_time_sum/num_batches*1000:.1f}ms "
            f"Avg compute: {compute_time_sum/num_batches*1000:.1f}ms"
        )

    return avg_loss



@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    epoch: int,
    args,
    logger: logging.Logger,
    rank: int = 0,
) -> float:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in val_loader:
        # Prepare batch data
        x, meta, orig_Ts, affines = prepare_batch_data(batch, device)

        # Forward pass
        loss, _, _, _ = model(
            x,
            mask_ratio=args.mask_ratio,
            meta=meta,
            orig_Ts=orig_Ts,
            affines=affines,
        )

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    if rank == 0:
        logger.info(f"Epoch {epoch} Validation Loss: {avg_loss:.6f}")

    return avg_loss


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    epoch: int,
    best_loss: float,
    checkpoint_dir: str,
    args=None,
    rank: int = 0,
) -> None:
    """Save model checkpoint with configuration."""
    if rank != 0:
        return

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Get model state dict (handle DDP wrapper)
    model_state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'best_loss': best_loss,
    }

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    # Save model configuration for downstream tasks
    if args is not None:
        checkpoint['config'] = {
            'model_type': args.model_type,
            'embed_dim': args.embed_dim,
            'depth': args.depth,
            'predictor_depth': args.predictor_depth,
            'drop_path_rate': args.drop_path_rate,
            'rms_norm': args.rms_norm,
            'fused_add_norm': args.fused_add_norm,
            'residual_in_fp32': args.residual_in_fp32,
            'bimamba_type': args.bimamba_type,
            'if_bimamba': args.if_bimamba,
            'mixer_type': args.mixer_type,
            'if_devide_out': args.if_devide_out,
            'predictor_hidden': args.predictor_hidden,
            'momentum': args.momentum,
            'norm_target': args.norm_target,
            'num_heads': args.num_heads,
            'mlp_ratio': args.mlp_ratio,
        }

    # Save latest checkpoint
    latest_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pt')
    torch.save(checkpoint, latest_path)

    # Save best checkpoint
    if best_loss is not None:
        best_path = os.path.join(checkpoint_dir, 'checkpoint_best.pt')
        torch.save(checkpoint, best_path)


def load_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[int, float]:
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model state dict (handle DDP wrapper)
    if isinstance(model, DDP):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    best_loss = checkpoint.get('best_loss', float('inf'))

    return epoch, best_loss


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Pre-train VolumeMambaJEPA or VolumeVitJEPA')

    # Data arguments
    parser.add_argument('--train_list', type=str, required=True,
                        help='Path to training list file(s)')
    parser.add_argument('--val_list', type=str, required=True,
                        help='Path to validation list file(s)')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size per GPU')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--memory_map', action='store_true', default=True,
                        help='Use memory mapping for NIfTI files')
    parser.add_argument('--T_prime', type=int, default=30,
                        help='Target number of time patches after TAPE')
    parser.add_argument('--tau_seconds', type=float, default=6.0,
                        help='Time window in seconds for TAPE kernel')

    # Model arguments
    parser.add_argument('--model_type', type=str, default='mamba', choices=['mamba', 'vit'],
                        help='Model type: mamba or vit')
    parser.add_argument('--embed_dim', type=int, default=512,
                        help='Embedding dimension')
    parser.add_argument('--depth', type=int, default=24,
                        help='Number of transformer blocks')
    parser.add_argument('--predictor_depth', type=int, default=4,
                        help='Number of predictor blocks')
    parser.add_argument('--drop_path_rate', type=float, default=0.1,
                        help='Drop path rate')
    parser.add_argument('--rms_norm', action='store_true', default=False,
                        help='Use RMSNorm instead of LayerNorm')
    parser.add_argument('--fused_add_norm', action='store_true', default=True,
                        help='Use fused add norm')
    parser.add_argument('--residual_in_fp32', action='store_true', default=True,
                        help='Keep residual in fp32')
    parser.add_argument('--bimamba_type', type=str, default='none',
                        help='BiMamba type')
    parser.add_argument('--if_bimamba', type=bool, default=False,
                        help='Use BiMamba')
    parser.add_argument('--mixer_type', type=str, default='mamba',
                        help='Mixer type')
    parser.add_argument('--if_devide_out', action='store_true', default=True,
                        help='Divide output')
    parser.add_argument('--predictor_hidden', type=int, default=None,
                        help='Predictor hidden dimension')

    # ViT specific arguments
    parser.add_argument('--num_heads', type=int, default=12,
                        help='Number of attention heads (ViT only)')
    parser.add_argument('--mlp_ratio', type=float, default=4.0,
                        help='MLP ratio (ViT only)')

    # EMA arguments
    parser.add_argument('--momentum', type=float, default=0.992,
                        help='Base momentum for EMA update')
    parser.add_argument('--final_momentum', type=float, default=0.9999,
                        help='Final momentum for EMA update (dynamic EMA)')
    parser.add_argument('--norm_target', action='store_true', default=True,
                        help='Normalize target features')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=3,
                        help='Number of warmup epochs')
    parser.add_argument('--mask_ratio', type=float, default=0.65,
                        help='Masking ratio')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping value')
    parser.add_argument('--grad_accumulation_steps', type=int, default=8,
                        help='Number of gradient accumulation steps')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # Logging and checkpoint arguments
    parser.add_argument('--log_interval', type=int, default=20,
                        help='Logging interval')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/mamba-moe-large-monitor-aux-balance',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs/mamba-moe-large-monitor-aux-balance',
                            help='Directory to save logs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    # MoE monitoring arguments
    parser.add_argument('--enable_moe_monitoring', action='store_true', default=True,
                        help='Enable MoE gradient monitoring')
    parser.add_argument('--moe_gradient_log_interval', type=int, default=120,
                        help='MoE gradient logging interval (in steps)')

    # Distributed training arguments
    parser.add_argument('--local_rank', type=int, default=0,
                        help='Local rank for distributed training')
    parser.add_argument('--world_size', type=int, default=1,
                        help='World size for distributed training')

    args = parser.parse_args()

    # Setup distributed training
    rank = args.local_rank
    world_size = args.world_size
    if world_size > 1:
        setup_distributed(rank, world_size)

    # Setup device
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

    # Setup logging
    logger = setup_logging(args.log_dir, rank=rank)

    if rank == 0:
        logger.info(f"Starting pre-training on device: {device}")
        logger.info(f"Arguments: {json.dumps(vars(args), indent=2)}")

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Build model
    model = build_model(args, device).to(device)

    # Wrap with DDP if distributed
    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        logger.info(f"Model created with {total_params:,} parameters")
        logger.info(f"  - Trainable: {trainable_params:,}")
        logger.info(f"  - Non-trainable: {non_trainable_params:,}")

    # Build dataloaders
    train_loader, val_loader = build_dataloaders(args, rank=rank, world_size=world_size)

    if rank == 0:
        logger.info(f"Train set size: {len(train_loader.dataset)}")
        logger.info(f"Val set size: {len(val_loader.dataset)}")

    # Initialize MoE gradient monitor (if enabled)
    moe_gradient_monitor = None
    if args.enable_moe_monitoring and rank == 0:
        # 获取实际的模型（如果是 DDP 包装的）
        actual_model = model.module if hasattr(model, 'module') else model

        # 检查模型是否有 MoE
        if hasattr(actual_model, 'moe'):
            logger.info("🔬 Initializing MoE gradient monitor...")
            moe_gradient_monitor = MoEGradientMonitor(actual_model.moe, logger=logger)
            logger.info(f"  ✓ MoE monitor enabled (log interval: {args.moe_gradient_log_interval} steps)")
        else:
            logger.warning("⚠️  MoE monitoring enabled but model has no 'moe' attribute")

    # Build optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Build learning rate scheduler
    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs

    
    def lr_lambda(step): # cos annealing
        if step < warmup_steps:
            return step / warmup_steps
        
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        num_cycles = 4
        cycle_progress = (progress * num_cycles) % 1.0
        
        if cycle_progress < 0.8:
            return 0.5 * (1 + math.cos(math.pi * cycle_progress / 0.8))
        else:
            return 0.5 * (1 + math.cos(math.pi))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Resume from checkpoint if specified
    start_epoch = 0
    best_loss = float('inf')
    if args.resume is not None:
        if rank == 0:
            logger.info(f"Resuming from checkpoint: {args.resume}")
        start_epoch, best_loss = load_checkpoint(
            model, optimizer, scheduler, args.resume, device
        )
        start_epoch += 1

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        if rank == 0:
            logger.info(f"Epoch {epoch + 1}/{args.epochs}")
            logger.info(f"{'='*60}")

        # Set epoch for distributed sampler
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            device, epoch, args.epochs, args, logger, rank=rank,
            moe_gradient_monitor=moe_gradient_monitor
        )

        # Validate
        val_loss = validate(
            model, val_loader, device, epoch, args, logger, rank=rank
        )

        # Save checkpoint
        if val_loss < best_loss:
            best_loss = val_loss
            if rank == 0:
                logger.info(f"New best validation loss: {best_loss:.6f}")

        save_checkpoint(
            model, optimizer, scheduler, epoch, best_loss,
            args.checkpoint_dir, args=args, rank=rank
        )

        if rank == 0:
            dynamic_momentum = get_dynamic_momentum(epoch, args.epochs, args.momentum, args.final_momentum)
            logger.info(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Momentum: {dynamic_momentum:.6f}, lr: {scheduler.get_last_lr()[0]:.6f}")

    if rank == 0:
        logger.info("\nTraining completed!")

    # Cleanup distributed training
    if world_size > 1:
        cleanup_distributed()


if __name__ == '__main__':
    main()
