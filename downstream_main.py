#!/usr/bin/env python3
"""
Downstream classification script for Volume JEPA models.

This script implements fine-tuning for binary classification tasks on pre-trained
VolumeMambaJEPA and VolumeVitJEPA models.
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

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Import models
from mamba_mae.models_vim_mae import VolumeMambaJEPA
from mamba_mae.models_vit_jepa import VolumeVitJEPA

# Import downstream components
from downstream_utils.dataset_downstream import ClassificationDataset, custom_collate_fn, prepare_batch_data
from downstream_utils.mamba import MambaJEPAClassifier, MambaJEPAClassifierAvgPool
from downstream_utils.vit import VolumeVitJEPAClassifierCLS, VolumeVitJEPAClassifierAvgPool
import re
from collections import defaultdict

def _is_norm_or_bias(name: str, module: nn.Module) -> bool:
    if name.endswith('bias'):
        return True
    # 典型归一化层的权重
    norm_types = (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)
    if isinstance(module, norm_types):
        return True
    # 常见 token/positional 参数也不做 wd
    bad_keywords = ['pos_embed', 'cls_token', 'rms_norm', 'norm', 'ln', 'bn']
    return any(k in name.lower() for k in bad_keywords)

def _infer_depth_from_name(param_name: str) -> Optional[int]:
    """
    从参数名推断 layer id（越小越靠近输入）。
    适配常见命名：blocks.i / layers.i / stages.i
    返回 int 或 None（无法推断时）。
    """
    for pat in [r'blocks\.(\d+)', r'layers\.(\d+)', r'stages\.(\d+)']:
        m = re.search(pat, param_name)
        if m:
            return int(m.group(1))
    return None

def _count_backbone_layers(model: nn.Module) -> int:
    """
    估计 backbone 的层数（用于 LLRD）。
    优先读 model.backbone.depth；否则扫描参数名的最大索引。
    """
    depth_candidates = []
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'depth'):
        return int(model.backbone.depth)
    for n, _ in model.named_parameters():
        if n.startswith('backbone.'):
            di = _infer_depth_from_name(n)
            if di is not None:
                depth_candidates.append(di)
    return (max(depth_candidates) + 1) if depth_candidates else 12  # 合理缺省

def build_param_groups_with_llrd(
    model: nn.Module,
    base_lr: float,
    lr_backbone: Optional[float],
    lr_head: Optional[float],
    layer_decay: float,
    weight_decay: float,
    zero_wd_on_norm_bias: bool = True,
    logger: Optional[logging.Logger] = None,
):
    """
    返回可直接传给 AdamW 的 param_groups 列表，实现：
      - backbone 分层 lr：lr_layer = (lr_backbone or base_lr) * layer_decay^(L-1-layer_id)
      - head 用 (lr_head or base_lr)
      - 对 norm/bias/pos_embed/cls_token 设 wd=0（可选）
    """
    if isinstance(model, DDP):
        model_ref = model.module
    else:
        model_ref = model

    L = _count_backbone_layers(model_ref)
    layer_scales = [layer_decay ** (L - 1 - i) for i in range(L + 1)]  # 0..L

    groups = defaultdict(lambda: {'params': [], 'lr': None, 'weight_decay': None})

    for n, p in model_ref.named_parameters():
        if not p.requires_grad:
            continue

        # 归类：head vs backbone
        is_backbone = n.startswith('backbone.')
        if not is_backbone:
            # 认为是 head（下游分类头）
            group_name = 'head'
            lr = (lr_head if lr_head is not None else base_lr)
            wd = 0.0 if (zero_wd_on_norm_bias and _is_norm_or_bias(n, None)) else weight_decay
        else:
            # backbone：根据层号做 LLRD
            lid = _infer_depth_from_name(n)
            # 层号失败时给“stem层”（0 层）
            lid = 0 if lid is None else (lid + 1)  # 预留 0 给 stem/patch_embed/pos_embed
            if any(k in n for k in ['patch_embed', 'stem', 'pos_embed', 'cls_token']):
                lid = 0
            scale = layer_scales[min(lid, L)]
            lr_b = (lr_backbone if lr_backbone is not None else base_lr)
            lr = lr_b * scale

            # 对 norm/bias/pos 置 0 wd（可选）
            wd = weight_decay
            if zero_wd_on_norm_bias:
                # 拿到 module 用来判断是否是norm层
                module = model_ref
                # 尽量获取 module 实例（若拿不到就用 name 关键词法）
                try:
                    mod = model_ref
                    for attr in n.split('.')[:-1]:
                        mod = getattr(mod, attr)
                    module = mod
                except Exception:
                    module = None
                if _is_norm_or_bias(n, module):
                    wd = 0.0

            group_name = f'backbone_layer_{lid:02d}'

        # 塞进 param group
        key = (group_name, lr, wd)
        if key not in groups:
            groups[key]['lr'] = lr
            groups[key]['weight_decay'] = wd
        groups[key]['params'].append(p)

    # 把 dict -> list
    param_groups = [{'params': v['params'], 'lr': v['lr'], 'weight_decay': v['weight_decay']}
                    for v in groups.values()]

    # 记录一下便于核对
    if logger is not None:
        logger.info(f"LLRD groups = {len(param_groups)}")
        for i, g in enumerate(sorted(param_groups, key=lambda x: x['lr'])):
            n_params = sum(p.numel() for p in g['params'])
            logger.info(f"  Group[{i:02d}] lr={g['lr']:.6e}, wd={g['weight_decay']}, #params={n_params:,}")

    return param_groups


def setup_logging(log_dir: str, rank: int = 0) -> logging.Logger:
    """Setup logging."""
    if rank == 0:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"downstream_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(level=logging.WARNING)
    
    return logging.getLogger(__name__)


def build_downstream_model(args, device, logger) -> nn.Module:
    """Build downstream classification model."""

    # Handle from_scratch training
    if args.from_scratch:
        logger.info(f"🔄 Training from scratch (not loading pre-trained weights)")
        checkpoint = None
        config = {}
    else:
        logger.info(f"Loading pre-trained {args.model_type.upper()} backbone from {args.pretrain_checkpoint}")
        # Load checkpoint
        checkpoint = torch.load(args.pretrain_checkpoint, map_location=device)
        config = checkpoint.get('config', {})

    # Extract configuration from checkpoint or use command-line args
    if config:
        logger.info(f"✓ Found config in checkpoint")
        # Use checkpoint config for model construction
        embed_dim = config.get('embed_dim', args.embed_dim)
        depth = config.get('depth', args.depth)
        predictor_depth = config.get('predictor_depth', args.predictor_depth)
        drop_path_rate = config.get('drop_path_rate', args.drop_path_rate)
        rms_norm = config.get('rms_norm', args.rms_norm)
        fused_add_norm = config.get('fused_add_norm', args.fused_add_norm)
        residual_in_fp32 = config.get('residual_in_fp32', args.residual_in_fp32)
        bimamba_type = config.get('bimamba_type', args.bimamba_type)
        if_bimamba = config.get('if_bimamba', args.if_bimamba)
        mixer_type = config.get('mixer_type', args.mixer_type)
        if_devide_out = config.get('if_devide_out', args.if_devide_out)
        predictor_hidden = config.get('predictor_hidden', args.predictor_hidden)
        momentum = config.get('momentum', args.momentum)
        norm_target = config.get('norm_target', args.norm_target)
        num_heads = config.get('num_heads', args.num_heads)
        mlp_ratio = config.get('mlp_ratio', args.mlp_ratio)

        # 🔍 智能检测：检查 checkpoint 中是否有双向参数
        if checkpoint is not None:
            model_state_key = 'model_state_dict' if 'model_state_dict' in checkpoint else 'model'
            state_dict = checkpoint[model_state_key]

            # 检查是否有后向扫描参数（BiMamba v2 特有）
            # 只检查 mixer 中的后向参数，避免误判
            backward_params = [k for k in state_dict.keys() if 'mixer.' in k and any(x in k for x in ['A_b_log', 'D_b', 'conv1d_b', 'x_proj_b', 'dt_proj_b'])]
            has_backward_params = len(backward_params) > 0

            if has_backward_params:
                logger.info(f"✓ 检测到双向参数 ({len(backward_params)} 个后向扫描参数)")
                logger.info(f"  → 使用 bimamba_type='{bimamba_type}', if_bimamba={if_bimamba}")
            else:
                logger.info(f"⚠️  未检测到双向参数，checkpoint 只包含单向 Mamba 参数")
                if bimamba_type in ['v1', 'v2']:
                    logger.warning(f"  → 自动切换到 bimamba_type='none', if_bimamba=False (单向模式)")
                    bimamba_type = 'none'
                    if_bimamba = False
                else:
                    logger.info(f"  → 保持 bimamba_type='{bimamba_type}', if_bimamba={if_bimamba}")

        # Log config mismatch warnings
        if embed_dim != args.embed_dim:
            logger.warning(f"⚠️  embed_dim mismatch: checkpoint={embed_dim}, args={args.embed_dim} → using checkpoint value")
        if depth != args.depth:
            logger.warning(f"⚠️  depth mismatch: checkpoint={depth}, args={args.depth} → using checkpoint value")
    else:
        if not args.from_scratch:
            logger.warning(f"⚠️  No config found in checkpoint, using command-line arguments")
        else:
            logger.info(f"✓ Using command-line arguments for model configuration")
        embed_dim = args.embed_dim
        depth = args.depth
        predictor_depth = args.predictor_depth
        drop_path_rate = args.drop_path_rate
        rms_norm = args.rms_norm
        fused_add_norm = args.fused_add_norm
        residual_in_fp32 = args.residual_in_fp32
        bimamba_type = args.bimamba_type
        if_bimamba = args.if_bimamba
        mixer_type = args.mixer_type
        if_devide_out = args.if_devide_out
        predictor_hidden = args.predictor_hidden
        momentum = args.momentum
        norm_target = args.norm_target
        num_heads = args.num_heads
        mlp_ratio = args.mlp_ratio

    # Load pre-trained backbone
    if args.model_type == 'mamba':
        backbone = VolumeMambaJEPA(
            embed_dim=embed_dim,
            depth=depth,
            predictor_depth=predictor_depth,
            ssm_cfg=None,
            encoder_attn_layer_idx=None,
            attn_cfg=None,
            drop_path_rate=drop_path_rate,
            norm_epsilon=1e-5,
            rms_norm=rms_norm,
            initializer_cfg=None,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            device=device,
            dtype=torch.float32,
            bimamba_type=bimamba_type,
            if_bimamba=if_bimamba,
            mixer_type=mixer_type,
            if_devide_out=if_devide_out,
            predictor_hidden=predictor_hidden,
            momentum=momentum,
            norm_target=norm_target,
        )

        # Load checkpoint with strict=False to allow version compatibility (if not from_scratch)
        if not args.from_scratch and checkpoint is not None:
            try:
                model_state_key = 'model_state_dict' if 'model_state_dict' in checkpoint else 'model'

                # First try strict=True
                try:
                    backbone.load_state_dict(checkpoint[model_state_key], strict=True)
                    logger.info(f"✓ All backbone weights loaded successfully (strict=True)")
                except RuntimeError as e:
                    # If strict=True fails, try strict=False for version compatibility
                    logger.warning(f"⚠ strict=True failed, trying strict=False for version compatibility")
                    logger.warning(f"  Error was: {str(e)[:200]}...")

                    incompatible_keys = backbone.load_state_dict(checkpoint[model_state_key], strict=False)

                    if incompatible_keys.missing_keys:
                        logger.warning(f"⚠ Missing keys in checkpoint ({len(incompatible_keys.missing_keys)}):")
                        # Check if all missing keys are backward scan parameters (bimamba v2)
                        missing_b_params = [k for k in incompatible_keys.missing_keys if any(x in k for x in ['_b.', '_b_', 'conv1d_b', 'x_proj_b', 'dt_proj_b', 'A_b_log', 'D_b'])]
                        if len(missing_b_params) == len(incompatible_keys.missing_keys):
                            logger.warning(f"  All missing keys are backward scan parameters (BiMamba v2)")
                            logger.warning(f"  This is likely due to version mismatch between training and inference code")
                            logger.warning(f"  The backward scan parameters will be randomly initialized")
                        else:
                            logger.warning(f"  First 10 missing keys: {incompatible_keys.missing_keys[:10]}")

                    if incompatible_keys.unexpected_keys:
                        logger.warning(f"⚠ Unexpected keys in checkpoint ({len(incompatible_keys.unexpected_keys)}):")
                        logger.warning(f"  First 10 unexpected keys: {incompatible_keys.unexpected_keys[:10]}")

                    logger.info(f"✓ Backbone weights loaded with strict=False (some parameters may be randomly initialized)")

            except Exception as e:
                logger.error(f"❌ Failed to load backbone weights: {e}")
                raise
        else:
            logger.info(f"✓ Backbone initialized with random weights (from_scratch=True)")

        # Build classifier based on head_type
        if args.head_type == 'transformer':
            model = MambaJEPAClassifier(
                backbone=backbone,
                num_classes=args.num_classes,
                head_depth=args.head_depth,
                head_num_heads=args.head_num_heads,
                head_mlp_ratio=args.head_mlp_ratio,
                head_proj_drop=args.head_proj_drop,
                head_drop_path=args.head_drop_path,
                mlp_hidden=args.mlp_hidden,
                mlp_depth=args.mlp_depth,
                mlp_dropout=args.mlp_dropout,
                freeze_backbone=args.freeze_backbone,
                device=device,
            )
            logger.info(f"✓ Using Transformer head (CLS + {args.head_depth} Transformer blocks)")
        elif args.head_type == 'avgpool':
            model = MambaJEPAClassifierAvgPool(
                backbone=backbone,
                num_classes=args.num_classes,
                mlp_hidden=args.mlp_hidden,
                mlp_depth=args.mlp_depth,
                mlp_dropout=args.mlp_dropout,
                freeze_backbone=args.freeze_backbone,
                device=device,
            )
            logger.info(f"✓ Using Average Pooling head ({args.mlp_depth} MLP layers)")
        else:
            raise ValueError(f"Unknown head_type: {args.head_type}")
        
    elif args.model_type == 'vit':
        backbone = VolumeVitJEPA(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            predictor_depth=predictor_depth,
            drop_path_rate=drop_path_rate,
            norm_epsilon=1e-5,
            rms_norm=rms_norm,
            device=device,
            dtype=torch.float32,
            momentum=momentum,
            norm_target=norm_target,
        )

        # Load checkpoint with strict=False to allow architecture mismatch (e.g., hybrid Mamba+ViT)
        if not args.from_scratch and checkpoint is not None:
            try:
                model_state_key = 'model_state_dict' if 'model_state_dict' in checkpoint else 'model'
                # Use strict=False to allow loading into different architectures (e.g., hybrid Mamba+ViT)
                incompatible_keys = backbone.load_state_dict(checkpoint[model_state_key], strict=False)

                if incompatible_keys.missing_keys:
                    logger.warning(f"⚠ Missing keys ({len(incompatible_keys.missing_keys)}): {incompatible_keys.missing_keys[:5]}...")
                if incompatible_keys.unexpected_keys:
                    logger.warning(f"⚠ Unexpected keys ({len(incompatible_keys.unexpected_keys)}): {incompatible_keys.unexpected_keys[:5]}...")

                logger.info(f"✓ Backbone weights loaded (strict=False, allows architecture mismatch)")
            except RuntimeError as e:
                logger.error(f"❌ Failed to load backbone weights: {e}")
                raise
        else:
            logger.info(f"✓ Backbone initialized with random weights (from_scratch=True)")

        # Build classifier based on head_type
        if args.head_type == 'transformer':
            model = VolumeVitJEPAClassifierCLS(
                backbone=backbone,
                num_classes=args.num_classes,
                head_depth=args.head_depth,
                head_num_heads=args.head_num_heads,
                head_mlp_ratio=args.head_mlp_ratio,
                head_proj_drop=args.head_proj_drop,
                head_drop_path=args.head_drop_path,
                mlp_hidden=args.mlp_hidden,
                mlp_depth=args.mlp_depth,
                mlp_dropout=args.mlp_dropout,
                freeze_backbone=args.freeze_backbone,
                device=device,
            )
            logger.info(f"✓ Using Transformer head (CLS + {args.head_depth} Transformer blocks)")
        elif args.head_type == 'avgpool':
            model = VolumeVitJEPAClassifierAvgPool(
                backbone=backbone,
                num_classes=args.num_classes,
                mlp_hidden=args.mlp_hidden,
                mlp_depth=args.mlp_depth,
                mlp_dropout=args.mlp_dropout,
                freeze_backbone=args.freeze_backbone,
                device=device,
            )
            logger.info(f"✓ Using Average Pooling head ({args.mlp_depth} MLP layers)")
        else:
            raise ValueError(f"Unknown head_type: {args.head_type}")
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    return model.to(device)


def build_downstream_dataloaders(args, rank: int = 0, world_size: int = 1) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Build downstream classification dataloaders (train, val, and optional test)."""
    # Build datasets
    train_set = ClassificationDataset(
        txt_files=args.train_list,
        # csv_path='ppmi.csv',
        return_torch=True,
        memory_map=args.memory_map,
        cache_meta=True,
        T_prime=args.T_prime,
        tau_seconds=args.tau_seconds,
    )

    val_set = ClassificationDataset(
        txt_files=args.val_list,
        # csv_path='ppmi.csv',
        return_torch=True,
        memory_map=args.memory_map,
        cache_meta=True,
        T_prime=args.T_prime,
        tau_seconds=args.tau_seconds,
    )

    # Build test set if provided
    test_set = None
    if args.test_list is not None:
        test_set = ClassificationDataset(
            txt_files=args.test_list,
            # csv_path='ppmi.csv',
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

    test_sampler = None
    if test_set is not None:
        test_sampler = DistributedSampler(
            test_set,
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

    test_loader = None
    if test_set is not None:
        test_loader = DataLoader(
            test_set,
            batch_size=args.batch_size,
            sampler=test_sampler,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=custom_collate_fn,
        )

    return train_loader, val_loader, test_loader


def train_downstream_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
    args,
    logger: logging.Logger,
    rank: int = 0,
) -> float:
    """Train downstream classifier for one epoch with gradient accumulation."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    criterion = nn.CrossEntropyLoss()
    accumulation_steps = args.grad_accumulation_steps

    for batch_idx, batch in enumerate(train_loader):
        # Prepare batch data
        x, meta, orig_Ts, labels, affines = prepare_batch_data(batch, device)

        # Forward pass
        with torch.cuda.amp.autocast(enabled=args.use_amp):
            logits = model(x, meta=meta, orig_Ts=orig_Ts, affines=affines)
            loss = criterion(logits, labels)
            # Scale loss by accumulation steps
            loss = loss / accumulation_steps

        # Backward pass
        loss.backward()

        # Update weights every accumulation_steps batches
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        num_batches += 1

        if rank == 0 and (batch_idx + 1) % args.log_interval == 0:
            logger.info(f"Epoch {epoch + 1} [{batch_idx + 1}/{len(train_loader)}] Loss: {loss.item() * accumulation_steps:.6f}")

    # Final optimizer step if there are remaining gradients
    if num_batches % accumulation_steps != 0:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def evaluate_downstream(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    args,
    logger: logging.Logger,
    split_name: str = "Validation",
    rank: int = 0,
) -> Dict[str, float]:
    """Evaluate downstream classifier on any dataset split."""
    model.eval()
    criterion = nn.CrossEntropyLoss()

    all_preds = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in data_loader:
            # Prepare batch data
            x, meta, orig_Ts, labels, affines = prepare_batch_data(batch, device)

            # Forward pass
            logits = model(x, meta=meta, orig_Ts=orig_Ts, affines=affines)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            num_batches += 1

            # Get predictions
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    avg_loss = total_loss / max(num_batches, 1)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

    if rank == 0:
        logger.info(f"{split_name} - Loss: {avg_loss:.6f}, Acc: {accuracy:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}, F1: {f1:.4f}")

    return metrics


def save_downstream_checkpoint(model, optimizer, scheduler, epoch, metrics, checkpoint_dir, rank=0):
    """Save downstream checkpoint."""
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint = {
            'epoch': epoch,
            'model': model.state_dict() if not isinstance(model, DDP) else model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'metrics': metrics,
        }
        path = os.path.join(checkpoint_dir, f"downstream_epoch_{epoch:03d}.pt")
        torch.save(checkpoint, path)


def main():
    """Main downstream training function."""
    parser = argparse.ArgumentParser(description='Downstream classification for Volume JEPA')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='mamba', choices=['mamba', 'vit'],
                        help='Model type: mamba or vit')
    parser.add_argument('--pretrain_checkpoint', type=str, default=None,
                        help='Path to pre-trained checkpoint (not required if --from_scratch is used)')
    parser.add_argument('--from_scratch', action='store_true',
                        help='Train from scratch without loading pre-trained weights')
    parser.add_argument('--embed_dim', type=int, default=512, help='Embedding dimension')
    parser.add_argument('--depth', type=int, default=24, help='Model depth')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads (ViT only)')
    parser.add_argument('--mlp_ratio', type=float, default=4.0, help='MLP ratio (ViT only)')
    parser.add_argument('--predictor_depth', type=int, default=4, help='Predictor depth')
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help='Drop path rate')
    parser.add_argument('--rms_norm', action='store_true', help='Use RMS norm')
    parser.add_argument('--fused_add_norm', type=bool, default=True, help='Use fused add norm')
    parser.add_argument('--residual_in_fp32', type=bool, default=True, help='Residual in fp32')
    parser.add_argument('--bimamba_type', type=str, default='none', help='BiMamba type')
    parser.add_argument('--if_bimamba', type=bool, default=False, help='Use BiMamba')
    parser.add_argument('--mixer_type', type=str, default='mamba', help='Mixer type')
    parser.add_argument('--if_devide_out', action='store_true', help='Divide output')
    parser.add_argument('--predictor_hidden', type=int, default=None, help='Predictor hidden dim')
    parser.add_argument('--momentum', type=float, default=0.996, help='EMA momentum')
    parser.add_argument('--norm_target', action='store_true', help='Normalize target')
    
    # Downstream classifier arguments
    parser.add_argument('--head_type', type=str, default='transformer',
                        choices=['transformer', 'avgpool'],
                        help='Head architecture type: transformer (CLS + Transformer blocks) or avgpool (average pooling + MLP)')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--head_depth', type=int, default=2, help='Head depth (for transformer head)')
    parser.add_argument('--head_num_heads', type=int, default=8, help='Head num heads (for transformer head)')
    parser.add_argument('--head_mlp_ratio', type=float, default=4.0, help='Head MLP ratio (for transformer head)')
    parser.add_argument('--head_proj_drop', type=float, default=0.1, help='Head projection dropout')
    parser.add_argument('--head_drop_path', type=float, default=0.05, help='Head drop path (for transformer head)')
    parser.add_argument('--mlp_hidden', type=int, default=1024, help='MLP hidden dimension')
    parser.add_argument('--mlp_depth', type=int, default=3, help='MLP depth')
    parser.add_argument('--mlp_dropout', type=float, default=0.1, help='MLP dropout')
    parser.add_argument('--freeze_backbone', action='store_true', help='Freeze backbone')
    
    # Data arguments
    parser.add_argument('--train_list', type=str, required=True, help='Training list file')
    parser.add_argument('--val_list', type=str, required=True, help='Validation list file')
    parser.add_argument('--test_list', type=str, default=None, help='Test list file (optional)')
    parser.add_argument('--T_prime', type=int, default=30, help='T_prime for dataset')
    parser.add_argument('--tau_seconds', type=float, default=6.0, help='Tau in seconds')
    parser.add_argument('--memory_map', type=bool, default=True, help='Use memory mapping')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers')

    # Training arguments
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay')
    parser.add_argument('--lr_backbone', type=float, default=6e-6,
                        help='Base LR for backbone (defaults to --lr if None)')
    parser.add_argument('--lr_head', type=float, default=6e-5,
                        help='Base LR for head (defaults to --lr if None)')
    parser.add_argument('--layer_decay', type=float, default=1,
                        help='Layer-wise LR decay factor (e.g., 0.8)')
    parser.add_argument('--no_wd_on_norm_and_bias', action='store_true',
                        help='Set weight_decay=0 for norm/bias/pos_embed/cls_token')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--warmup_epochs', type=int, default=3, help='Warmup epochs')
    parser.add_argument('--grad_accumulation_steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision')
    parser.add_argument('--log_interval', type=int, default=20, help='Log interval')
    
    # Checkpoint arguments
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/downstream/mamba-moe-base-adni-visual',
                        help='Checkpoint directory')
    parser.add_argument('--log_dir', type=str, default='./logs/downstream/mamba-moe-base-adni-visual',
                        help='Log directory')
    
    # Distributed arguments
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank')
    parser.add_argument('--world_size', type=int, default=1, help='World size')
    
    args = parser.parse_args()

    # Validate arguments
    if not args.from_scratch and args.pretrain_checkpoint is None:
        raise ValueError("Either --pretrain_checkpoint must be provided or --from_scratch must be set")

    if args.from_scratch and args.pretrain_checkpoint is not None:
        print("⚠️  Warning: Both --from_scratch and --pretrain_checkpoint provided. Using --from_scratch (ignoring checkpoint)")

    # Setup device
    rank = args.local_rank
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

    # Setup logging
    logger = setup_logging(args.log_dir, rank=rank)

    if rank == 0:
        logger.info(f"Starting downstream classification on device: {device}")
        if args.from_scratch:
            logger.info(f"🔄 Training mode: FROM SCRATCH (no pre-trained weights)")
        else:
            logger.info(f"📦 Training mode: FINE-TUNING (loading pre-trained weights)")
        logger.info(f"Arguments: {json.dumps(vars(args), indent=2)}")
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Build model
    model = build_downstream_model(args, device, logger)
    
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model created with {total_params:,} parameters")
        logger.info(f"  - Trainable: {trainable_params:,}")
    
    # Build dataloaders
    train_loader, val_loader, test_loader = build_downstream_dataloaders(args, rank=rank, world_size=args.world_size)

    if rank == 0:
        logger.info(f"Train set size: {len(train_loader.dataset)}")
        logger.info(f"Val set size: {len(val_loader.dataset)}")
        if test_loader is not None:
            logger.info(f"Test set size: {len(test_loader.dataset)}")


    # Build optimizer
    # param_groups = build_param_groups_with_llrd(
    # model=model,
    # base_lr=args.lr,
    # lr_backbone=args.lr_backbone,
    # lr_head=args.lr_head,
    # layer_decay=args.layer_decay,
    # weight_decay=args.weight_decay,
    # zero_wd_on_norm_bias=args.no_wd_on_norm_and_bias,
    # logger=logger if args.local_rank == 0 else None,
    # )
    # optimizer = optim.AdamW(param_groups)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # Build learning rate scheduler
    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.0, (total_steps - step) / (total_steps - warmup_steps))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training loop
    best_f1 = 0.0
    for epoch in range(args.epochs):
        if rank == 0:
            logger.info(f"Epoch {epoch + 1}/{args.epochs}")
            logger.info(f"{'='*60}")

        # Train
        train_loss = train_downstream_one_epoch(
            model, train_loader, optimizer, scheduler,
            device, epoch, args, logger, rank=rank
        )

        # Validate
        val_metrics = evaluate_downstream(
            model, val_loader, device, args, logger, split_name="Validation", rank=rank
        )

        # Test (if test set is provided)
        test_metrics = None
        if test_loader is not None:
            test_metrics = evaluate_downstream(
                model, test_loader, device, args, logger, split_name="Test", rank=rank
            )

        # Save checkpoint
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            if rank == 0:
                logger.info(f"New best F1 score: {best_f1:.4f}")

        # Prepare metrics for checkpoint
        checkpoint_metrics = {'val': val_metrics}
        if test_metrics is not None:
            checkpoint_metrics['test'] = test_metrics

        save_downstream_checkpoint(
            model, optimizer, scheduler, epoch, checkpoint_metrics,
            args.checkpoint_dir, rank=rank
        )

        if rank == 0:
            logger.info(f"Train Loss: {train_loss:.6f}")
            logger.info(f"Val Metrics: {val_metrics}")
            if test_metrics is not None:
                logger.info(f"Test Metrics: {test_metrics}")


if __name__ == '__main__':
    main()

