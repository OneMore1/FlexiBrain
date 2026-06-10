from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

from flexibrain.config import ModelConfig, apply_checkpoint_config
from flexibrain.models.mamba_jepa import VolumeMambaJEPA
from flexibrain.models.classifier import MambaJEPAClassifier, MambaJEPAClassifierAvgPool


def build_mamba_backbone(cfg: ModelConfig, device: torch.device, dtype=torch.float32) -> VolumeMambaJEPA:
    return VolumeMambaJEPA(
        embed_dim=cfg.embed_dim,
        depth=cfg.depth,
        predictor_depth=cfg.predictor_depth,
        ssm_cfg=None,
        encoder_attn_layer_idx=None,
        attn_cfg=None,
        drop_path_rate=cfg.drop_path_rate,
        norm_epsilon=1e-5,
        rms_norm=cfg.rms_norm,
        initializer_cfg=None,
        fused_add_norm=cfg.fused_add_norm,
        residual_in_fp32=cfg.residual_in_fp32,
        device=device,
        dtype=dtype,
        bimamba_type=cfg.bimamba_type,
        if_bimamba=cfg.if_bimamba,
        mixer_type=cfg.mixer_type,
        if_devide_out=cfg.if_devide_out,
        momentum=cfg.momentum,
        norm_target=cfg.norm_target,
    )


def build_pretrain_model(cfg: ModelConfig, device: torch.device) -> nn.Module:
    if cfg.model_type != "mamba":
        raise ValueError("This cleaned Flexibrain build currently keeps only the Mamba pretrain/downstream path")
    return build_mamba_backbone(cfg, device=device, dtype=torch.float32).to(device)


def load_checkpoint(path: str, device: torch.device):
    return torch.load(path, map_location=device)


def state_dict_from_checkpoint(checkpoint: dict):
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if "model" in checkpoint:
        return checkpoint["model"]
    raise KeyError("Checkpoint has neither model_state_dict nor model")


def build_downstream_model(cfg: ModelConfig, device: torch.device, logger: Optional[logging.Logger] = None, checkpoint_path: Optional[str] = None, from_scratch: bool = False, use_checkpoint_config: bool = True) -> nn.Module:
    checkpoint = None
    if checkpoint_path and not from_scratch:
        checkpoint = load_checkpoint(checkpoint_path, device)
        if use_checkpoint_config:
            apply_checkpoint_config(cfg, checkpoint.get("config", {}))
            if logger:
                logger.info("Backbone config restored from checkpoint: %s", checkpoint.get("config", {}))
    if cfg.model_type != "mamba":
        raise ValueError("This cleaned Flexibrain build currently keeps only the Mamba downstream path")
    backbone = build_mamba_backbone(cfg, device=device, dtype=torch.float32)
    if checkpoint is not None:
        state = state_dict_from_checkpoint(checkpoint)
        try:
            backbone.load_state_dict(state, strict=True)
            if logger:
                logger.info("Loaded pretrained backbone strictly from %s", checkpoint_path)
        except RuntimeError as exc:
            incompatible = backbone.load_state_dict(state, strict=False)
            backward_markers = ["_b", "conv1d_b", "x_proj_b", "dt_proj_b", "A_b_log", "D_b"]
            missing = list(incompatible.missing_keys)
            only_backward = missing and all(any(marker in key for marker in backward_markers) for key in missing)
            if not only_backward or incompatible.unexpected_keys:
                raise exc
            if logger:
                logger.warning("Strict load missed %d backward-scan BiMamba keys; loaded checkpoint with strict=False compatibility", len(missing))
    elif logger:
        logger.info("Backbone initialized from scratch")
    if cfg.head_type == "transformer":
        model = MambaJEPAClassifier(
            backbone=backbone,
            num_classes=cfg.num_classes,
            head_depth=cfg.head_depth,
            head_num_heads=cfg.head_num_heads,
            head_mlp_ratio=cfg.head_mlp_ratio,
            head_proj_drop=cfg.head_proj_drop,
            head_drop_path=cfg.head_drop_path,
            mlp_hidden=cfg.mlp_hidden,
            mlp_depth=cfg.mlp_depth,
            mlp_dropout=cfg.mlp_dropout,
            freeze_backbone=cfg.freeze_backbone,
            device=device,
        )
    elif cfg.head_type == "avgpool":
        model = MambaJEPAClassifierAvgPool(backbone=backbone, num_classes=cfg.num_classes, mlp_hidden=cfg.mlp_hidden, mlp_depth=cfg.mlp_depth, mlp_dropout=cfg.mlp_dropout, freeze_backbone=cfg.freeze_backbone, device=device)
    else:
        raise ValueError(f"Unknown head_type: {cfg.head_type}")
    return model.to(device)
