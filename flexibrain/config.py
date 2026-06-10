from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ModelConfig:
    model_type: str = "mamba"
    embed_dim: int = 512
    depth: int = 24
    predictor_depth: int = 2
    drop_path_rate: float = 0.1
    rms_norm: bool = False
    fused_add_norm: bool = True
    residual_in_fp32: bool = True
    bimamba_type: str = "v2"
    if_bimamba: bool = False
    mixer_type: str = "mamba"
    if_devide_out: bool = True
    momentum: float = 0.996
    final_momentum: float = 0.9999
    norm_target: bool = True
    num_heads: int = 8
    mlp_ratio: float = 4.0
    head_type: str = "transformer"
    num_classes: int = 3
    head_depth: int = 2
    head_num_heads: int = 8
    head_mlp_ratio: float = 4.0
    head_proj_drop: float = 0.1
    head_drop_path: float = 0.1
    mlp_hidden: int = 512
    mlp_depth: int = 4
    mlp_dropout: float = 0.1
    freeze_backbone: bool = False


@dataclass
class DataConfig:
    train_list: str = ""
    val_list: str = ""
    test_list: Optional[str] = None
    csv: Optional[str] = None
    id_column: str = "Subject"
    label_column: str = "Group_idx"
    label_mode: str = "multiclass"
    path_id_mode: str = "auto"
    normal_label: int = 2
    batch_size: int = 8
    num_workers: int = 8
    memory_map: bool = True
    T_prime: int = 30
    tau_seconds: float = 6.0


@dataclass
class TrainingConfig:
    epochs: int = 30
    lr: float = 5e-4
    lr_backbone: Optional[float] = None
    lr_head: Optional[float] = None
    weight_decay: float = 0.05
    warmup_epochs: int = 2
    mask_ratio: float = 0.65
    grad_clip: float = 1.0
    grad_accumulation_steps: int = 1
    seed: int = 42
    use_amp: bool = False
    local_rank: int = 0
    world_size: int = 1


@dataclass
class LoggingConfig:
    log_interval: int = 20
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    resume: Optional[str] = None


@dataclass
class RunConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    pretrain_checkpoint: Optional[str] = None
    from_scratch: bool = False
    use_checkpoint_config: bool = True


def _update_dataclass(obj, values: dict):
    for key, value in values.items():
        if hasattr(obj, key):
            setattr(obj, key, value)


def load_config(path: Optional[str]) -> RunConfig:
    cfg = RunConfig()
    if not path:
        return cfg
    data = yaml.safe_load(Path(path).read_text()) or {}
    if "model" in data:
        _update_dataclass(cfg.model, data["model"] or {})
    if "data" in data:
        _update_dataclass(cfg.data, data["data"] or {})
    if "training" in data:
        _update_dataclass(cfg.training, data["training"] or {})
    if "logging" in data:
        _update_dataclass(cfg.logging, data["logging"] or {})
    for key in ["pretrain_checkpoint", "from_scratch", "use_checkpoint_config"]:
        if key in data:
            setattr(cfg, key, data[key])
    return cfg


def apply_checkpoint_config(model_cfg: ModelConfig, checkpoint_config: dict) -> None:
    keys = [
        "model_type", "embed_dim", "depth", "predictor_depth", "drop_path_rate",
        "rms_norm", "fused_add_norm", "residual_in_fp32", "bimamba_type",
        "if_bimamba", "mixer_type", "if_devide_out", "momentum", "norm_target",
        "num_heads", "mlp_ratio",
    ]
    for key in keys:
        if key in checkpoint_config:
            setattr(model_cfg, key, checkpoint_config[key])
