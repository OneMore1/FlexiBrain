from __future__ import annotations

from typing import Optional, Tuple

from torch.utils.data import DataLoader, DistributedSampler

from flexibrain.config import DataConfig, TrainingConfig
from flexibrain.data.nifti import NiftiTxtDataset
from flexibrain.data.classification import ClassificationDataset, custom_collate_fn as downstream_collate
from flexibrain.data.collate import custom_collate_fn as pretrain_collate


def build_pretrain_dataloaders(data: DataConfig, training: TrainingConfig, rank: int = 0, world_size: int = 1) -> Tuple[DataLoader, DataLoader]:
    train_set = NiftiTxtDataset(data.train_list, return_torch=True, memory_map=data.memory_map, cache_meta=True, T_prime=data.T_prime, tau_seconds=data.tau_seconds)
    val_set = NiftiTxtDataset(data.val_list, return_torch=True, memory_map=data.memory_map, cache_meta=True, T_prime=data.T_prime, tau_seconds=data.tau_seconds)
    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True, seed=training.seed) if world_size > 1 else None
    val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=rank, shuffle=False, seed=training.seed) if world_size > 1 else None
    train_loader = DataLoader(train_set, batch_size=data.batch_size, sampler=train_sampler, shuffle=train_sampler is None, num_workers=data.num_workers, pin_memory=True, drop_last=True, collate_fn=pretrain_collate)
    val_loader = DataLoader(val_set, batch_size=data.batch_size, sampler=val_sampler, shuffle=False, num_workers=data.num_workers, pin_memory=True, drop_last=False, collate_fn=pretrain_collate)
    return train_loader, val_loader


def _classification_dataset(txt_file: Optional[str], data: DataConfig):
    if not txt_file:
        return None
    if not data.csv:
        raise ValueError("data.csv is required for downstream classification")
    return ClassificationDataset(
        txt_files=txt_file,
        csv_path=data.csv,
        id_column=data.id_column,
        label_column=data.label_column,
        label_mode=data.label_mode,
        path_id_mode=data.path_id_mode,
        normal_label=data.normal_label,
        return_torch=True,
        memory_map=data.memory_map,
        cache_meta=True,
        T_prime=data.T_prime,
        tau_seconds=data.tau_seconds,
    )


def build_downstream_dataloaders(data: DataConfig, training: TrainingConfig, rank: int = 0, world_size: int = 1):
    train_set = _classification_dataset(data.train_list, data)
    val_set = _classification_dataset(data.val_list, data)
    test_set = _classification_dataset(data.test_list, data)
    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True, seed=training.seed) if world_size > 1 else None
    val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=rank, shuffle=False, seed=training.seed) if world_size > 1 else None
    test_sampler = DistributedSampler(test_set, num_replicas=world_size, rank=rank, shuffle=False, seed=training.seed) if world_size > 1 and test_set is not None else None
    train_loader = DataLoader(train_set, batch_size=data.batch_size, sampler=train_sampler, shuffle=train_sampler is None, num_workers=data.num_workers, pin_memory=True, drop_last=True, collate_fn=downstream_collate)
    val_loader = DataLoader(val_set, batch_size=data.batch_size, sampler=val_sampler, shuffle=False, num_workers=data.num_workers, pin_memory=True, drop_last=False, collate_fn=downstream_collate)
    test_loader = None
    if test_set is not None:
        test_loader = DataLoader(test_set, batch_size=data.batch_size, sampler=test_sampler, shuffle=False, num_workers=data.num_workers, pin_memory=True, drop_last=False, collate_fn=downstream_collate)
    return train_loader, val_loader, test_loader
