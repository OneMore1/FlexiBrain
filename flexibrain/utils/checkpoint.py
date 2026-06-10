import torch.nn as nn
import torch
from typing import Tuple
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import os

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