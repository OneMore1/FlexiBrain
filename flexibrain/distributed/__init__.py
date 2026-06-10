import os
import torch.distributed as dist


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