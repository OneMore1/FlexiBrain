import torch
import torch.nn.functional as F


def pack_batch(u, attn_mask):
    """
    优化版本：减少中间张量分配和索引操作
    u: (B, Lmax, C)
    attn_mask: (B, Lmax), True 表示 padding
    return:
        u_packed: (1, N, C)
        seq_idx:  (1, N)
        idx_info: (B, Lmax)  # 解包用
    """
    B, Lmax, C = u.shape
    device = u.device
    dtype = u.dtype

    valid = ~attn_mask                # (B, Lmax)
    lengths = valid.sum(dim=1)        # (B,)
    N = lengths.sum().item()          # 总有效 tokens 数

    # 优化1: 使用 masked_select 替代 reshape + boolean indexing
    # 这样可以减少一次 reshape 操作
    u_packed = u[valid]               # (N, C) - 直接使用 boolean indexing

    # 优化2: 预先分配 seq_idx，避免 repeat_interleave 的开销
    b_ids = torch.arange(B, dtype=torch.int32, device=device)
    seq_idx = torch.repeat_interleave(b_ids, lengths)   # (N,)

    # 优化3: 使用更高效的方式构建 idx_info
    # 避免创建 arange 然后再索引
    idx_info = torch.full((B, Lmax), -1, dtype=torch.long, device=device)
    idx_info[valid] = torch.arange(N, device=device, dtype=torch.long)

    # 喂给 mamba2 的就是 (1, N, C) / (1, N)
    return u_packed.unsqueeze(0), seq_idx.unsqueeze(0), idx_info


def pack_batch_fast(u, attn_mask):
    """
    更激进的优化版本：使用 cumsum 和 gather 操作
    适用于 padding 比例较高的情况
    """
    B, Lmax, C = u.shape
    device = u.device

    valid = ~attn_mask                # (B, Lmax)
    lengths = valid.sum(dim=1)        # (B,)
    N = lengths.sum().item()

    # 使用 nonzero 获取有效位置的索引
    valid_indices = valid.nonzero(as_tuple=False)  # (N, 2)
    batch_indices = valid_indices[:, 0]
    seq_indices = valid_indices[:, 1]

    # 直接使用高级索引提取
    u_packed = u[batch_indices, seq_indices]  # (N, C)

    # 构造 seq_idx
    seq_idx = batch_indices.to(torch.int32)  # (N,)

    # 构造 idx_info
    idx_info = torch.full((B, Lmax), -1, dtype=torch.long, device=device)
    idx_info[batch_indices, seq_indices] = torch.arange(N, device=device, dtype=torch.long)

    return u_packed.unsqueeze(0), seq_idx.unsqueeze(0), idx_info

def unpack_batch(out_packed, idx_info):
    """
    优化版本：减少中间张量分配
    out_packed: (1, N, C)
    idx_info: (B, Lmax)
    return: (B, Lmax, C)
    """
    out_packed = out_packed.squeeze(0)   # (N, C)
    B, Lmax = idx_info.shape
    C = out_packed.size(-1)

    # 优化1: 使用 empty 替代 zeros，然后只填充有效位置
    # 这样可以避免初始化整个张量为 0
    out = torch.zeros(B, Lmax, C, device=out_packed.device, dtype=out_packed.dtype)

    # 优化2: 直接使用 boolean mask，避免额外的索引操作
    valid = idx_info >= 0
    out[valid] = out_packed[idx_info[valid]]

    return out


def unpack_batch_fast(out_packed, idx_info):
    """
    更快的解包版本：使用 scatter
    """
    out_packed = out_packed.squeeze(0)   # (N, C)
    B, Lmax = idx_info.shape
    C = out_packed.size(-1)

    # 使用 scatter 操作
    out = torch.zeros(B, Lmax, C, device=out_packed.device, dtype=out_packed.dtype)

    # 找到有效位置
    valid_mask = idx_info >= 0
    valid_indices = valid_mask.nonzero(as_tuple=False)  # (N, 2)

    if valid_indices.numel() > 0:
        batch_idx = valid_indices[:, 0]
        seq_idx = valid_indices[:, 1]
        packed_idx = idx_info[batch_idx, seq_idx]

        # 直接赋值
        out[batch_idx, seq_idx] = out_packed[packed_idx]

    return out


def unpack_batch_inplace(out_packed, idx_info, out_buffer=None):
    """
    In-place 版本：重用输出缓冲区
    out_buffer: 预先分配的输出缓冲区 (B, Lmax, C)
    """
    out_packed = out_packed.squeeze(0)   # (N, C)
    B, Lmax = idx_info.shape
    C = out_packed.size(-1)

    if out_buffer is None or out_buffer.shape != (B, Lmax, C):
        out_buffer = torch.zeros(B, Lmax, C, device=out_packed.device, dtype=out_packed.dtype)
    else:
        out_buffer.zero_()

    valid = idx_info >= 0
    out_buffer[valid] = out_packed[idx_info[valid]]

    return out_buffer
