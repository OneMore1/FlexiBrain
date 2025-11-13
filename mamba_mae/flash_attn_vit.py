import torch
import torch.nn as nn
from typing import Optional
from flash_attn import flash_attn_varlen_func
from flash_attn.bert_padding import unpad_input, pad_input


# ----------------------------
# Stochastic Depth (DropPath)
# ----------------------------
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        # [B, 1, 1] / [B, 1, 1, 1] 按批正则
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        return x / keep_prob * random_tensor


# ----------------------------
# MLP（无 LayerNorm，符合 MAE/ViT 规范）
# ----------------------------
class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


# ----------------------------
# 标准 ViT Attention（支持可选 attn_mask + Flash-Attn varlen）
# 输入/输出：x: [B, L, C] ；attn_mask: [B, L]，1/True=有效，0/False=padding
# ----------------------------
class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        causal: bool = False,     # MAE 为 False
        use_unpad_utils: bool = True,  # 有 mask 时建议 True
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.causal = causal
        self.use_unpad_utils = use_unpad_utils

    @torch.no_grad()
    def _make_dense_cu(self, B: int, L: int, device) -> torch.Tensor:
        # cu = [0, L, 2L, ..., B*L]  (int32)
        return torch.arange(0, (B + 1) * L, L, device=device, dtype=torch.int32)

    def _flash_varlen(
        self,
        q: torch.Tensor,  # [T, H, D]
        k: torch.Tensor,  # [T, H, D]
        v: torch.Tensor,  # [T, H, D]
        cu_q: torch.Tensor,
        cu_k: torch.Tensor,
        max_sq: int,
        max_sk: int,
        training: bool,
    ) -> torch.Tensor:
        dropout_p = self.attn_drop.p if training else 0.0
        out = flash_attn_varlen_func(
            q, k, v,
            cu_q, cu_k,
            max_sq, max_sk,
            dropout_p=dropout_p,
            softmax_scale=None,
            causal=self.causal
        )
        return out  # [T, H, D]

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [B, L, C]
        attn_mask: [B, L]，1/True=有效；None 则视为全有效、等长序列
        """
        assert x.dim() == 3, "x should be [B, L, C]"
        B, L, C = x.shape
        device = x.device

        # qkv 计算
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 1, 3, 4)
        q, k, v = qkv.unbind(0)  # [B, L, H, D]

        if attn_mask is None:
            # 等长、全有效：flatten + 构造等长 cu
            T = B * L
            q = q.reshape(T, self.num_heads, self.head_dim)
            k = k.reshape(T, self.num_heads, self.head_dim)
            v = v.reshape(T, self.num_heads, self.head_dim)

            cu = self._make_dense_cu(B, L, device)
            out = self._flash_varlen(q, k, v, cu, cu, L, L, self.training)   # [T, H, D]
            out = out.reshape(B, L, self.num_heads * self.head_dim)          # [B, L, C]
        else:
            mask_bool = attn_mask.bool()
            if self.use_unpad_utils:
                # 用官方工具去 pad
                q_unpad, idx_q, cu_q, max_sq = unpad_input(q, mask_bool)  # [T, H, D], [B+1]
                k_unpad, idx_k, cu_k, max_sk = unpad_input(k, mask_bool)
                v_unpad, _,     _,     _     = unpad_input(v, mask_bool)

                out_unpad = self._flash_varlen(q_unpad, k_unpad, v_unpad, cu_q, cu_k, max_sq, max_sk, self.training)
                out_unpad = out_unpad.reshape(-1, self.num_heads * self.head_dim)  # [T, C]
                out = pad_input(out_unpad, idx_q, B, L)                             # [B, L, C]
            else:
                # 手工 varlen（慢但易读）
                lengths = mask_bool.to(torch.int32).sum(dim=1)  # [B]
                max_seqlen = int(lengths.max().item())
                cu = torch.zeros(B + 1, dtype=torch.int32, device=device)
                cu[1:] = torch.cumsum(lengths, dim=0)

                flats = []
                for tensor in (q, k, v):
                    chunks = []
                    for i in range(B):
                        li = int(lengths[i])
                        if li > 0:
                            valid_idx = mask_bool[i].nonzero(as_tuple=False).squeeze(-1)
                            chunks.append(tensor[i, valid_idx])  # [li, H, D]
                    flats.append(torch.cat(chunks, dim=0) if chunks else tensor.new_zeros((0, self.num_heads, self.head_dim)))
                q_unpad, k_unpad, v_unpad = flats

                out_unpad = self._flash_varlen(q_unpad, k_unpad, v_unpad, cu, cu, max_seqlen, max_seqlen, self.training)
                out_unpad = out_unpad.reshape(-1, self.num_heads * self.head_dim)

                out = x.new_zeros((B, L, C))
                start = 0
                for i in range(B):
                    li = int(lengths[i])
                    if li > 0:
                        valid_idx = mask_bool[i].nonzero(as_tuple=False).squeeze(-1)
                        out[i, valid_idx] = out_unpad[start:start + li]
                        start += li

        # 输出线性
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


# ----------------------------
# 标准 MAE/ViT Block（Pre-LN, 可选 DropPath）
# ----------------------------
class ViTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        ffn_drop: float = 0.0,
        drop_path: float = 0.0,
        causal: bool = False,            # MAE: False
        use_unpad_utils: bool = True,    # 有 mask 时建议 True
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        self.ln1 = norm_layer(dim)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            causal=causal,
            use_unpad_utils=use_unpad_utils,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        self.ln2 = norm_layer(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout=ffn_drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-LN: x = x + Attn(LN1(x))
        x = x + self.drop_path1(self.attn(self.ln1(x), attn_mask=attn_mask))
        # Pre-LN: x = x + MLP(LN2(x))
        x = x + self.drop_path2(self.mlp(self.ln2(x)))
        return x
