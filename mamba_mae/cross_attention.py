import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict

# 你原本代码里的环境标记
try:
    import torch
    is_torch2 = int(torch.__version__.split('.')[0]) >= 2
except Exception:
    is_torch2 = False
from flex_patch_utils import pi_resize_weight_1d, pi_resize_weight_3d


class CrossAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = decoder_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(decoder_dim, decoder_dim, bias=qkv_bias)
        self.kv = nn.Linear(encoder_dim, decoder_dim * 2, bias=qkv_bias)

        if is_torch2:
            # 在 SDPA 分支里这是一个 float 概率
            self.attn_drop = attn_drop
        else:
            # 在旧分支里是一个模块
            self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(decoder_dim, decoder_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _normalize_bool_mask(self, mask, *, B, L, device):
        """
        将 mask 规整为 (B, L) 的 bool 张量；允许传入 (B, L) 或 (B, L, 1) 或 None。
        语义：True = 屏蔽/忽略
        """
        if mask is None:
            return torch.zeros(B, L, dtype=torch.bool, device=device)
        if mask.dtype != torch.bool:
            # 兼容 0/1、float 等；非零视为 True
            mask = mask.to(device).ne(0)
        else:
            mask = mask.to(device)
        # 去掉可能多出来的最后一维
        if mask.dim() == 3 and mask.size(-1) == 1:
            mask = mask.squeeze(-1)
        assert mask.shape == (B, L), f"mask shape should be (B,{L}), got {tuple(mask.shape)}"
        return mask

    def forward(self, x, y, q_mask=None, kv_mask=None):
        """
        x: (B, N, C)  —— decoder 序列，作为 query
        y: (B, Ny, *) —— encoder 序列，作为 key/value
        q_mask:  (B, N)   bool，True=屏蔽该 query 位置
        kv_mask: (B, Ny)  bool，True=屏蔽该 key/value 位置
        """
        B, N, C = x.shape
        Ny = y.shape[1]
        device = x.device

        # 规整掩码（True = 屏蔽）
        q_mask = self._normalize_bool_mask(q_mask, B=B, L=N, device=device) if q_mask is not None else None
        kv_mask = self._normalize_bool_mask(kv_mask, B=B, L=Ny, device=device) if kv_mask is not None else None

        # 计算投影并做多头拆分
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)   # (B, H, N, Dh)
        kv = self.kv(y).reshape(B, Ny, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # (B, H, Ny, Dh)

        # 组合 pairwise 掩码： (B, N, Ny) —— True 的对 (i,j) 被禁止关注
        pair_mask = None
        if (q_mask is not None) or (kv_mask is not None):
            if q_mask is None:
                q_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
            if kv_mask is None:
                kv_mask = torch.zeros(B, Ny, dtype=torch.bool, device=device)
            pair_mask = q_mask[:, :, None] | kv_mask[:, None, :]  # (B, N, Ny)

        if is_torch2:
            # PyTorch 2.x: F.scaled_dot_product_attention
            # 注意：bool attn_mask 里 True=不允许关注（屏蔽）
            if pair_mask is not None:
                attn_mask = pair_mask[:, None, :, :]  # (B, 1, N, Ny)，按头广播
            else:
                attn_mask = None

            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,   # 可以是 None 或 bool/float
                dropout_p=self.attn_drop,
                is_causal=False
            )  # (B, H, N, Dh)

            x = out.transpose(1, 2).reshape(B, N, C)

        else:
            # 旧分支：手工打分 + mask + softmax
            attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, Ny)

            if pair_mask is not None:
                # True 的地方加 -inf
                minus_inf = torch.finfo(attn.dtype).min
                attn = attn.masked_fill(pair_mask[:, None, :, :], minus_inf)

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # 输出投影
        x = self.proj(x)
        x = self.proj_drop(x)

        # 为稳妥起见：把被 q_mask=True 的行输出清零（若未提供 q_mask 则不做）
        if q_mask is not None:
            x = x.masked_fill(q_mask[:, :, None], 0.0)

        return x


class PIAdaptiveLinear4D(nn.Module):
    """
    将 decoder 向量（decoder_dim）投影为每样本各自的 F_i = C*pH*pW*pD*pT。
    共享一套“基准权重 + 参数化插值”，类似 STAPE 的核重采样。

    基准参数：
      W_base: [decoder_dim, C, kH0, kW0, kD0, kT0]
      b_base: [C, kH0, kW0, kD0, kT0]  （可选）

    前向：
      dec:       [B, L_max, decoder_dim]
      attn_mask: [B, L_max]  (True=pad)
      patch_sizes: List[Tuple[pH,pW,pD,pT]]  每样本一组
      C: 通道数（与 F_i 中的 C 一致，通常 1 或 3）

    返回：
      y:         [B, L_max, F_max]   （按样本 Fi 右侧 zero-pad 到 F_max）
      feat_mask: [B, L_max, F_max]   （True=pad；有效区域为 False）
      Fi_list:   List[int]           每样本的输出维（C*pH*pW*pD*pT）
    """
    def __init__(
        self,
        decoder_dim:     int,
        C:               int,
        kH0: int, kW0: int, kD0: int, kT0: int,
        use_bias: bool = True
    ):
        super().__init__()
        self.decoder_dim = decoder_dim
        self.C = int(C)
        self.kH0, self.kW0, self.kD0, self.kT0 = int(kH0), int(kW0), int(kD0), int(kT0)

        # 基准权重：D x C x H0 x W0 x D0 x T0
        self.W_base = nn.Parameter(
            torch.randn(decoder_dim, C, kH0, kW0, kD0, kT0) * (1.0 / decoder_dim) ** 0.5
        )
        if use_bias:
            self.b_base = nn.Parameter(torch.zeros(C, kH0, kW0, kD0, kT0))
        else:
            self.b_base = None

    def _resize_weight_for_size(
        self,
        pH: int, pW: int, pD: int, pT: int,
        dtype: torch.dtype, device: torch.device
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        将基准 (H0,W0,D0,T0) 插值到目标 (pH,pW,pD,pT)。
        返回：
          W_lin: [F_i, decoder_dim]
          b_lin: [F_i] 或 None
        注意：不缓存已插值结果（避免断梯度）；在一个 forward 内会做“按尺寸去重”的小缓存。
        """
        D = self.decoder_dim
        C = self.C
        H0, W0, D0, T0 = self.kH0, self.kW0, self.kD0, self.kT0

        W0 = self.W_base.to(dtype=dtype, device=device)  # [D, C, H0, W0, D0, T0]

        # --- 先做空间 3D 插值（对每个 T 切片独立）---
        # 视作 (D*C*T0) 个 3D 核，每个大小 H0xW0xD0
        W_sp_in = W0.permute(0, 1, 5, 2, 3, 4).contiguous()                 # [D, C, T0, H0, W0, D0]
        W_sp_in = W_sp_in.view(D * C * T0, 1, H0, W0, D0)                   # [D*C*T0, 1, H0, W0, D0]
        W_sp = pi_resize_weight_3d(W_sp_in, pH, pW, pD)                      # [D*C*T0, 1, pH, pW, pD]
        W_sp = W_sp.view(D, C, T0, pH, pW, pD).permute(0, 1, 3, 4, 5, 2)    # [D, C, pH, pW, pD, T0]

        # --- 再做时间 1D 插值 ---
        W_t_in = W_sp.view(D * C * pH * pW * pD, 1, T0)                     # [D*C*pH*pW*pD, 1, T0]
        W_t  = pi_resize_weight_1d(W_t_in, pT)                               # [D*C*pH*pW*pD, 1, pT]
        W_t  = W_t.view(D, C, pH, pW, pD, pT)                                # [D, C, pH, pW, pD, pT]

        # 变成线性层权重（out_features, in_features）
        W_lin = W_t.permute(1, 2, 3, 4, 5, 0).contiguous()                   # [C, pH, pW, pD, pT, D]
        W_lin = W_lin.view(C * pH * pW * pD * pT, D)                         # [F_i, D]

        # 偏置同样插值（若需要）
        if self.b_base is not None:
            b0 = self.b_base.to(dtype=dtype, device=device)                  # [C, H0, W0, D0, T0]
            b_sp_in = b0.permute(0, 4, 1, 2, 3).contiguous()                 # [C, T0, H0, W0, D0]
            b_sp_in = b_sp_in.view(C * T0, 1, H0, W0, D0)                    # [C*T0, 1, H0, W0, D0]
            b_sp = pi_resize_weight_3d(b_sp_in, pH, pW, pD)                  # [C*T0, 1, pH, pW, pD]
            b_sp = b_sp.view(C, T0, pH, pW, pD).permute(0, 2, 3, 4, 1)       # [C, pH, pW, pD, T0]

            b_t_in = b_sp.view(C * pH * pW * pD, 1, T0)                      # [C*pH*pW*pD, 1, T0]
            b_t  = pi_resize_weight_1d(b_t_in, pT)                            # [C*pH*pW*pD, 1, pT]
            b_t  = b_t.view(C, pH, pW, pD, pT)                                # [C, pH, pW, pD, pT]
            b_lin = b_t.reshape(C * pH * pW * pD * pT)                        # [F_i]
        else:
            b_lin = None

        return W_lin, b_lin

    def forward(
        self,
        dec: torch.Tensor,                              # [B, L_max, decoder_dim]
        attn_mask: torch.Tensor,                        # [B, L_max]  True=pad
        patch_sizes: List[Tuple[int,int,int,int]],      # 每样本 (pH,pW,pD,pT)
    ):
        B, Lmax, D = dec.shape
        assert D == self.decoder_dim, f"decoder_dim mismatch: got {D}, expect {self.decoder_dim}"
        assert attn_mask.shape == (B, Lmax)
        assert len(patch_sizes) == B

        device, dtype = dec.device, dec.dtype

        # 统计每样本输出维 Fi，并找 F_max 以便右侧 pad
        Fi_list = [self.C * int(pH) * int(pW) * int(pD) * int(pT) for (pH,pW,pD,pT) in patch_sizes]
        F_max = max(Fi_list) if Fi_list else 0

        # 结果容器
        y = dec.new_zeros((B, Lmax, F_max), dtype=dtype)
        feat_mask = torch.ones((B, Lmax, F_max), dtype=torch.bool, device=device)

        # 同尺寸样本只插值一次（保持可导，勿 .detach）
        unique_keys = {}
        for sz in set(patch_sizes):
            pH,pW,pD,pT = map(int, sz)
            W_lin, b_lin = self._resize_weight_for_size(pH,pW,pD,pT, dtype=dtype, device=device)  # [Fi, D], [Fi]
            unique_keys[sz] = (W_lin, b_lin)

        # 逐样本应用线性层到有效 token
        for b in range(B):
            pH,pW,pD,pT = map(int, patch_sizes[b])
            Fi = Fi_list[b]
            if Fi == 0:
                continue  # 理论上不会出现

            W_lin, b_lin = unique_keys[(pH,pW,pD,pT)]  # [Fi, D], [Fi] or None

            # 有效 token 数（排除 pad）
            Lb = int((~attn_mask[b]).sum().item())
            if Lb == 0:
                continue

            xb = dec[b, :Lb, :]                        # [Lb, D]
            # yb = xb @ W^T + b
            yb = torch.matmul(xb, W_lin.t())           # [Lb, Fi]
            if b_lin is not None:
                yb = yb + b_lin.unsqueeze(0)

            y[b, :Lb, :Fi] = yb
            feat_mask[b, :Lb, :Fi] = False

        # 保持原有 attn_mask 不变（只新增特征维掩码）
        return y, feat_mask, Fi_list
