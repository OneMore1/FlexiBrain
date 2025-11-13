# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import copy
import sys
import os
from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import DropPath

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Brain-Harmony'))

from .models_vim import create_block, _init_weights
from libs.flex_transformer import Block

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from flex_patch_utils.visualize import maybe_visualize_batch

# 使用本地修改版本的mamba
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'mamba2'))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flex_patchembed import STAPE4D_TimeToSpace
from cross_attention import CrossAttention, PIAdaptiveLinear4D


class WeightedFeatureMaps(nn.Module):
    def __init__(self, k, embed_dim, *, norm_layer=nn.LayerNorm, decoder_depth):
        super(WeightedFeatureMaps, self).__init__()
        self.linear = nn.Linear(k, decoder_depth, bias=False)
        
        std_dev = 1. / math.sqrt(k)
        nn.init.normal_(self.linear.weight, mean=0., std=std_dev)

    def forward(self, feature_maps):
        # Ensure the input is a list
        assert isinstance(feature_maps, list), "Input should be a list of feature maps"
        # Ensure the list has the same length as the number of weights
        assert len(feature_maps) == (self.linear.weight.shape[1]), "Number of feature maps and weights should match"
        stacked_feature_maps = torch.stack(feature_maps, dim=-1)  # shape: (B, L, C, k)
        # compute a weighted average of the feature maps
        # decoder_depth is denoted as j
        output = self.linear(stacked_feature_maps)
        return output

class VolumeMambaJEPA(nn.Module):
    """ JEPA with VisionMamba backbone
    """
    def __init__(self,
                embed_dim= 512,
                depth=24,
                ssm_cfg=None, 
                encoder_attn_layer_idx=None,
                attn_cfg=None,
                drop_path_rate=0.1,
                norm_epsilon: float = 1e-5, 
                rms_norm: bool = False, 
                initializer_cfg=None,
                fused_add_norm=True,
                residual_in_fp32=False,
                device=None,
                dtype=None,
                bimamba_type="v2",
                mixer_type="mamba",
                if_devide_out=True,
                norm_target: bool = True,

                decoder_depth: int = 8,
                decoder_dim: int = 256,

                use_fm: list = [-1],
                weight_fm: bool = True,
                use_input: bool = False,

                **kwargs
                ):
        factory_kwargs = {"device": device, "dtype": dtype}
        # add factory_kwargs into kwargs
        kwargs.update(factory_kwargs)
        super().__init__()

        self.embed_dim = embed_dim
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.norm_target = bool(norm_target)   

        self.patch_embed = STAPE4D_TimeToSpace(
            d_mid=16,
            d_out=embed_dim,
            kt_base=6,
            kx_base=6,
            ky_base=6,
            kz_base=6,
            tau_seconds=6.0,
            rho_mm=(12.0, 12.0, 12.0),
        )
        if device is not None:
            self.patch_embed = self.patch_embed.to(device=device)

        # -------- 编码器（可训练）--------
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.blocks = nn.ModuleList()
        block_idx = 0
        for i in range(depth):
            self.blocks.append(
                create_block(
                    embed_dim,
                    ssm_cfg=ssm_cfg,
                    attn_layer_idx=encoder_attn_layer_idx,
                    attn_cfg=attn_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=block_idx,
                    bimamba_type=bimamba_type,
                    drop_path=inter_dpr[i],
                    if_devide_out=if_devide_out,   # 若你已统一成 if_divide_out，这里也同步
                    mixer_type=mixer_type,
                    **factory_kwargs,
                )
            )
            block_idx += 1

        self.norm_layer = (nn.LayerNorm if not rms_norm else RMSNorm)(
            embed_dim, eps=norm_epsilon, **factory_kwargs
        )
        
        # decoder
        self.decoder_depth = decoder_depth
        self.decoder_dim = decoder_dim
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.decoder_blocks = nn.ModuleList([
            CrossAttention(embed_dim, decoder_dim, decoder_heads, mlp_ratio, qkv_bias=True, qk_scale=None)
            for i in range(decoder_depth)])
        self.decoder_pred = PIAdaptiveLinear4D(decoder_dim, 1, 
                                                kH0=6, kW0=6, kD0=6, kT0=6)

        # -------- mask token --------
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        torch.nn.init.normal_(self.mask_token, std=0.02)

        # -------- 加权特征图 --------
        self.weight_fm = weight_fm
        self.use_input = use_input  # 是否将输入特征也作为加权对象
        if len(use_fm) == 1 and use_fm[0] == -1:
            self.use_fm = list(range(depth))
        else:
            self.use_fm = [i if i >= 0 else depth + i for i in use_fm]
        if self.weight_fm:
            dec_norms = []
            for i in range(decoder_depth):
                norm_layer_i = nn.LayerNorm(embed_dim)
                dec_norms.append(norm_layer_i)
            self.dec_norms = nn.ModuleList(dec_norms)

            # feature weighting
            self.wfm = WeightedFeatureMaps(len(self.use_fm) + (1 if self.use_input else 0), embed_dim, norm_layer=nn.LayerNorm, decoder_depth=decoder_depth)


        # ---- 初始化 ----
        self.apply(self._init_linear_ln)
        self.initialize_jepa_weights()
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def _init_linear_ln(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, RMSNorm)) and hasattr(m, 'weight'):
            nn.init.constant_(m.weight, 1.0)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def initialize_jepa_weights(self):

        #  mask token
        torch.nn.init.normal_(self.mask_token, std=0.02)

        #  上下文编码器头尾（线性 + LN）
        self.blocks.apply(self._init_linear_ln)
        self._init_linear_ln(self.norm_layer)

        #  patch-embed（STAPE4D）如果内部没有 reset_parameters，可做保守初始化：
        if hasattr(self.patch_embed, 'reset_parameters'):
            self.patch_embed.reset_parameters()
        else:
            for m in self.patch_embed.modules():
                if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                    torch.nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    torch.nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
    
    # ---- 随机遮挡（复用你现有的思路；保留 ragged 长度）----
    def random_masking(self, x, attn_mask, lengths, mask_ratio, keep_mask_ratio):
        """
        x: [B, Lmax, D]
        attn_mask: [B, Lmax] (True=pad)   # 这里保持原始接口，不更改含义
        lengths: list[int]                # 每样本有效长度 (不含 pad)
        mask_ratio: float or tensor       # 基础的 mask 比例
        keep_mask_ratio: float or tensor  # 参考代码中的 kept_mask_ratio，
                                        # 表示最终实际的 mask 比例（0=全保留, 1=全移除）

        Return (与原函数完全一致):
            x_keep: [B, Lk_max, D]
            mask_full: [B, Lmax] (0=keep, 1=remove; pad 仍为1)
            ids_restore: [B, Lmax]
            attn_keep: [B, Lk_max] (True=pad)
            keep_lengths: list[int]
            ids_keep_pad: [B, Lk_max]  # 每样本保留的原索引（按采样后的顺序）
        """

        B, Lmax, D = x.shape
        device = x.device

        # 小工具：拿到按样本的标量
        def _get_ratio(val, i):
            if isinstance(val, (float, int)):
                return float(val)
            if torch.is_tensor(val):
                if val.ndim == 0:
                    return float(val.item())
                return float(val[i].item())
            raise TypeError("mask_ratio / keep_mask_ratio 必须为 float/int 或 tensor")

        x_keep_list, mask_list, ids_restore_list = [], [], []
        ids_keep_list, keep_lengths, Lk_max = [], [], 0

        for i in range(B):
            Li = int(lengths[i])  # 有效长度（不含 pad）
            if Li <= 0:
                x_keep_list.append(torch.empty(0, D, device=device, dtype=x.dtype))
                mask_list.append(torch.ones(Lmax, device=device))
                ids_restore_list.append(torch.arange(Lmax, device=device))
                ids_keep_list.append(torch.empty(0, dtype=torch.long, device=device))
                keep_lengths.append(0)
                continue

            # 读取并夹紧两种比例到 [0, 1]
            mr_i  = max(0.0, min(1.0, _get_ratio(mask_ratio, i)))
            kmr_i = max(0.0, min(1.0, _get_ratio(keep_mask_ratio, i)))

            # 参考你给的第二段代码：
            # len_keep = int(L*(1 - mask_ratio))
            # len_masked = int(L*(mask_ratio - kept_mask_ratio))
            # 最终保留数 = len_keep + len_masked = int(L*(1 - kept_mask_ratio))
            # 注意：若 kept_mask_ratio > mask_ratio，则 len_masked 为负数，最终保留数会少于 len_keep —— 与参考代码一致。
            Lk_target = int(Li * (1.0 - kmr_i))

            # 至少保留 1 个（与原实现一致），且不超过有效长度
            Lk = max(1, min(Li, Lk_target))
            keep_lengths.append(Lk)
            Lk_max = max(Lk_max, Lk)

            # 只在有效区间内采样并洗牌
            noise = torch.rand(Li, device=device)
            ids_shuffle = torch.argsort(noise, dim=0)      # 升序：前面的更“保留”
            ids_restore_valid = torch.argsort(ids_shuffle) # 恢复到原顺序的索引
            ids_keep = ids_shuffle[:Lk]                    # 按采样后的顺序保留的“原索引”（0..Li-1）

            # 收集保留的特征
            x_keep_list.append(x[i, ids_keep])
            ids_keep_list.append(ids_keep)

            # 生成 mask（0=keep, 1=remove；pad 仍为 1）
            mask_i = torch.ones(Lmax, device=device)
            valid_mask = torch.ones(Li, device=device)
            valid_mask[:Lk] = 0
            mask_i[:Li] = torch.gather(valid_mask, 0, ids_restore_valid)
            mask_list.append(mask_i)

            # ids_restore：把有效区的恢复索引放回到 Lmax 中
            ids_restore_full = torch.arange(Lmax, device=device)
            ids_restore_full[:Li] = ids_restore_valid
            ids_restore_list.append(ids_restore_full)

        # 统一 pad 到批次的最大保留长度
        Lk_pad = max(1, Lk_max)
        x_keep = x.new_zeros((B, Lk_pad, D))
        ids_keep_pad = torch.full((B, Lk_pad), -1, dtype=torch.long, device=device)
        attn_keep = torch.ones(B, Lk_pad, dtype=torch.bool, device=device)

        for i, (xi, ik) in enumerate(zip(x_keep_list, ids_keep_list)):
            if xi.numel() > 0:
                Lk_i = xi.size(0)
                x_keep[i, :Lk_i] = xi
                ids_keep_pad[i, :Lk_i] = ik
                attn_keep[i, :Lk_i] = False  # 非 pad

        mask_full = torch.stack(mask_list, dim=0)
        ids_restore = torch.stack(ids_restore_list, dim=0)

        return x_keep, mask_full, ids_restore, attn_keep, keep_lengths, ids_keep_pad
    
    def patchify(
            self,
            imgs: torch.Tensor,                          # [B, C, H, W, D, T]
            patch_sizes: "list[tuple[int,int,int,int]]", # 每样本 (pH,pW,pD,pT)
            T_trues: "list[int|None]" = None,            # 可选：每样本真实 T（若 None 自动检测）
            crop_to_fit: bool = True                     # True: 裁剪到可整除；False: 断言可整除
        ):
            """
            返回：
            tokens_out : [B, L_max, F_max]   （F_max = max_i C*pH_i*pW_i*pD_i*pT_i）
            attn_mask  : [B, L_max]          True=pad（序列维）
            lengths    : list[int]           每样本 token 数 L_i = h*w*d*t
            feat_dims  : list[int]           每样本向量维 F_i = C*pH*pW*pD*pT
            feat_mask  : [B, L_max, F_max]   True=pad（特征维；可选使用）
            说明：
            - patch 顺序与你给的 4D patchify 完全一致；
            - 对每个样本独立 patchify，再在 batch 维做双向 pad；
            - 如果某样本没有有效帧（T_true==0），该样本 L_i=0。
            """
            assert imgs.dim() == 6, "imgs 需为 [B, C, H, W, D, T]"
            B, C, H, W, Dz, T = imgs.shape
            device, dtype = imgs.device, imgs.dtype
            assert len(patch_sizes) == B, "patch_sizes 长度需等于 B"
            if T_trues is not None:
                assert len(T_trues) == B, "T_trues 长度需等于 B"

            per_tokens = []
            lengths, feat_dims = [], []

            for b in range(B):
                pH, pW, pD, pT = map(int, patch_sizes[b])
                x_b = imgs[b]  # [C, H, W, D, T]

                # ---- 处理有效时间长度 ----
                if T_trues is not None and T_trues[b] is not None:
                    T_true = int(T_trues[b])
                else:
                    T_true = self._detect_true_T_5d(x_b)

                if T_true == 0:
                    per_tokens.append(imgs.new_zeros((0, 0), dtype=dtype))
                    lengths.append(0)
                    feat_dims.append(C * pH * pW * pD * pT)  # 记录理论F_i；即便L=0
                    continue

                # 有效 T 取能被 pT 整除的部分
                if crop_to_fit:
                    T_eff = (T_true // pT) * pT
                    if T_eff == 0:
                        per_tokens.append(imgs.new_zeros((0, 0), dtype=dtype))
                        lengths.append(0)
                        feat_dims.append(C * pH * pW * pD * pT)
                        continue
                    if T_eff < T:
                        x_b = x_b[..., :T_eff].contiguous()
                else:
                    assert T_true == T and (T % pT == 0), \
                        f"样本{b}: T={T} 与 pT={pT} 不整除，且 crop_to_fit=False"
                    T_eff = T

                # ---- 空间整除性检查 ----
                assert H % pH == 0 and W % pW == 0 and Dz % pD == 0, \
                    f"样本{b}: (H,W,D)=({H},{W},{Dz}) 不能被 (pH,pW,pD)=({pH},{pW},{pD}) 整除"

                h, w, d, t = H // pH, W // pW, Dz // pD, T_eff // pT
                F_i = C * pH * pW * pD * pT
                feat_dims.append(F_i)

                # ---- 按与你原始 4D patchify 一致的顺序展开 ----
                # [C, H, W, D, T_eff] -> [C, h, pH, w, pW, d, pD, t, pT]
                xb = x_b.reshape(C, h, pH, w, pW, d, pD, t, pT)
                # -> [h, w, d, t, C, pH, pW, pD, pT]
                xb = xb.permute(1, 3, 5, 7, 0, 2, 4, 6, 8).contiguous()
                # -> [L_i, F_i]
                xb = xb.reshape(h * w * d * t, F_i)

                per_tokens.append(xb)
                lengths.append(xb.size(0))

            # ---- 双向 pad：序列维 & 特征维 ----
            L_max = max(lengths) if lengths else 0
            F_max = max(feat_dims) if feat_dims else 0

            # 空 batch 的兜底
            if L_max == 0 or F_max == 0:
                tokens_out = imgs.new_zeros((B, 0, 0), dtype=dtype)
                attn_mask = torch.ones((B, 0), dtype=torch.bool, device=device)
                feat_mask = torch.ones((B, 0, 0), dtype=torch.bool, device=device)
                return tokens_out, attn_mask, lengths, feat_dims, feat_mask

            tokens_out = imgs.new_zeros((B, L_max, F_max), dtype=dtype)
            attn_mask  = torch.ones((B, L_max), dtype=torch.bool, device=device)      # 序列 pad
            feat_mask  = torch.ones((B, L_max, F_max), dtype=torch.bool, device=device)  # 特征 pad

            for b, xb in enumerate(per_tokens):
                Li = xb.size(0)
                Fi = feat_dims[b]
                if Li > 0:
                    tokens_out[b, :Li, :Fi] = xb
                    attn_mask[b, :Li] = False
                    feat_mask[b, :Li, :Fi] = False  # 有效特征维

            return tokens_out, attn_mask, lengths, feat_dims, feat_mask

    # ---- 通用 encoder 运行----
    def _run_blocks(self, x, attn_mask, blocks, norm_layer, inference_params=None):
        residual = None
        hidden_states = x
        x_feats = []
        if self.use_input:
            x_feats.append(hidden_states)
        for layer in blocks:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params,
                attn_mask=attn_mask
            )
            if self.weight_fm and layer in self.use_fm:
                x_feats.append(hidden_states)

        if self.weight_fm:
            return x_feats
        else:
            if not self.fused_add_norm:
                if residual is None:
                    residual = hidden_states
                else:
                    residual = residual + self.drop_path(hidden_states)
                hidden_states = norm_layer(residual.to(dtype=norm_layer.weight.dtype))
            else:
                fused_add_norm_fn = rms_norm_fn if isinstance(norm_layer, RMSNorm) else layer_norm_fn
                hidden_states = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    norm_layer.weight,
                    norm_layer.bias,
                    residual=residual,
                    prenorm=False,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=norm_layer.eps,
                )
            return hidden_states

    def mask_tokens_grid(self, mask, ids_restore, pos_encoding, attn_ctx):
        """
        Args:
            mask:        [B, Lmax]  (0=keep, 1=remove; pad 仍为1，但会被排除)
            ids_restore: [B, Lmax]  每样本的“原->采样序”排名（来自 random_masking）
            pos_encoding:[B, Lmax, C] 与最终 tokens 对齐的 PE
            attn_ctx:    [B, Lmax]  True=padding（上下文级别的 pad）

        Returns:
            x_masked:    [B, Lm_max, C]  仅包含“被遮蔽”的位置编码 + mask_token，按批次 pad
            attn_mask:   [B, Lm_max]     True=pad（与 x_masked 对齐）
        """

        B, Lmax = mask.shape
        assert pos_encoding.dim() == 3 and pos_encoding.size(0) == B and pos_encoding.size(1) == Lmax
        assert ids_restore.shape[0] == B and ids_restore.shape[1] == Lmax
        assert attn_ctx.shape[0] == B and attn_ctx.shape[1] == Lmax

        device = pos_encoding.device
        dtype  = pos_encoding.dtype
        C      = pos_encoding.size(-1)

        # 规范化/广播 mask_token 到 (1, C)
        mt = self.mask_token
        if mt.dim() == 1:                  # (C,)
            mt = mt.view(1, C)
        elif mt.dim() == 2:                # (1, C) or (B, C) 都可广播
            pass
        elif mt.dim() == 3:                # (1,1,C) -> (1,C)
            assert mt.size(-1) == C
            mt = mt.view(1, C)
        else:
            raise ValueError("mask_token 的形状不被支持，期望 (C,), (1,C) 或 (1,1,C)")
        mt = mt.to(device=device, dtype=dtype)

        per_sample_x = []
        per_sample_len = []

        for b in range(B):
            # 有效区长度（排除 pad）
            Li = int((~attn_ctx[b]).sum().item())
            if Li <= 0:
                per_sample_x.append(pos_encoding.new_zeros((0, C)))
                per_sample_len.append(0)
                continue

            # 只在有效区内，选出 mask==1 的原索引
            # 注意：mask 的 pad 位虽为 1，但被 Li 截断后自然排除了
            valid_remove = (mask[b, :Li] == 1)
            if valid_remove.any():
                idx = torch.nonzero(valid_remove, as_tuple=False).squeeze(-1)  # 原序的被遮蔽位置

                # 用 ids_restore 的“采样排名”做稳定排序（与 x_keep / 复原过程一致）
                order_rank = ids_restore[b, :Li][idx]           # 每个被遮蔽位置在采样序中的排名
                order = torch.argsort(order_rank, dim=0)        # 升序
                idx_sorted = idx[order]

                # 取该顺序下的位置编码，并加上 mask_token
                pe = pos_encoding[b, idx_sorted, :]             # [Lm_i, C]
                x_b = pe + mt                                   # 广播到 [Lm_i, C]
            else:
                x_b = pos_encoding.new_zeros((0, C))

            per_sample_x.append(x_b)
            per_sample_len.append(x_b.size(0))

        # 批次 pad
        Lm_max = max(1, max(per_sample_len) if len(per_sample_len) > 0 else 0)
        x_out = pos_encoding.new_zeros((B, Lm_max, C))
        attn_out = torch.ones((B, Lm_max), dtype=torch.bool, device=device)

        for b, xb in enumerate(per_sample_x):
            Lm = xb.size(0)
            if Lm > 0:
                x_out[b, :Lm] = xb
                attn_out[b, :Lm] = False

        return x_out, attn_out
    
    def forward(self, x, mask_ratio=0.6, meta=None, orig_Ts=None, affines=None, inference_params=None):
        """
        返回: loss, pred_feat(被遮挡位), tgt_feat(被遮挡位), mask_full
        """
        # flex patch-embed：返回 patch_embed 后的全长原位置
        x_full, attn_pad, lengths, pos_encoding, patch_sizes = self.patch_embed(x, meta, orig_Ts, affines, return_grid_info=False)

        # random mask
        x_keep, mask_full, ids_restore, attn_keep, _, ids_keep_pad = self.random_masking(x_full, attn_pad, lengths, mask_ratio)

        ctx_keep_out = self._run_blocks(x_keep, attn_keep, 
            blocks=self.blocks,     
            norm_layer=self.norm_layer,
        ) # [B, Lk_max, D]

        device = x_full.device
        B, Lmax, D = x_full.shape

        # 集结mask token
        x_mask, attn_out = self.mask_tokens_grid(mask_full, ids_restore, pos_encoding, attn_pad)

        if self.weight_fm:
            y = self.wfm(ctx_keep_out)

        for i, blk in enumerate(self.decoder_blocks):
            if self.weight_fm:
                x_mask = blk(x_mask, self.dec_norms[i](y[..., i]))
            else:
                x_mask = blk(x_mask, y)

        x_mask = self.decoder_norm(x_mask)
        x_pred, feat_mask, Fi_list = self.decoder_pred(x_mask, attn_out, patch_sizes) 

        target, target_mask, target_lengths, target_feat_dims, target_feat_mask = self.patchify(x, patch_sizes, T_trues=orig_Ts, crop_to_fit=False)

        # 计算 loss
        

        return loss, pred_masked, tgt_feat, mask_full


