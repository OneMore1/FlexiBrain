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
from flex_patch_utils.mamba_pack import pack_batch_fast as pack_batch, unpack_batch_inplace as unpack_batch

# 使用本地修改版本的mamba
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'mamba2'))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flex_patchembed_timetospace import STAPE4D_TimeToSpace
# from flex_patchembed import STAPE4D_TimeToSpace
from .MoE_models import MoE


class VolumeMambaJEPA(nn.Module):
    """ JEPA with VisionMamba backbone
    """
    def __init__(self,
                embed_dim=512,
                depth=24,
                predictor_depth=2,
                ssm_cfg=None, 
                encoder_attn_layer_idx=None,
                attn_cfg=None,
                drop_path_rate=0.1,
                norm_epsilon: float = 1e-5, 
                rms_norm: bool = False, 
                initializer_cfg=None,
                fused_add_norm=True,
                residual_in_fp32=True,
                device=None,
                dtype=None,
                bimamba_type="none",
                if_bimamba=False,
                mixer_type="mamba",
                if_devide_out=False,
                momentum: float = 0.996,
                norm_target: bool = True,

                **kwargs
                ):
        factory_kwargs = {"device": device, "dtype": dtype}
        # add factory_kwargs into kwargs
        kwargs.update(factory_kwargs)
        super().__init__()

        self.embed_dim = embed_dim
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.momentum = float(momentum)
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
        # 将patch_embed移到正确的设备
        if device is not None:
            self.patch_embed = self.patch_embed.to(device=device)

        # -------- 上下文编码器（可训练）--------
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
                    if_bimamba=if_bimamba,
                    drop_path=inter_dpr[i],
                    if_devide_out=if_devide_out,   # 若你已统一成 if_divide_out，这里也同步
                    mixer_type=mixer_type,
                    **factory_kwargs,
                )
            )
            block_idx += 1

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            embed_dim, eps=norm_epsilon, **factory_kwargs
        )

        # -------- 目标编码器（动量 EMA，同结构）--------
        self.target_blocks = copy.deepcopy(self.blocks) 
        self.target_norm = (nn.LayerNorm if not rms_norm else RMSNorm)(embed_dim, eps=norm_epsilon, **factory_kwargs)
        for p in self.target_blocks.parameters():
            p.requires_grad = False
        for p in self.target_norm.parameters():
            p.requires_grad = False


        # ------- MoE & ResolutionxTR embed-------
        self.moe_aux_coef = float(kwargs.get("moe_aux_coef", 0.1))
        self.moe = MoE(
            dim=embed_dim,
            hidden_dim=embed_dim * 4,
            num_indep=3,
            aux_loss_coef=self.moe_aux_coef,
            device=device,
            dtype=dtype,
        )

        # -------- 上下文侧的 mask token --------
        self.mask_token_ctx = nn.Parameter(torch.zeros(1, 1, embed_dim))
        torch.nn.init.normal_(self.mask_token_ctx, std=0.02)

        # -------- 预测头（在被遮挡位置回归目标表征）--------
        self.pred_depth = predictor_depth
        self.pred_dpr = [0.0 for _ in range(self.pred_depth)]
        self.predictor_blocks = nn.ModuleList()
        for i in range(self.pred_depth):
            self.predictor_blocks.append(
                create_block(
                    embed_dim,
                    ssm_cfg=ssm_cfg,
                    attn_layer_idx=None,   # 这里可留空或按需设置
                    attn_cfg=attn_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    bimamba_type=bimamba_type,
                    if_bimamba=if_bimamba,
                    drop_path=self.pred_dpr[i],
                    if_devide_out=if_devide_out,
                    mixer_type=mixer_type,
                    **factory_kwargs,
                )
            )

        self.predictor_norm = (nn.LayerNorm if not rms_norm else RMSNorm)(
            embed_dim, eps=norm_epsilon, **factory_kwargs)

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
        # 1) mask token
        torch.nn.init.normal_(self.mask_token_ctx, std=0.02)

        # 2) 预测器（用 block 堆或 MLP 的都做一次线性/LN 初始化）
        if hasattr(self, 'predictor_blocks'):
            self.predictor_blocks.apply(self._init_linear_ln)
        if hasattr(self, 'predictor_norm'):
            self._init_linear_ln(self.predictor_norm)

        # 3) 上下文编码器头尾（线性 + LN）
        self.blocks.apply(self._init_linear_ln)
        self._init_linear_ln(self.norm_f)

        # 4) patch-embed（STAPE4D）如果内部没有 reset_parameters，可做保守初始化：
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

        # 5) 目标编码器：不要随机初始化；直接对齐为上下文的权重（或在 forward 中做 EMA）
        # 如果是 deepcopy 创建的 target_blocks/target_norm，这里只需一次“硬拷贝”
        with torch.no_grad():
            for ps, pt in zip(self.blocks.parameters(), self.target_blocks.parameters()):
                pt.copy_(ps)
            if hasattr(self, 'target_norm'):
                for ps, pt in zip(self.norm_f.parameters(), self.target_norm.parameters()):
                    pt.copy_(ps)
    
    @torch.no_grad()
    def update_target_encoder(self, m: float = None):
        """EMA update of the target encoder"""
        m = float(m or self.momentum)
        for ps, pt in zip(self.blocks.parameters(), self.target_blocks.parameters()):
            pt.data.mul_(m).add_(ps.data, alpha=1 - m)
        for ps, pt in zip(self.norm_f.parameters(), self.target_norm.parameters()):
            pt.data.mul_(m).add_(ps.data, alpha=1 - m)
    
    
    # ---- 随机遮挡（复用你现有的思路；保留 ragged 长度）----
    def random_masking(self, x, attn_mask, lengths, mask_ratio):
        """
        x: [B, Lmax, D], attn_mask: [B, Lmax] (True=pad), lengths: list[int]
        Return:
        x_keep: [B, Lk_max, D]
        mask_full: [B, Lmax] (0=keep, 1=remove; pad 仍为1)
        ids_restore: [B, Lmax]
        attn_keep: [B, Lk_max] (True=pad)
        keep_lengths: list[int]
        ids_keep_pad: [B, Lk_max]  # <--- 新增: 每样本保留的原索引（按采样后的顺序）
        """
        N, Lmax, D = x.shape
        device = x.device

        x_keep_list, mask_list, ids_restore_list = [], [], []
        ids_keep_list, keep_lengths, Lk_max = [], [], 0

        for i in range(N):
            Li = lengths[i]
            if Li == 0:
                x_keep_list.append(torch.empty(0, D, device=device, dtype=x.dtype))
                mask_list.append(torch.ones(Lmax, device=device))
                ids_restore_list.append(torch.arange(Lmax, device=device))
                ids_keep_list.append(torch.empty(0, dtype=torch.long, device=device))
                keep_lengths.append(0)
                continue

            Lk = max(1, int(Li * (1 - mask_ratio)))
            keep_lengths.append(Lk)
            Lk_max = max(Lk_max, Lk)

            noise = torch.rand(Li, device=device)
            ids_shuffle = torch.argsort(noise)           # [Li]
            ids_restore_valid = torch.argsort(ids_shuffle)
            ids_keep = ids_shuffle[:Lk]                  # <--- 保留的原索引（相对 0..Li-1）

            x_keep_list.append(x[i, ids_keep])
            ids_keep_list.append(ids_keep)

            mask_i = torch.ones(Lmax, device=device)
            valid_mask = torch.ones(Li, device=device)
            valid_mask[:Lk] = 0
            mask_i[:Li] = torch.gather(valid_mask, 0, ids_restore_valid)
            mask_list.append(mask_i)

            ids_restore_full = torch.arange(Lmax, device=device)
            ids_restore_full[:Li] = ids_restore_valid
            ids_restore_list.append(ids_restore_full)

        # pad keep tensors
        x_keep = x.new_zeros((N, max(1, Lk_max), D))
        ids_keep_pad = torch.full((N, max(1, Lk_max)), -1, dtype=torch.long, device=device)
        attn_keep = torch.ones(N, max(1, Lk_max), dtype=torch.bool, device=device)

        for i, (xi, ik) in enumerate(zip(x_keep_list, ids_keep_list)):
            if xi.numel() > 0:
                Lk = xi.size(0)
                x_keep[i, :Lk] = xi
                ids_keep_pad[i, :Lk] = ik
                attn_keep[i, :Lk] = False

        mask_full = torch.stack(mask_list, dim=0)
        ids_restore = torch.stack(ids_restore_list, dim=0)

        return x_keep, mask_full, ids_restore, attn_keep, keep_lengths, ids_keep_pad

    def _build_context_visible(self, x_keep, attn_keep):
        """
        仅用于 context encoder：
        x_keep:   [B, Lk_max, D]  (拼成等长批)
        attn_keep:[B, Lk_max] (True=pad, False=valid)
        直接原样返回，交由 encoder 在可见序列上计算。
        """
        return x_keep, attn_keep
    
    def _build_target_masked(self, x_full, mask_full, lengths):
        """
        x_full: [B, Lmax, D]（这里用 patch_embed 的原始 token，对应原顺序）
        返回:
          x_tgt_pad:   [B, Lt_max, D]
          attn_tgt:    [B, Lt_max] (True=padding)
          tgt_lengths: list[int]
        """
        B, Lmax, D = x_full.shape
        device, dtype = x_full.device, x_full.dtype
        per_sample, tgt_lengths, Lt_max = [], [], 0
        for i in range(B):
            Li = lengths[i]
            if Li == 0:
                per_sample.append(x_full.new_empty((0, D)))
                tgt_lengths.append(0)
                continue
            sel = (mask_full[i, :Li] == 1) if mask_full.dtype != torch.bool else mask_full[i, :Li]
            xi = x_full[i, :Li][sel]
            per_sample.append(xi)
            tgt_lengths.append(xi.size(0))
            Lt_max = max(Lt_max, xi.size(0))

        x_tgt_pad = x_full.new_zeros((B, Lt_max, D))
        attn_tgt = torch.ones(B, Lt_max, dtype=torch.bool, device=device)
        for i, xi in enumerate(per_sample):
            if xi.numel() > 0:
                Lti = xi.size(0)
                x_tgt_pad[i, :Lti] = xi
                attn_tgt[i, :Lti] = False

        return x_tgt_pad, attn_tgt, tgt_lengths

    # ---- 通用 encoder 运行----
    def _run_blocks(self, x, attn_mask, blocks, norm_layer, inference_params=None, unpack_buffer=None):
        residual = None
        # x, seq_idx, idx_info = pack_batch(x, attn_mask)
        hidden_states = x
        for layer in blocks:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params,
                attn_mask=attn_mask
            )
            # hidden_states, residual = layer(
            #     hidden_states, residual, inference_params=inference_params,
            #     seq_idx=seq_idx
            # )
        # 只对 mamba blocks 应用最后的 norm（vit blocks 已经内部处理）
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
        # hidden_states = unpack_batch(hidden_states, idx_info, unpack_buffer)
        return hidden_states

    def forward(self, x, mask_ratio=0.6, meta=None, orig_Ts=None, affines=None, inference_params=None, return_moe_features=False):
        """
        返回: loss, pred_feat(被遮挡位), tgt_feat(被遮挡位), mask_full
        如果 return_moe_features=True, 返回: moe_features, attn_mask, lengths, meta
        """
        # flex patch-embed：返回 patch_embed 后的全长原位置
        x_full, attn_pad, lengths, _ = self.patch_embed(x, meta, orig_Ts, affines, return_grid_info=False, explain_mode=False)

        # random mask
        x_keep, mask_full, ids_restore, attn_keep, keep_lengths, ids_keep_pad = self.random_masking(x_full, attn_pad, lengths, mask_ratio)

        # build ctx visible
        x_ctx, attn_ctx = self._build_context_visible(x_keep, attn_keep)
        ctx_keep_out = self._run_blocks(x_ctx, attn_ctx,
            blocks=self.blocks,
            norm_layer=self.norm_f,
        ) # [B, Lk_max, D]

        device = x_full.device
        B, Lmax, D = x_full.shape
        ctx_keep_out, moe_aux, gates = self.moe(ctx_keep_out, attn_mask=attn_ctx, cond_vec=None, return_gates=True)

        # 如果只需要MoE特征，直接返回
        if return_moe_features:
            return ctx_keep_out, attn_keep, keep_lengths, meta

        # 将编码后的可见特征“散射”回全长原位置；被遮挡位置放入 mask token
        ctx_full = x_full.new_zeros((B, Lmax, D))
        attn_full = torch.ones(B, Lmax, dtype=torch.bool, device=device)
        for i in range(B):
            Li = lengths[i]
            if Li == 0:
                continue

            # 1) 写回可见位置
            Lk_i = int((~attn_keep[i]).sum().item())
            if Lk_i > 0:
                keep_idx = ids_keep_pad[i, :Lk_i].long()          # [Lk_i]
                ctx_full[i, keep_idx] = ctx_keep_out[i, :Lk_i]    # [Lk_i, D]

            # 2) 用 mask token 填充被遮挡位置（一次 index_copy_，不做二次切片）
            masked_sel = (mask_full[i, :Li] == 1) if mask_full.dtype != torch.bool else mask_full[i, :Li]
            if masked_sel.any():
                idx = torch.nonzero(masked_sel, as_tuple=False).squeeze(1)   # [n_mask]
                token_rows = self.mask_token_ctx[0, 0].expand(idx.numel(), D)  # [n_mask, D]
                ctx_full[i, :Li].index_copy_(0, idx, token_rows)

            # 3) 有效区间标记为可见（非 padding）
            attn_full[i, :Li] = False

        # (e) predictor：在“可见特征 + mask token”的全长序列上做信息传播与预测
        # 轻量 predictor：两层 block + norm（可复用 create_block）
        pred_full = self._run_blocks(
            ctx_full, attn_full, 
            blocks=self.predictor_blocks,
            norm_layer=self.predictor_norm,
        ) # [B, Lmax, D]

        # (f) build target masked
        x_tgt_pad, attn_tgt, tgt_lengths = self._build_target_masked(x_full, mask_full, lengths)
        with torch.no_grad():
            self.update_target_encoder(self.momentum)
            tgt_feat = self._run_blocks(
                x_tgt_pad, attn_tgt, 
                blocks=self.target_blocks,
                norm_layer=self.target_norm,
            ) # [B, Lt_max, D]

        #  从 pred_full 里抽取被遮挡位置输出，pad 成与 x_tgt_pad 对齐
        Lt_max = x_tgt_pad.size(1)
        pred_masked = pred_full.new_zeros((B, Lt_max, D))
        attn_pred = torch.ones(B, Lt_max, dtype=torch.bool, device=device)
        for i in range(B):
            Li = lengths[i]
            if Li == 0:
                continue
            sel = (mask_full[i, :Li] == 1) if mask_full.dtype != torch.bool else mask_full[i, :Li]
            vi = pred_full[i, :Li][sel]
            if vi.numel() > 0:
                Lti = vi.size(0)
                pred_masked[i, :Lti] = vi
                attn_pred[i, :Lti] = False

        # (g) 计算 loss
        if self.norm_target:
            # def norm_sg(x, eps=1e-6):
            #     # 归一化，但不让范数参与反传；只学“方向”
            #     return x / (x.norm(dim=-1, keepdim=True).clamp_min(eps).detach())
            # tgt_feat   = norm_sg(tgt_feat)
            # pred_masked = norm_sg(pred_masked)
            # 对target特征进行L2归一化
            tgt_norm = torch.linalg.norm(tgt_feat, dim=-1, keepdim=True).clamp_min(1e-6)
            tgt_feat = tgt_feat / tgt_norm
            # 对pred特征也进行L2归一化，保持scale一致
            pred_norm = torch.linalg.norm(pred_masked, dim=-1, keepdim=True).clamp_min(1e-6)
            pred_masked = pred_masked / pred_norm

        # maybe_visualize_batch(
        #     context_features_proj=ctx_full, 
        #     target_features_proj=tgt_feat, 
        #     pred_target=pred_masked, 
        #     out_root='/mnt/dataset4/yewh/temp-free-model/visual_col_mamba/mamba-moe-colorbar', 
        #     max_samples=2,
        #     share_vrange_raw=True,
        # )

        valid = ~attn_pred
        denom = valid.sum().clamp_min(1)
        loss = (pred_masked[valid] - tgt_feat[valid]).pow(2).sum() / denom

        loss = loss


        return loss, pred_masked, tgt_feat, mask_full


