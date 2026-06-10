"""Mamba sequence blocks used by Flexibrain.

References:
- Mamba: https://github.com/state-spaces/mamba
- 3D Mamba MAE: https://github.com/ydchen0806/TokenUnify

This file keeps only the block factory pieces needed by the Flexibrain
Mamba-JEPA backbone, instead of vendoring the full upstream training project.
"""

from functools import partial
import inspect
import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from timm.models.layers import DropPath

from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mha import MHA
from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn


class Block(nn.Module):
    def __init__(self, dim, mixer_cls, mlp_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False, drop_path=0.0):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.norm = norm_cls(dim)
        self.mixer = mixer_cls(dim)
        try:
            self._mixer_kwset = set(inspect.signature(self.mixer.forward).parameters.keys())
        except Exception:
            self._mixer_kwset = set()
        if mlp_cls is not nn.Identity:
            self.norm2 = norm_cls(dim)
            self.mlp = mlp_cls(dim)
        else:
            self.mlp = None
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import failed"
            assert isinstance(self.norm, (nn.LayerNorm, RMSNorm))

    def forward(self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None, **mixer_kwargs):
        if not self.fused_add_norm:
            residual = (self.drop_path(hidden_states) + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            hidden_states, residual = layer_norm_fn(
                self.drop_path(hidden_states), self.norm.weight, self.norm.bias,
                residual=residual, prenorm=True, residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps, is_rms_norm=isinstance(self.norm, RMSNorm),
            )
        filtered_kwargs = {k: v for k, v in mixer_kwargs.items() if k in self._mixer_kwset}
        hidden_states = self.mixer(hidden_states, inference_params=inference_params, **filtered_kwargs)
        if self.mlp is not None:
            if not self.fused_add_norm:
                residual = self.drop_path(hidden_states) + residual
                residual = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)
            else:
                hidden_states, residual = layer_norm_fn(
                    self.drop_path(hidden_states), self.norm2.weight, self.norm2.bias,
                    residual=residual, prenorm=True, residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm2.eps, is_rms_norm=isinstance(self.norm2, RMSNorm),
                )
            hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block(d_model, ssm_cfg=None, attn_layer_idx=None, attn_cfg=None, norm_epsilon=1e-5, drop_path=0.0, rms_norm=False, residual_in_fp32=False, fused_add_norm=False, layer_idx=None, device=None, dtype=None, if_bimamba=False, bimamba_type="none", if_devide_out=False, init_layer_scale=None, mixer_type="mamba"):
    if if_bimamba and bimamba_type == "none":
        bimamba_type = "v1"
    if ssm_cfg is None:
        ssm_cfg = {}
    if attn_cfg is None:
        attn_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    if (attn_layer_idx is None) or (layer_idx not in attn_layer_idx):
        if mixer_type == "mamba":
            mixer_cls = partial(Mamba, layer_idx=layer_idx, init_layer_scale=init_layer_scale, bimamba_type=bimamba_type, if_devide_out=if_devide_out, **ssm_cfg, **factory_kwargs)
        elif mixer_type == "mamba2":
            mixer_cls = partial(Mamba2, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
        else:
            raise ValueError(f"Unknown mixer_type: {mixer_type}")
    else:
        mixer_cls = partial(MHA, layer_idx=layer_idx, **attn_cfg, **factory_kwargs)
    norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs)
    block = Block(d_model, mixer_cls, nn.Identity, norm_cls=norm_cls, drop_path=drop_path, fused_add_norm=fused_add_norm, residual_in_fp32=residual_in_fp32)
    block.layer_idx = layer_idx
    return block


def _init_weights(module, n_layer, initializer_range=0.02, rescale_prenorm_residual=True, n_residuals_per_layer=1):
    if isinstance(module, nn.Linear):
        if module.bias is not None and not getattr(module.bias, "_no_reinit", False):
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)
    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)
