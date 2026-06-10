import math
from typing import Tuple

import torch
import torch.nn as nn


def stape_patch_world_coords_physical(
    X:int, Y:int, Z:int,
    kx:int, ky:int, kz:int,
    affine: torch.Tensor,
    rho_mm: Tuple[float, float, float], 
    device=None, dtype=None
):
    if device is None: device = affine.device
    if dtype is None:  dtype = torch.float32

    A = affine[:3, :3].to(device=device, dtype=dtype)  # [3,3]
    t = affine[:3, 3].to(device=device, dtype=dtype)   # [3]

    Lx, Ly, Lz = X//kx, Y//ky, Z//kz
    icx = torch.arange(Lx, device=device, dtype=dtype)*kx + (kx-1)*0.5
    icy = torch.arange(Ly, device=device, dtype=dtype)*ky + (ky-1)*0.5
    icz = torch.arange(Lz, device=device, dtype=dtype)*kz + (kz-1)*0.5

    gx, gy, gz = torch.meshgrid(icx, icy, icz, indexing='ij')     # [Lx,Ly,Lz]
    idx = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)        # [N,3]
    coords = idx @ A.T + t                                        # [N,3]
    return coords



class FixedSinCos3DPE(nn.Module):
    def __init__(self, embed_dim:int, num_freq:int=12,
                 space_scale:float=1.0, learnable_proj: bool=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_freq = num_freq
        freq = torch.exp(torch.linspace(0, math.log(10000.0), num_freq)) / 10000.0  
        self.register_buffer('freq', freq)  # [num_freq]
        self.space_scale = space_scale
        in_dim = 3 * 2 * num_freq
        self.proj = nn.Linear(in_dim, embed_dim, bias=False) if learnable_proj else nn.Identity()

    def forward(self, xyz: torch.Tensor):
        """
        xyz: [B, L, 3]
        return: [B, L, embed_dim]
        """
        B, L, _ = xyz.shape
        x = xyz[..., 0] * self.space_scale
        y = xyz[..., 1] * self.space_scale
        z = xyz[..., 2] * self.space_scale

        freq = self.freq.to(device=xyz.device, dtype=xyz.dtype)

        def enc(u):
            u = u[..., None] * freq   # [B,L,num_freq]
            return torch.cat([torch.sin(u), torch.cos(u)], dim=-1)  # [B,L,2*num_freq]

        feats = torch.cat([enc(x), enc(y), enc(z)], dim=-1)  # [B,L, 3*2*num_freq]
        return self.proj(feats)  # [B,L,C]
