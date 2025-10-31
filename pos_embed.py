import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# # --- 从 affine 抽取每轴线性映射 x(i)=a*i+b （支持旋转/切变，使用对角线元素） ---
# def _axis_linear_from_affine(affine: torch.Tensor):
#     """
#     从仿射变换矩阵提取每轴的线性映射参数。

#     对于包含旋转/剪切的矩阵，直接使用对角线元素作为缩放因子。
#     这是一个近似，但对于医学影像通常足够准确。
#     """
#     A = affine
#     R = A[:3, :3]
#     t = A[:3, 3]

#     # 直接使用对角线元素作为缩放因子（即使有旋转/剪切）
#     ax, ay, az = R[0,0].item(), R[1,1].item(), R[2,2].item()
#     bx, by, bz = t[0].item(), t[1].item(), t[2].item()
#     return (ax,bx), (ay,by), (az,bz)

# # --- 对“物理窗口”求一维闭式均值（各向异性自然支持） ---
# def _axis_patch_means_physical(length_vox: int, k_axis: int,
#                                a: float, b: float, rho_axis_mm: float):
#     """
#     返回每个 patch 在该轴上的世界坐标均值（长度 L_axis = length_vox//k_axis）
#     """
#     L_axis = length_vox // k_axis
#     # 该轴上 patch 的体素索引中心
#     ic = torch.arange(L_axis, dtype=torch.float32) * k_axis + (k_axis - 1) * 0.5  # [L_axis]
#     xc = a * ic + b  # 世界中心 [L_axis]

#     half = rho_axis_mm * 0.5
#     x_lo = xc - half
#     x_hi = xc + half

#     # 反到索引空间（a 可为负）
#     i_lo = (x_lo - b) / a
#     i_hi = (x_hi - b) / a
#     i_min = torch.minimum(i_lo, i_hi)
#     i_max = torch.maximum(i_lo, i_hi)

#     # 离散闭区间并裁到有效索引
#     i0 = torch.ceil(i_min).clamp_(0, length_vox - 1)
#     i1 = torch.floor(i_max).clamp_(0, length_vox - 1)

#     # 极端：窗口完全落在网格外 → 回退到最近体素中心
#     bad = i0 > i1
#     if bad.any():
#         ic_near = ((xc - b) / a).round().clamp_(0, length_vox - 1)
#         i0[bad] = ic_near[bad]
#         i1[bad] = ic_near[bad]

#     # 闭式均值： E[x] = a * (i0+i1)/2 + b
#     mean = a * (i0 + i1) * 0.5 + b  # [L_axis]
#     return mean

# # --- 生成每个空间 patch 的 (x,y,z)（严格物理窗口，支持各向异性与各轴rho不同） ---
# def stape_patch_world_coords_physical(X:int, Y:int, Z:int,
#                                       kx:int, ky:int, kz:int,
#                                       affine: torch.Tensor,
#                                       rho_mm=(12.0, 12.0, 12.0)):
#     """
#     输出: [Lx*Ly*Lz, 3]，顺序与 STAPE 的空间展平顺序一致（这里用 x→y→z 的行主）
#     说明: 对于 3×3×4 mm 体素，形参里不会写分辨率，分辨率信息体现在 affine 的 (a_x,a_y,a_z)
#     """
#     (ax,bx),(ay,by),(az,bz) = _axis_linear_from_affine(affine)
#     Lx, Ly, Lz = X//kx, Y//ky, Z//kz

#     mx = _axis_patch_means_physical(X, kx, ax, bx, rho_mm[0])  # [Lx]
#     my = _axis_patch_means_physical(Y, ky, ay, by, rho_mm[1])  # [Ly]
#     mz = _axis_patch_means_physical(Z, kz, az, bz, rho_mm[2])  # [Lz]

#     Xg = mx.view(Lx, 1, 1).expand(Lx, Ly, Lz)
#     Yg = my.view(1, Ly, 1).expand(Lx, Ly, Lz)
#     Zg = mz.view(1, 1, Lz).expand(Lx, Ly, Lz)
#     coords = torch.stack([Xg, Yg, Zg], dim=-1).reshape(Lx*Ly*Lz, 3)
#     return coords  # [N,3]

def stape_patch_world_coords_physical(
    X:int, Y:int, Z:int,
    kx:int, ky:int, kz:int,
    affine: torch.Tensor,
    rho_mm: Tuple[float, float, float], # 占位
    device=None, dtype=None
):
    """
    返回每个空间 patch 的世界坐标中心 [N,3]（mm），严格使用完整仿射 A,t。
    忽略 rho（因为这是中心点，不是窗口均值）。
    """
    if device is None: device = affine.device
    if dtype is None:  dtype = torch.float32

    A = affine[:3, :3].to(device=device, dtype=dtype)  # [3,3]
    t = affine[:3, 3].to(device=device, dtype=dtype)   # [3]

    Lx, Ly, Lz = X//kx, Y//ky, Z//kz
    icx = torch.arange(Lx, device=device, dtype=dtype)*kx + (kx-1)*0.5
    icy = torch.arange(Ly, device=device, dtype=dtype)*ky + (ky-1)*0.5
    icz = torch.arange(Lz, device=device, dtype=dtype)*kz + (kz-1)*0.5

    gx, gy, gz = torch.meshgrid(icx, icy, icz, indexing='ij')     # [Lx,Ly,Lz]
    idx = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)        # [N,3] 索引中心
    coords = idx @ A.T + t                                        # [N,3] 世界坐标中心（mm）
    return coords



class FixedSinCos3DPE(nn.Module):
    def __init__(self, embed_dim:int, num_freq:int=12,
                 space_scale:float=1.0, learnable_proj: bool=True):
        """
        num_freq: 每个维度的频率数；总特征维 = 3 * 2 * num_freq
        space_scale: 对 mm 的缩放，防止量纲差异
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_freq = num_freq
        freq = torch.exp(torch.linspace(0, math.log(10000.0), num_freq)) / 10000.0  # 频率按 log 均匀分布
        self.register_buffer('freq', freq)  # [num_freq]
        self.space_scale = space_scale
        in_dim = 3 * 2 * num_freq
        self.proj = nn.Linear(in_dim, embed_dim, bias=False) if learnable_proj else nn.Identity()

    def forward(self, xyz: torch.Tensor):
        """
        xyz: [B, L, 3]，单位 mm
        返回: [B, L, embed_dim]
        """
        B, L, _ = xyz.shape
        x = xyz[..., 0] * self.space_scale
        y = xyz[..., 1] * self.space_scale
        z = xyz[..., 2] * self.space_scale

        # 确保freq在正确的设备上
        freq = self.freq.to(device=xyz.device, dtype=xyz.dtype)

        def enc(u):
            u = u[..., None] * freq   # [B,L,num_freq]
            return torch.cat([torch.sin(u), torch.cos(u)], dim=-1)  # [B,L,2*num_freq]

        feats = torch.cat([enc(x), enc(y), enc(z)], dim=-1)  # [B,L, 3*2*num_freq]
        return self.proj(feats)  # [B,L,C]
