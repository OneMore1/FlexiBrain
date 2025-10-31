# stape_time_to_space.py
import math
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pos_embed import FixedSinCos3DPE, stape_patch_world_coords_physical
from flex_patch_utils import pi_resize_weight_1d, pi_resize_weight_3d


# # ===== PI-resize 函数 =====
# def _linear_resize_matrix_1d(from_len: int, to_len: int, device=None, dtype=torch.float32):
#     device = device or torch.device("cpu")
#     R = torch.zeros((to_len, from_len), device=device, dtype=dtype)
#     if from_len == 1:
#         R[:, 0] = 1.0
#         return R
#     for i in range(to_len):
#         pos = i * (from_len - 1) / (to_len - 1) if to_len > 1 else 0.0
#         j0 = int(pos)
#         j1 = min(j0 + 1, from_len - 1)
#         w1 = float(pos - j0)
#         R[i, j0] = 1.0 - w1
#         R[i, j1] += w1
#     return R

# def pi_resize_weight_1d(w_star: torch.Tensor, k_new: int) -> torch.Tensor:
#     Dm, Cin, Kb = w_star.shape
#     if k_new == Kb:
#         return w_star
#     R = _linear_resize_matrix_1d(from_len=k_new, to_len=Kb, device=w_star.device, dtype=w_star.dtype)
#     R_pinv = torch.linalg.pinv(R)          # [k_new, Kb]
#     W = w_star.reshape(-1, Kb)             # [(Dm*Cin), Kb]
#     Wt = (R_pinv @ W.T).T                  # [(Dm*Cin), k_new]
#     return Wt.reshape(Dm, Cin, k_new)

# def pi_resize_weight_3d(w_star: torch.Tensor, kx: int, ky: int, kz: int) -> torch.Tensor:
#     Do, Dm, Kx0, Ky0, Kz0 = w_star.shape
#     if (kx, ky, kz) == (Kx0, Ky0, Kz0):
#         return w_star
#     dev, dt = w_star.device, w_star.dtype

#     # x
#     Rx = _linear_resize_matrix_1d(from_len=kx, to_len=Kx0, device=dev, dtype=dt)
#     Rx_p = torch.linalg.pinv(Rx)
#     W = w_star.permute(2, 0, 1, 3, 4).reshape(Kx0, -1)
#     W = (Rx_p @ W).reshape(kx, Do, Dm, Ky0, Kz0).permute(1, 2, 0, 3, 4)

#     # y
#     Ry = _linear_resize_matrix_1d(from_len=ky, to_len=Ky0, device=dev, dtype=dt)
#     Ry_p = torch.linalg.pinv(Ry)
#     W = W.permute(3, 0, 1, 2, 4).reshape(Ky0, -1)
#     W = (Ry_p @ W).reshape(ky, Do, Dm, kx, Kz0).permute(1, 2, 3, 0, 4)

#     # z
#     Rz = _linear_resize_matrix_1d(from_len=kz, to_len=Kz0, device=dev, dtype=dt)
#     Rz_p = torch.linalg.pinv(Rz)
#     W = W.permute(4, 0, 1, 2, 3).reshape(Kz0, -1)
#     W = (Rz_p @ W).reshape(kz, Do, Dm, kx, ky).permute(1, 2, 3, 4, 0)
#     return W


# ============================================================
#  时间先 STAPE（1D），把 T' 融到通道，再做空间 STAPE（3D）
# ============================================================
class STAPE4D_TimeToSpace(nn.Module):
    """
    输入:
        x: (B, 96, 96, 96, T_max) —— 其他样本在末尾用 0 pad
        meta: {subject_idx: {"voxel": (vx,vy,vz) mm, "tr": float 秒}}
        orig_Ts: (可选) 长度 B 的原始 T（若缺省则自动从尾部 0 检测），或列表
        affines:  长度 B 的仿射变换矩阵，或列表
    输出:
        tokens:   (B, L_max, D_out)
        attn_mask:(B, L_max)  True=padding
        lengths:  List[int] 每样本有效 token 数
    过程:
        - 按数据集分组；
        - 组内：先做时间 STAPE（Conv1d, stride=kt），得 T'；把 [Dm, T'] 折叠成通道 Dm*T'；
        - 做空间 STAPE（Conv3d, stride=(kx,ky,kz)），仅保留非零空间块，得到纯空间 tokens；
        - batch 内按最长 L 做 pad + mask。
    """
    def __init__(self,
                 d_mid: int = 128,     # 空间 STAPE 的中间通道
                 d_out: int = 256,
                 kt_base: int = 6,
                 kx_base: int = 6,
                 ky_base: int = 6,
                 kz_base: int = 6,
                 tau_seconds: float = 6.0,
                 rho_mm: Tuple[float, float, float] = (12., 12., 12.),
                 ):
        super().__init__()
        self.Dm = d_mid
        self.Do = d_out
        self.kt0, self.kx0, self.ky0, self.kz0 = kt_base, kx_base, ky_base, kz_base
        self.tau = float(tau_seconds)
        self.rho = tuple(float(r) for r in rho_mm)

        # --- 时间 STAPE 的基准核: [Dm, 1, kt0] ---
        self.w_t_first_base = nn.Parameter(
            torch.randn(d_mid, 1, kt_base) * (1.0 / (1 * kt_base)) ** 0.5
        )
        self.b_t_first = nn.Parameter(torch.zeros(d_mid))

        # --- 空间 STAPE 的基准核: [Do, Dm, kx0, ky0, kz0] ---
        #  in_channels=Dm（时间折叠前的中间通道）
        self.w_xyz_after_base = nn.Parameter(
            torch.randn(d_out, d_mid, kx_base, ky_base, kz_base) *
            (1.0 / (d_mid * kx_base * ky_base * kz_base)) ** 0.5
        )
        self.b_xyz_after = nn.Parameter(torch.zeros(d_out))

        # 缓存（按尺度与 dtype）以避免重复 PI 重采样
        self._cache_t = {}     # key=(kt,dtype) -> w_t_first
        self._cache_xyz = {}   # key=(kx,ky,kz,dtype) -> w_xyz_after

        # 添加3D空间位置编码模块
        self.pos_embed = FixedSinCos3DPE(
            embed_dim=d_out,           # 输出维度
            num_freq=12,               # 频率数量
            space_scale=0.01,          # 空间缩放因子
            learnable_proj=True        # 使用可学习的投影层
        )

    # ---- 工具 ----
    @torch.no_grad()
    def _k_from_meta(self, tr: float, voxel: Tuple[float, float, float]) -> Tuple[int, int, int, int]:
        vx, vy, vz = voxel
        kt = max(1, round(self.tau / tr))
        kx = max(1, round(self.rho[0] / vx))
        ky = max(1, round(self.rho[1] / vy))
        kz = max(1, round(self.rho[2] / vz))
        return int(kt), int(kx), int(ky), int(kz)

    def _get_wt_first(self, kt:int, device, dtype):
        # 注意：不缓存detached版本，因为这会断开梯度连接
        # 每次都重新计算以保持梯度流
        wt = pi_resize_weight_1d(self.w_t_first_base.to(dtype), kt)  # [Dm,1,kt]
        return wt.to(device)

    def _get_wxyz_after(self, kx:int, ky:int, kz:int, device, dtype):
        # 注意：不缓存detached版本，因为这会断开梯度连接
        # 每次都重新计算以保持梯度流
        w = pi_resize_weight_3d(self.w_xyz_after_base.to(dtype), kx, ky, kz)  # [Do,Dm,kx,ky,kz]
        return w.to(device)

    @staticmethod
    def _detect_true_T(x_b: torch.Tensor) -> int:
        """
        从尾部 0 pad 恢复原始长度:
        x_b: [96,96,96,T_max]
        """
        with torch.no_grad():
            s = x_b.abs().sum(dim=(0,1,2))
            nz = torch.nonzero(s > 0, as_tuple=False)
            if nz.numel() == 0:
                return 0
            return int(nz.max().item() + 1)

    @staticmethod
    def _spatial_keep_mask_alltime(x_b: torch.Tensor, kx:int, ky:int, kz:int, T_true:int) -> torch.Tensor:
        """
        x_b: [96,96,96,T_max] 仅使用前 T_true 帧
        返回: [Lx*Ly*Lz] bool，True=保留（全时间任一体素非零）
        """
        X=Y=Z=96
        Lx, Ly, Lz = X//kx, Y//ky, Z//kz
        if T_true == 0:
            return torch.zeros(Lx*Ly*Lz, dtype=torch.bool, device=x_b.device)
        vol = (x_b[:,:,:,:T_true] != 0).any(dim=-1).float()  # [96,96,96] -> 1/0
        vol = vol[:Lx*kx, :Ly*ky, :Lz*kz].unsqueeze(0).unsqueeze(0)  # [1,1,X,Y,Z]
        keep = F.max_pool3d(vol, kernel_size=(kx,ky,kz), stride=(kx,ky,kz)) > 0  # [1,1,Lx,Ly,Lz]
        return keep.squeeze(0).squeeze(0).reshape(-1)

    def _compute_spatial_coords_for_group(self,
                                        group_idxs: List[int],
                                        affines: List[torch.Tensor],
                                        kx: int, ky: int, kz: int,
                                        device: torch.device) -> torch.Tensor:
        """
        为组内样本计算空间patch的物理坐标

        Args:
            group_idxs: 组内样本的索引列表
            affines: 对应的仿射变换矩阵列表
            kx, ky, kz: 空间patch大小
            device: 设备

        Returns:
            coords: [G, Lx*Ly*Lz, 3] 物理坐标 (mm)
        """
        G = len(group_idxs)
        X, Y, Z = 96, 96, 96  # 固定输入尺寸
        Lx, Ly, Lz = X//kx, Y//ky, Z//kz

        coords_list = []
        for g, affine in enumerate(affines):
            # 使用现有函数计算物理坐标
            coords = stape_patch_world_coords_physical(
                X=X, Y=Y, Z=Z,
                kx=kx, ky=ky, kz=kz,
                affine=affine,
                rho_mm=self.rho  # 使用类的rho参数
            )  # [Lx*Ly*Lz, 3]
            coords_list.append(coords)

        # 堆叠为 [G, Lx*Ly*Lz, 3]
        coords_batch = torch.stack(coords_list, dim=0)
        return coords_batch.to(device)

    def _add_positional_encoding(self,
                               tokens_all: torch.Tensor,    # [G, Lx*Ly*Lz, Do]
                               group_idxs: List[int],
                               affines: List[torch.Tensor],
                               kx: int, ky: int, kz: int) -> torch.Tensor:
        """
        为空间tokens添加位置编码

        Args:
            tokens_all: 空间patch embedding结果
            group_idxs: 组内样本索引
            affines: 仿射变换矩阵
            kx, ky, kz: patch大小

        Returns:
            tokens_with_pos: 添加位置编码后的tokens
        """
        device = tokens_all.device

        # 计算物理坐标
        coords = self._compute_spatial_coords_for_group(
            group_idxs, affines, kx, ky, kz, device
        )  # [G, Lx*Ly*Lz, 3]

        # 生成位置编码
        pos_encoding = self.pos_embed(coords)  # [G, Lx*Ly*Lz, Do]  # 位置编码
        pos_encoding = pos_encoding.to(tokens_all.dtype)

        # 添加到tokens
        tokens_with_pos = tokens_all + pos_encoding

        return tokens_with_pos, pos_encoding

    # ---- 核心：组内处理（时间→通道折叠→空间）----
    def _run_group_time_first(self,
                              x_group: torch.Tensor,            # [G,96,96,96,T_max]
                              orig_Ts: List[int],               # 组内每样本的原始 T_i
                              kt:int, kx:int, ky:int, kz:int,
                              group_idxs: List[int],            # 组内样本索引
                              affines: List[torch.Tensor],      # 仿射矩阵
                              return_grid_info: bool = False,
                              explain_mode: bool = False) -> Tuple[List[torch.Tensor], List[int], Dict]:
        device, dtype = x_group.device, x_group.dtype
        G, X, Y, Z, T_max = x_group.shape
        assert X==96 and Y==96 and Z==96

        # 统一 pad 到组内一致的 T_pad'（kt 的倍数）
        T_true_max = max(orig_Ts) if len(orig_Ts)>0 else 0
        T_pad = math.ceil(T_true_max / kt) * kt
        T_prime = T_pad // kt   # 组内统一的 T′

        # 时间核
        w_t = self._get_wt_first(kt, device, dtype)   # [Dm,1,kt]

        # 构造组内 pad 后的张量
        xg = x_group.clone()
        if T_max < T_pad:
            pad_len = T_pad - T_max
            xg = F.pad(xg, (0, pad_len), mode='constant', value=0.0)  # [G,96,96,96,T_pad]
        # 若 T_max > T_pad，剪裁到 T_pad（多出来是全 0 也无影响）
        xg = xg[..., :T_pad]

        # ---- 时间 STAPE（Conv1d stride=kt，独立于每个体素）----
        # 重排为 [N=G*X*Y*Z, C=1, T_pad]
        xlin = xg.permute(0,1,2,3,4).contiguous().view(G*X*Y*Z, 1, T_pad)
        b_t_first = self.b_t_first.to(device=device, dtype=dtype)
        tfeat = F.conv1d(xlin, w_t, bias=b_t_first, stride=kt)  # [N, Dm, T′]
        # 还原并把时间折叠到通道: [G, Dm*T′, X, Y, Z]
        tfeat = tfeat.view(G, X, Y, Z, self.Dm, T_prime).permute(0,4,5,1,2,3).contiguous()
        x_sp_in = tfeat.view(G, self.Dm*T_prime, X, Y, Z)  # [G, C_in, X,Y,Z]

        # ---- 空间 STAPE（Conv3d stride=patch）----
        w_xyz = self._get_wxyz_after(kx,ky,kz, device, dtype)   # [Do, Dm, kx,ky,kz]
        # 将同一个空间核对每个时间块共享（把通道扩成 Dm*T′）：重复 T′ 次
        w_xyz_rep = w_xyz.repeat(1, T_prime, 1, 1, 1)           # [Do, Dm*T′, kx,ky,kz]
        b_xyz_after = self.b_xyz_after.to(device=device, dtype=dtype)
        sfeat = F.conv3d(x_sp_in, w_xyz_rep, bias=b_xyz_after, stride=(kx,ky,kz))  # [G, Do, Lx,Ly,Lz]

        Lx, Ly, Lz = X//kx, Y//ky, Z//kz
        # 展平空间 -> tokens
        tokens_all = sfeat.permute(0,2,3,4,1).contiguous().view(G, Lx*Ly*Lz, self.Do)

        # 添加位置编码 (在删除背景之前)
        if affines is not None and len(affines) == len(group_idxs):
            tokens_all, pos_group = self._add_positional_encoding(tokens_all, group_idxs, affines, kx, ky, kz)

        # ---- 物理删除零背景（跨全时间）或保持完整网格（解释模式）----
        tokens_list, lengths, pos_list = [], [], []
        grid_data = {}

        for g in range(G):
            T_true = orig_Ts[g]

            if explain_mode:
                # 解释模式：保持完整的patch网格，不删除背景
                toks = tokens_all[g]  # [Lx*Ly*Lz, Do] 保持完整网格
                pe = pos_group[g]     # [Lx*Ly*Lz, Do] 保持完整位置编码
                keep_mask = torch.ones(Lx*Ly*Lz, dtype=torch.bool, device=tokens_all[g].device)  # 全部保留
            else:
                # 正常模式：删除背景patch
                keep_mask = self._spatial_keep_mask_alltime(x_group[g], kx,ky,kz, T_true)  # [Lx*Ly*Lz]
                if keep_mask.any():
                    toks = tokens_all[g][keep_mask]   # [N_valid, Do]
                    pe = pos_group[g][keep_mask]
                else:
                    toks = tokens_all[g].new_zeros((0, self.Do))
                    pe = pos_group[g].new_zeros((0, self.Do))

            tokens_list.append(toks)
            pos_list.append(pe)
            lengths.append(int(toks.size(0)))

            # 记录网格信息
            if return_grid_info:
                sample_idx = group_idxs[g]
                grid_data[sample_idx] = {
                    'Lx': Lx,
                    'Ly': Ly,
                    'Lz': Lz,
                    'kx': kx,
                    'ky': ky,
                    'kz': kz,
                    'keep_mask': keep_mask.cpu(),  # [Lx*Ly*Lz] bool
                    'grid_to_token_idx': torch.nonzero(keep_mask, as_tuple=False).squeeze(-1).cpu(),  # 有效 token 在网格中的索引
                    'explain_mode': explain_mode  # 记录是否为解释模式
                }

        return tokens_list, lengths, grid_data, pos_list

    # ---- 前向 ----
    def forward(self,
                x: torch.Tensor,                     # (B,96,96,96,T_max) —— 已做 0 pad
                meta: Dict[int, Dict],               # 每个被试的元信息: {subject_idx: {"voxel": (x,y,z), "tr": float}}
                orig_Ts: Sequence[int] = None,
                affines: Sequence[torch.Tensor] = None,
                return_grid_info: bool = False,      # 是否返回网格信息
                explain_mode: bool = False):         # 解释模式：不删除背景patch，保持完整网格
        assert x.dim()==5 and x.size(1)==96 and x.size(2)==96 and x.size(3)==96, \
            "期望输入为 (B,96,96,96,T_max)"
        B = x.size(0)
        device, dtype = x.device, x.dtype

        # 1) 准备每样本原始 T（若未提供则自动检测）
        if orig_Ts is None:
            orig_Ts = [self._detect_true_T(x[b]) for b in range(B)]
        else:
            orig_Ts = [int(t) for t in orig_Ts]

        # 2) 准备affines
        if affines is None:
            print("WARNING: 未提供affines，使用单位矩阵")
            affines = [torch.eye(4, device=device, dtype=dtype) for _ in range(B)]
        else:
            affines = [aff.to(device=device, dtype=dtype) for aff in affines]
            assert len(affines) == B, f"affines长度({len(affines)})与batch_size({B})不匹配"

        # 3) 按相同的voxel和tr参数分组处理
        # 将具有相同voxel和tr的被试分到同一组，以便批量处理
        groups = defaultdict(list)
        for i in range(B):
            assert i in meta and "voxel" in meta[i] and "tr" in meta[i], f"元信息缺失: subject {i}"
            voxel = tuple(meta[i]["voxel"])
            tr = float(meta[i]["tr"])
            group_key = (voxel, tr)
            groups[group_key].append(i)

        per_sample_tokens: List[torch.Tensor] = [None]*B
        per_sample_pos:    List[torch.Tensor] = [None]*B
        lengths: List[int] = [0]*B
        grid_info: Dict[int, Dict] = {}  # 存储网格信息

        # 4) 逐组处理（时间→空间）
        for (voxel, tr), idxs in groups.items():
            kt, kx, ky, kz = self._k_from_meta(tr, voxel)

            x_group = x[idxs, ...]                              # [G,96,96,96,T_max]
            Ts_group = [orig_Ts[i] for i in idxs]               # 组内原始长度
            affines_group = [affines[i] for i in idxs]          # 组内仿射矩阵

            # 传递额外参数
            toks, lens, grid_data, pos_list = self._run_group_time_first(
                x_group, Ts_group, kt, kx, ky, kz, idxs, affines_group,
                return_grid_info=return_grid_info, explain_mode=explain_mode
            )
            for g_idx, (tok, ln) in enumerate(zip(toks, lens)):
                loc = idxs[g_idx]
                per_sample_tokens[loc] = tok
                lengths[loc] = ln
                per_sample_pos[loc]    = pos_list[g_idx]
                if return_grid_info and loc in grid_data:
                    grid_info[loc] = grid_data[loc]

        # 5) 对齐到 L_max，生成 (B,L_max,Do) 与 attn_mask
        L_max = max(lengths) if lengths else 0
        out = x.new_zeros((B, L_max, self.Do))
        pos_out    = x.new_zeros((B, L_max, self.Do))
        attn_mask = torch.ones((B, L_max), dtype=torch.bool, device=device)

        for b, tok in enumerate(per_sample_tokens):
            n = lengths[b]
            if n > 0:
                out[b, :n] = tok
                pos_out[b, :n] = per_sample_pos[b]
                attn_mask[b, :n] = False

        if return_grid_info:
            return out, attn_mask, lengths, grid_info, pos_out
        else:
            return out, attn_mask, lengths, pos_out
