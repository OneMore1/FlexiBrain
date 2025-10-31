import math
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pos_embed import FixedSinCos3DPE, stape_patch_world_coords_physical
from flex_patch_utils import pi_resize_weight_1d, pi_resize_weight_3d

# -------------------------
#  基础：PI-resize（1D/3D）
# -------------------------
def _linear_resize_matrix_1d(from_len: int, to_len: int, device=None, dtype=torch.float32):
    """构造 1D 线性插值矩阵 R (to_len x from_len)，用于 y = R @ x"""
    device = device or torch.device("cpu")
    R = torch.zeros((to_len, from_len), device=device, dtype=dtype)
    if from_len == 1:
        R[:, 0] = 1.0
        return R
    for i in range(to_len):
        pos = i * (from_len - 1) / (to_len - 1) if to_len > 1 else 0.0
        j0 = int(pos)
        j1 = min(j0 + 1, from_len - 1)
        w1 = float(pos - j0)
        R[i, j0] = 1.0 - w1
        R[i, j1] += w1
    return R

def pi_resize_weight_1d(w_star: torch.Tensor, k_new: int) -> torch.Tensor:
    """时间核伪逆重采样: [Dmid, Cin, Kb] -> [Dmid, Cin, k_new]"""
    Dm, Cin, Kb = w_star.shape
    if k_new == Kb:
        return w_star
    R = _linear_resize_matrix_1d(from_len=k_new, to_len=Kb, device=w_star.device, dtype=w_star.dtype)  # (Kb,k_new 的前向算子^T)
    R_pinv = torch.linalg.pinv(R)  # [k_new, Kb]
    W = w_star.reshape(-1, Kb)     # [(Dm*Cin), Kb]
    Wt = (R_pinv @ W.T).T          # [(Dm*Cin), k_new]
    return Wt.reshape(Dm, Cin, k_new)

def pi_resize_weight_3d(w_star: torch.Tensor, kx: int, ky: int, kz: int) -> torch.Tensor:
    """空间核伪逆重采样: [Dout, Dmid, Kx0, Ky0, Kz0] -> [Dout, Dmid, kx, ky, kz]"""
    Do, Dm, Kx0, Ky0, Kz0 = w_star.shape
    if (kx, ky, kz) == (Kx0, Ky0, Kz0):
        return w_star
    dev, dt = w_star.device, w_star.dtype

    # x 轴
    Rx = _linear_resize_matrix_1d(from_len=kx, to_len=Kx0, device=dev, dtype=dt)  # [Kx0, kx]
    Rx_p = torch.linalg.pinv(Rx)                                                  # [kx, Kx0]
    W = w_star.permute(2, 0, 1, 3, 4).reshape(Kx0, -1)
    W = (Rx_p @ W).reshape(kx, Do, Dm, Ky0, Kz0).permute(1, 2, 0, 3, 4)

    # y 轴
    Ry = _linear_resize_matrix_1d(from_len=ky, to_len=Ky0, device=dev, dtype=dt)
    Ry_p = torch.linalg.pinv(Ry)
    W = W.permute(3, 0, 1, 2, 4).reshape(Ky0, -1)
    W = (Ry_p @ W).reshape(ky, Do, Dm, kx, Kz0).permute(1, 2, 3, 0, 4)

    # z 轴
    Rz = _linear_resize_matrix_1d(from_len=kz, to_len=Kz0, device=dev, dtype=dt)
    Rz_p = torch.linalg.pinv(Rz)
    W = W.permute(4, 0, 1, 2, 3).reshape(Kz0, -1)
    W = (Rz_p @ W).reshape(kz, Do, Dm, kx, ky).permute(1, 2, 3, 4, 0)
    return W


# --------------------------------
#  STAPE 主模块：4D（t,x,y,z）
# --------------------------------
class STAPE4D(nn.Module):
    """
    输入  : x 形状 (B, 96, 96, 96, T) —— 无显式通道，隐含 C=1
            dataset_ids: (B,) 的长整型张量/列表，表明每条样本来自哪个数据集
            meta: dict[dataset_id] -> {"voxel": (vx,vy,vz) mm, "tr": float 秒}
    设定  : tau 秒、rho mm（统一物理时空尺度）
    过程  : 按数据集分组做 STAPE；其余 patch_size 的样本逐一 for 循环
            分块后“物理删除”全 0 背景 patch
            用 batch 内最长 token 长度对齐 + attention mask
    输出  : tokens: (B, L_max, D_out), attn_mask: (B, L_max)
            以及每样本的元信息（可选）
    """
    def __init__(self,
                 d_mid: int = 128,
                 d_out: int = 256,
                 kt_base: int = 6,
                 kx_base: int = 6,
                 ky_base: int = 6,
                 kz_base: int = 6,
                 tau_seconds: float = 6.0,
                 rho_mm: Tuple[float, float, float] = (12., 12., 12.)):
        super().__init__()
        self.Dm = d_mid
        self.Do = d_out
        self.kt0, self.kx0, self.ky0, self.kz0 = kt_base, kx_base, ky_base, kz_base
        self.tau = float(tau_seconds)
        self.rho = tuple(float(r) for r in rho_mm)

        # 基准核（C=1）- 注意：Space-First模式下核的顺序调整
        # 空间核：用于对每帧做空间卷积 [D_mid, 1, kx, ky, kz]
        self.w_xyz_base = nn.Parameter(torch.randn(d_mid, 1, kx_base, ky_base, kz_base) *
                                       (1.0 / (1 * kx_base * ky_base * kz_base)) ** 0.5)
        self.b_xyz = nn.Parameter(torch.zeros(d_mid))

        # 时间核：用于对空间patch序列做时间卷积 [D_out, D_mid, kt]
        self.w_t_base = nn.Parameter(torch.randn(d_out, d_mid, kt_base) *
                                     (1.0 / (d_mid * kt_base)) ** 0.5)
        self.b_t = nn.Parameter(torch.zeros(d_out))

        # 缓存不同比例的核
        self._cache = {}  # key=(kt,kx,ky,kz) -> (w_t, w_xyz)

    # ------- 工具函数 -------
    @torch.no_grad()
    def _k_from_meta(self, tr: float, voxel: Tuple[float, float, float]) -> Tuple[int, int, int, int]:
        vx, vy, vz = voxel
        kt = max(1, round(self.tau / tr))
        kx = max(1, round(self.rho[0] / vx))
        ky = max(1, round(self.rho[1] / vy))
        kz = max(1, round(self.rho[2] / vz))
        return int(kt), int(kx), int(ky), int(kz)

    def _get_resized_kernels(self, kt, kx, ky, kz, device, dtype):
        key = (kt, kx, ky, kz, dtype)
        if key in self._cache:
            wxyz, wt = self._cache[key]
            return wxyz.to(device), wt.to(device)
        # Space-First: 空间核用于单帧卷积 [Dm, 1, kx, ky, kz]
        wxyz = pi_resize_weight_3d(self.w_xyz_base.to(dtype), kx, ky, kz)      # [Dm,1,kx,ky,kz]
        # 时间核用于patch序列卷积 [Do, Dm, kt]
        wt = pi_resize_weight_1d(self.w_t_base.to(dtype), kt)                  # [Do,Dm,kt]
        self._cache[key] = (wxyz.detach().cpu(), wt.detach().cpu())
        return wxyz.to(device), wt.to(device)

    @staticmethod
    def _zero_block_mask_4d(x_pad: torch.Tensor, kt, kx, ky, kz) -> torch.Tensor:
        """
        x_pad: [B, 1, T', X, Y, Z]
        返回每样本的块有效性: [B, Lt, Lx, Ly, Lz] （True=保留）
        """
        B, C, Tp, X, Y, Z = x_pad.shape
        Lt, Lx, Ly, Lz = Tp // kt, X // kx, Y // ky, Z // kz
        xv = x_pad.view(B, C, Lt, kt, Lx, kx, Ly, ky, Lz, kz)
        keep = (xv.abs().sum(dim=(1, 3, 5, 7, 9)) > 0)  # [B, Lt, Lx, Ly, Lz]
        return keep

    @staticmethod
    def _pad_time_to_multiple(x_btxyz: torch.Tensor, kt: int) -> torch.Tensor:
        """
        x_btxyz: [B, 1, T, X, Y, Z] -> pad T 到 kt 的倍数
        """
        B, C, T, X, Y, Z = x_btxyz.shape
        Tpad = math.ceil(T / kt) * kt
        if Tpad == T:
            return x_btxyz
        pad_len = Tpad - T
        # F.pad 在时间维（dim=2）前后各 pad；我们 pad 在尾部
        pad_spec = (0, 0, 0, 0, 0, 0, 0, pad_len)  # Z,Y,X 无 pad；T 末尾 pad
        return F.pad(x_btxyz, pad=pad_spec, mode="constant", value=0.0)

    @staticmethod
    def _get_spatial_nonzero_mask(x_chunk: torch.Tensor, kx: int, ky: int, kz: int) -> torch.Tensor:
        """
        检测空间patch是否为全零
        x_chunk: [X, Y, Z, kt] 单个样本的时间块
        返回: [Lx*Ly*Lz] bool mask，True表示非零patch
        """
        X, Y, Z, kt = x_chunk.shape
        Lx, Ly, Lz = X // kx, Y // ky, Z // kz

        # 裁剪到patch边界
        x_cropped = x_chunk[:Lx*kx, :Ly*ky, :Lz*kz, :]  # [Lx*kx, Ly*ky, Lz*kz, kt]

        # 重排为patch格式 [Lx, Ly, Lz, kx, ky, kz, kt]
        patches = x_cropped.view(Lx, kx, Ly, ky, Lz, kz, kt).permute(0, 2, 4, 1, 3, 5, 6)

        # 展平为 [Lx*Ly*Lz, kx*ky*kz*kt]
        patches_flat = patches.reshape(Lx * Ly * Lz, -1)

        # 检测非零patch
        nonzero_mask = (patches_flat.abs().sum(dim=1) > 0)
        return nonzero_mask

    # ------- 核心：Space-First流式处理一组同数据集的样本 -------
    def _run_group(self, x_group: torch.Tensor,                 # [G, 96, 96, 96, T]
                   kt: int, kx: int, ky: int, kz: int) -> Tuple[List[torch.Tensor], List[int]]:
        """
        Space-First + 流式时间处理
        返回：tokens_list（每样本已删除全0块的 token [Ni, Do]）、lengths（Ni）
        """
        device, dtype = x_group.device, x_group.dtype
        G, X, Y, Z, T = x_group.shape

        # 1) 获取重采样后的核
        w_xyz, w_t = self._get_resized_kernels(kt, kx, ky, kz, device, dtype)

        # 2) 计算空间输出尺寸
        Lx, Ly, Lz = X // kx, Y // ky, Z // kz
        total_spatial_patches = Lx * Ly * Lz  # 例如 4,096 instead of 884,736

        # 3) 时间分块参数
        Lt_total = (T + kt - 1) // kt  # 总时间块数

        # 4) 为每个样本初始化token收集器
        sample_tokens = [[] for _ in range(G)]

        # 5) 流式处理：逐个时间块处理
        for t_block in range(Lt_total):
            t_start = t_block * kt
            t_end = min(t_start + kt, T)
            actual_kt = t_end - t_start

            # 提取当前时间块
            x_chunk = x_group[:, :, :, :, t_start:t_end]  # [G, X, Y, Z, actual_kt]

            # 如果不足kt帧，进行零填充
            if actual_kt < kt:
                pad_frames = kt - actual_kt
                x_chunk = F.pad(x_chunk, (0, pad_frames), mode='constant', value=0)

            # 处理当前时间块
            block_tokens = self._process_time_block(x_chunk, w_xyz, w_t, kx, ky, kz, kt)

            # 收集每个样本的tokens
            for g in range(G):
                if block_tokens[g] is not None and block_tokens[g].size(0) > 0:
                    sample_tokens[g].append(block_tokens[g])

        # 6) 合并每个样本的所有时间块tokens
        tokens_list, lengths = [], []
        for g in range(G):
            if sample_tokens[g]:
                all_tokens = torch.cat(sample_tokens[g], dim=0)  # [N_total, Do]
            else:
                all_tokens = torch.empty(0, self.Do, device=device, dtype=dtype)
            tokens_list.append(all_tokens)
            lengths.append(all_tokens.size(0))

        return tokens_list, lengths

    def _process_time_block(self, x_chunk: torch.Tensor,        # [G, X, Y, Z, kt]
                           w_xyz: torch.Tensor,                 # [Dm, 1, kx, ky, kz]
                           w_t: torch.Tensor,                   # [Do, Dm, kt]
                           kx: int, ky: int, kz: int, kt: int) -> List[torch.Tensor]:
        """
        处理单个时间块：Space-First卷积
        返回：每个样本的tokens列表 [G个Tensor，每个形状为[N_valid, Do]]
        """
        G, X, Y, Z, kt_actual = x_chunk.shape
        device, dtype = x_chunk.device, x_chunk.dtype
        Lx, Ly, Lz = X // kx, Y // ky, Z // kz

        # 对每一帧做空间卷积
        frame_features = []
        for t_frame in range(kt):
            # 单帧数据 [G, X, Y, Z]
            frame = x_chunk[:, :, :, :, t_frame].unsqueeze(1)  # [G, 1, X, Y, Z]

            # 空间3D卷积
            spatial_feat = F.conv3d(
                frame, w_xyz, bias=self.b_xyz,
                stride=(kx, ky, kz)
            )  # [G, Dm, Lx, Ly, Lz]

            frame_features.append(spatial_feat)

        # 堆叠时间维度
        temporal_stack = torch.stack(frame_features, dim=2)  # [G, Dm, kt, Lx, Ly, Lz]

        # 对每个空间位置做时间卷积
        # 重排为 [G*Lx*Ly*Lz, Dm, kt]
        temp_input = temporal_stack.permute(0, 3, 4, 5, 1, 2).reshape(
            G * Lx * Ly * Lz, self.Dm, kt
        )

        # 时间1D卷积 (stride=kt，输出长度为1)
        time_feat = F.conv1d(
            temp_input, w_t, bias=self.b_t, stride=kt
        )  # [G*Lx*Ly*Lz, Do, 1]

        # 重排回 [G, Lx, Ly, Lz, Do]
        time_feat = time_feat.squeeze(-1).view(G, Lx, Ly, Lz, self.Do)

        # 检测并过滤全零块
        block_tokens = []
        for g in range(G):
            # 检查原始数据块是否有非零内容
            chunk_sum = x_chunk[g].abs().sum()
            if chunk_sum > 0:  # 非全零块
                # 展平空间维度得到tokens
                tokens = time_feat[g].reshape(Lx * Ly * Lz, self.Do)  # [Lx*Ly*Lz, Do]

                # 进一步过滤空间上的全零patch
                spatial_mask = self._get_spatial_nonzero_mask(
                    x_chunk[g], kx, ky, kz
                )  # [Lx*Ly*Lz]

                valid_tokens = tokens[spatial_mask]  # [N_valid, Do]
                block_tokens.append(valid_tokens)
            else:
                # 全零块，返回空tensor
                block_tokens.append(torch.empty(0, self.Do, device=device, dtype=dtype))

        return block_tokens

    # ------- 对整个 batch 执行 -------
    def forward(self,
                x: torch.Tensor,                    # (B, 96,96,96,T)
                dataset_ids: Sequence[int],         # 长度 B
                meta: Dict[int, Dict]):             # {ds_id: {"voxel":(vx,vy,vz), "tr":float}}
        assert x.dim() == 5 and x.size(1) == 96 and x.size(2) == 96 and x.size(3) == 96, \
            "期望输入为 (B,96,96,96,T)"
        B = x.size(0)
        device, dtype = x.device, x.dtype

        # 1) 依据数据集分组
        groups = defaultdict(list)
        for i, ds in enumerate(dataset_ids):
            groups[int(ds)].append(i)

        # 2) 逐组跑 STAPE（相同数据集 -> 一次性；其余天然 for 循环）
        per_sample_tokens: List[torch.Tensor] = [None] * B
        lengths: List[int] = [0] * B

        for ds, idxs in groups.items():
            assert ds in meta and "voxel" in meta[ds] and "tr" in meta[ds], f"元信息缺失: dataset {ds}"
            voxel = meta[ds]["voxel"]
            tr = float(meta[ds]["tr"])
            kt, kx, ky, kz = self._k_from_meta(tr, voxel)

            x_group = x[idxs, ...]  # [G,96,96,96,T]
            toks, lens = self._run_group(x_group, kt, kx, ky, kz)
            for loc, (tok, ln) in zip(idxs, zip(toks, lens)):
                per_sample_tokens[loc] = tok
                lengths[loc] = ln

        # 3) 以最长样本对齐，构建 (B, L_max, D) + attention mask
        L_max = max(lengths)
        out = x.new_zeros((B, L_max, self.Do))
        attn_mask = torch.ones((B, L_max), dtype=torch.bool, device=device)  # True=padding

        for b, tok in enumerate(per_sample_tokens):
            n = lengths[b]
            if n == 0:
                continue
            out[b, :n] = tok
            attn_mask[b, :n] = False

        return out, attn_mask, lengths  # 形状：(B, L_max, D_out), (B, L_max), [B]
