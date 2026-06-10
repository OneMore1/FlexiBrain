# stape_time_to_space.py
import math
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from flexibrain.models.layers.pos_embed import FixedSinCos3DPE, stape_patch_world_coords_physical
from flexibrain.utils.weight_resize import pi_resize_weight_1d, pi_resize_weight_3d

class STAPE4D_TimeToSpace(nn.Module):
    """
    input:
        x: (B, 96, 96, 96, T_max) 
        meta: {subject_idx: {"voxel": (vx,vy,vz) mm, "tr": float s}}
        orig_T
        affine
    output:
        tokens:   (B, L_max, D_out)
        attn_mask:(B, L_max)  True=padding
        lengths:  List[int] 
    """
    def __init__(self,
                 d_mid: int = 128,     
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

        # time based kernal [Dm, 1, kt0] 
        self.w_t_first_base = nn.Parameter(
            torch.randn(d_mid, 1, kt_base) * (1.0 / (1 * kt_base)) ** 0.5
        )
        self.b_t_first = nn.Parameter(torch.zeros(d_mid))

        # space base kernal: [Do, Dm, kx0, ky0, kz0] 
        self.w_xyz_after_base = nn.Parameter(
            torch.randn(d_out, d_mid, kx_base, ky_base, kz_base) *
            (1.0 / (d_mid * kx_base * ky_base * kz_base)) ** 0.5
        )
        self.b_xyz_after = nn.Parameter(torch.zeros(d_out))

        self._cache_t = {}     # key=(kt,dtype) -> w_t_first
        self._cache_xyz = {}   # key=(kx,ky,kz,dtype) -> w_xyz_after

        # physical pe
        self.pos_embed = FixedSinCos3DPE(
            embed_dim=d_out,           
            num_freq=12,               
            space_scale=0.01,          
            learnable_proj=True        
        )

    @torch.no_grad()
    def _k_from_meta(self, tr: float, voxel: Tuple[float, float, float]) -> Tuple[int, int, int, int]:
        vx, vy, vz = voxel
        kt = max(1, round(self.tau / tr))
        kx = max(1, round(self.rho[0] / vx))
        ky = max(1, round(self.rho[1] / vy))
        kz = max(1, round(self.rho[2] / vz))
        return int(kt), int(kx), int(ky), int(kz)

    def _get_wt_first(self, kt:int, device, dtype):
        wt = pi_resize_weight_1d(self.w_t_first_base.to(dtype), kt)  # [Dm,1,kt]
        return wt.to(device)

    def _get_wxyz_after(self, kx:int, ky:int, kz:int, device, dtype):
        w = pi_resize_weight_3d(self.w_xyz_after_base.to(dtype), kx, ky, kz)  # [Do,Dm,kx,ky,kz]
        return w.to(device)

    @staticmethod
    def _detect_true_T(x_b: torch.Tensor) -> int:

        with torch.no_grad():
            s = x_b.abs().sum(dim=(0,1,2))
            nz = torch.nonzero(s > 0, as_tuple=False)
            if nz.numel() == 0:
                return 0
            return int(nz.max().item() + 1)

    @staticmethod
    def _spatial_keep_mask_alltime(x_b: torch.Tensor, kx:int, ky:int, kz:int, T_true:int) -> torch.Tensor:
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
        compute patch physical coordinate
        """
        G = len(group_idxs)
        X, Y, Z = 96, 96, 96  
        Lx, Ly, Lz = X//kx, Y//ky, Z//kz

        coords_list = []
        for g, affine in enumerate(affines):
            coords = stape_patch_world_coords_physical(
                X=X, Y=Y, Z=Z,
                kx=kx, ky=ky, kz=kz,
                affine=affine,
                rho_mm=self.rho  
            )  # [Lx*Ly*Lz, 3]
            coords_list.append(coords)

        # [G, Lx*Ly*Lz, 3]
        coords_batch = torch.stack(coords_list, dim=0)
        return coords_batch.to(device)

    def _add_positional_encoding(self,
                               tokens_all: torch.Tensor,    # [G, Lx*Ly*Lz, Do]
                               group_idxs: List[int],
                               affines: List[torch.Tensor],
                               kx: int, ky: int, kz: int) -> torch.Tensor:
        device = tokens_all.device

        coords = self._compute_spatial_coords_for_group(
            group_idxs, affines, kx, ky, kz, device
        )  # [G, Lx*Ly*Lz, 3]

        pos_encoding = self.pos_embed(coords)  # [G, Lx*Ly*Lz, Do]  
        pos_encoding = pos_encoding.to(tokens_all.dtype)
        tokens_with_pos = tokens_all + pos_encoding

        return tokens_with_pos, pos_encoding

    def _run_group_time_first(self,
                              x_group: torch.Tensor,            # [G,96,96,96,T_max]
                              orig_Ts: List[int],               
                              kt:int, kx:int, ky:int, kz:int,
                              group_idxs: List[int],          
                              affines: List[torch.Tensor],    
                              return_grid_info: bool = False) -> Tuple[List[torch.Tensor], List[int], Dict]:
        device, dtype = x_group.device, x_group.dtype
        G, X, Y, Z, T_max = x_group.shape
        assert X==96 and Y==96 and Z==96

        T_true_max = max(orig_Ts) if len(orig_Ts)>0 else 0
        T_pad = math.ceil(T_true_max / kt) * kt
        T_prime = T_pad // kt   

        w_t = self._get_wt_first(kt, device, dtype)   # [Dm,1,kt]

        xg = x_group.clone()
        if T_max < T_pad:
            pad_len = T_pad - T_max
            xg = F.pad(xg, (0, pad_len), mode='constant', value=0.0)  # [G,96,96,96,T_pad]
        xg = xg[..., :T_pad]

        xlin = xg.permute(0,1,2,3,4).contiguous().view(G*X*Y*Z, 1, T_pad)
        b_t_first = self.b_t_first.to(device=device, dtype=dtype)
        tfeat = F.conv1d(xlin, w_t, bias=b_t_first, stride=kt)  # [N, Dm, T′]
        tfeat = tfeat.view(G, X, Y, Z, self.Dm, T_prime).permute(0,4,5,1,2,3).contiguous()
        x_sp_in = tfeat.view(G, self.Dm*T_prime, X, Y, Z)  # [G, C_in, X,Y,Z]

        w_xyz = self._get_wxyz_after(kx,ky,kz, device, dtype)   # [Do, Dm, kx,ky,kz]
        w_xyz_rep = w_xyz.repeat(1, T_prime, 1, 1, 1)           # [Do, Dm*T′, kx,ky,kz]
        b_xyz_after = self.b_xyz_after.to(device=device, dtype=dtype)
        sfeat = F.conv3d(x_sp_in, w_xyz_rep, bias=b_xyz_after, stride=(kx,ky,kz))  # [G, Do, Lx,Ly,Lz]

        Lx, Ly, Lz = X//kx, Y//ky, Z//kz
        tokens_all = sfeat.permute(0,2,3,4,1).contiguous().view(G, Lx*Ly*Lz, self.Do)

        if affines is not None and len(affines) == len(group_idxs):
            tokens_all, pos_group = self._add_positional_encoding(tokens_all, group_idxs, affines, kx, ky, kz)

        tokens_list, lengths, pos_list = [], [], []
        grid_data = {}

        for g in range(G):
            T_true = orig_Ts[g]

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
                    'grid_to_token_idx': torch.nonzero(keep_mask, as_tuple=False).squeeze(-1).cpu(),  
                }

        return tokens_list, lengths, grid_data, pos_list
    
    def forward(self,
                x: torch.Tensor,                     
                meta: Dict[int, Dict],               
                orig_Ts: Sequence[int] = None,
                affines: Sequence[torch.Tensor] = None,
                return_grid_info: bool = False):         
        B = x.size(0)
        device, dtype = x.device, x.dtype

        if orig_Ts is None:
            orig_Ts = [self._detect_true_T(x[b]) for b in range(B)]
        else:
            orig_Ts = [int(t) for t in orig_Ts]

        if affines is None:
            print("WARNING: not provide affine")
            affines = [torch.eye(4, device=device, dtype=dtype) for _ in range(B)]
        else:
            affines = [aff.to(device=device, dtype=dtype) for aff in affines]

        groups = defaultdict(list)
        for i in range(B):
            voxel = tuple(meta[i]["voxel"])
            tr = float(meta[i]["tr"])
            group_key = (voxel, tr)
            groups[group_key].append(i)

        per_sample_tokens: List[torch.Tensor] = [None]*B
        per_sample_pos:    List[torch.Tensor] = [None]*B
        lengths: List[int] = [0]*B
        grid_info: Dict[int, Dict] = {}  

        for (voxel, tr), idxs in groups.items():
            kt, kx, ky, kz = self._k_from_meta(tr, voxel)

            x_group = x[idxs, ...]                              # [G,96,96,96,T_max]
            Ts_group = [orig_Ts[i] for i in idxs]               
            affines_group = [affines[i] for i in idxs]          

            toks, lens, grid_data, pos_list = self._run_group_time_first(
                x_group, Ts_group, kt, kx, ky, kz, idxs, affines_group,
                return_grid_info=return_grid_info
            )
            for g_idx, (tok, ln) in enumerate(zip(toks, lens)):
                loc = idxs[g_idx]
                per_sample_tokens[loc] = tok
                lengths[loc] = ln
                per_sample_pos[loc]    = pos_list[g_idx]
                if return_grid_info and loc in grid_data:
                    grid_info[loc] = grid_data[loc]

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
