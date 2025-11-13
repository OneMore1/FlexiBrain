import torch
import torch.nn as nn
import math


class CondLoRA(nn.Module):
    def __init__(self, dim, cond_in, rank=8, cond_hidden=32, device=None, dtype=None):
        super().__init__()
        fk={}
        if device is not None: fk["device"]=device
        if dtype  is not None: fk["dtype"]=dtype
        self.U = nn.Parameter(torch.randn(dim, rank, **fk)*0.02)
        self.V = nn.Parameter(torch.randn(dim, rank, **fk)*0.02)
        self.cond = nn.Sequential(
            nn.LayerNorm(cond_in),
            nn.Linear(cond_in, cond_hidden, **fk), nn.GELU(),
            nn.Linear(cond_hidden, rank, **fk)
        )

    def forward(self, x, cond_vec, attn_mask=None):   # x:[B,L,D], cond_vec:[B,C]
        B,L,D = x.shape
        a = self.cond(cond_vec)                        # [B,r]
        xU = torch.einsum('bld,dr->blr', x, self.U)    # [B,L,r]
        add = torch.einsum('blr,br,dr->bld', xU, a, self.V)  # [B,L,D]
        y = x + add
        if attn_mask is not None:
            valid = (~attn_mask).unsqueeze(-1)
            y = torch.where(valid, y, x)
        return y


# 先验注入resolution + tr
class ResoPrior_E(nn.Module):
    def __init__(self, k_space=0.5, k_time=0.5, gamma=0.5):
        super().__init__()
        self.ks, self.kt, self.g = k_space, k_time, gamma
    def forward(self, rxyztr):
        rx, ry, rz, tr = torch.clamp(rxyztr, 1e-6).unbind(-1)
        sig = torch.stack([self.ks*rx, self.ks*ry, self.ks*rz, self.kt*tr], dim=-1)  # [B,4]
        snr = (rx*ry*rz) / (tr ** self.g + 1e-6)                                     # [B]
        z = torch.cat([torch.log(sig+1e-6), torch.log(snr+1e-6).unsqueeze(-1)], dim=-1)  # [B,5]
        return z


class ExpertMLP(nn.Module):
    def __init__(self, dim, hidden_dim=None, device=None, dtype=None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        factory_kwargs = {"device": device, "dtype": dtype}
        self.fc1 = nn.Linear(dim, hidden_dim, **factory_kwargs)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim, **factory_kwargs)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class MoE(nn.Module):
    def __init__(self,
                 dim,
                 hidden_dim=None,   
                 num_indep=3,
                 aux_loss_coef=0.00,
                 device=None,
                 dtype=None,
                 load_balance_coef: float = 0.01,

                 use_res_cond: bool = False,
                 cond_dim: int = 5,
                 cond_hidden_dim: int = 16,
                 cond_tanh_scale: float = 0.5):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.num_shared = 1
        self.num_indep = int(num_indep)
        self.num_experts = self.num_shared + self.num_indep
        self.aux_loss_coef = float(aux_loss_coef)
        self.load_balance_coef = load_balance_coef  # 存储负载均衡系数

        # 专家
        experts = [ExpertMLP(dim, hidden_dim, **factory_kwargs)]
        for _ in range(self.num_indep):
            experts.append(ExpertMLP(dim, hidden_dim, **factory_kwargs))
        self.experts = nn.ModuleList(experts)

        # 原 token 路由器
        # 注意：如果启用条件路由，我们把 token 部分与条件部分分开建
        self.router_token = nn.Linear(dim, self.num_experts, bias=False, **factory_kwargs)

        # -------------- 分辨率条件 --------------
        self.use_res_cond = bool(use_res_cond)
        self.cond_tanh_scale = float(cond_tanh_scale)
        if self.use_res_cond:
            self.cond_proj = nn.Sequential(
                nn.LayerNorm(cond_dim, **factory_kwargs),
                nn.Linear(cond_dim, cond_hidden_dim, **factory_kwargs),
                nn.GELU(),
                nn.LayerNorm(cond_hidden_dim, **factory_kwargs),
            )
            self.router_scale = nn.Linear(cond_hidden_dim, self.num_experts, bias=False, **factory_kwargs)
            self.router_bias  = nn.Linear(cond_hidden_dim, self.num_experts, bias=False, **factory_kwargs)
        else:
            self.router_scale = None
            self.router_bias  = None

        # FiLM到路由输入的对角调制
        self.use_router_film = False          
        if self.use_router_film and self.use_res_cond:
            self.film_gamma = nn.Linear(cond_hidden_dim, dim, **factory_kwargs)
            self.film_beta  = nn.Linear(cond_hidden_dim, dim, **factory_kwargs)


    def forward(self, x, attn_mask=None, cond_vec: torch.Tensor = None, return_gates: bool = False):
        """
        x: [B, L, D]
        attn_mask: [B, L]  True=pad, False=valid
        cond_venc: [B, 3]  每个样本的体素间距 (mm)
        return_gates: 是否返回 gates 用于监测
        return: y: [B, L, D], aux: scalar, (gates: [B, L, E] if return_gates=True)
        """
        B, L, D = x.shape

        if self.use_res_cond:
            cond = self.cond_proj(cond_vec)  # [B,cond_dim]
            if self.use_router_film:
                gamma = torch.tanh(self.film_gamma(cond))  # [B,D]
                beta  = self.film_beta(cond)
                x = x * (1 + 0.3 * gamma.unsqueeze(1)) + 0.3 * beta.unsqueeze(1)


        token_logits = self.router_token(x)  # [B, L, E]

        if self.use_res_cond:
            scale = torch.tanh(self.router_scale(cond))           # [B,E]
            bias  = self.router_bias(cond)                        # [B,E]
            token_logits = token_logits * (1 + self.cond_tanh_scale * scale.unsqueeze(1)) \
                                         + bias.unsqueeze(1)      # [B,L,E]

        gates = torch.softmax(token_logits, dim=-1)             # [B, L, E]

        if attn_mask is not None:
            valid = ~attn_mask                            # [B, L]
            gates = gates * valid.unsqueeze(-1)
            gates = gates / gates.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        # 专家前向并加权
        expert_outs = torch.stack([expert(x) for expert in self.experts], dim=-2)  # [B, L, E, D]
        y = (gates.unsqueeze(-1) * expert_outs).sum(dim=-2)                        # [B, L, D]


        # 负载均衡正则化
        imp = gates.sum(dim=(0, 1))                      # [E], 计算每个专家的激活次数
        imp = imp / imp.sum().clamp_min(1e-6)             # 归一化激活次数
        uniform = torch.full_like(imp, 1.0 / self.num_experts)  # 均匀分布
        load_balance_loss = ((imp - uniform) ** 2).sum() * self.load_balance_coef  # L2正则，鼓励均匀分布

        # 均衡正则（可选）
        aux = x.new_zeros(())
        if self.aux_loss_coef > 0.0:
            imp = gates.sum(dim=(0, 1))                      # [E]
            imp = imp / imp.sum().clamp_min(1e-6)
            uniform = torch.full_like(imp, 1.0 / self.num_experts)
            aux = ((imp - uniform) ** 2).sum() * self.aux_loss_coef

        if return_gates:
            return y, load_balance_loss, gates
        return y, load_balance_loss
