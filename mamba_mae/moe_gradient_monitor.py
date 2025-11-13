"""
MoE 专家梯度监测工具

用于监测 MoE 中每个专家的梯度分布，帮助诊断训练问题：
- 专家是否被充分训练
- 梯度是否消失/爆炸
- 专家之间的梯度分布是否均衡
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional
import logging


class MoEGradientMonitor:
    """
    监测 MoE 专家的梯度统计信息
    """
    def __init__(self, moe_module: nn.Module, logger: Optional[logging.Logger] = None):
        """
        Args:
            moe_module: MoE 模块
            logger: 日志记录器
        """
        self.moe = moe_module
        self.logger = logger or logging.getLogger(__name__)
        self.num_experts = moe_module.num_experts
        
        # 存储梯度统计信息
        self.grad_stats = {
            'expert_grad_norm': [[] for _ in range(self.num_experts)],
            'expert_grad_mean': [[] for _ in range(self.num_experts)],
            'expert_grad_std': [[] for _ in range(self.num_experts)],
            'expert_grad_max': [[] for _ in range(self.num_experts)],
            'router_grad_norm': [],
        }
        
    def compute_gradient_stats(self) -> Dict[str, torch.Tensor]:
        """
        计算当前的梯度统计信息
        
        Returns:
            包含梯度统计的字典
        """
        stats = {}
        
        # 1. 计算每个专家的梯度统计
        for expert_idx, expert in enumerate(self.moe.experts):
            grad_norms = []
            grad_values = []
            
            for name, param in expert.named_parameters():
                if param.grad is not None:
                    grad = param.grad.detach()
                    grad_norms.append(grad.norm().item())
                    grad_values.append(grad.flatten())
            
            if grad_values:
                all_grads = torch.cat(grad_values)
                stats[f'expert_{expert_idx}_grad_norm'] = sum(grad_norms)
                stats[f'expert_{expert_idx}_grad_mean'] = all_grads.mean().item()
                stats[f'expert_{expert_idx}_grad_std'] = all_grads.std().item()
                stats[f'expert_{expert_idx}_grad_max'] = all_grads.abs().max().item()
                stats[f'expert_{expert_idx}_grad_min'] = all_grads.abs().min().item()
            else:
                stats[f'expert_{expert_idx}_grad_norm'] = 0.0
                stats[f'expert_{expert_idx}_grad_mean'] = 0.0
                stats[f'expert_{expert_idx}_grad_std'] = 0.0
                stats[f'expert_{expert_idx}_grad_max'] = 0.0
                stats[f'expert_{expert_idx}_grad_min'] = 0.0
        
        # 2. 计算路由器的梯度统计
        router_grad_norms = []
        router_grad_values = []
        
        # Token router
        if self.moe.router_token.weight.grad is not None:
            grad = self.moe.router_token.weight.grad.detach()
            router_grad_norms.append(grad.norm().item())
            router_grad_values.append(grad.flatten())
        
        # Conditional router (if exists)
        if self.moe.use_res_cond:
            if self.moe.router_scale.weight.grad is not None:
                grad = self.moe.router_scale.weight.grad.detach()
                router_grad_norms.append(grad.norm().item())
                router_grad_values.append(grad.flatten())
            
            if self.moe.router_bias.weight.grad is not None:
                grad = self.moe.router_bias.weight.grad.detach()
                router_grad_norms.append(grad.norm().item())
                router_grad_values.append(grad.flatten())
        
        if router_grad_values:
            all_router_grads = torch.cat(router_grad_values)
            stats['router_grad_norm'] = sum(router_grad_norms)
            stats['router_grad_mean'] = all_router_grads.mean().item()
            stats['router_grad_std'] = all_router_grads.std().item()
            stats['router_grad_max'] = all_router_grads.abs().max().item()
        else:
            stats['router_grad_norm'] = 0.0
            stats['router_grad_mean'] = 0.0
            stats['router_grad_std'] = 0.0
            stats['router_grad_max'] = 0.0
        
        return stats
    
    def log_gradient_stats(self, step: int, prefix: str = ""):
        """
        记录梯度统计信息到日志
        
        Args:
            step: 当前训练步数
            prefix: 日志前缀
        """
        stats = self.compute_gradient_stats()
        
        # 格式化输出
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"{prefix}MoE Gradient Statistics (Step {step})")
        self.logger.info(f"{'='*80}")
        
        # 专家梯度统计
        self.logger.info(f"\n📊 Expert Gradients:")
        self.logger.info(f"{'Expert':<10} {'Norm':<12} {'Mean':<12} {'Std':<12} {'Max':<12} {'Min':<12}")
        self.logger.info(f"{'-'*70}")
        
        expert_norms = []
        for expert_idx in range(self.num_experts):
            norm = stats[f'expert_{expert_idx}_grad_norm']
            mean = stats[f'expert_{expert_idx}_grad_mean']
            std = stats[f'expert_{expert_idx}_grad_std']
            max_val = stats[f'expert_{expert_idx}_grad_max']
            min_val = stats[f'expert_{expert_idx}_grad_min']
            
            expert_norms.append(norm)
            
            expert_name = f"Expert {expert_idx}" if expert_idx > 0 else "Shared"
            self.logger.info(
                f"{expert_name:<10} {norm:<12.6f} {mean:<12.6f} {std:<12.6f} {max_val:<12.6f} {min_val:<12.6f}"
            )
        
        # 专家梯度均衡性分析
        if len(expert_norms) > 1:
            expert_norms_tensor = torch.tensor(expert_norms)
            norm_mean = expert_norms_tensor.mean().item()
            norm_std = expert_norms_tensor.std().item()
            norm_max = expert_norms_tensor.max().item()
            norm_min = expert_norms_tensor.min().item()
            
            self.logger.info(f"\n📈 Expert Gradient Balance:")
            self.logger.info(f"  Mean Norm: {norm_mean:.6f}")
            self.logger.info(f"  Std Norm:  {norm_std:.6f}")
            self.logger.info(f"  Max Norm:  {norm_max:.6f}")
            self.logger.info(f"  Min Norm:  {norm_min:.6f}")
            self.logger.info(f"  Ratio (Max/Min): {norm_max / (norm_min + 1e-8):.2f}")
            
            # 警告：梯度不均衡
            if norm_std / (norm_mean + 1e-8) > 0.5:
                self.logger.warning(f"  ⚠️  High gradient imbalance detected! (CV={norm_std / (norm_mean + 1e-8):.2f})")
            
            # 警告：某些专家梯度过小
            for expert_idx, norm in enumerate(expert_norms):
                if norm < norm_mean * 0.1:
                    expert_name = f"Expert {expert_idx}" if expert_idx > 0 else "Shared"
                    self.logger.warning(f"  ⚠️  {expert_name} has very small gradients ({norm:.6f})")
        
        # 路由器梯度统计
        self.logger.info(f"\n🎯 Router Gradients:")
        self.logger.info(f"  Norm: {stats['router_grad_norm']:.6f}")
        self.logger.info(f"  Mean: {stats['router_grad_mean']:.6f}")
        self.logger.info(f"  Std:  {stats['router_grad_std']:.6f}")
        self.logger.info(f"  Max:  {stats['router_grad_max']:.6f}")
        
        # 警告：梯度消失/爆炸
        if stats['router_grad_norm'] < 1e-6:
            self.logger.warning(f"  ⚠️  Router gradients are vanishing!")
        elif stats['router_grad_norm'] > 100:
            self.logger.warning(f"  ⚠️  Router gradients are exploding!")
        
        self.logger.info(f"{'='*80}\n")
        
        return stats
    
    def get_expert_usage_stats(self, gates: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        计算专家使用统计
        
        Args:
            gates: [B, L, E] 专家门控权重
            attn_mask: [B, L] 注意力掩码 (True=pad)
            
        Returns:
            专家使用统计字典
        """
        if attn_mask is not None:
            valid = ~attn_mask  # [B, L]
            gates = gates * valid.unsqueeze(-1)
        
        # 每个专家的平均权重
        expert_weights = gates.mean(dim=(0, 1))  # [E]
        
        # 每个专家被选为主要专家的频率
        primary_expert = gates.argmax(dim=-1)  # [B, L]
        expert_counts = torch.zeros(self.num_experts, device=gates.device)
        for expert_idx in range(self.num_experts):
            expert_counts[expert_idx] = (primary_expert == expert_idx).float().sum()
        
        total_tokens = primary_expert.numel()
        if attn_mask is not None:
            total_tokens = (~attn_mask).sum().item()
        
        expert_usage = expert_counts / (total_tokens + 1e-8)
        
        stats = {}
        for expert_idx in range(self.num_experts):
            expert_name = f"expert_{expert_idx}" if expert_idx > 0 else "shared"
            stats[f'{expert_name}_avg_weight'] = expert_weights[expert_idx].item()
            stats[f'{expert_name}_usage_rate'] = expert_usage[expert_idx].item()
        
        return stats
    
    def log_expert_usage(self, gates: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, step: int = 0):
        """
        记录专家使用情况
        
        Args:
            gates: [B, L, E] 专家门控权重
            attn_mask: [B, L] 注意力掩码
            step: 当前步数
        """
        stats = self.get_expert_usage_stats(gates, attn_mask)
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"MoE Expert Usage (Step {step})")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"{'Expert':<10} {'Avg Weight':<15} {'Usage Rate':<15}")
        self.logger.info(f"{'-'*40}")
        
        for expert_idx in range(self.num_experts):
            expert_name = f"Expert {expert_idx}" if expert_idx > 0 else "Shared"
            key_prefix = f"expert_{expert_idx}" if expert_idx > 0 else "shared"
            
            avg_weight = stats[f'{key_prefix}_avg_weight']
            usage_rate = stats[f'{key_prefix}_usage_rate']
            
            self.logger.info(f"{expert_name:<10} {avg_weight:<15.4f} {usage_rate:<15.2%}")
        
        # 检查专家使用是否均衡
        usage_rates = [stats[f'expert_{i}_usage_rate'] if i > 0 else stats['shared_usage_rate'] 
                       for i in range(self.num_experts)]
        usage_std = torch.tensor(usage_rates).std().item()
        
        if usage_std > 0.2:
            self.logger.warning(f"  ⚠️  Expert usage is highly imbalanced (std={usage_std:.4f})")
        
        self.logger.info(f"{'='*80}\n")
        
        return stats

