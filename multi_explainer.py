
"""
多方法可解释性工具 - 使用Captum调包实现
支持: IntegratedGradients, GradientShap, Saliency, DeepLift, GradCAM, LRP等
"""

import torch
import numpy as np
import nibabel as nib
import argparse
import os
from typing import Dict, Optional, Tuple
from scipy.ndimage import gaussian_filter, label as scipy_label
from captum.attr import (
    IntegratedGradients,
    GradientShap,
    Saliency,
    DeepLift,
    DeepLiftShap,
    InputXGradient,
    GuidedBackprop,
    Deconvolution,
    LRP,
)
from downstream_utils.mamba import MambaJEPAClassifier
from dataset import NiftiTxtDataset
import tempfile


def load_single_nifti(nifti_path: str, T_prime: int = None, tau_seconds: float = 6.0):
    """加载单个NIfTI文件"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(nifti_path + '\n')
        temp_txt = f.name

    try:
        dataset = NiftiTxtDataset([temp_txt], return_torch=True, memory_map=True,
                                 T_prime=T_prime, tau_seconds=tau_seconds)
        sample = dataset[0]
        x_data = sample['data'].unsqueeze(0)
        meta = {0: {"voxel": sample['voxel'], "tr": sample['tr']}}
        orig_Ts = np.array([sample['T_selected']])
        nifti_img = nib.load(nifti_path)
        
        print(f"  数据形状: {x_data.shape}")
        print(f"  Voxel: {sample['voxel']}, TR: {sample['tr']}")
        return x_data, meta, orig_Ts, nifti_img
    finally:
        os.unlink(temp_txt)


class MultiExplainer:
    """多方法可解释性计算器"""

    def __init__(self, model, device, target_class=1):
        self.model = model
        self.device = device
        self.target_class = target_class
        self.meta = None
        self.orig_Ts = None
        self.affines = None


    def _forward_func(self, x: torch.Tensor) -> torch.Tensor:
        """Captum要求的前向函数"""
        x = x.requires_grad_(True)
        batch_size = x.shape[0]
        
        # 动态扩展meta, orig_Ts, affines
        current_meta = {i: self.meta[0] for i in range(batch_size)}
        current_orig_Ts = np.tile(self.orig_Ts, batch_size)
        current_affines = self.affines * batch_size if isinstance(self.affines, list) else [self.affines[0]] * batch_size
        
        with torch.enable_grad():
            logits = self.model.forward(x, meta=current_meta, orig_Ts=current_orig_Ts,
                                       affines=current_affines, inference_params=None, explain_mode=True)
            return logits[:, self.target_class]
    
    def _get_baseline(self, x: torch.Tensor, method: str = 'zero'):
        """生成baseline"""
        if method == 'zero':
            return torch.zeros_like(x)
        elif method == 'mean':
            return torch.full_like(x, x.mean().item())
        elif method == 'random':
            return torch.randn_like(x) * x.std().item()
        else:
            raise ValueError(f"未知baseline方法: {method}")
    
    def compute(self, x: torch.Tensor, meta: Dict, orig_Ts: np.ndarray, affines: list,
                method: str = 'ig', n_steps: int = 50, baseline_method: str = 'zero',
                internal_batch_size: int = 5, use_memory_efficient: bool = False):
        """
        计算可解释性梯度

        Args:
            x: 输入数据 [B, H, W, D, T]
            meta: 元数据字典
            orig_Ts: 原始时间长度
            affines: 仿射矩阵列表
            method: 解释方法 ('ig', 'saliency', 'deeplift', 'gradshap', etc.)
            n_steps: 积分步数（用于IG等方法）
            baseline_method: 基线方法 ('zero', 'mean', 'random')
            internal_batch_size: 内部批次大小（用于IG等方法）
            use_memory_efficient: 是否使用内存高效模式

        Returns:
            gradients: 梯度张量 [B, H, W, D, T]
        """
        print(f"\n{'='*60}")
        print(f"方法: {method.upper()}")
        print(f"{'='*60}")
        
        # 存储模型参数
        self.meta = meta
        self.orig_Ts = orig_Ts
        self.affines = affines
        
        # 自动设置internal_batch_size（更保守的策略）
        if internal_batch_size is None and method in ['ig', 'gradshap', 'deeplift', 'deepliftshap']:
            # 根据数据大小和步数自动设置
            data_size_gb = x.element_size() * x.nelement() / (1024**3)
            if data_size_gb > 1.0:  # 数据大于1GB
                internal_batch_size = max(1, min(5, n_steps // 20))
            elif data_size_gb > 0.5:  # 数据大于0.5GB
                internal_batch_size = max(1, min(10, n_steps // 10))
            else:
                internal_batch_size = max(1, min(20, n_steps // 5))
            print(f"自动设置 internal_batch_size={internal_batch_size} (数据大小: {data_size_gb:.2f} GB)")
        
        # 清理显存
        torch.cuda.empty_cache()

        # 选择方法
        if method == 'ig':
            explainer = IntegratedGradients(self._forward_func)
            baseline = self._get_baseline(x, baseline_method)

            # 如果使用内存高效模式，进一步减小batch size
            if use_memory_efficient and internal_batch_size is not None:
                internal_batch_size = max(1, internal_batch_size // 2)
                print(f"内存高效模式: 减小 internal_batch_size 到 {internal_batch_size}")

            attributions = explainer.attribute(x, baselines=baseline, n_steps=n_steps,
                                              internal_batch_size=internal_batch_size)
        
        elif method == 'gradshap':
            explainer = GradientShap(self._forward_func)
            baseline = self._get_baseline(x, baseline_method)
            attributions = explainer.attribute(x, baselines=baseline, n_samples=n_steps)
        
        elif method == 'saliency':
            explainer = Saliency(self._forward_func)
            attributions = explainer.attribute(x, abs=False)
        
        elif method == 'deeplift':
            explainer = DeepLift(self.model)
            baseline = self._get_baseline(x, baseline_method)
            # DeepLift需要直接使用模型，不能用_forward_func
            print("⚠ DeepLift需要模型支持，可能不适用于当前模型架构")
            attributions = explainer.attribute(x, baselines=baseline)
        
        elif method == 'deepliftshap':
            explainer = DeepLiftShap(self.model)
            baseline = self._get_baseline(x, baseline_method)
            print("⚠ DeepLiftShap需要模型支持，可能不适用于当前模型架构")
            attributions = explainer.attribute(x, baselines=baseline)
        
        elif method == 'inputxgrad':
            explainer = InputXGradient(self._forward_func)
            attributions = explainer.attribute(x)
        
        elif method == 'guidedbackprop':
            explainer = GuidedBackprop(self.model)
            print("⚠ GuidedBackprop需要模型支持，可能不适用于当前模型架构")
            attributions = explainer.attribute(x)
        
        elif method == 'deconv':
            explainer = Deconvolution(self.model)
            print("⚠ Deconvolution需要模型支持，可能不适用于当前模型架构")
            attributions = explainer.attribute(x)
        
        elif method == 'lrp':
            explainer = LRP(self.model)
            print("⚠ LRP需要模型支持，可能不适用于当前模型架构")
            attributions = explainer.attribute(x)



        else:
            raise ValueError(f"未知方法: {method}")
        
        print(f"✓ 计算完成")
        print(f"  梯度范围: [{attributions.min():.8f}, {attributions.max():.8f}]")
        print(f"  非零比例: {torch.count_nonzero(attributions).item() / attributions.numel():.4f}")

        return attributions.detach()

    def filter_activations(self, gradients: torch.Tensor,
                          filter_method: str = 'topk',
                          threshold_percentile: float = 95.0,
                          topk_percent: float = 10.0,
                          n_components: int = 3,
                          min_component_size: int = 100) -> torch.Tensor:
        """
        过滤激活区域，减少噪声和不重要的激活

        Args:
            gradients: 梯度张量 [B, H, W, D, T] 或 [H, W, D]
            filter_method: 过滤方法
                - 'threshold': 阈值过滤（保留高于百分位数的值）
                - 'topk': Top-K 过滤（保留最大的 K% 值）
                - 'connected': 连通域过滤（保留最大的 N 个连通区域）
                - 'combined': 组合方法（threshold + connected）
            threshold_percentile: 阈值百分位数 (0-100)
            topk_percent: Top-K 百分比 (0-100)
            n_components: 保留的连通域数量
            min_component_size: 最小连通域大小（体素数）

        Returns:
            过滤后的梯度张量
        """
        print(f"\n{'='*60}")
        print(f"激活区域过滤")
        print(f"{'='*60}")
        print(f"  方法: {filter_method}")
        print(f"  原始范围: [{gradients.min():.8f}, {gradients.max():.8f}]")

        # 取绝对值
        abs_grad = torch.abs(gradients)

        if filter_method == 'threshold':
            # 阈值过滤
            threshold = np.percentile(abs_grad.cpu().numpy(), threshold_percentile)
            print(f"  阈值 ({threshold_percentile}%): {threshold:.8f}")

            mask = abs_grad >= threshold
            filtered = gradients * mask.float()

            kept_ratio = mask.float().mean().item()
            print(f"  保留比例: {kept_ratio:.2%}")

        elif filter_method == 'topk':
            # Top-K 过滤
            k = int(gradients.numel() * topk_percent / 100.0)
            print(f"  Top-K: {k} / {gradients.numel()} ({topk_percent}%)")

            # 展平并找到 top-k
            flat_abs = abs_grad.flatten()
            topk_values, topk_indices = torch.topk(flat_abs, k)

            # 创建掩码
            mask = torch.zeros_like(flat_abs, dtype=torch.bool)
            mask[topk_indices] = True
            mask = mask.reshape(gradients.shape)

            filtered = gradients * mask.float()

            threshold_val = topk_values[-1].item()
            print(f"  Top-K 阈值: {threshold_val:.8f}")

        elif filter_method == 'connected':
            # 连通域过滤
            print(f"  保留最大的 {n_components} 个连通域")
            print(f"  最小连通域大小: {min_component_size} 体素")

            # 需要先聚合时间维度（如果有）
            if gradients.ndim == 5:  # [B, H, W, D, T]
                grad_3d = abs_grad.squeeze(0).mean(dim=-1)  # [H, W, D]
            else:
                grad_3d = abs_grad.squeeze(0) if gradients.ndim == 4 else abs_grad

            # 二值化
            threshold = np.percentile(grad_3d.cpu().numpy(), 90)
            binary = (grad_3d.cpu().numpy() > threshold).astype(np.uint8)

            # 连通域分析
            labeled, num_features = scipy_label(binary)
            print(f"  检测到 {num_features} 个连通域")

            # 计算每个连通域的大小
            component_sizes = []
            for i in range(1, num_features + 1):
                size = (labeled == i).sum()
                if size >= min_component_size:
                    component_sizes.append((i, size))

            # 按大小排序，保留最大的 N 个
            component_sizes.sort(key=lambda x: x[1], reverse=True)
            keep_components = [c[0] for c in component_sizes[:n_components]]

            print(f"  保留的连通域: {len(keep_components)}")
            for idx, (comp_id, size) in enumerate(component_sizes[:n_components]):
                print(f"    #{idx+1}: {size} 体素")

            # 创建掩码
            mask_3d = np.zeros_like(labeled, dtype=bool)
            for comp_id in keep_components:
                mask_3d |= (labeled == comp_id)

            mask = torch.from_numpy(mask_3d).to(gradients.device)

            # 扩展到原始形状
            if gradients.ndim == 5:
                mask = mask.unsqueeze(0).unsqueeze(-1).expand_as(gradients)
            elif gradients.ndim == 4:
                mask = mask.unsqueeze(0).expand_as(gradients)

            filtered = gradients * mask.float()

        elif filter_method == 'combined':
            # 组合方法：先阈值过滤，再连通域过滤
            print(f"  步骤 1: 阈值过滤 ({threshold_percentile}%)")
            threshold = np.percentile(abs_grad.cpu().numpy(), threshold_percentile)
            mask1 = abs_grad >= threshold
            temp_filtered = gradients * mask1.float()

            print(f"  步骤 2: 连通域过滤 (保留 {n_components} 个)")

            # 聚合时间维度
            if temp_filtered.ndim == 5:
                grad_3d = torch.abs(temp_filtered).squeeze(0).mean(dim=-1)
            else:
                grad_3d = torch.abs(temp_filtered).squeeze(0) if temp_filtered.ndim == 4 else torch.abs(temp_filtered)

            # 二值化
            binary = (grad_3d.cpu().numpy() > 0).astype(np.uint8)

            # 连通域分析
            labeled, num_features = scipy_label(binary)
            print(f"    检测到 {num_features} 个连通域")

            # 保留最大的 N 个
            component_sizes = [(i, (labeled == i).sum()) for i in range(1, num_features + 1)]
            component_sizes.sort(key=lambda x: x[1], reverse=True)
            keep_components = [c[0] for c in component_sizes[:n_components]]

            mask_3d = np.zeros_like(labeled, dtype=bool)
            for comp_id in keep_components:
                mask_3d |= (labeled == comp_id)

            mask2 = torch.from_numpy(mask_3d).to(gradients.device)

            # 扩展到原始形状
            if gradients.ndim == 5:
                mask2 = mask2.unsqueeze(0).unsqueeze(-1).expand_as(gradients)
            elif gradients.ndim == 4:
                mask2 = mask2.unsqueeze(0).expand_as(gradients)

            filtered = gradients * mask2.float()

            kept_ratio = mask2.float().mean().item()
            print(f"  最终保留比例: {kept_ratio:.2%}")

        else:
            raise ValueError(f"未知过滤方法: {filter_method}")

        print(f"  过滤后范围: [{filtered.min():.8f}, {filtered.max():.8f}]")
        print(f"  非零比例: {torch.count_nonzero(filtered).item() / filtered.numel():.4f}")
        print(f"{'='*60}\n")

        return filtered
    
    def save_to_nifti(self, gradients: torch.Tensor, original_nifti_path: str,
                      output_path: str, scaling_method: str = 'percentile',
                      temporal_aggregation: str = 'mean', smooth_sigma: float = 1.0,
                      filter_method: Optional[str] = None,
                      threshold_percentile: float = 95.0,
                      topk_percent: float = 10.0,
                      n_components: int = 3,
                      min_component_size: int = 100):
        """
        保存梯度到NIfTI文件

        Args:
            gradients: 梯度张量 [B, H, W, D, T]
            original_nifti_path: 原始NIfTI文件路径
            output_path: 输出路径
            scaling_method: 缩放方法 ('percentile', 'minmax', 'std', 'none')
            temporal_aggregation: 时间聚合方法 ('mean', 'max', 'sum', 'none')
            smooth_sigma: 3D Gaussian平滑的sigma值 (0表示不平滑，推荐0.5-2.0)
            filter_method: 激活过滤方法 ('threshold', 'topk', 'connected', 'combined', None)
            threshold_percentile: 阈值百分位数
            topk_percent: Top-K 百分比
            n_components: 保留的连通域数量
            min_component_size: 最小连通域大小
        """
        print(f"\n保存梯度到NIfTI...")
        print(f"  原始梯度范围: [{gradients.min():.8f}, {gradients.max():.8f}]")

        # 应用激活过滤（如果指定）
        if filter_method is not None:
            gradients = self.filter_activations(
                gradients,
                filter_method=filter_method,
                threshold_percentile=threshold_percentile,
                topk_percent=topk_percent,
                n_components=n_components,
                min_component_size=min_component_size
            )

        # 移除batch维度 [B, H, W, D, T] -> [H, W, D, T]
        grad_data = gradients.squeeze(0)

        # 时间维度聚合
        if temporal_aggregation == 'mean':
            print(f"  时间聚合: 平均 (T={grad_data.shape[-1]} -> 1)")
            grad_data = grad_data.mean(dim=-1)  # [H, W, D]
        elif temporal_aggregation == 'max':
            print(f"  时间聚合: 最大值 (T={grad_data.shape[-1]} -> 1)")
            grad_data = grad_data.abs().max(dim=-1)[0]  # 取绝对值的最大值
        elif temporal_aggregation == 'sum':
            print(f"  时间聚合: 求和 (T={grad_data.shape[-1]} -> 1)")
            grad_data = grad_data.sum(dim=-1)
        else:
            print(f"  保留时间维度: T={grad_data.shape[-1]}")

        print(f"  聚合后范围: [{grad_data.min():.8f}, {grad_data.max():.8f}]")

        # 3D Gaussian平滑 (去除网格状伪影)
        if smooth_sigma > 0:
            print(f"  应用3D Gaussian平滑 (sigma={smooth_sigma})...")
            grad_np = grad_data.cpu().numpy()
            grad_smoothed = gaussian_filter(grad_np, sigma=smooth_sigma, mode='constant')
            grad_data = torch.from_numpy(grad_smoothed).to(grad_data.device)
            print(f"  平滑后范围: [{grad_data.min():.8f}, {grad_data.max():.8f}]")

        # 缩放梯度到合适区间
        if scaling_method == 'percentile':
            # 使用百分位数缩放到 [-1, 1]
            abs_grad = torch.abs(grad_data)
            p995 = np.percentile(abs_grad.cpu().numpy(), 99.5)
            if p995 > 1e-10:
                scaled = torch.clamp(grad_data / p995, -1, 1)
                print(f"  缩放方法: percentile (99.5%={p995:.8f})")
            else:
                scaled = grad_data
                print(f"  ⚠ 梯度过小，跳过缩放")

        elif scaling_method == 'minmax':
            # 最小-最大缩放到 [-1, 1]
            min_val, max_val = grad_data.min(), grad_data.max()
            range_val = max(abs(min_val), abs(max_val))
            if range_val > 1e-10:
                scaled = grad_data / range_val
                print(f"  缩放方法: minmax (范围={range_val:.8f})")
            else:
                scaled = grad_data
                print(f"  ⚠ 梯度过小，跳过缩放")

        elif scaling_method == 'std':
            # 标准化到均值0，标准差1，然后clip到 [-3, 3]
            mean_val = grad_data.mean()
            std_val = grad_data.std()
            if std_val > 1e-10:
                scaled = (grad_data - mean_val) / std_val
                scaled = torch.clamp(scaled, -3, 3) / 3  # 归一化到 [-1, 1]
                print(f"  缩放方法: std (mean={mean_val:.8f}, std={std_val:.8f})")
            else:
                scaled = grad_data
                print(f"  ⚠ 梯度标准差过小，跳过缩放")

        elif scaling_method == 'abs_percentile':
            # 取绝对值后用百分位数缩放到 [0, 1]
            abs_grad = torch.abs(grad_data)
            p995 = np.percentile(abs_grad.cpu().numpy(), 99)
            if p995 > 1e-10:
                scaled = torch.clamp(abs_grad / p995, 0, 1)
                print(f"  缩放方法: abs_percentile (99.5%={p995:.8f})")
            else:
                scaled = abs_grad
                print(f"  ⚠ 梯度过小，跳过缩放")
        else:
            scaled = grad_data
            print(f"  缩放方法: none (保持原始值)")

        print(f"  缩放后范围: [{scaled.min():.8f}, {scaled.max():.8f}]")

        # 转换为numpy
        scaled_np = scaled.cpu().numpy()

        # 加载原始NIfTI获取affine和header
        original_img = nib.load(original_nifti_path)

        # 创建新的NIfTI图像
        new_img = nib.Nifti1Image(scaled_np, original_img.affine, original_img.header)

        # 保存
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        nib.save(new_img, output_path)
        print(f"✓ 已保存到: {output_path}")
        print(f"  输出形状: {scaled_np.shape}")


def load_model(checkpoint_path: str, device: str):
    """加载模型"""
    print("加载模型...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model', checkpoint.get('model_state_dict', checkpoint))
    
    from mamba_mae.models_vim_mae import VolumeMambaJEPA
    backbone = VolumeMambaJEPA(
        in_chans=1, embed_dim=512, depth=24, num_heads=8, mlp_ratio=4.0,
        qkv_bias=True, norm_layer=torch.nn.LayerNorm, mask_ratio=0.0,
    )
    
    model = MambaJEPAClassifier(backbone=backbone, num_classes=2, head_depth=2,
                                mlp_depth=3, mlp_hidden=1024)
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    print("✓ 模型加载完成")
    return model


def main():
    parser = argparse.ArgumentParser(description='多方法可解释性计算')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点')
    parser.add_argument('--data_path', type=str, help='输入nifti文件路径')
    parser.add_argument('--data_list', type=str, help='包含多个nifti文件路径的txt文件')
    parser.add_argument('--output_dir', type=str, default='/mnt/dataset4/yewh/temp-free-model/explainer_outputs/abide1/connected', help='输出目录')
    parser.add_argument('--methods', type=str, nargs='+',
                       default=['ig', 'inputxgrad'],
                       help='可解释性方法列表 (支持: ig, saliency, inputxgrad, etc.)')
    parser.add_argument('--target_class', type=int, default=1, help='目标类别')
    parser.add_argument('--n_steps', type=int, default=128, help='积分步数(IG/GradShap)')
    parser.add_argument('--device', type=str, default='cuda', help='计算设备')
    parser.add_argument('--baseline', type=str, default='zero',
                       choices=['zero', 'mean', 'random'], help='baseline方法')
    parser.add_argument('--scaling', type=str, default='minmax',
                       choices=['percentile', 'minmax', 'std', 'abs_percentile', 'none'],
                       help='梯度缩放方法')
    parser.add_argument('--temporal_agg', type=str, default='mean',
                       choices=['mean', 'max', 'sum', 'none'],
                       help='时间维度聚合方法')
    parser.add_argument('--smooth_sigma', type=float, default=0.5,
                       help='3D Gaussian平滑sigma值 (0=不平滑, 推荐0.5-2.0)')
    parser.add_argument('--T_prime', type=int, default=30, help='时间patch数量')
    parser.add_argument('--tau_seconds', type=float, default=6.0, help='时间窗口(秒)')
    parser.add_argument('--internal_batch_size', type=int, default=1,
                       help='内部批处理大小(减少显存，建议1-5)')
    parser.add_argument('--memory_efficient', action='store_true',
                       help='启用内存高效模式(进一步减少显存占用)')

    # 激活过滤参数
    parser.add_argument('--filter_method', type=str, default='connected',
                       choices=['threshold', 'topk', 'connected', 'combined', 'none'],
                       help='激活区域过滤方法 (减少热力图噪声)')
    parser.add_argument('--threshold_percentile', type=float, default=90.0,
                       help='阈值过滤的百分位数 (0-100)')
    parser.add_argument('--topk_percent', type=float, default=3.0,
                       help='Top-K 过滤的百分比 (0-100)')
    parser.add_argument('--n_components', type=int, default=5,
                       help='连通域过滤保留的区域数量')
    parser.add_argument('--min_component_size', type=int, default=200,
                       help='最小连通域大小（体素数）')

    args = parser.parse_args()

    # 检查输入参数
    if not args.data_path and not args.data_list:
        parser.error("必须指定 --data_path 或 --data_list 之一")

    # 获取数据文件列表
    if args.data_list:
        print(f"从文件读取数据列表: {args.data_list}")
        with open(args.data_list, 'r') as f:
            data_paths = [line.strip() for line in f if line.strip()]
        print(f"找到 {len(data_paths)} 个数据文件")
    else:
        data_paths = [args.data_path]

    # 加载模型
    model = load_model(args.checkpoint, args.device)

    # 创建解释器
    explainer = MultiExplainer(model, args.device, target_class=args.target_class)

    # 批量处理每个数据文件
    success_count = 0
    fail_count = 0

    for idx, data_path in enumerate(data_paths, 1):
        print(f"\n{'='*80}")
        print(f"处理 [{idx}/{len(data_paths)}]: {data_path}")
        print(f"{'='*80}")

        try:
            # 加载数据
            x_data, meta, orig_Ts, nifti_img = load_single_nifti(
                data_path, T_prime=args.T_prime, tau_seconds=args.tau_seconds
            )

            affine_matrix = torch.from_numpy(nifti_img.affine).float()
            affines = [affine_matrix]

            # 获取文件名（不含扩展名）
            file_name = os.path.basename(data_path).replace('.nii.gz', '').replace('.nii', '')

            # 对每个方法计算
            file_success = True
            for method in args.methods:
                try:
                    print(f"\n方法: {method.upper()}")
                    gradients = explainer.compute(
                        x_data.to(args.device), meta, orig_Ts, affines,
                        method=method, n_steps=args.n_steps, baseline_method=args.baseline,
                        internal_batch_size=args.internal_batch_size,
                        use_memory_efficient=args.memory_efficient
                    )

                    # 确定过滤方法
                    filter_method = None if args.filter_method == 'none' else args.filter_method

                    # 保存结果（以文件名命名）
                    suffix = f"_{method}"
                    if filter_method:
                        suffix += f"_{filter_method}"
                    output_path = os.path.join(args.output_dir, f'{file_name}{suffix}.nii.gz')

                    explainer.save_to_nifti(
                        gradients, data_path, output_path,
                        scaling_method=args.scaling,
                        temporal_aggregation=args.temporal_agg,
                        smooth_sigma=args.smooth_sigma,
                        filter_method=filter_method,
                        threshold_percentile=args.threshold_percentile,
                        topk_percent=args.topk_percent,
                        n_components=args.n_components,
                        min_component_size=args.min_component_size
                    )

                except Exception as e:
                    print(f"✗ 方法 {method} 失败: {e}")
                    file_success = False
                    continue

            if file_success:
                success_count += 1
                print(f"✓ 文件处理成功: {file_name}")
            else:
                fail_count += 1
                print(f"⚠ 文件部分失败: {file_name}")

        except Exception as e:
            print(f"✗ 文件处理失败: {data_path}")
            print(f"  错误: {e}")
            fail_count += 1
            continue

    # 总结
    print(f"\n{'='*80}")
    print(f"批量处理完成！")
    print(f"  成功: {success_count}/{len(data_paths)}")
    print(f"  失败: {fail_count}/{len(data_paths)}")
    print(f"  输出目录: {args.output_dir}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

