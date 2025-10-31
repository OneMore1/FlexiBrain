#!/usr/bin/env python3
"""
使用官方Captum IntegratedGradients的端到端可解释性实现
更稳定、更高效的IG计算
"""

import torch
import numpy as np
import nibabel as nib
import argparse
import os
from typing import Dict, Callable
from captum.attr import IntegratedGradients
from downstream_utils.mamba import MambaJEPAClassifier, MambaJEPAClassifierAvgPool
from dataset import NiftiTxtDataset
import tempfile


def load_single_nifti(nifti_path: str, T_prime: int = None, tau_seconds: float = 6.0):
    """使用dataset.py加载单个NIfTI文件，支持时间选择逻辑"""
    # 创建临时txt文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(nifti_path + '\n')
        temp_txt = f.name

    try:
        # 使用NiftiTxtDataset加载，包含时间选择逻辑
        dataset = NiftiTxtDataset(
            [temp_txt],
            return_torch=True,
            memory_map=True,
            T_prime=T_prime,  # 目标时间patch数量
            tau_seconds=tau_seconds  # 时间窗口大小
        )
        sample = dataset[0]

        # 提取数据
        x_data = sample['data'].unsqueeze(0)  # 添加batch维度 [1, H, W, D, T]

        # 创建meta信息
        # meta需要是字典格式: {subject_idx: {"voxel": (x,y,z), "tr": float}}
        meta = {0: {"voxel": sample['voxel'], "tr": sample['tr']}}

        # 原始时间步长（这里是T_selected，经过时间选择后的长度）
        orig_Ts = np.array([sample['T_selected']])

        # 加载原始nifti图像用于保存
        nifti_img = nib.load(nifti_path)

        print(f"  数据形状: {x_data.shape}")
        print(f"  Voxel size: {sample['voxel']}")
        print(f"  TR: {sample['tr']}")
        if T_prime is not None:
            print(f"  时间选择: T_total={nifti_img.get_fdata().shape[-1]} -> T_selected={sample['T_selected']} (T_prime={T_prime})")
            kt = max(1, round(tau_seconds / sample['tr']))
            print(f"  时间patch参数: kt={kt}, tau_seconds={tau_seconds}")

        return x_data, meta, orig_Ts, nifti_img

    finally:
        # 清理临时文件
        os.unlink(temp_txt)


class CaptumIG:
    """使用官方Captum IntegratedGradients的端到端可解释性"""
    
    def __init__(self, model, device, target_class=1):
        self.model = model
        self.device = device
        self.target_class = target_class
        
        # 存储模型前向传播所需的额外参数
        self.meta = None
        self.orig_Ts = None
        self.affines = None
        
    def _forward_func(self, x: torch.Tensor) -> torch.Tensor:
        """
        Captum要求的前向函数格式
        只接受输入张量，返回目标类别的logits
        """
        # 确保输入需要梯度
        x = x.requires_grad_(True)

        # 获取当前batch_size（Captum会改变batch_size）
        current_batch_size = x.shape[0]

        # 动态扩展meta以匹配当前batch_size
        current_meta = {}
        for i in range(current_batch_size):
            current_meta[i] = self.meta[0]  # 复制原始meta信息

        # 动态扩展orig_Ts以匹配当前batch_size
        current_orig_Ts = np.tile(self.orig_Ts, current_batch_size)

        # 动态扩展affines以匹配当前batch_size
        if isinstance(self.affines, list):
            current_affines = self.affines * current_batch_size
        else:
            current_affines = [self.affines[0]] * current_batch_size

        with torch.enable_grad():
            # 使用explain_mode=True确保整个过程可导
            logits = self.model.forward(
                x,
                meta=current_meta,
                orig_Ts=current_orig_Ts,
                affines=current_affines,
                inference_params=None,
                explain_mode=True
            )
            # 返回目标类别的logits
            return logits[:, self.target_class]
    
    def compute_ig(self, x: torch.Tensor, meta: Dict, orig_Ts: np.ndarray,
                   affines: list, n_steps: int = 50, baseline_method: str = 'mean',
                   internal_batch_size: int = None, use_cpu_offload: bool = False) -> torch.Tensor:
        """
        使用Captum计算Integrated Gradients

        Args:
            x: 输入数据 [B, H, W, D, T]
            meta: 元信息
            orig_Ts: 原始时间步长
            affines: 仿射矩阵列表
            n_steps: IG积分步数
            baseline_method: baseline方法 ('zero', 'mean', 'random')
            internal_batch_size: Captum内部批处理大小，用于内存优化
            use_cpu_offload: 是否使用CPU offloading减少显存

        Returns:
            gradients: IG梯度 [B, H, W, D, T]
        """
        print(f"开始Captum IG计算 (n_steps={n_steps})...")

        # 存储模型前向传播所需的参数
        self.meta = meta
        self.orig_Ts = orig_Ts
        self.affines = affines

        # 选择baseline
        if baseline_method == 'zero':
            baseline = torch.zeros_like(x)
        elif baseline_method == 'mean':
            baseline = torch.full_like(x, x.mean().item())
        elif baseline_method == 'random':
            baseline = torch.randn_like(x) * x.std().item()
        else:
            raise ValueError(f"未知的baseline方法: {baseline_method}")

        print(f"使用{baseline_method} baseline (值={baseline.mean().item():.6f})")

        # 如果没有指定internal_batch_size，自动设置一个保守的值
        if internal_batch_size is None:
            internal_batch_size = max(1, n_steps // 10)  # 默认将步数分成10批
            print(f"自动设置 internal_batch_size={internal_batch_size}")
        else:
            print(f"使用指定的 internal_batch_size={internal_batch_size}")

        # 创建IntegratedGradients对象
        ig = IntegratedGradients(self._forward_func)

        try:
            # 如果使用CPU offloading，先将数据移到CPU
            if use_cpu_offload:
                print("使用CPU offloading模式...")
                return self._compute_ig_with_cpu_offload(x, baseline, n_steps, internal_batch_size)

            # 计算integrated gradients
            # Captum会自动处理积分过程和梯度计算
            attributions = ig.attribute(
                inputs=x,
                baselines=baseline,
                n_steps=n_steps,
                internal_batch_size=internal_batch_size,  # 用于内存优化
                return_convergence_delta=False  # 不返回收敛性指标
            )

            print(f"IG完成")
            print(f"梯度范围: [{attributions.min():.8f}, {attributions.max():.8f}]")
            print(f"非零比例: {torch.count_nonzero(attributions).item() / attributions.numel():.4f}")

            return attributions.detach()

        except Exception as e:
            print(f"Captum IG计算失败: {e}")
            raise  # 直接抛出异常，不再回退到手动实现

    def _compute_ig_with_cpu_offload(self, x: torch.Tensor, baseline: torch.Tensor,
                                     n_steps: int, internal_batch_size: int) -> torch.Tensor:
        """使用CPU offloading计算IG，减少显存占用"""
        print("使用CPU offloading计算IG...")

        integrated_gradients = torch.zeros_like(x).cpu()  # 在CPU上累积

        for batch_start in range(0, n_steps, internal_batch_size):
            batch_end = min(batch_start + internal_batch_size, n_steps)
            batch_size = batch_end - batch_start

            print(f"处理步骤 {batch_start+1}-{batch_end}/{n_steps}...")

            # 为这一批创建插值输入
            alphas = torch.linspace(batch_start/n_steps, batch_end/n_steps, batch_size+1)[1:]

            batch_grads = torch.zeros_like(x).cpu()

            for alpha in alphas:
                x_interp = baseline + alpha * (x - baseline)
                x_interp = x_interp.requires_grad_(True)

                # 前向传播
                score = self._forward_func(x_interp)
                score.sum().backward()

                # 将梯度移到CPU累积
                if x_interp.grad is not None:
                    batch_grads += x_interp.grad.cpu()

                # 清理GPU显存
                del x_interp, score
                torch.cuda.empty_cache()

            integrated_gradients += batch_grads

        # 计算最终梯度
        final_gradients = integrated_gradients * (x.cpu() - baseline.cpu()) / n_steps

        return final_gradients.to(self.device)
    
    def compute_ig_with_convergence(self, x: torch.Tensor, meta: Dict, orig_Ts: np.ndarray,
                                   affines: list, n_steps: int = 50, baseline_method: str = 'mean',
                                   internal_batch_size: int = None) -> tuple:
        """
        计算IG并返回收敛性指标

        Returns:
            tuple: (attributions, convergence_delta)
        """
        print(f"开始Captum IG计算（含收敛性检查）...")

        # 存储模型前向传播所需的参数
        self.meta = meta
        self.orig_Ts = orig_Ts
        self.affines = affines

        # 选择baseline
        if baseline_method == 'zero':
            baseline = torch.zeros_like(x)
        elif baseline_method == 'mean':
            baseline = torch.full_like(x, x.mean().item())
        elif baseline_method == 'random':
            baseline = torch.randn_like(x) * x.std().item()
        else:
            raise ValueError(f"未知的baseline方法: {baseline_method}")

        print(f"使用{baseline_method} baseline (值={baseline.mean().item():.6f})")

        # 如果没有指定internal_batch_size，自动设置一个保守的值
        if internal_batch_size is None:
            internal_batch_size = max(1, n_steps // 10)
            print(f"自动设置 internal_batch_size={internal_batch_size}")

        # 创建IntegratedGradients对象
        ig = IntegratedGradients(self._forward_func)

        # 计算integrated gradients with convergence delta
        attributions, convergence_delta = ig.attribute(
            inputs=x,
            baselines=baseline,
            n_steps=n_steps,
            internal_batch_size=internal_batch_size,
            return_convergence_delta=True
        )

        print(f"IG完成，收敛性指标: {convergence_delta.item():.8f}")
        print(f"梯度范围: [{attributions.min():.8f}, {attributions.max():.8f}]")

        return attributions.detach(), convergence_delta.item()
    
    def scale_gradients(self, gradients: torch.Tensor, method: str = 'percentile') -> torch.Tensor:
        """梯度缩放用于可视化"""
        if method == 'none':
            return gradients
            
        print(f"使用{method}方法缩放梯度...")
        abs_grad = torch.abs(gradients)
        
        if method == 'percentile':
            # 基于99.9%百分位数缩放
            abs_grad_np = abs_grad.cpu().numpy()
            p999 = np.percentile(abs_grad_np, 99.9)
            scaled = torch.clamp(abs_grad / p999, 0, 1)
            scaled = torch.sign(gradients) * scaled
            print(f"  99.9%分位数: {p999:.8f} -> 1.0")
            
        elif method == 'minmax':
            # 最小-最大缩放到[-1, 1]
            min_val = torch.min(gradients)
            max_val = torch.max(gradients)
            range_val = max_val - min_val
            scaled = 2 * (gradients - min_val) / (range_val + 1e-8) - 1
            print(f"  范围: [{min_val:.8f}, {max_val:.8f}] -> [-1, 1]")
            
        else:
            raise ValueError(f"未知的缩放方法: {method}")
        
        print(f"  缩放后范围: [{scaled.min():.6f}, {scaled.max():.6f}]")
        return scaled
    
    def save_to_nifti(self, gradients: torch.Tensor, original_nifti_path: str, 
                      output_path: str, scaling_method: str = 'percentile'):
        """保存梯度到NIfTI文件"""
        print("保存梯度到NIfTI文件...")
        
        # 缩放梯度
        scaled_gradients = self.scale_gradients(gradients, scaling_method)
        
        # 加载原始NIfTI文件获取header和affine
        original_img = nib.load(original_nifti_path)
        
        # 转换为numpy并调整维度顺序
        grad_data = scaled_gradients.squeeze(0).cpu().numpy()  # 移除batch维度
        
        # 创建新的NIfTI图像
        new_img = nib.Nifti1Image(grad_data, original_img.affine, original_img.header)
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存
        nib.save(new_img, output_path)
        print(f"✓ 梯度已保存到: {output_path}")


def load_model(checkpoint_path: str, device: str) -> torch.nn.Module:
    """加载模型"""
    print("加载模型...")

    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 检测模型状态字典的键名
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
        print("使用模型状态字典键名: model")
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print("使用模型状态字典键名: model_state_dict")
    else:
        state_dict = checkpoint
        print("直接使用checkpoint作为状态字典")

    # 创建backbone
    from mamba_mae.models_vim_mae import VolumeMambaJEPA
    backbone = VolumeMambaJEPA(
        in_chans=1,
        embed_dim=512,
        depth=24,
        num_heads=8,    # 调整为匹配512维度
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=torch.nn.LayerNorm,
        mask_ratio=0.6,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        decoder_mlp_ratio=4.0,
        norm_pix_loss=False
    )

    # 创建分类器
    model = MambaJEPAClassifier(
        backbone=backbone,
        num_classes=2,
        head_depth=2,
        mlp_depth=3,
        mlp_hidden=1024
    )

    # 加载权重
    try:
        model.load_state_dict(state_dict, strict=True)
        print("✓ 所有模型权重加载成功 (strict=True)")
    except Exception as e:
        print(f"⚠ strict=True失败，尝试strict=False进行版本兼容")
        print(f"  错误信息: {e}")
        model.load_state_dict(state_dict, strict=False)
        print("✓ 模型权重加载完成 (strict=False)")

    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description='使用Captum的端到端IG梯度计算')
    parser.add_argument('--checkpoint', type=str, required=True, help='分类模型检查点')
    parser.add_argument('--data_path', type=str, required=True, help='输入nifti文件路径')
    parser.add_argument('--output_dir', type=str, default='./captum_ig_outputs', help='输出目录')
    parser.add_argument('--target_class', type=int, default=1, help='目标类别')
    parser.add_argument('--n_steps', type=int, default=50, help='IG积分步数')
    parser.add_argument('--device', type=str, default='cuda', help='计算设备')
    parser.add_argument('--baseline', type=str, default='mean',
                       choices=['zero', 'mean', 'random'], help='baseline方法')
    parser.add_argument('--scaling', type=str, default='percentile',
                       choices=['percentile', 'minmax', 'none'], help='梯度缩放方法')
    parser.add_argument('--T_prime', type=int, default=30,
                       help='目标时间patch数量（用于时间选择）')
    parser.add_argument('--tau_seconds', type=float, default=6.0,
                       help='时间窗口大小（秒）')
    parser.add_argument('--internal_batch_size', type=int, default=None,
                       help='Captum内部批处理大小（用于内存优化，建议设置为5-10）')
    parser.add_argument('--check_convergence', action='store_true',
                       help='检查IG收敛性')
    parser.add_argument('--use_cpu_offload', action='store_true',
                       help='使用CPU offloading减少显存占用（速度较慢但更省显存）')

    args = parser.parse_args()

    # 加载模型
    model = load_model(args.checkpoint, args.device)

    # 加载数据
    print(f"加载数据: {args.data_path}")
    x_data, meta, orig_Ts, nifti_img = load_single_nifti(
        args.data_path,
        T_prime=args.T_prime,
        tau_seconds=args.tau_seconds
    )

    # 准备affine矩阵
    affine_matrix = torch.from_numpy(nifti_img.affine).float()
    affines = [affine_matrix]

    # 创建Captum IG计算器
    ig_calculator = CaptumIG(model, args.device, target_class=args.target_class)

    # 计算IG
    if args.check_convergence:
        print("计算IG（含收敛性检查）...")
        gradients, convergence_delta = ig_calculator.compute_ig_with_convergence(
            x_data.to(args.device), meta, orig_Ts, affines,
            n_steps=args.n_steps, baseline_method=args.baseline,
            internal_batch_size=args.internal_batch_size
        )

        # 保存收敛性信息
        convergence_info = {
            'convergence_delta': convergence_delta,
            'n_steps': args.n_steps,
            'baseline_method': args.baseline,
            'target_class': args.target_class
        }

        import json
        convergence_path = os.path.join(args.output_dir, 'convergence_info.json')
        os.makedirs(args.output_dir, exist_ok=True)
        with open(convergence_path, 'w') as f:
            json.dump(convergence_info, f, indent=2)
        print(f"✓ 收敛性信息已保存到: {convergence_path}")

    else:
        print("计算IG...")
        gradients = ig_calculator.compute_ig(
            x_data.to(args.device), meta, orig_Ts, affines,
            n_steps=args.n_steps, baseline_method=args.baseline,
            internal_batch_size=args.internal_batch_size
        )

    # 保存结果
    output_path = os.path.join(args.output_dir, 'captum_ig_gradients.nii.gz')
    ig_calculator.save_to_nifti(gradients, args.data_path, output_path, args.scaling)

    print("完成！")
    print(f"输出目录: {args.output_dir}")


if __name__ == '__main__':
    main()
