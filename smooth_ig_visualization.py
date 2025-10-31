#!/usr/bin/env python3
"""
平滑IG梯度可视化，消除patch网格效应
"""

import nibabel as nib
import numpy as np
from scipy import ndimage
import argparse
import os

def smooth_patch_gradients(gradients, patch_size=(3, 3, 4), method='gaussian', sigma=1.0):
    """
    平滑patch级梯度，消除网格效应
    
    Args:
        gradients: [H, W, D, T] 梯度数组
        patch_size: (kx, ky, kz) patch大小
        method: 平滑方法 ('gaussian', 'uniform', 'median')
        sigma: 高斯滤波的标准差
    
    Returns:
        smoothed_gradients: 平滑后的梯度
    """
    print(f"使用{method}方法平滑梯度，patch大小: {patch_size}")
    
    H, W, D, T = gradients.shape
    kx, ky, kz = patch_size
    
    smoothed = np.zeros_like(gradients)
    
    for t in range(T):
        time_slice = gradients[:, :, :, t]
        
        if method == 'gaussian':
            # 3D高斯滤波
            smoothed_slice = ndimage.gaussian_filter(time_slice, sigma=sigma)
            
        elif method == 'uniform':
            # 均匀滤波（移动平均）
            kernel_size = max(kx, ky, kz)
            smoothed_slice = ndimage.uniform_filter(time_slice, size=kernel_size)
            
        elif method == 'median':
            # 中值滤波
            kernel_size = max(kx, ky, kz)
            smoothed_slice = ndimage.median_filter(time_slice, size=kernel_size)
            
        elif method == 'bilateral':
            # 双边滤波（保边平滑）
            from skimage import restoration
            smoothed_slice = restoration.denoise_bilateral(
                time_slice, sigma_color=0.1, sigma_spatial=sigma
            )
            
        else:
            raise ValueError(f"未知的平滑方法: {method}")
        
        smoothed[:, :, :, t] = smoothed_slice
    
    return smoothed

def interpolate_patch_boundaries(gradients, patch_size=(3, 3, 4), method='linear'):
    """
    在patch边界处进行插值，减少突变
    
    Args:
        gradients: [H, W, D, T] 梯度数组
        patch_size: (kx, ky, kz) patch大小
        method: 插值方法
    
    Returns:
        interpolated_gradients: 插值后的梯度
    """
    print(f"在patch边界处进行{method}插值")
    
    H, W, D, T = gradients.shape
    kx, ky, kz = patch_size
    
    result = gradients.copy()
    
    for t in range(T):
        time_slice = result[:, :, :, t]
        
        # X方向边界插值
        for i in range(kx, H, kx):
            if i < H - 1:
                # 在边界处插值
                left_val = time_slice[i-1, :, :]
                right_val = time_slice[i, :, :]
                time_slice[i, :, :] = (left_val + right_val) / 2
        
        # Y方向边界插值
        for j in range(ky, W, ky):
            if j < W - 1:
                up_val = time_slice[:, j-1, :]
                down_val = time_slice[:, j, :]
                time_slice[:, j, :] = (up_val + down_val) / 2
        
        # Z方向边界插值
        for k in range(kz, D, kz):
            if k < D - 1:
                front_val = time_slice[:, :, k-1]
                back_val = time_slice[:, :, k]
                time_slice[:, :, k] = (front_val + back_val) / 2
        
        result[:, :, :, t] = time_slice
    
    return result

def create_super_resolution_gradients(gradients, patch_size=(3, 3, 4), scale_factor=2):
    """
    创建超分辨率梯度图，提高空间分辨率
    
    Args:
        gradients: [H, W, D, T] 梯度数组
        patch_size: (kx, ky, kz) patch大小
        scale_factor: 上采样倍数
    
    Returns:
        sr_gradients: 超分辨率梯度
    """
    print(f"创建{scale_factor}x超分辨率梯度图")
    
    H, W, D, T = gradients.shape
    kx, ky, kz = patch_size
    
    # 计算patch级梯度
    Lx, Ly, Lz = H//kx, W//ky, D//kz
    patch_gradients = np.zeros((Lx, Ly, Lz, T))
    
    for i in range(Lx):
        for j in range(Ly):
            for k in range(Lz):
                patch_data = gradients[i*kx:(i+1)*kx, j*ky:(j+1)*ky, k*kz:(k+1)*kz, :]
                patch_gradients[i, j, k, :] = np.mean(patch_data, axis=(0, 1, 2))
    
    # 上采样patch级梯度
    new_shape = (Lx * scale_factor, Ly * scale_factor, Lz * scale_factor, T)
    sr_patch_gradients = np.zeros(new_shape)
    
    for t in range(T):
        sr_patch_gradients[:, :, :, t] = ndimage.zoom(
            patch_gradients[:, :, :, t], 
            (scale_factor, scale_factor, scale_factor), 
            order=1  # 线性插值
        )
    
    # 扩展回原始分辨率
    final_kx = kx // scale_factor if kx >= scale_factor else 1
    final_ky = ky // scale_factor if ky >= scale_factor else 1
    final_kz = kz // scale_factor if kz >= scale_factor else 1
    
    sr_gradients = np.repeat(np.repeat(np.repeat(
        sr_patch_gradients, final_kx, axis=0), final_ky, axis=1), final_kz, axis=2)
    
    # 裁剪到原始大小
    sr_gradients = sr_gradients[:H, :W, :D, :]
    
    return sr_gradients

def analyze_patch_structure(gradients, patch_size=(3, 3, 4)):
    """
    分析梯度的patch结构
    """
    print("=== Patch结构分析 ===")
    
    H, W, D, T = gradients.shape
    kx, ky, kz = patch_size
    Lx, Ly, Lz = H//kx, W//ky, D//kz
    
    # 检查patch内一致性
    patch_consistency = []
    for t in range(min(5, T)):  # 只检查前5个时间点
        time_slice = gradients[:, :, :, t]
        consistencies = []
        
        for i in range(Lx):
            for j in range(Ly):
                for k in range(Lz):
                    patch_data = time_slice[i*kx:(i+1)*kx, j*ky:(j+1)*ky, k*kz:(k+1)*kz]
                    if patch_data.size > 0:
                        patch_std = np.std(patch_data)
                        patch_mean = np.abs(np.mean(patch_data))
                        if patch_mean > 1e-10:
                            consistency = 1 - (patch_std / patch_mean)
                            consistencies.append(consistency)
        
        if consistencies:
            patch_consistency.append(np.mean(consistencies))
    
    avg_consistency = np.mean(patch_consistency) if patch_consistency else 0
    print(f"Patch内一致性: {avg_consistency:.4f} (1.0=完全一致)")
    
    # 检查边界跳跃
    boundary_jumps = []
    time_slice = gradients[:, :, :, gradients.shape[3]//2]  # 中间时间点
    
    # X方向边界
    for i in range(kx, H, kx):
        if i < H - 1:
            left_patch = np.mean(time_slice[i-kx:i, :, :])
            right_patch = np.mean(time_slice[i:i+kx, :, :])
            boundary_jumps.append(abs(right_patch - left_patch))
    
    avg_jump = np.mean(boundary_jumps) if boundary_jumps else 0
    print(f"边界平均跳跃: {avg_jump:.8f}")
    
    return avg_consistency, avg_jump

def main():
    parser = argparse.ArgumentParser(description='平滑IG梯度可视化')
    parser.add_argument('--input', type=str, default='ig_outputs/ig_grad_sample_000.nii.gz', 
                       help='输入IG梯度文件')
    parser.add_argument('--output_dir', type=str, default='ig_outputs/smoothed', 
                       help='输出目录')
    parser.add_argument('--methods', nargs='+', default=['gaussian', 'bilateral'], 
                       choices=['gaussian', 'uniform', 'median', 'bilateral', 'interpolate', 'super_res'],
                       help='平滑方法')
    parser.add_argument('--sigma', type=float, default=1.0, help='高斯滤波标准差')
    parser.add_argument('--patch_size', nargs=3, type=int, default=[3, 3, 4], 
                       help='Patch大小 (kx ky kz)')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载梯度数据
    print(f"加载梯度文件: {args.input}")
    img = nib.load(args.input)
    gradients = img.get_fdata()
    
    print(f"梯度形状: {gradients.shape}")
    print(f"梯度范围: [{gradients.min():.6f}, {gradients.max():.6f}]")
    
    # 分析原始patch结构
    consistency, jump = analyze_patch_structure(gradients, tuple(args.patch_size))
    
    # 应用不同的平滑方法
    for method in args.methods:
        print(f"\n=== 应用{method}平滑 ===")
        
        if method == 'interpolate':
            smoothed = interpolate_patch_boundaries(gradients, tuple(args.patch_size))
        elif method == 'super_res':
            smoothed = create_super_resolution_gradients(gradients, tuple(args.patch_size))
        else:
            smoothed = smooth_patch_gradients(gradients, tuple(args.patch_size), method, args.sigma)
        
        # 分析平滑后的结果
        smooth_consistency, smooth_jump = analyze_patch_structure(smoothed, tuple(args.patch_size))
        
        print(f"平滑前: 一致性={consistency:.4f}, 边界跳跃={jump:.8f}")
        print(f"平滑后: 一致性={smooth_consistency:.4f}, 边界跳跃={smooth_jump:.8f}")
        
        improvement = (jump - smooth_jump) / jump * 100 if jump > 0 else 0
        print(f"边界跳跃改善: {improvement:.1f}%")
        
        # 保存结果
        output_path = os.path.join(args.output_dir, f'ig_grad_smoothed_{method}.nii.gz')
        smoothed_img = nib.Nifti1Image(smoothed, img.affine, img.header)
        nib.save(smoothed_img, output_path)
        print(f"保存到: {output_path}")
    
    print(f"\n所有平滑结果保存在: {args.output_dir}")

if __name__ == '__main__':
    main()
