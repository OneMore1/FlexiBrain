"""
测试脚本：统计 patch embed 中删除背景操作的效果
分析不同样本删除了多少百分比的 patches
"""

import sys
import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import NiftiTxtDataset
from flex_patchembed_timetospace import STAPE4D_TimeToSpace


def custom_collate_fn(batch):
    """自定义 collate 函数，处理不同形状的数据"""
    return batch


class BackgroundRemovalAnalyzer:
    """分析背景删除统计"""
    
    def __init__(self, patch_embed: STAPE4D_TimeToSpace, device: str = 'cuda'):
        self.patch_embed = patch_embed.to(device)
        self.device = device
        self.stats = []
        
    def analyze_sample(self, sample: Dict) -> Dict:
        """分析单个样本的背景删除情况"""
        data = sample['data']  # [X, Y, Z, T]
        voxel = sample['voxel']  # (vx, vy, vz)
        tr = sample['tr']
        affine = sample['affine']  # [4, 4]

        # 转换为 torch tensor 并添加 batch 维度
        if not isinstance(data, torch.Tensor):
            data = torch.from_numpy(data)
        if not isinstance(affine, torch.Tensor):
            affine = torch.from_numpy(affine)

        # 保存原始数据用于验证
        data_cpu = data.clone()

        # [X, Y, Z, T] -> [1, X, Y, Z, T]
        x = data.unsqueeze(0).to(self.device)

        # 准备 meta 信息（按照 forward 函数的要求）
        meta = {
            0: {
                'voxel': voxel,
                'tr': tr,
            }
        }

        # 准备 affines 列表
        affines = [affine.to(self.device)]

        # 调用 patch_embed 的 forward，需要获取 grid_info 来知道哪些 patch 被删除
        with torch.no_grad():
            # 正常模式：删除背景，获取 grid_info
            tokens_normal, attn_mask_normal, lengths_normal, grid_info_normal, pos_normal = self.patch_embed(
                x, meta=meta, affines=affines, orig_Ts=None,
                return_grid_info=True, explain_mode=False
            )

            # 解释模式：保留所有 patches，获取 grid_info
            tokens_explain, attn_mask_explain, lengths_explain, grid_info_explain, pos_explain = self.patch_embed(
                x, meta=meta, affines=affines, orig_Ts=None,
                return_grid_info=True, explain_mode=True
            )

        # 统计信息
        total_patches = lengths_explain[0]  # 解释模式下的总 patch 数
        kept_patches = lengths_normal[0]    # 正常模式下保留的 patch 数
        removed_patches = total_patches - kept_patches
        removal_ratio = removed_patches / total_patches if total_patches > 0 else 0.0

        # 从 grid_info 获取 keep_mask
        grid_normal = grid_info_normal[0]
        grid_explain = grid_info_explain[0]
        keep_mask = grid_normal['keep_mask']  # [Lx*Ly*Lz] bool

        # 获取 patch 参数
        kx, ky, kz = grid_normal['kx'], grid_normal['ky'], grid_normal['kz']
        Lx, Ly, Lz = grid_normal['Lx'], grid_normal['Ly'], grid_normal['Lz']

        # 检查原始数据中被删除的 patch 区域是否为0
        # 使用与 _spatial_keep_mask_alltime 相同的逻辑
        T_true = data_cpu.shape[-1]
        X, Y, Z = 96, 96, 96

        # 检查每个 patch 在原始数据中是否全为0
        patch_is_zero_in_data = []
        patch_token_norms = []

        all_tokens = tokens_explain[0, :total_patches].cpu()  # [total_patches, embed_dim]

        for patch_idx in range(Lx * Ly * Lz):
            # 计算 patch 在 3D 网格中的位置
            lz = patch_idx % Lz
            ly = (patch_idx // Lz) % Ly
            lx = patch_idx // (Ly * Lz)

            # 提取原始数据中对应的 patch 区域
            x_start, x_end = lx * kx, (lx + 1) * kx
            y_start, y_end = ly * ky, (ly + 1) * ky
            z_start, z_end = lz * kz, (lz + 1) * kz

            patch_data = data_cpu[x_start:x_end, y_start:y_end, z_start:z_end, :T_true]

            # 检查是否全为0
            is_zero = (patch_data == 0).all().item()
            patch_is_zero_in_data.append(is_zero)

            # 计算 token 范数
            token_norm = torch.norm(all_tokens[patch_idx], p=2).item()
            patch_token_norms.append(token_norm)

        # 统计被删除和保留的 patches
        kept_indices = torch.nonzero(keep_mask, as_tuple=False).squeeze(-1).tolist()
        removed_indices = torch.nonzero(~keep_mask, as_tuple=False).squeeze(-1).tolist()

        # 统计被删除的 patches 中有多少在原始数据中是零
        removed_zero_in_data = sum(1 for idx in removed_indices if patch_is_zero_in_data[idx])
        removed_nonzero_in_data = len(removed_indices) - removed_zero_in_data

        # 统计被保留的 patches 中有多少在原始数据中是零
        kept_zero_in_data = sum(1 for idx in kept_indices if patch_is_zero_in_data[idx])
        kept_nonzero_in_data = len(kept_indices) - kept_zero_in_data

        # Token 范数统计
        kept_token_norms = [patch_token_norms[idx] for idx in kept_indices]
        removed_token_norms = [patch_token_norms[idx] for idx in removed_indices]

        result = {
            'path': str(sample['path']),
            'subject_idx': sample['subject_idx'],
            'voxel': voxel,
            'tr': tr,
            'data_shape': tuple(data.shape),
            'total_patches': int(total_patches),
            'kept_patches': int(kept_patches),
            'removed_patches': int(removed_patches),
            'removal_ratio': float(removal_ratio),
            'removal_percentage': float(removal_ratio * 100),
            # 新增：原始数据中的零 patch 统计
            'total_zero_patches_in_data': int(sum(patch_is_zero_in_data)),
            'total_nonzero_patches_in_data': int(len(patch_is_zero_in_data) - sum(patch_is_zero_in_data)),
            'removed_zero_patches_in_data': int(removed_zero_in_data),
            'removed_nonzero_patches_in_data': int(removed_nonzero_in_data),
            'kept_zero_patches_in_data': int(kept_zero_in_data),
            'kept_nonzero_patches_in_data': int(kept_nonzero_in_data),
            'removed_are_all_zero_in_data': removed_nonzero_in_data == 0,
            # 范数统计
            'kept_token_norm_mean': float(np.mean(kept_token_norms)) if kept_token_norms else 0.0,
            'kept_token_norm_std': float(np.std(kept_token_norms)) if kept_token_norms else 0.0,
            'removed_token_norm_mean': float(np.mean(removed_token_norms)) if removed_token_norms else 0.0,
            'removed_token_norm_std': float(np.std(removed_token_norms)) if removed_token_norms else 0.0,
        }

        return result
    
    def analyze_dataset(self, dataloader: DataLoader, max_samples: int = None) -> List[Dict]:
        """分析整个数据集"""
        results = []
        
        pbar = tqdm(dataloader, desc="Analyzing samples")
        for batch_idx, batch in enumerate(pbar):
            if max_samples is not None and batch_idx >= max_samples:
                break
            
            # batch 是一个列表，每个元素是一个样本
            for sample in batch:
                try:
                    result = self.analyze_sample(sample)
                    results.append(result)
                    
                    # 更新进度条显示
                    pbar.set_postfix({
                        'removed': f"{result['removal_percentage']:.1f}%",
                        'kept': result['kept_patches'],
                        'total': result['total_patches']
                    })
                    
                except Exception as e:
                    print(f"\nError processing {sample.get('path', 'unknown')}: {e}")
                    continue
        
        return results
    
    def print_summary(self, results: List[Dict]):
        """打印统计摘要"""
        if not results:
            print("No results to summarize.")
            return

        removal_percentages = [r['removal_percentage'] for r in results]
        kept_patches = [r['kept_patches'] for r in results]
        total_patches = [r['total_patches'] for r in results]

        # 零 patch 统计（基于原始数据）
        total_zero_patches = [r['total_zero_patches_in_data'] for r in results]
        removed_zero_patches = [r['removed_zero_patches_in_data'] for r in results]
        removed_nonzero_patches = [r['removed_nonzero_patches_in_data'] for r in results]
        kept_zero_patches = [r['kept_zero_patches_in_data'] for r in results]
        kept_nonzero_patches = [r['kept_nonzero_patches_in_data'] for r in results]
        all_removed_are_zero = all(r['removed_are_all_zero_in_data'] for r in results)

        print("\n" + "="*80)
        print("背景删除统计摘要")
        print("="*80)
        print(f"总样本数: {len(results)}")

        print(f"\n删除比例统计:")
        print(f"  平均删除: {np.mean(removal_percentages):.2f}%")
        print(f"  中位数删除: {np.median(removal_percentages):.2f}%")
        print(f"  标准差: {np.std(removal_percentages):.2f}%")
        print(f"  最小删除: {np.min(removal_percentages):.2f}%")
        print(f"  最大删除: {np.max(removal_percentages):.2f}%")

        print(f"\nPatch 数量统计:")
        print(f"  平均总 patches: {np.mean(total_patches):.1f}")
        print(f"  平均保留 patches: {np.mean(kept_patches):.1f}")
        print(f"  平均删除 patches: {np.mean(total_patches) - np.mean(kept_patches):.1f}")

        print(f"\n原始数据中的零 Patch 统计（验证删除正确性）:")
        print(f"  平均总零 patches（原始数据）: {np.mean(total_zero_patches):.1f}")
        print(f"  平均删除的零 patches（原始数据）: {np.mean(removed_zero_patches):.1f}")
        print(f"  平均删除的非零 patches（原始数据）: {np.mean(removed_nonzero_patches):.1f}")
        print(f"  平均保留的零 patches（原始数据）: {np.mean(kept_zero_patches):.1f}")
        print(f"  平均保留的非零 patches（原始数据）: {np.mean(kept_nonzero_patches):.1f}")
        print(f"  所有样本的删除都是零 patch（原始数据）: {'是' if all_removed_are_zero else '否'}")

        # 检查是否有误删非零 patch（原始数据中非零）
        samples_with_nonzero_removed = sum(1 for r in results if r['removed_nonzero_patches_in_data'] > 0)
        if samples_with_nonzero_removed > 0:
            print(f"  ⚠ 警告: {samples_with_nonzero_removed}/{len(results)} 个样本删除了原始数据中非零的 patches!")
            print(f"\n  误删非零 patch 的样本:")
            for r in results:
                if r['removed_nonzero_patches_in_data'] > 0:
                    print(f"    - {Path(r['path']).name}: 删除了 {r['removed_nonzero_patches_in_data']} 个非零 patches")
                    print(f"      删除 patch 的平均 token 范数: {r['removed_token_norm_mean']:.6f}")
        else:
            print(f"  ✓ 所有删除的 patches 在原始数据中都是零 patch（背景）")

        # Token 范数统计
        kept_norms_mean = [r['kept_token_norm_mean'] for r in results]
        removed_norms_mean = [r['removed_token_norm_mean'] for r in results]
        print(f"\nToken 范数统计:")
        print(f"  保留 patches 的平均范数: {np.mean(kept_norms_mean):.6f} ± {np.std(kept_norms_mean):.6f}")
        print(f"  删除 patches 的平均范数: {np.mean(removed_norms_mean):.6f} ± {np.std(removed_norms_mean):.6f}")

        # 分位数统计
        print(f"\n删除比例分位数:")
        for q in [10, 25, 50, 75, 90, 95, 99]:
            val = np.percentile(removal_percentages, q)
            print(f"  {q}th percentile: {val:.2f}%")

        # 找出删除最多和最少的样本
        print(f"\n删除最多的 5 个样本:")
        sorted_results = sorted(results, key=lambda x: x['removal_ratio'], reverse=True)
        for i, r in enumerate(sorted_results[:5]):
            print(f"  {i+1}. {Path(r['path']).name}: {r['removal_percentage']:.2f}% "
                  f"(kept {r['kept_patches']}/{r['total_patches']}, "
                  f"removed_nonzero_in_data={r['removed_nonzero_patches_in_data']})")

        print(f"\n删除最少的 5 个样本:")
        for i, r in enumerate(sorted_results[-5:]):
            print(f"  {i+1}. {Path(r['path']).name}: {r['removal_percentage']:.2f}% "
                  f"(kept {r['kept_patches']}/{r['total_patches']}, "
                  f"removed_nonzero_in_data={r['removed_nonzero_patches_in_data']})")

        print("="*80)


def main():
    parser = argparse.ArgumentParser(description='统计 patch embed 背景删除效果')
    parser.add_argument('--train_list', type=str, 
                       default='/mnt/dataset4/DATASETS/fsl_fmri/split/train.txt',
                       help='训练集列表文件')
    parser.add_argument('--val_list', type=str,
                       default='/mnt/dataset4/DATASETS/fsl_fmri/split/val.txt',
                       help='验证集列表文件')
    parser.add_argument('--dataset', type=str, default='train', choices=['train', 'val', 'both'],
                       help='分析哪个数据集')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='最大分析样本数（None=全部）')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载线程数')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备 (cuda/cpu)')
    parser.add_argument('--T_prime', type=int, default=15,
                       help='目标时间 patch 数量')
    parser.add_argument('--tau_seconds', type=float, default=6.0,
                       help='时间窗口（秒）')
    parser.add_argument('--output', type=str, default='background_removal_stats.json',
                       help='输出 JSON 文件路径')
    
    args = parser.parse_args()
    
    # 创建 patch_embed
    print("创建 STAPE4D_TimeToSpace...")
    patch_embed = STAPE4D_TimeToSpace(
        d_mid=16,
        d_out=256,  # 使用与模型相同的配置
        kt_base=6,
        kx_base=6,
        ky_base=6,
        kz_base=6,
        tau_seconds=args.tau_seconds,
        rho_mm=(12.0, 12.0, 12.0),
    )
    
    # 创建分析器
    analyzer = BackgroundRemovalAnalyzer(patch_embed, device=args.device)
    
    # 加载数据集
    datasets_to_analyze = []
    if args.dataset in ['train', 'both']:
        print(f"\n加载训练集: {args.train_list}")
        train_set = NiftiTxtDataset(
            txt_files=args.train_list,
            return_torch=True,
            memory_map=True,
            cache_meta=True,
            T_prime=args.T_prime,
            tau_seconds=args.tau_seconds,
        )
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=custom_collate_fn,
        )
        datasets_to_analyze.append(('train', train_loader))
    
    if args.dataset in ['val', 'both']:
        print(f"\n加载验证集: {args.val_list}")
        val_set = NiftiTxtDataset(
            txt_files=args.val_list,
            return_torch=True,
            memory_map=True,
            cache_meta=True,
            T_prime=args.T_prime,
            tau_seconds=args.tau_seconds,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=custom_collate_fn,
        )
        datasets_to_analyze.append(('val', val_loader))
    
    # 分析每个数据集
    all_results = {}
    for dataset_name, dataloader in datasets_to_analyze:
        print(f"\n{'='*80}")
        print(f"分析 {dataset_name.upper()} 数据集")
        print(f"{'='*80}")
        
        results = analyzer.analyze_dataset(dataloader, max_samples=args.max_samples)
        all_results[dataset_name] = results
        
        analyzer.print_summary(results)
    
    # 保存结果到 JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n结果已保存到: {output_path}")


if __name__ == '__main__':
    main()

