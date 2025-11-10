"""
提取MoE特征并进行t-SNE可视化分析
保留所有token，分析时空分辨率聚类情况
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import pickle

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import NiftiTxtDataset
from mamba_mae.models_vim_mae import VolumeMambaJEPA


def collate_fn(batch):
    """自定义collate函数，处理变长时间序列"""
    # 找到最大时间长度
    max_T = max(sample['data'].shape[-1] for sample in batch)
    batch_size = len(batch)
    
    # 初始化
    x = torch.zeros(batch_size, 96, 96, 96, max_T)
    affines = []
    voxels = []
    trs = []
    T_selected = []
    
    for i, sample in enumerate(batch):
        data = sample['data']
        T = data.shape[-1]
        x[i, :, :, :, :T] = data
        affines.append(sample['affine'])
        voxels.append(sample['voxel'])
        trs.append(sample['tr'])
        T_selected.append(sample['T_selected'])
    
    # 构建meta字典
    meta = {}
    for i in range(batch_size):
        meta[i] = {"voxel": voxels[i], "tr": trs[i]}
    
    return {
        'data': x,
        'affine': torch.stack(affines),
        'meta': meta,
        'T_selected': np.array(T_selected),
        'voxel': voxels,
        'tr': trs,
    }


def load_model(checkpoint_path, device):
    """加载预训练模型"""
    print(f"加载模型: {checkpoint_path}")
    
    # 创建模型
    model = VolumeMambaJEPA(
        embed_dim=512,
        depth=24,
        predictor_depth=2,
        ssm_cfg=None,
        encoder_attn_layer_idx=None,
        attn_cfg=None,
        drop_path_rate=0.1,
        norm_epsilon=1e-5,
        rms_norm=False,
        initializer_cfg=None,
        fused_add_norm=True,
        residual_in_fp32=False,
        device=device,
        dtype=None,
        bimamba_type="none",
        if_bimamba=False,
        mixer_type="mamba",
        if_devide_out=True,
        momentum=0.996,
        norm_target=True,
    )
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))
    
    # 尝试加载
    try:
        model.load_state_dict(state_dict, strict=True)
        print("✓ 权重加载成功 (strict=True)")
    except RuntimeError as e:
        print(f"⚠ strict=True失败，尝试strict=False")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"  Missing keys: {len(missing)}")
        print(f"  Unexpected keys: {len(unexpected)}")
    
    model.to(device)
    model.eval()
    return model


def extract_features(model, dataloader, device, max_samples=None):
    """
    提取MoE特征（保留所有token）
    
    返回:
        all_tokens: List[Tensor] - 每个样本的token特征 [L_i, D]
        resolution_info: List[dict] - 每个样本的分辨率信息
    """
    all_tokens = []
    resolution_info = []
    
    total_samples = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="提取特征")):
            # 准备输入
            x = batch['data'].to(device)
            meta = batch['meta']
            orig_Ts = batch['T_selected']
            affines = batch['affine'].to(device)
            
            # 提取MoE特征
            moe_features, attn_mask, lengths, _ = model(
                x, 
                mask_ratio=0.0,  # 不mask，使用所有token
                meta=meta,
                orig_Ts=orig_Ts,
                affines=affines,
                return_moe_features=True
            )
            
            # moe_features: [B, L_max, D]
            # attn_mask: [B, L_max] (True=padding)
            # lengths: List[int]
            
            B = moe_features.size(0)
            
            # 对每个样本，提取有效token
            for i in range(B):
                L_i = lengths[i]
                if L_i > 0:
                    # 提取有效token
                    tokens = moe_features[i, :L_i, :].cpu()  # [L_i, D]
                    all_tokens.append(tokens)
                    
                    # 保存分辨率信息
                    voxel = batch['voxel'][i]
                    tr = batch['tr'][i]
                    resolution_info.append({
                        'voxel': voxel,
                        'tr': tr,
                        'voxel_volume': voxel[0] * voxel[1] * voxel[2],
                        'num_tokens': L_i,
                    })
                    
                    total_samples += 1
                    if max_samples and total_samples >= max_samples:
                        break
            
            if max_samples and total_samples >= max_samples:
                break
    
    print(f"\n✓ 提取完成: {total_samples} 个样本")
    print(f"  总token数: {sum(len(t) for t in all_tokens)}")
    
    return all_tokens, resolution_info


def prepare_for_tsne(all_tokens, resolution_info, strategy='mean'):
    """
    准备t-SNE输入 - 每个样本一个点

    Args:
        all_tokens: List[Tensor] - 每个样本的token [L_i, D]
        resolution_info: List[dict] - 分辨率信息
        strategy: 'mean' | 'max' | 'median'

    Returns:
        features: np.ndarray [N_samples, D]
        labels: np.ndarray [N_samples] - 样本索引
        res_values: dict - 用于着色的分辨率值
    """
    # 每个样本聚合为一个向量
    features_list = []

    for tokens in all_tokens:
        if strategy == 'mean':
            # 平均池化
            feat = tokens.mean(dim=0).numpy()
        elif strategy == 'max':
            # 最大池化
            feat = tokens.max(dim=0)[0].numpy()
        elif strategy == 'median':
            # 中位数池化
            feat = tokens.median(dim=0)[0].numpy()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        features_list.append(feat)

    features = np.stack(features_list, axis=0)  # [N_samples, D]
    labels = np.arange(len(all_tokens))  # 样本索引

    # 提取分辨率信息用于着色
    voxel_volumes = np.array([info['voxel_volume'] for info in resolution_info])
    trs = np.array([info['tr'] for info in resolution_info])
    voxels_x = np.array([info['voxel'][0] for info in resolution_info])
    voxels_y = np.array([info['voxel'][1] for info in resolution_info])
    voxels_z = np.array([info['voxel'][2] for info in resolution_info])

    res_values = {
        'voxel_volume': voxel_volumes,
        'tr': trs,
        'voxel_x': voxels_x,
        'voxel_y': voxels_y,
        'voxel_z': voxels_z,
        'sample_idx': labels,
    }

    print(f"策略: {strategy}池化 (每个样本一个点)")
    print(f"  特征矩阵: {features.shape}")
    print(f"  样本数: {len(all_tokens)}")

    return features, labels, res_values


def main():
    parser = argparse.ArgumentParser(description='提取MoE特征 - 每个样本一个特征向量')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型checkpoint路径')
    parser.add_argument('--data_list', type=str, required=True, help='数据列表txt文件')
    parser.add_argument('--output', type=str, default='moe_features.pkl', help='输出文件')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--max_samples', type=int, default=None, help='最大样本数')
    parser.add_argument('--strategy', type=str, default='mean', choices=['mean', 'max', 'median'],
                        help='样本内token聚合策略: mean(平均), max(最大), median(中位数)')
    parser.add_argument('--device', type=str, default='cuda:0', help='设备')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    model = load_model(args.checkpoint, device)
    
    # 创建数据集
    print(f"\n加载数据集: {args.data_list}")
    dataset = NiftiTxtDataset(
        txt_files=args.data_list,
        return_torch=True,
        memory_map=True,
        cache_meta=True,
        T_prime=15,
        tau_seconds=6.0,
    )
    print(f"  数据集大小: {len(dataset)}")
    
    # 创建dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
    )
    
    # 提取特征
    print("\n开始提取MoE特征...")
    all_tokens, resolution_info = extract_features(
        model, dataloader, device, max_samples=args.max_samples
    )
    
    # 准备t-SNE输入
    print("\n准备t-SNE输入...")
    features, labels, res_values = prepare_for_tsne(
        all_tokens, resolution_info, strategy=args.strategy
    )
    
    # 保存结果
    output_data = {
        'features': features,
        'labels': labels,
        'res_values': res_values,
        'resolution_info': resolution_info,
        'strategy': args.strategy,
    }

    # 创建输出目录
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n保存到: {args.output}")
    with open(args.output, 'wb') as f:
        pickle.dump(output_data, f)
    
    print("\n✓ 完成!")
    print(f"  特征维度: {features.shape}")
    print(f"  唯一分辨率组数: {len(set(tuple(info['voxel']) + (info['tr'],) for info in resolution_info))}")


if __name__ == '__main__':
    main()

