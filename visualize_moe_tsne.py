"""
t-SNE可视化MoE特征，分析时空分辨率聚类
"""

import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
from pathlib import Path
import matplotlib as mpl


def compute_tsne(features, perplexity=30, n_iter=1000, random_state=42, max_samples=10000):
    """计算t-SNE降维"""
    print(f"\n计算t-SNE...")
    print(f"  输入维度: {features.shape}")

    # 如果数据点太多，进行采样
    sample_indices = None
    if len(features) > max_samples:
        print(f"  数据点过多({len(features)})，采样到{max_samples}个点")
        np.random.seed(random_state)
        sample_indices = np.random.choice(len(features), max_samples, replace=False)
        features_sampled = features[sample_indices]
    else:
        features_sampled = features
        sample_indices = np.arange(len(features))

    # 调整perplexity
    max_perplexity = (len(features_sampled) - 1) // 3
    perplexity = min(perplexity, max_perplexity)
    print(f"  perplexity: {perplexity}")
    print(f"  实际处理: {len(features_sampled)} 个点")

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=200,
        n_iter=n_iter,
        random_state=random_state,
        verbose=1,
        n_jobs=4,
    )

    coords_2d = tsne.fit_transform(features_sampled)
    print(f"✓ t-SNE完成: {coords_2d.shape}")

    return coords_2d, sample_indices


def create_resolution_groups(resolution_info):
    """创建分辨率分组标签"""
    unique_groups = {}
    group_labels = []
    
    for info in resolution_info:
        # 使用(voxel, tr)作为唯一键
        voxel = info['voxel']
        tr = info['tr']
        key = (round(voxel[0], 2), round(voxel[1], 2), round(voxel[2], 2), round(tr, 2))
        
        if key not in unique_groups:
            unique_groups[key] = {
                'id': len(unique_groups),
                'voxel': voxel,
                'tr': tr,
                'count': 0,
            }
        
        unique_groups[key]['count'] += 1
        group_labels.append(unique_groups[key]['id'])
    
    return group_labels, unique_groups


def plot_tsne_multi(coords_2d, res_values, resolution_info, output_dir):
    """创建多子图t-SNE可视化 - 每个点代表一个样本"""

    # 创建分辨率分组
    group_labels, unique_groups = create_resolution_groups(resolution_info)
    group_labels = np.array(group_labels)
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('MoE Features t-SNE Visualization', fontsize=16, y=0.995)
    
    # 子图1: 按空间分辨率着色
    ax = axes[0, 0]
    voxel_volumes = res_values['voxel_volume']
    scatter = ax.scatter(
        coords_2d[:, 0], coords_2d[:, 1],
        c=voxel_volumes, cmap='twilight', s=50, alpha=0.7,
        edgecolors='black', linewidths=0.5
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Voxel Volume (mm³)', fontsize=10)
    ax.set_title('Colored by Spatial Resolution', fontsize=12)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.grid(True, alpha=0.3)

    # 子图2: 按时间分辨率(TR)着色
    ax = axes[0, 1]
    trs = res_values['tr']
    scatter = ax.scatter(
        coords_2d[:, 0], coords_2d[:, 1],
        c=trs, cmap='twilight', s=50, alpha=0.7,
        edgecolors='black', linewidths=0.5
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('TR (seconds)', fontsize=10)
    ax.set_title('Colored by Temporal Resolution', fontsize=12)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.grid(True, alpha=0.3)
    
    # 子图3: 按分辨率组着色（离散）
    ax = axes[1, 0]
    n_groups = len(unique_groups)
    cmap = plt.cm.get_cmap('tab20' if n_groups <= 20 else 'hsv', n_groups)  # 颜色映射

    for group_id in range(n_groups):
        mask = group_labels == group_id
        if mask.sum() > 0:
            ax.scatter(
                coords_2d[mask, 0], coords_2d[mask, 1],
                c=[cmap(group_id)], s=50, alpha=0.7,
                label=f'Group {group_id}',
                edgecolors='black', linewidths=0.5
            )

    ax.set_title(f'Colored by Resolution Groups (n={n_groups})', fontsize=12)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.grid(True, alpha=0.3)
    if n_groups <= 10:
        ax.legend(fontsize=8, loc='best', markerscale=1.5)
    
    # 子图4: 样本标注图
    ax = axes[1, 1]

    # 按组着色，并标注样本编号
    for group_id in range(n_groups):
        mask = group_labels == group_id
        if mask.sum() > 0:
            ax.scatter(
                coords_2d[mask, 0], coords_2d[mask, 1],
                c=[cmap(group_id)], s=100, alpha=0.7,
                edgecolors='black', linewidths=0.5
            )
            # 标注样本编号（如果样本不太多）
            if len(coords_2d) <= 50:
                indices = np.where(mask)[0]
                for idx in indices:
                    ax.annotate(str(idx), (coords_2d[idx, 0], coords_2d[idx, 1]),
                               fontsize=6, alpha=0.7)

    ax.set_title('Sample Indices', fontsize=12)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()

    # 保存合并图
    output_path = Path(output_dir) / 'tsne_multi_view.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 保存合并图像: {output_path}")

    plt.close()

    # 单独保存每个子图
    # 子图1: 空间分辨率
    fig1, ax1 = plt.subplots(figsize=(8, 7))
    scatter1 = ax1.scatter(coords_2d[:, 0], coords_2d[:, 1],
        c=voxel_volumes, cmap='twilight', s=50, alpha=0.7,
        edgecolors='black', linewidths=0.5)
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('Voxel Volume (mm³)', fontsize=10)
    ax1.set_title('Colored by Spatial Resolution', fontsize=12)
    ax1.set_xlabel('t-SNE 1')
    ax1.set_ylabel('t-SNE 2')
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'tsne_view_1_spatial.png', dpi=300, bbox_inches='tight')
    print(f"✓ 保存单独图像 1: tsne_view_1_spatial.png")
    plt.close()

    # 子图2: 时间分辨率
    fig2, ax2 = plt.subplots(figsize=(8, 7))
    scatter2 = ax2.scatter(coords_2d[:, 0], coords_2d[:, 1],
        c=trs, cmap='twilight', s=50, alpha=0.7,
        edgecolors='black', linewidths=0.5)
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('TR (seconds)', fontsize=10)
    ax2.set_title('Colored by Temporal Resolution', fontsize=12)
    ax2.set_xlabel('t-SNE 1')
    ax2.set_ylabel('t-SNE 2')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'tsne_view_2_temporal.png', dpi=300, bbox_inches='tight')
    print(f"✓ 保存单独图像 2: tsne_view_2_temporal.png")
    plt.close()

    # 子图3: 分辨率组（使用离散化的 twilight colorbar）
    fig3, ax3 = plt.subplots(figsize=(8, 7))
    # 创建离散化的 twilight 色表
    cmap_twilight = plt.cm.get_cmap('twilight', n_groups)
    boundaries = np.arange(n_groups + 1) - 0.5
    norm = BoundaryNorm(boundaries, cmap_twilight.N)

    scatter3 = ax3.scatter(coords_2d[:, 0], coords_2d[:, 1],
        c=group_labels, cmap=cmap_twilight, norm=norm, s=50, alpha=0.7,
        edgecolors='black', linewidths=0.5)

    # 添加 colorbar
    sm = ScalarMappable(cmap=cmap_twilight, norm=norm)
    sm.set_array([])
    cbar3 = plt.colorbar(sm, ax=ax3, ticks=np.arange(n_groups))
    cbar3.set_label('Group ID', fontsize=10)

    ax3.set_title(f'Colored by Resolution Groups (n={n_groups})', fontsize=12)
    ax3.set_xlabel('t-SNE 1')
    ax3.set_ylabel('t-SNE 2')
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'tsne_view_3_groups.png', dpi=300, bbox_inches='tight')
    print(f"✓ 保存单独图像 3: tsne_view_3_groups.png")
    plt.close()

    # 子图4: 样本标注（使用离散化的 twilight colorbar）
    fig4, ax4 = plt.subplots(figsize=(8, 7))
    # 创建离散化的 twilight 色表
    cmap_twilight = plt.cm.get_cmap('twilight', n_groups)
    boundaries = np.arange(n_groups + 1) - 0.5
    norm = BoundaryNorm(boundaries, cmap_twilight.N)

    scatter4 = ax4.scatter(coords_2d[:, 0], coords_2d[:, 1],
        c=group_labels, cmap=cmap_twilight, norm=norm, s=100, alpha=0.7,
        edgecolors='black', linewidths=0.5)

    # 添加样本标注（如果样本不太多）
    if len(coords_2d) <= 50:
        for idx in range(len(coords_2d)):
            ax4.annotate(str(idx), (coords_2d[idx, 0], coords_2d[idx, 1]),
                       fontsize=6, alpha=0.7)

    # 添加 colorbar
    sm = ScalarMappable(cmap=cmap_twilight, norm=norm)
    sm.set_array([])
    cbar4 = plt.colorbar(sm, ax=ax4, ticks=np.arange(n_groups))
    cbar4.set_label('Group ID', fontsize=10)

    ax4.set_title('Sample Indices', fontsize=12)
    ax4.set_xlabel('t-SNE 1')
    ax4.set_ylabel('t-SNE 2')
    ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'tsne_view_4_indices.png', dpi=300, bbox_inches='tight')
    print(f"✓ 保存单独图像 4: tsne_view_4_indices.png")
    plt.close()
    
    # 打印分组信息
    print("\n分辨率分组信息:")
    for key, info in sorted(unique_groups.items(), key=lambda x: x[1]['id']):
        voxel = info['voxel']
        print(f"  Group {info['id']}: voxel={voxel}, TR={info['tr']:.2f}s, count={info['count']}")


def compute_clustering_metrics(features, group_labels):
    """计算聚类质量指标"""
    print("\n计算聚类指标...")
    
    # 轮廓系数
    if len(set(group_labels)) > 1:
        silhouette = silhouette_score(features, group_labels)
        print(f"  Silhouette Score: {silhouette:.4f}")
    else:
        silhouette = None
        print(f"  Silhouette Score: N/A (只有一个组)")
    
    # 组内/组间距离
    unique_groups = list(set(group_labels))
    intra_dists = []
    inter_dists = []
    
    for i in range(len(features)):
        for j in range(i+1, min(i+1000, len(features))):  # 采样以节省时间
            dist = np.linalg.norm(features[i] - features[j])
            if group_labels[i] == group_labels[j]:
                intra_dists.append(dist)
            else:
                inter_dists.append(dist)
    
    if intra_dists and inter_dists:
        intra_mean = np.mean(intra_dists)
        inter_mean = np.mean(inter_dists)
        ratio = intra_mean / inter_mean
        print(f"  组内平均距离: {intra_mean:.4f}")
        print(f"  组间平均距离: {inter_mean:.4f}")
        print(f"  组内/组间比率: {ratio:.4f} {'(良好聚类)' if ratio < 1 else '(聚类较弱)'}")
    
    # 最近邻分析
    k = min(10, len(features) - 1)
    knn = NearestNeighbors(n_neighbors=k+1)
    knn.fit(features)
    distances, indices = knn.kneighbors(features)
    
    # 计算最近邻中同组的比例
    same_group_ratios = []
    for i in range(len(features)):
        neighbors = indices[i, 1:]  # 排除自己
        same_group = sum(group_labels[n] == group_labels[i] for n in neighbors)
        same_group_ratios.append(same_group / k)
    
    print(f"  最近邻同组比例: {np.mean(same_group_ratios):.4f} (随机为 {1/len(unique_groups):.4f})")
    
    return {
        'silhouette': silhouette,
        'intra_dist': intra_mean if intra_dists else None,
        'inter_dist': inter_mean if inter_dists else None,
        'same_group_ratio': np.mean(same_group_ratios),
    }


def main():
    parser = argparse.ArgumentParser(description='t-SNE可视化MoE特征 - 每个点代表一个样本')
    parser.add_argument('--input', type=str, required=True, help='特征文件(.pkl)')
    parser.add_argument('--output_dir', type=str, default='tsne_results', help='输出目录')
    parser.add_argument('--perplexity', type=int, default=10, help='t-SNE perplexity')
    parser.add_argument('--n_iter', type=int, default=1000, help='t-SNE迭代次数')
    parser.add_argument('--max_samples', type=int, default=10000, help='最大样本数（超过则采样）')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 加载特征
    print(f"加载特征: {args.input}")
    with open(args.input, 'rb') as f:
        data = pickle.load(f)
    
    features = data['features']
    labels = data['labels']
    res_values = data['res_values']
    resolution_info = data['resolution_info']
    strategy = data['strategy']
    
    print(f"  策略: {strategy}")
    print(f"  特征形状: {features.shape}")
    print(f"  样本数: {len(resolution_info)}")
    
    # 计算t-SNE
    coords_2d, sample_indices = compute_tsne(
        features,
        perplexity=args.perplexity,
        n_iter=args.n_iter,
        max_samples=args.max_samples
    )

    # 如果进行了采样，更新相关数据
    if len(sample_indices) < len(features):
        res_values = {k: v[sample_indices] for k, v in res_values.items()}
        resolution_info = [resolution_info[i] for i in sample_indices]
    
    # 保存t-SNE结果
    tsne_output = output_dir / 'tsne_coords.pkl'
    with open(tsne_output, 'wb') as f:
        pickle.dump({
            'coords_2d': coords_2d,
            'res_values': res_values,
            'labels': labels,
            'resolution_info': resolution_info,
        }, f)
    print(f"✓ 保存t-SNE坐标: {tsne_output}")
    
    # 可视化
    print("\n创建可视化...")
    plot_tsne_multi(coords_2d, res_values, resolution_info, output_dir)

    # 计算聚类指标
    group_labels, _ = create_resolution_groups(resolution_info)
    group_labels = np.array(group_labels)

    # 使用采样后的特征计算指标
    features_for_metrics = features[sample_indices] if len(sample_indices) < len(features) else features
    metrics = compute_clustering_metrics(features_for_metrics, group_labels)
    
    # 保存指标
    metrics_output = output_dir / 'clustering_metrics.txt'
    with open(metrics_output, 'w') as f:
        f.write("聚类质量指标\n")
        f.write("=" * 50 + "\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    print(f"✓ 保存指标: {metrics_output}")
    
    print("\n✓ 完成!")


if __name__ == '__main__':
    main()

