import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime

import torch
import seaborn as sns

def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().float().cpu().numpy()

def _imshow_square(ax, arr2d, vlim=None):
    im = ax.imshow(
        arr2d, origin="lower", aspect="auto",
        vmin=None if vlim is None else vlim[0],
        vmax=None if vlim is None else vlim[1],
    )
    try: ax.set_box_aspect(1) 
    except Exception: pass
    ax.set_xlabel("Channels (C)")
    ax.set_ylabel("Tokens (L)")
    return im

def _save_panel_png(arrs, titles, save_path, suptitle=None, share_vrange=False):
    n = len(arrs)
    fig, axes = plt.subplots(1, n, figsize=(3.2*n, 3.2), dpi=140)
    if n == 1: axes = [axes]
    vlim = None
    if share_vrange:
        vmin = min(float(a.min()) for a in arrs)
        vmax = max(float(a.max()) for a in arrs)
        vlim = (vmin, vmax)
    for ax, arr, t in zip(axes, arrs, titles):
        im = _imshow_square(ax, arr, vlim=vlim)
        ax.set_title(t, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    if suptitle: fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout(pad=1.0)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), bbox_inches="tight")
    plt.close(fig)

def _save_single_png(arr, title, save_path, vlim=None):
    """保存单个特征图"""
    fig, ax = plt.subplots(figsize=(6, 5), dpi=140)
    im = _imshow_square(ax, arr, vlim=vlim)
    ax.set_title(title, fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), bbox_inches="tight")
    plt.close(fig)

def _compute_covariance_spectrum(arr):
    """
    计算特征协方差矩阵的特征值谱
    arr: [L, C] 形状的特征
    返回: 排序后的特征值（降序）
    """
    # 中心化
    arr_centered = arr - arr.mean(axis=0, keepdims=True)
    # 计算协方差矩阵 C x C
    cov = np.cov(arr_centered.T)
    # 计算特征值
    eigenvalues = np.linalg.eigvalsh(cov)
    # 降序排序
    eigenvalues = np.sort(eigenvalues)[::-1]
    return eigenvalues

def _save_spectrum_png(eigenvalues, title, save_path):
    """保存协方差谱图"""
    fig, ax = plt.subplots(figsize=(6, 4), dpi=140)
    ax.plot(eigenvalues, 'o-', markersize=3, linewidth=1)
    ax.set_xlabel('Eigenvalue Index')
    ax.set_ylabel('Eigenvalue')
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), bbox_inches="tight")
    plt.close(fig)

def _compute_collapse_metrics(arr):
    """
    计算模型坍缩相关的定量指标
    arr: [L, C] 形状的特征
    返回: 指标字典
    """
    L, C = arr.shape

    # 1. 有效秩 (Effective Rank) - 基于奇异值的熵
    # 参考: Roy & Vetterli (2007)
    arr_centered = arr - arr.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(arr_centered, full_matrices=False)

    # 归一化奇异值
    S_normalized = S / (S.sum() + 1e-10)
    # 计算熵
    entropy = -np.sum(S_normalized * np.log(S_normalized + 1e-10))
    effective_rank = np.exp(entropy)

    # 2. 条件数 (Condition Number) - 最大奇异值 / 最小奇异值
    # 高条件数表示矩阵接近奇异（坍缩）
    condition_number = S[0] / (S[-1] + 1e-10) if len(S) > 0 else 0.0

    # 3. 特征标准差的统计 - 检测特征是否趋向相同值
    feature_stds = arr.std(axis=0)  # [C]
    std_mean = float(feature_stds.mean())
    std_std = float(feature_stds.std())
    std_min = float(feature_stds.min())

    # 4. Token 范数统计 - 检测是否有 token 消失
    token_norms = np.linalg.norm(arr, axis=1)  # [L]
    norm_mean = float(token_norms.mean())
    norm_std = float(token_norms.std())
    norm_min = float(token_norms.min())
    norm_max = float(token_norms.max())

    # 5. 特征多样性 - 基于协方差矩阵的迹和 Frobenius 范数
    cov = np.cov(arr_centered.T)
    trace = np.trace(cov)
    frobenius = np.linalg.norm(cov, 'fro')

    # 6. 奇异值比例 - 前k个奇异值占总能量的比例
    S_cumsum = np.cumsum(S)
    S_total = S.sum()
    top1_ratio = S[0] / S_total if S_total > 0 else 0.0
    top5_ratio = S_cumsum[min(4, len(S)-1)] / S_total if S_total > 0 else 0.0
    top10_ratio = S_cumsum[min(9, len(S)-1)] / S_total if S_total > 0 else 0.0

    # 7. Token 间平均距离 - 检测 token 是否聚集
    # 随机采样以提高效率
    n_samples = min(100, L)
    if L > n_samples:
        indices = np.random.choice(L, n_samples, replace=False)
        arr_sampled = arr[indices]
    else:
        arr_sampled = arr

    # 计算成对欧氏距离
    from scipy.spatial.distance import pdist
    pairwise_dists = pdist(arr_sampled, metric='euclidean')
    dist_mean = float(pairwise_dists.mean())
    dist_std = float(pairwise_dists.std())
    dist_min = float(pairwise_dists.min())

    metrics = {
        # 秩相关
        'effective_rank': float(effective_rank),
        'effective_rank_ratio': float(effective_rank / min(L, C)),  # 归一化到 [0, 1]
        'condition_number': float(condition_number),
        'singular_value_top1_ratio': float(top1_ratio),
        'singular_value_top5_ratio': float(top5_ratio),
        'singular_value_top10_ratio': float(top10_ratio),

        # 特征统计
        'feature_std_mean': std_mean,
        'feature_std_std': std_std,
        'feature_std_min': std_min,

        # Token 范数统计
        'token_norm_mean': norm_mean,
        'token_norm_std': norm_std,
        'token_norm_min': norm_min,
        'token_norm_max': norm_max,
        'token_norm_cv': float(norm_std / (norm_mean + 1e-10)),  # 变异系数

        # 协方差统计
        'covariance_trace': float(trace),
        'covariance_frobenius': float(frobenius),

        # Token 间距离
        'token_distance_mean': dist_mean,
        'token_distance_std': dist_std,
        'token_distance_min': dist_min,
    }

    return metrics


def _save_cosine_similarity_heatmap(arr, title, save_path):
    """
    保存特征的余弦相似度热力图
    arr: [L, C] 形状的特征
    返回: 量化指标字典 {'mean': float, 'std': float, 'min': float, 'max': float}
    """
    # 计算余弦相似度矩阵 [L, L]
    arr_tensor = torch.tensor(arr, dtype=torch.float32)
    cos_sim = torch.nn.functional.cosine_similarity(
        arr_tensor[:, None, :], arr_tensor[None, :, :], dim=-1
    ).numpy()

    # 计算量化指标（排除对角线）
    mask = ~np.eye(cos_sim.shape[0], dtype=bool)
    off_diagonal = cos_sim[mask]
    metrics = {
        'mean': float(off_diagonal.mean()),
        'std': float(off_diagonal.std()),
        'min': float(off_diagonal.min()),
        'max': float(off_diagonal.max()),
    }

    # 绘制热力图，标题中包含统计信息
    fig, ax = plt.subplots(figsize=(7, 6), dpi=140)
    sns.heatmap(cos_sim, cmap='twilight', vmin=0, vmax=1,
                square=True, cbar_kws={'label': 'Cosine Similarity'},
                ax=ax)
    title_with_stats = f"{title}\nMean={metrics['mean']:.4f}, Std={metrics['std']:.4f}"
    ax.set_title(title_with_stats, fontsize=11)
    ax.set_xlabel('Token Index')
    ax.set_ylabel('Token Index')
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), bbox_inches="tight")
    plt.close(fig)

    return metrics


def _compute_summary_statistics(collapse_metrics, cosine_metrics):
    """
    计算所有样本的汇总统计
    """
    summary = {
        "collapse_summary": {},
        "cosine_summary": {},
        "collapse_warnings": [],
    }

    # 汇总坍缩指标
    for feature_type in ["context", "prediction", "target"]:
        # 收集所有样本的指标
        effective_ranks = []
        effective_rank_ratios = []
        condition_numbers = []
        top1_ratios = []
        token_norm_cvs = []

        for sample_key in collapse_metrics:
            metrics = collapse_metrics[sample_key][feature_type]
            effective_ranks.append(metrics['effective_rank'])
            effective_rank_ratios.append(metrics['effective_rank_ratio'])
            condition_numbers.append(metrics['condition_number'])
            top1_ratios.append(metrics['singular_value_top1_ratio'])
            token_norm_cvs.append(metrics['token_norm_cv'])

        summary["collapse_summary"][feature_type] = {
            "effective_rank_mean": float(np.mean(effective_ranks)),
            "effective_rank_std": float(np.std(effective_ranks)),
            "effective_rank_ratio_mean": float(np.mean(effective_rank_ratios)),
            "condition_number_mean": float(np.mean(condition_numbers)),
            "condition_number_max": float(np.max(condition_numbers)),
            "top1_singular_value_ratio_mean": float(np.mean(top1_ratios)),
            "token_norm_cv_mean": float(np.mean(token_norm_cvs)),
        }

        # 检测坍缩警告
        avg_eff_rank_ratio = np.mean(effective_rank_ratios)
        avg_top1_ratio = np.mean(top1_ratios)
        avg_condition = np.mean(condition_numbers)

        if avg_eff_rank_ratio < 0.1:
            summary["collapse_warnings"].append(
                f"{feature_type}: 有效秩比例过低 ({avg_eff_rank_ratio:.4f} < 0.1) - 可能发生严重坍缩"
            )
        elif avg_eff_rank_ratio < 0.3:
            summary["collapse_warnings"].append(
                f"{feature_type}: 有效秩比例较低 ({avg_eff_rank_ratio:.4f} < 0.3) - 可能发生轻微坍缩"
            )

        if avg_top1_ratio > 0.9:
            summary["collapse_warnings"].append(
                f"{feature_type}: 第一奇异值占比过高 ({avg_top1_ratio:.4f} > 0.9) - 特征高度集中"
            )

        if avg_condition > 1000:
            summary["collapse_warnings"].append(
                f"{feature_type}: 条件数过高 ({avg_condition:.2f} > 1000) - 矩阵接近奇异"
            )

    # 汇总余弦相似度指标
    for feature_type in ["context", "prediction", "target"]:
        cosine_means = []
        cosine_stds = []

        for sample_key in cosine_metrics:
            metrics = cosine_metrics[sample_key][feature_type]
            cosine_means.append(metrics['mean'])
            cosine_stds.append(metrics['std'])

        summary["cosine_summary"][feature_type] = {
            "mean_similarity_mean": float(np.mean(cosine_means)),
            "mean_similarity_std": float(np.std(cosine_means)),
            "std_similarity_mean": float(np.mean(cosine_stds)),
        }

        # 检测高相似度警告
        avg_cosine = np.mean(cosine_means)
        if avg_cosine > 0.95:
            summary["collapse_warnings"].append(
                f"{feature_type}: 平均余弦相似度过高 ({avg_cosine:.4f} > 0.95) - token 高度相似"
            )

    return summary


def maybe_visualize_batch(context_features_proj, target_features_proj, pred_target,
                          out_root, max_samples=3, share_vrange_raw=True):
    """
    每次调用都可视化一批样本（最多 max_samples 个）。
    输出目录：{out_root}/batch_{YYYYmmdd_HHMMSS_micro}

    为每个样本输出：
    - 4个单独的特征图：context, pred, target, error
    - 3个协方差谱图：context_spectrum, pred_spectrum, target_spectrum
    - 3个余弦相似度热力图：context_cosine, pred_cosine, target_cosine
    - 1个合并的panel图（可选）

    总计每个样本11个图片文件。
    """
    B = context_features_proj.size(0)
    n_show = min(B, max_samples)

    # 用时间戳做批次目录，避免覆盖
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_dir = Path(out_root) / f"batch_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 先创建初始 meta.json（稍后会更新）
    meta_path = out_dir / "meta.json"
    initial_meta = {
        "timestamp": ts,
        "context_shape": list(context_features_proj.shape),
        "target_shape": list(target_features_proj.shape),
        "pred_shape": list(pred_target.shape),
        "n_show": n_show
    }

    ctx_np = _to_numpy(context_features_proj)
    tgt_np = _to_numpy(target_features_proj)
    prd_np = _to_numpy(pred_target)

    # 存储所有样本的指标
    cosine_metrics = {}
    collapse_metrics = {}

    for b in range(n_show):
        ctx_raw = ctx_np[b]
        tgt_raw = tgt_np[b]
        prd_raw = prd_np[b]
        if prd_raw.shape == tgt_raw.shape:
            err_raw = prd_raw - tgt_raw
        else:
            Lt = min(prd_raw.shape[0], tgt_raw.shape[0])
            err_raw = prd_raw[:Lt] - tgt_raw[:Lt]

        # 计算共享的值域（如果需要）
        vlim = None
        if share_vrange_raw:
            vmin = min(ctx_raw.min(), prd_raw.min(), tgt_raw.min())
            vmax = max(ctx_raw.max(), prd_raw.max(), tgt_raw.max())
            vlim = (vmin, vmax)

        # 单独保存每个特征图
        _save_single_png(ctx_raw, f"Context b{b}",
                        out_dir / f"sample{b}_context.png", vlim=vlim)
        _save_single_png(prd_raw, f"Prediction b{b}",
                        out_dir / f"sample{b}_pred.png", vlim=vlim)
        _save_single_png(tgt_raw, f"Target b{b}",
                        out_dir / f"sample{b}_target.png", vlim=vlim)
        _save_single_png(err_raw, f"Error (Pred - Target) b{b}",
                        out_dir / f"sample{b}_error.png", vlim=None)

        # 计算并保存协方差谱
        ctx_spectrum = _compute_covariance_spectrum(ctx_raw)
        prd_spectrum = _compute_covariance_spectrum(prd_raw)
        tgt_spectrum = _compute_covariance_spectrum(tgt_raw)

        _save_spectrum_png(ctx_spectrum, f"Context Covariance Spectrum b{b}",
                          out_dir / f"sample{b}_context_spectrum.png")
        _save_spectrum_png(prd_spectrum, f"Prediction Covariance Spectrum b{b}",
                          out_dir / f"sample{b}_pred_spectrum.png")
        _save_spectrum_png(tgt_spectrum, f"Target Covariance Spectrum b{b}",
                          out_dir / f"sample{b}_target_spectrum.png")

        # 计算并保存余弦相似度热力图，收集指标
        ctx_cosine_metrics = _save_cosine_similarity_heatmap(
            ctx_raw, f"Context Cosine Similarity b{b}",
            out_dir / f"sample{b}_context_cosine.png")
        prd_cosine_metrics = _save_cosine_similarity_heatmap(
            prd_raw, f"Prediction Cosine Similarity b{b}",
            out_dir / f"sample{b}_pred_cosine.png")
        tgt_cosine_metrics = _save_cosine_similarity_heatmap(
            tgt_raw, f"Target Cosine Similarity b{b}",
            out_dir / f"sample{b}_target_cosine.png")

        # 计算模型坍缩相关指标
        ctx_collapse = _compute_collapse_metrics(ctx_raw)
        prd_collapse = _compute_collapse_metrics(prd_raw)
        tgt_collapse = _compute_collapse_metrics(tgt_raw)

        # 存储该样本的指标
        cosine_metrics[f"sample_{b}"] = {
            "context": ctx_cosine_metrics,
            "prediction": prd_cosine_metrics,
            "target": tgt_cosine_metrics,
        }

        collapse_metrics[f"sample_{b}"] = {
            "context": ctx_collapse,
            "prediction": prd_collapse,
            "target": tgt_collapse,
        }

        # 可选：保存合并的panel图
        _save_panel_png(
            [ctx_raw, prd_raw, tgt_raw, err_raw],
            [f"context b{b}", f"pred b{b}", f"target b{b}", f"error b{b}"],
            out_dir / f"sample{b}_panel.png",
            suptitle=f"sample {b}",
            share_vrange=share_vrange_raw
        )

    # 更新 meta.json，添加所有指标
    initial_meta["cosine_similarity_metrics"] = cosine_metrics
    initial_meta["collapse_metrics"] = collapse_metrics

    # 添加汇总统计
    initial_meta["summary"] = _compute_summary_statistics(collapse_metrics, cosine_metrics)

    with open(meta_path, "w") as f:
        json.dump(initial_meta, f, indent=2)