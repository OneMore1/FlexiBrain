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

    # 存储所有样本的余弦相似度指标
    cosine_metrics = {}

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

        # 存储该样本的余弦相似度指标
        cosine_metrics[f"sample_{b}"] = {
            "context": ctx_cosine_metrics,
            "prediction": prd_cosine_metrics,
            "target": tgt_cosine_metrics,
        }

        # 可选：保存合并的panel图
        _save_panel_png(
            [ctx_raw, prd_raw, tgt_raw, err_raw],
            [f"context b{b}", f"pred b{b}", f"target b{b}", f"error b{b}"],
            out_dir / f"sample{b}_panel.png",
            suptitle=f"sample {b}",
            share_vrange=share_vrange_raw
        )

    # 更新 meta.json，添加余弦相似度指标
    initial_meta["cosine_similarity_metrics"] = cosine_metrics
    with open(meta_path, "w") as f:
        json.dump(initial_meta, f, indent=2)