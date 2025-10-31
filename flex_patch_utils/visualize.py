import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime

import torch

def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().float().cpu().numpy()

def _norm_2d(arr: np.ndarray, mode: str = "z"):
    if mode == "z":
        mean = arr.mean(axis=0, keepdims=True)
        std = arr.std(axis=0, keepdims=True) + 1e-6
        return (arr - mean) / std
    elif mode == "minmax":
        mn = arr.min(axis=0, keepdims=True)
        mx = arr.max(axis=0, keepdims=True)
        return (arr - mn) / (mx - mn + 1e-6)
    return arr

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


def maybe_visualize_batch(context_features_proj, target_features_proj, pred_target,
                          out_root, max_samples=3,
                          norm_mode="z", also_raw=True, share_vrange_raw=True):
    """
    每次调用都可视化一批样本（最多 max_samples 个）。
    输出目录：{out_root}/batch_{YYYYmmdd_HHMMSS_micro}
    """
    # 不再基于 every/batch_idx 控制，改为每次都画
    B = context_features_proj.size(0)
    n_show = min(B, max_samples)

    # 用时间戳做批次目录，避免覆盖
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_dir = Path(out_root) / f"batch_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "meta.json", "w") as f:
        json.dump({
            "timestamp": ts,
            "context_shape": list(context_features_proj.shape),
            "target_shape": list(target_features_proj.shape),
            "pred_shape": list(pred_target.shape),
            "norm_mode": norm_mode,
            "n_show": n_show
        }, f, indent=2)

    ctx_np = _to_numpy(context_features_proj)
    tgt_np = _to_numpy(target_features_proj)
    prd_np = _to_numpy(pred_target)

    for b in range(n_show):
        ctx_raw = ctx_np[b]
        tgt_raw = tgt_np[b]
        prd_raw = prd_np[b]
        if prd_raw.shape == tgt_raw.shape:
            err_raw = prd_raw - tgt_raw
        else:
            Lt = min(prd_raw.shape[0], tgt_raw.shape[0])
            err_raw = prd_raw[:Lt] - tgt_raw[:Lt]

        if norm_mode not in (None, "raw"):
            ctx = _norm_2d(ctx_raw, norm_mode)
            tgt = _norm_2d(tgt_raw, norm_mode)
            prd = _norm_2d(prd_raw, norm_mode)
            if prd.shape == tgt.shape:
                err = prd - tgt
            else:
                Lt = min(prd.shape[0], tgt.shape[0])
                err = prd[:Lt] - tgt[:Lt]

            _save_panel_png(
                [ctx, prd, tgt, err],
                [f"context b{b}", f"pred b{b}", f"target b{b}", f"pred-target b{b}"],
                out_dir / f"sample{b}_panel_norm.png",
                suptitle=f"sample {b} (norm={norm_mode})",
                share_vrange=False
            )

        if also_raw:
            _save_panel_png(
                [ctx_raw, prd_raw, tgt_raw, err_raw],
                [f"context b{b} (raw)", f"pred b{b} (raw)", f"target b{b} (raw)", f"pred-target b{b} (raw)"],
                out_dir / f"sample{b}_panel_raw.png",
                suptitle=f"sample {b} (raw)",
                share_vrange=share_vrange_raw
            )