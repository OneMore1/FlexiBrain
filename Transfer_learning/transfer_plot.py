import os, json, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Rectangle
from pathlib import Path
from typing import Tuple, Optional, List

# === seaborn 仅用于散点图 ===
import seaborn as sns

# 固定行列顺序（与 gold 顺序一致）
DATASETS = ["fsl", "t1", "mni"]
CLASSES  = ["ad", "mci", "cn"]

# 目录名：{src_ds}_adni_{src_cls}-to-(tgt_ds)_adni_(tgt_cls)
DIR_PAT = re.compile(r"^(fsl|t1|mni)_adni_(ad|cn|mci)-to-(fsl|t1|mni)_adni_(ad|cn|mci)$")

# gold（按 CLASSES = ["ad","mci","cn"] 的顺序）
fsl_gold = [1.3421610832214355, 1.4787631218249981, 1.5497631311416626]
t1_gold  = [0.9680524667104086, 1.1347211599349976, 1.3187918066978455]
mni_gold = [1.3083481788635254, 1.567249337832133, 1.592444618542989]

# ========= 配色 =========
def _rgb255(*xs): return tuple([v/255.0 for v in xs])
LOW_BLUE = _rgb255(51,114,188)
HIGH_RED = _rgb255(199,106,122)
MID_WHITE = (1.0, 1.0, 1.0)
DIAG_GRAY = (0.75, 0.75, 0.75)  # 对角线灰色

def make_diverging_cmap(low=LOW_BLUE, high=HIGH_RED, mid=MID_WHITE, name="custom_div"):
    colors = [low, mid, high] if mid is not None else [low, high]
    return LinearSegmentedColormap.from_list(name, colors, N=256)

CUSTOM_CMAP = make_diverging_cmap()  # 蓝-白-红

# ========= 工具 =========
def key_to_label(ds: str, cls: str) -> str:
    return f"{ds}_{cls}"

def label_order() -> list:
    return [key_to_label(ds, cls) for ds in DATASETS for cls in CLASSES]

def parse_dirname(name: str) -> Optional[Tuple[str, str, str, str]]:
    m = DIR_PAT.match(name)
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3), m.group(4)

def read_avg_loss_from_subdir(subdir: Path) -> Optional[float]:
    for p in subdir.glob("*.json"):
        try:
            with p.open("r") as f:
                obj = json.load(f)
            return float(obj.get("avg_loss"))
        except Exception:
            continue
    return None

def build_transfer_matrix(root_dir: str) -> pd.DataFrame:
    labels = label_order()
    mat = np.full((len(labels), len(labels)), np.nan, dtype=float)
    idx_map = {lbl: i for i, lbl in enumerate(labels)}

    root = Path(root_dir)
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        parsed = parse_dirname(entry.name)
        if not parsed:
            continue
        src_ds, src_cls, tgt_ds, tgt_cls = parsed
        val = read_avg_loss_from_subdir(entry)
        if val is None:
            continue
        r = idx_map.get(key_to_label(src_ds, src_cls))
        c = idx_map.get(key_to_label(tgt_ds, tgt_cls))
        if r is not None and c is not None:
            mat[r, c] = val

    return pd.DataFrame(mat, index=labels, columns=labels)

def build_gold_map() -> dict:
    gold_map = {}
    per_ds = {"fsl": fsl_gold, "t1": t1_gold, "mni": mni_gold}
    for ds in DATASETS:
        for i, cls in enumerate(CLASSES):
            gold_map[f"{ds}_{cls}"] = float(per_ds[ds][i])
    return gold_map

def zscore_global(a: np.ndarray) -> np.ndarray:
    mu = np.nanmean(a); sigma = np.nanstd(a)
    if not np.isfinite(sigma) or sigma <= 1e-12:
        return np.full_like(a, np.nan)
    return ((a - mu) / sigma).astype(np.float32)

def minmax_global(a: np.ndarray) -> np.ndarray:
    amin, amax = np.nanmin(a), np.nanmax(a)
    out = np.full_like(a, np.nan, dtype=np.float32)
    if not np.isfinite(amin) or not np.isfinite(amax): return out
    rng = amax - amin
    if rng <= 1e-12:
        out[~np.isnan(a)] = 0.0; return out
    return ((a - amin) / rng).astype(np.float32)

def minmax_global_pm1(a: np.ndarray) -> np.ndarray:
    amin, amax = np.nanmin(a), np.nanmax(a)
    out = np.full_like(a, np.nan, dtype=np.float32)
    if not np.isfinite(amin) or not np.isfinite(amax): return out
    rng = amax - amin
    if rng <= 1e-12:
        out[~np.isnan(a)] = 0.0; return out
    return (2.0 * (a - amin) / rng - 1.0).astype(np.float32)

def symmetric_vmin_vmax(a: np.ndarray) -> Tuple[float, float]:
    absmax = np.nanmax(np.abs(a))
    return -absmax, absmax

def pick_indices(labels: List[str], keep_prefixes=("fsl_", "mni_")) -> List[int]:
    return [i for i, lb in enumerate(labels) if any(lb.startswith(p) for p in keep_prefixes)]

# ========= 对角线覆盖（稳定版）=========
def _overlay_diag_rectangles(ax, n: int, facecolor=DIAG_GRAY, edgecolor="white", lw=0.6):
    # 使用数据坐标，明确边界，置于上层
    for i in range(n):
        ax.add_patch(
            Rectangle((i - 0.5, i - 0.5), 1.0, 1.0,
                      facecolor=facecolor, edgecolor=edgecolor, linewidth=lw,
                      transform=ax.transData, zorder=10, clip_on=False)
        )
    # 保证坐标边界与单元格对齐
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)

# ========= Matplotlib 热图（带对角线灰色覆盖 & 小号 colorbar）=========
def plot_heatmap(mat, labels, title, out_svg, vmin=None, vmax=None,
                 cbar_label="", diag_color=DIAG_GRAY, cmap=CUSTOM_CMAP):
    m = np.ma.masked_invalid(mat)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(m, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, shrink=0.6, aspect=25)
    if cbar_label:
        cbar.ax.set_ylabel(cbar_label, rotation=270, labelpad=12)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(title)

    # 细白网格
    ax.set_xlim(-0.5, len(labels)-0.5)
    ax.set_ylim(len(labels)-0.5, -0.5)
    for x in range(len(labels)+1):
        ax.axhline(x-0.5, color="white", lw=0.6)
        ax.axvline(x-0.5, color="white", lw=0.6)

    # 覆盖对角线颜色（不改变数值）
    _overlay_diag_rectangles(ax, len(labels), facecolor=diag_color, edgecolor="white", lw=0.6)

    plt.tight_layout()
    plt.savefig(out_svg, format="svg")
    plt.close(fig)
    print(f"Saved: {out_svg}")

# ====================== MAIN ======================
if __name__ == "__main__":
    ROOT = "/mnt/dataset4/yewh/temp-free-model/eval/all/transfer/0.1"

    df = build_transfer_matrix(ROOT)
    labels = df.index.tolist()
    mat = df.values.astype(np.float32)

    gold_map = build_gold_map()
    gold_vec = np.array([gold_map[col] for col in df.columns], dtype=np.float32)
    mat_minus = gold_vec[np.newaxis, :] - mat

    # ===== 6×6（fsl + mni） =====
    keep_idx = pick_indices(labels, ("fsl_", "mni_"))
    labels_6 = [labels[i] for i in keep_idx]
    mat_6 = mat[np.ix_(keep_idx, keep_idx)]
    mat_minus_6 = mat_minus[np.ix_(keep_idx, keep_idx)]

    # ---- 保存原始矩阵与 minus 版本 ----
    np.save("transfer_avg_loss_6x6.npy", mat_6)
    pd.DataFrame(mat_6, index=labels_6, columns=labels_6).to_csv("transfer_avg_loss_6x6.csv")

    np.save("transfer_avg_loss_minus_gold_6x6.npy", mat_minus_6)
    pd.DataFrame(mat_minus_6, index=labels_6, columns=labels_6).to_csv("transfer_avg_loss_minus_gold_6x6.csv")

    # ---- 原始 avg_loss（对角线灰色覆盖）
    raw_vmin, raw_vmax = np.nanmin(mat_6), np.nanmax(mat_6)
    plot_heatmap(
        mat_6, labels_6,
        "Transfer avg_loss (6x6 fsl+mni, diag gray)",
        "heatmap_transfer_avg_loss_6x6.svg",
        raw_vmin, raw_vmax, "avg_loss",
        diag_color=DIAG_GRAY, cmap=CUSTOM_CMAP
    )

    # ---- minus gold（对称色阶）
    delta_vmin, delta_vmax = symmetric_vmin_vmax(mat_minus_6)
    plot_heatmap(
        mat_minus_6, labels_6,
        "Transfer avg_loss minus GOLD (6x6, diag gray)",
        "heatmap_transfer_minus_gold_6x6.svg",
        delta_vmin, delta_vmax, "Δ loss vs GOLD",
        diag_color=DIAG_GRAY, cmap=CUSTOM_CMAP
    )

    # ---- 全局 z-score
    z6 = zscore_global(mat_minus_6)
    np.save("transfer_minus_gold_6x6_zscore.npy", z6)
    pd.DataFrame(z6, index=labels_6, columns=labels_6).to_csv("transfer_minus_gold_6x6_zscore.csv")
    plot_heatmap(
        z6, labels_6,
        "Δ loss vs GOLD (6x6) Global Z-score (diag gray)",
        "heatmap_transfer_minus_gold_6x6_zscore.svg",
        *symmetric_vmin_vmax(z6), "Z (global)",
        diag_color=DIAG_GRAY, cmap=CUSTOM_CMAP
    )

    # ---- Min-Max 0–1（双色，无白中点）
    norm6_01 = minmax_global(mat_minus_6)
    np.save("transfer_minus_gold_6x6_minmax01.npy", norm6_01)
    pd.DataFrame(norm6_01, index=labels_6, columns=labels_6).to_csv("transfer_minus_gold_6x6_minmax01.csv")
    plot_heatmap(
        norm6_01, labels_6,
        "Δ loss vs GOLD (6x6) Min-Max 0–1 (diag gray)",
        "heatmap_transfer_minus_gold_6x6_minmax01.svg",
        0.0, 1.0, "Min-Max (0–1)",
        diag_color=DIAG_GRAY, cmap=make_diverging_cmap(low=LOW_BLUE, high=HIGH_RED, mid=None)
    )

    # ---- Min-Max [-1,1]（蓝-白-红）
    norm6_pm1_vis = minmax_global_pm1(mat_minus_6)
    np.save("transfer_minus_gold_6x6_minmax_pm1_vis.npy", norm6_pm1_vis)
    pd.DataFrame(norm6_pm1_vis, index=labels_6, columns=labels_6).to_csv("transfer_minus_gold_6x6_minmax_pm1_vis.csv")
    plot_heatmap(
        norm6_pm1_vis, labels_6,
        "Δ loss vs GOLD (6x6) Min-Max [-1,1] (diag gray)",
        "heatmap_transfer_minus_gold_6x6_minmax_pm1_vis.svg",
        -1.0, 1.0, "Min-Max (-1..1)",
        diag_color=DIAG_GRAY, cmap=CUSTOM_CMAP
    )

    # ===== seaborn 版本（仅作对照，可选）=====
    sns.set_theme(style="whitegrid")
    df_mat = pd.DataFrame(norm6_pm1_vis, index=labels_6, columns=labels_6)

    plt.figure(figsize=(7, 6))
    ax = sns.heatmap(
        df_mat,
        cmap=CUSTOM_CMAP, vmin=-1.0, vmax=1.0, center=0.0,
        square=True, linewidths=0.6, linecolor="white",
        cbar_kws={"shrink": 0.6, "aspect": 25, "label": "Min-Max (-1..1)"},
    )
    ax.set_title("Δ loss vs GOLD (6x6) Min-Max [-1,1] (seaborn, diag gray)")
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    # 覆盖对角线灰色
    # _overlay_diag_rectangles(ax, len(labels_6), facecolor=DIAG_GRAY, edgecolor="white", lw=0.6)

    plt.tight_layout()
    plt.savefig("heatmap_transfer_minus_gold_6x6_minmax_pm1_vis.svg", format="svg")
    plt.close()
    print("Saved: heatmap_transfer_minus_gold_6x6_minmax_pm1_vis.svg")

    # ===== 散点图（保持原逻辑；不转置）=====
    sns.set_theme(style="whitegrid")
    df_scatter = (
        pd.DataFrame(norm6_pm1_vis, index=labels_6, columns=labels_6)
        .stack()
        .reset_index(name="val")
        .rename(columns={"level_0": "src", "level_1": "tgt"})
    )
    g = sns.relplot(
        data=df_scatter,
        x="src", y="tgt", hue="val", size="val",
        palette=CUSTOM_CMAP,
        hue_norm=(-1, 1), edgecolor=".1",
        height=10, sizes=(200, 500), size_norm=(-1, 1)
    )
    g.set(xlabel="", ylabel="", aspect="equal")
    g.despine(left=True, bottom=True)
    g.ax.margins(.02)
    for label in g.ax.get_xticklabels():
        label.set_rotation(90)

    g.fig.savefig("heat_scatter_transfer_minus_gold_6x6_minmax_pm1_vis.svg", format="svg")
    plt.close(g.fig)
    print("Saved: heat_scatter_transfer_minus_gold_6x6_minmax_pm1_vis.svg")
