import json
import os
import numpy as np
import matplotlib.pyplot as plt
def read_avg_loss(json_path: str) -> float:
    """读取json文件中的avg_loss值"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('avg_loss', None)

golden_folder = "/mnt/dataset4/yewh/temp-free-model/eval/fsl/gold"
transfer_folder = "/mnt/dataset4/yewh/temp-free-model/eval/fsl/transfer"
#adhd adni cn
# gold_list =np.array([1.379220712184906,1.2791148602962494,1.425817620754242]) 

# # adhd_adni adhd_cn
# # adni_adhd adni cn
# # cn_adhd cn_adni
# transfer_list = np.array([[0,1.383071231842041,1.4128315925598145],
#                  [1.3328654885292053,0,1.353811550140381],
#                  [1.400253164768219,1.3644853115081788,0]])



# # fsl
# #ad mci cn
# gold_list =np.array([1.3421610832214355,1.4787631218249981,1.5497631311416626]) 

# # ad_mci ad_cn
# # mci_ad mci cn
# # cn_ad cn_adni
# transfer_list = np.array([[0,1.4347382692190318,1.532806658744812],
#                  [1.3602228164672852,0,1.5598352193832397],
#                  [1.3769065618515015,1.4786796019627497,0]])


# mni
#ad mci cn
gold_list =np.array([1.5916521946589153,1.4070260524749756,1.6933365265528362]) 

# ad_mci ad_cn
# mci_ad mci cn
# cn_ad cn_adni
transfer_list = np.array([[0,1.4380200703938801 ,1.4045379956563313],
                 [ 1.1913891633351643,0,1.302407701810201],
                 [1.6490651766459148,1.7438360452651978,0]])


# gold_list =np.array([0.9680524667104086,1.1347211599349976,1.3187918066978455]) 

# # ad_mci ad_cn
# # mci_ad mci cn
# # cn_ad cn_adni
# transfer_list = np.array([[0,0.958565870920817,1.0187365611394246],
#                  [1.1534908612569172,0,1.1826779047648113],
#                  [1.3280282020568848,1.2286128997802734,0]])


delta = gold_list - transfer_list
np.fill_diagonal(delta, 0)  # 对角线为0（自身不算迁移）

print("ΔLoss 矩阵：\n", delta)

# === 按列进行 z-score 归一化 ===
mean_col = delta.mean(axis=0, keepdims=True)
std_col  = delta.std(axis=0, keepdims=True) + 1e-8
zscore_mat = (delta - mean_col) / std_col

print("\nZ-score 归一化后矩阵：\n", zscore_mat)

# === 任务标签（按你的顺序：行=源，列=目标）===
# tasks = ["ADHD", "ADNI", "CN"]
tasks = ["AD", "MCI", "CN"]

# （可选）把对角线遮蔽，不参与显示的颜色映射
mask = np.zeros_like(delta, dtype=bool)
np.fill_diagonal(mask, True)

def plot_heatmap(matrix, title, fname, fmt="{:.4f}", vmin=None, vmax=None, cmap="viridis"):
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    data = np.array(matrix, dtype=float).copy()
    # 对被遮蔽元素设为 NaN，以便用 set_bad 颜色
    data_masked = np.ma.array(data, mask=mask)
    cm = plt.cm.get_cmap(cmap).copy()
    cm.set_bad(color="#f0f0f0")  # 对角显示为浅灰

    im = ax.imshow(data_masked, cmap=cm, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("value", rotation=90)

    ax.set_xticks(range(len(tasks)))
    ax.set_yticks(range(len(tasks)))
    ax.set_xticklabels(tasks)
    ax.set_yticklabels(tasks)
    ax.set_xlabel("Target Task")
    ax.set_ylabel("Source Task")
    ax.set_title(title)

    # 数值标注（跳过对角线）
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if i == j: 
                continue
            ax.text(j, i, fmt.format(data[i, j]), ha="center", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    # plt.show()  # 如需交互查看再打开
    plt.close(fig)

# === 绘制并保存 ===
plot_heatmap(
    delta,
    title="ΔLoss = gold_target - transfer(source→target)",
    fname="mni_0.1_transfer_delta_heatmap.png",
    fmt="{:.4f}",
    cmap="RdBu_r"
)

plot_heatmap(
    zscore_mat,
    title="Z-score (column-wise) of ΔLoss",
    fname="mni_0.1transfer_zscore_heatmap.png",
    fmt="{:.2f}",
    cmap="RdBu_r"  # z-score 常用双向色图
)

print("Saved: transfer_delta_heatmap.png, transfer_zscore_heatmap.png")


