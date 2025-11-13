#!/bin/bash
# MoE特征提取与t-SNE可视化分析脚本

# 配置参数
CHECKPOINT="/mnt/dataset4/yewh/temp-free-model/checkpoints/mamba_moe1/checkpoint_best.pt"  # 修改为你的checkpoint路径
DATA_LIST="/mnt/dataset4/DATASETS/fsl_fmri/split/train.txt"  # 修改为你的数据列表
OUTPUT_DIR="moe_tsne_results/perx150_lr200_twlight"
BATCH_SIZE=4
MAX_SAMPLES=2000  # 限制样本数以加快速度，设为None使用全部数据
STRATEGY="mean"  # 'mean' 平均池化

# 步骤1: 提取MoE特征
echo "========================================="
echo "步骤1: 提取MoE特征"
echo "========================================="

python extract_moe_features.py \
    --checkpoint ${CHECKPOINT} \
    --data_list ${DATA_LIST} \
    --output ${OUTPUT_DIR}/moe_features.pkl \
    --batch_size ${BATCH_SIZE} \
    --max_samples ${MAX_SAMPLES} \
    --strategy ${STRATEGY} \
    --device cuda:0

if [ $? -ne 0 ]; then
    echo "错误: 特征提取失败"
    exit 1
fi

# 步骤2: t-SNE可视化
echo ""
echo "========================================="
echo "步骤2: t-SNE可视化"
echo "========================================="

python visualize_moe_tsne.py \
    --input ${OUTPUT_DIR}/moe_features.pkl \
    --output_dir ${OUTPUT_DIR} \
    --perplexity 30 \
    --n_iter 1000

if [ $? -ne 0 ]; then
    echo "错误: 可视化失败"
    exit 1
fi

echo ""
echo "========================================="
echo "✓ 完成! 结果保存在: ${OUTPUT_DIR}"
echo "========================================="
echo "  - moe_features.pkl: 提取的特征"
echo "  - tsne_coords.pkl: t-SNE坐标"
echo "  - tsne_multi_view.png: 可视化图像"
echo "  - clustering_metrics.txt: 聚类指标"

