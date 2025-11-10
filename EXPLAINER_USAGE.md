# Multi-Explainer 使用指南

## 概述

`multi_explainer.py` 提供了多种可解释性方法和激活区域过滤功能，用于生成更清晰、更有意义的热力图。

## 功能特性

### 激活区域过滤

**问题**：原始热力图激活区域太多，包含大量噪声和不重要的激活。

**解决方案**：提供 4 种过滤方法来减少噪声，突出重要区域。

#### 过滤方法

1. **threshold** - 阈值过滤
   - 只保留高于指定百分位数的激活值
   - 参数：`--threshold_percentile` (默认 99.0)
   - 适用场景：快速去除低强度噪声

2. **topk** - Top-K 过滤
   - 只保留最大的 K% 激活值
   - 参数：`--topk_percent` (默认 5.0)
   - 适用场景：严格控制激活区域大小

3. **connected** - 连通域过滤
   - 保留最大的 N 个连通区域
   - 参数：
     - `--n_components` (默认 3): 保留的区域数量
     - `--min_component_size` (默认 100): 最小区域大小（体素数）
   - 适用场景：去除分散的小激活点，保留主要激活区域

4. **combined** - 组合过滤（推荐）
   - 先阈值过滤，再连通域过滤
   - 结合了两种方法的优点
   - 适用场景：大多数情况下的最佳选择

## 使用示例

### 示例 1: 基础使用（带组合过滤）

```bash
python multi_explainer.py \
    --checkpoint /path/to/checkpoint.pt \
    --data_path /path/to/data.nii.gz \
    --methods ig saliency \
    --filter_method combined \
    --threshold_percentile 95 \
    --n_components 3 \
    --output_dir ./outputs
```

**效果**：
- 先去除低于 95% 百分位的激活
- 再保留最大的 3 个连通区域
- 大幅减少噪声，突出主要激活区域

### 示例 2: 严格过滤（Top-K）

```bash
python multi_explainer.py \
    --checkpoint /path/to/checkpoint.pt \
    --data_path /path/to/data.nii.gz \
    --methods ig \
    --filter_method topk \
    --topk_percent 2.0 \
    --output_dir ./outputs
```

**效果**：
- 只保留最强的 2% 激活值
- 热力图非常稀疏，只显示最重要的区域

### 示例 3: 连通域过滤（保留单个最大区域）

```bash
python multi_explainer.py \
    --checkpoint /path/to/checkpoint.pt \
    --data_path /path/to/data.nii.gz \
    --methods ig \
    --filter_method connected \
    --n_components 1 \
    --min_component_size 200 \
    --output_dir ./outputs
```

**效果**：
- 只保留最大的 1 个连通区域
- 去除所有小于 200 体素的区域
- 适合寻找单一最重要的激活区域

### 示例 4: 批量处理多个文件

```bash
python multi_explainer.py \
    --checkpoint /path/to/checkpoint.pt \
    --data_list /path/to/file_list.txt \
    --methods ig saliency deeplift \
    --filter_method combined \
    --threshold_percentile 99 \
    --n_components 3 \
    --output_dir ./outputs
```

## 参数调优建议

### 减少激活区域的策略

1. **轻度过滤**（保留较多信息）
   ```bash
   --filter_method threshold \
   --threshold_percentile 90
   ```

2. **中度过滤**（推荐）
   ```bash
   --filter_method combined \
   --threshold_percentile 95 \
   --n_components 3
   ```

3. **重度过滤**（只保留最重要区域）
   ```bash
   --filter_method topk \
   --topk_percent 1.0
   ```
   或
   ```bash
   --filter_method combined \
   --threshold_percentile 98 \
   --n_components 1
   ```

### 不同任务的推荐设置

#### 脑区定位任务
```bash
--filter_method connected \
--n_components 5 \
--min_component_size 100
```
- 保留多个独立的激活区域
- 适合识别多个相关脑区

#### 病灶检测任务
```bash
--filter_method combined \
--threshold_percentile 98 \
--n_components 1 \
--min_component_size 50
```
- 严格过滤，只保留最显著的异常区域

#### 探索性分析
```bash
--filter_method threshold \
--threshold_percentile 90
```
- 保留较多信息，避免过度过滤

## 输出文件命名

输出文件名格式：`{原文件名}_{方法}_{过滤方法}.nii.gz`

示例：
- `sub001_ig_combined.nii.gz` - IG 方法 + 组合过滤
- `sub001_layerig_topk.nii.gz` - LayerIG + Top-K 过滤
- `sub001_saliency.nii.gz` - Saliency 方法，无过滤

## 完整参数列表

### 基础参数
- `--checkpoint`: 模型检查点路径
- `--data_path`: 单个 NIfTI 文件路径
- `--data_list`: 包含多个文件路径的 txt 文件
- `--output_dir`: 输出目录
- `--methods`: 可解释性方法列表
- `--target_class`: 目标类别 (默认 1)
- `--device`: 计算设备 (默认 cuda)

### 方法参数
- `--n_steps`: 积分步数 (默认 128)
- `--baseline`: baseline 方法 (zero/mean/random)
- `--internal_batch_size`: 内部批处理大小 (默认 1)
- `--memory_efficient`: 启用内存高效模式

### 后处理参数
- `--scaling`: 缩放方法 (percentile/minmax/std/abs_percentile/none)
- `--temporal_agg`: 时间聚合 (mean/max/sum/none)
- `--smooth_sigma`: 高斯平滑 sigma (默认 0.5)

### 过滤参数
- `--filter_method`: 过滤方法 (threshold/topk/connected/combined/none)
- `--threshold_percentile`: 阈值百分位数 (默认 95.0)
- `--topk_percent`: Top-K 百分比 (默认 5.0)
- `--n_components`: 保留的连通域数量 (默认 3)
- `--min_component_size`: 最小连通域大小 (默认 100)

## 常见问题

### Q1: 热力图还是太密集怎么办？

**A**: 尝试以下方法：
1. 提高阈值：`--threshold_percentile 99.5`
2. 减少 Top-K：`--topk_percent 1.0`
3. 减少连通域数量：`--n_components 1`
4. 增加最小区域大小：`--min_component_size 500`

### Q2: 过滤后没有激活区域了？

**A**: 说明过滤太严格，尝试：
1. 降低阈值：`--threshold_percentile 95`
2. 增加 Top-K：`--topk_percent 10.0`
3. 增加连通域数量：`--n_components 5`
4. 减小最小区域大小：`--min_component_size 50`

### Q3: 不同方法的结果差异很大？

**A**: 这是正常的，不同方法有不同的特点：
- **IG**: 最稳定，推荐作为基准
- **Saliency**: 最快，但可能有噪声
- **DeepLift**: 适合深度网络，计算效率高
- **GradientShap**: 结合梯度和Shapley值，更鲁棒

## 性能优化

### 减少显存占用
```bash
--internal_batch_size 1 \
--memory_efficient \
--n_steps 50
```

### 加快计算速度
```bash
--methods saliency inputxgrad \
--n_steps 50
```

### 平衡质量和速度
```bash
--methods ig \
--n_steps 100 \
--internal_batch_size 5
```

