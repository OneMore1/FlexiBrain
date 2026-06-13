<div align="center">
FlexiBrain: Resolution-Agnostic Voxel-Level Encoding for Native fMRI



[![arXiv](https://img.shields.io/badge/arXiv-2606.11500-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2606.11500)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=flat-square&logo=github)](https://github.com/OneMore1/FlexiBrain)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue)](https://huggingface.co/OneMore1/FlexiBrain)

</div>

Flexibrain is a voxel-level fMRI representation learning framework for pretraining and downstream classification. It keeps fMRI volumes in a fixed 96 x 96 x 96 input grid, reads each sample's voxel spacing and TR from the NIfTI header, and resizes patch embedding kernels in physical spatial and temporal units before learning with a Mamba-JEPA backbone.

<p align="center">
  <img src="assets/pipeline.png" width="900" alt="FlexiBrain framework pipeline">
</p>

## Installation

The code was tested on l40 with Python 3.10, PyTorch 2.1.2, CUDA 12.1, `causal-conv1d`, `mamba-ssm`, and `flash-attn`.

```bash
conda create -n flexibrain python=3.10
conda activate flexibrain
pip install -r requirements.txt
pip install -e .
```

Check the CLI:

```bash
python -m flexibrain --help
python -m flexibrain pretrain --help
python -m flexibrain downstream --help
```

## Data Preparation

Each sample should be a 4D NIfTI file shaped as:

```text
96 x 96 x 96 x T
```

Flexibrain uses the NIfTI header to read voxel spacing and TR. If a dataset has missing TR metadata, fix the header before training or pass an explicit fallback with `--default-tr` / `data.default_tr`.

`T_prime` and `tau_seconds` control the selected temporal length:

```text
kt = round(tau_seconds / TR)
T_selected = T_prime * kt
```

The preprocessing script can convert native/T1/MNI-space inputs to 96 x 96 x 96, apply sample-wise global z-score normalization over foreground voxels, and write 4D NIfTI outputs:

```bash
python data_process.py \
  --input-root /path/to/input_root \
  --output-root /path/to/output_root \
  --spaces all \
  --groups class0,class1,class2
```

Expected grouped input layout:

```text
input_root/
|-- nativespace/class0/*.nii.gz
|-- t1space/class0/*.nii.gz
`-- mnispace/class0/*.nii.gz
```

If files are not organized by group subfolders, omit `--groups`. For MNI-space inputs, provide `--template-mask` when the default mask is not available.

Pretraining list files contain one NIfTI path per line:

```text
/path/to/sub-0001_bold.nii.gz
/path/to/sub-0002_bold.nii.gz
```

Downstream classification uses the same list format plus a CSV label table:

```csv
Subject,Group_idx
003_S_0908,2
011_S_0002,1
1001,0
```

Default label fields are `Subject` and `Group_idx`. `path_id_mode=auto` supports ADNI-style IDs such as `003_S_0908`, ADHD-style filenames, and fallback digit IDs.

## Pretraining

Run from a config:

```bash
python -m flexibrain pretrain --config configs/pretrain_example.yaml
```

Or use CLI arguments:

```bash
python -m flexibrain pretrain \
  --train-list /path/to/pretrain_train.txt \
  --val-list /path/to/pretrain_val.txt \
  --checkpoint-dir ./checkpoints/pretrain/example \
  --log-dir ./logs/pretrain/example \
  --embed-dim 512 \
  --depth 24 \
  --predictor-depth 2 \
  --bimamba-type v2 \
  --if-devide-out \
  --batch-size 4 \
  --epochs 30 \
  --lr 5e-4 \
  --weight-decay 0.05 \
  --warmup-epochs 3 \
  --mask-ratio 0.65 \
  --grad-accumulation-steps 4 \
  --t-prime 30 \
  --tau-seconds 6.0 \
  --use-amp
```

Outputs:

```text
checkpoint_latest.pt
checkpoint_best.pt
pretrain_*.log
```

## Downstream Classification

Run from a config:

```bash
python -m flexibrain downstream --config configs/downstream_example.yaml
```

Or use CLI arguments:

```bash
python -m flexibrain downstream \
  --train-list /path/to/downstream_train.txt \
  --val-list /path/to/downstream_val.txt \
  --test-list /path/to/downstream_test.txt \
  --csv /path/to/labels.csv \
  --pretrain-checkpoint /path/to/checkpoint_best.pt \
  --num-classes 3 \
  --head-type transformer \
  --batch-size 8 \
  --epochs 30 \
  --lr 1e-5 \
  --lr-backbone 6e-6 \
  --lr-head 6e-5 \
  --checkpoint-dir ./checkpoints/downstream/example \
  --log-dir ./logs/downstream/example \
  --use-amp
```

During downstream training, validation metrics select `downstream_best.pt`. The test set is evaluated once at the end after loading that best validation checkpoint, and the final metrics are written to `test_metrics.json`.

## Configuration

YAML config mirrors the CLI options. Keep private paths in local config files and leave shared configs as portable examples. The provided examples use placeholder paths under `data/`:

```text
configs/pretrain_example.yaml
configs/downstream_example.yaml
```

## Checkpoint Compatibility

Checkpoint is available at https://huggingface.co/OneMore1/FlexiBrain/blob/main/checkpoint_best.pt.

The downstream loader can initialize from the original pretraining checkpoint path format:

```text
/path/to/checkpoint_best.pt
```

When `use_checkpoint_config: true`, model-shape settings stored in the checkpoint are applied before loading the backbone.

