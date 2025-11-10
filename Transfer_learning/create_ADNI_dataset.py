#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split a folder of files into train/val/test (7:1:2) and write absolute paths into .txt files.

Usage:
  python split_712.py /path/to/folder
  # 指定输出目录与随机种子、递归搜索、文件模式
  python split_712.py /data/myset --out /data/splits --seed 42 --recursive \
      --patterns "*.nii,*.nii.gz,*.npz"

Notes:
- 默认匹配: *.nii, *.nii.gz, *.npz, *.npy, *.pt
- --patterns 逗号分隔，优先于默认
- --recursive 开启后使用 rglob 递归检索
"""

import argparse
from pathlib import Path
import random
import os
from typing import List

def collect_files(root: Path, recursive: bool, patterns: List[str]) -> List[Path]:
    files = []
    if patterns:
        for pat in patterns:
            pat = pat.strip()
            if not pat:
                continue
            if recursive:
                files += list(root.rglob(pat))
            else:
                files += list(root.glob(pat))
    else:
        # 默认常见科研/医学文件
        default_pats = ["*.nii", "*.nii.gz", "*.npz", "*.npy", "*.pt"]
        for pat in default_pats:
            if recursive:
                files += list(root.rglob(pat))
            else:
                files += list(root.glob(pat))
    # 只保留文件
    files = [p for p in files if p.is_file()]
    # 去重并排序（稳定）
    files = sorted(set(files))
    return files

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", type=str, help="待划分的数据文件夹路径")
    ap.add_argument("--out", type=str, default=None, help="输出目录（默认：folder 同级的 splits/）")
    ap.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）")
    ap.add_argument("--recursive", action="store_true", help="递归检索文件")
    ap.add_argument("--patterns", type=str, default="",
                    help="逗号分隔的通配（如 \"*.nii,*.nii.gz\"），留空则使用默认集合")
    ap.add_argument("--train_ratio", type=float, default=0.7, help="训练集比例（默认 0.7）")
    ap.add_argument("--val_ratio", type=float, default=0.1, help="验证集比例（默认 0.1）")
    args = ap.parse_args()

    root = Path(args.folder).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Not a directory: {root}")

    out_dir = Path(args.out).resolve() if args.out else (root.parent / "splits")
    out_dir.mkdir(parents=True, exist_ok=True)

    patterns = [p for p in args.patterns.split(",") if p.strip()] if args.patterns else []

    files = collect_files(root, args.recursive, patterns)
    if not files:
        raise RuntimeError("No files found. 请检查目录与匹配模式 (--patterns)。")

    # 打乱与划分
    rnd = random.Random(args.seed)
    rnd.shuffle(files)

    n = len(files)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    n_test = n - n_train - n_val  # 余数给 test

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    # 写入绝对路径
    def write_txt(paths: List[Path], out_path: Path):
        with out_path.open("w", encoding="utf-8") as f:
            for p in paths:
                f.write(str(p.resolve()) + "\n")

    base = root.name
    train_txt = out_dir / f"{base}_train.txt"
    val_txt   = out_dir / f"{base}_val.txt"
    test_txt  = out_dir / f"{base}_test.txt"

    write_txt(train_files, train_txt)
    write_txt(val_files,   val_txt)
    write_txt(test_files,  test_txt)

    print(f"Total: {n}  Train/Val/Test: {len(train_files)}/{len(val_files)}/{len(test_files)}")
    print(f"Wrote:\n  {train_txt}\n  {val_txt}\n  {test_txt}")

if __name__ == "__main__":
    main()
