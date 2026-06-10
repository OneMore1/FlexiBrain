#!/usr/bin/env python3
"""Unified ADNI preprocessing for native, T1, and MNI global z-score NIfTIs."""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import nibabel as nib
import numpy as np

try:
    from nilearn.image import resample_to_img
except ImportError:  # MNI processing will report a clear error if requested.
    resample_to_img = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


SPACE_DIRS = {
    "native": "nativespace",
    "t1": "t1space",
    "mni": "mnispace",
}


@dataclass
class Counts:
    seen: int = 0
    processed: int = 0
    dry_run: int = 0
    skipped_existing: int = 0
    skipped_input_dir: int = 0
    skipped_non4d: int = 0
    skipped_no_mask: int = 0
    skipped_empty_foreground: int = 0
    failed: int = 0

    def add(self, other: "Counts") -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))


def parse_target_shape(value: str) -> Tuple[int, int, int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("target shape must be formatted as X,Y,Z")
    try:
        shape = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("target shape values must be integers") from exc
    if any(dim <= 0 for dim in shape):
        raise argparse.ArgumentTypeError("target shape values must be positive")
    return shape  # type: ignore[return-value]


def parse_groups(value: str) -> List[str]:
    groups = [group.strip() for group in value.split(",") if group.strip()]
    if not groups:
        raise argparse.ArgumentTypeError("at least one group is required")
    return groups


def selected_spaces(value: str) -> List[str]:
    if value == "all":
        return ["native", "t1", "mni"]
    return [value]


def is_nifti_file(path: Path) -> bool:
    name = path.name.lower()
    return path.is_file() and (name.endswith(".nii") or name.endswith(".nii.gz"))


def nifti_stem(path: Path) -> str:
    name = path.name
    if name.lower().endswith(".nii.gz"):
        return name[:-7]
    if name.lower().endswith(".nii"):
        return name[:-4]
    return path.stem


def output_path(out_root: Path, space: str, group: str, in_file: Path) -> Path:
    stem = nifti_stem(in_file)
    return out_root / SPACE_DIRS[space] / group / f"ADNI_{stem}_{group}_global_zscore.nii.gz"


def iter_nifti_files(input_dir: Path) -> List[Path]:
    return sorted((path for path in input_dir.iterdir() if is_nifti_file(path)), key=lambda p: p.name)


def progress(items: Iterable[Path], desc: str) -> Iterable[Path]:
    if tqdm is None:
        return items
    return tqdm(list(items), desc=desc)


def pad_crop_keep_world(
    img: nib.Nifti1Image,
    target_xyz: Tuple[int, int, int] = (96, 96, 96),
    fill_value: float = 0.0,
) -> Tuple[nib.Nifti1Image, Dict[str, Any]]:
    """Center pad/crop the first three axes and update affine translation."""
    data = np.asarray(img.dataobj)
    affine = img.affine.copy()
    linear = affine[:3, :3].copy()
    old_translation = affine[:3, 3].copy()
    original_ndim = data.ndim

    if data.ndim == 3:
        x, y, z = data.shape
        t = 1
        data = data[..., None]
    elif data.ndim == 4:
        x, y, z, t = data.shape
    else:
        raise ValueError(f"Expect 3D/4D NIfTI, got shape {data.shape}")

    def split_plan(old: int, new: int) -> Tuple[int, int, str]:
        if old == new:
            return 0, 0, "same"
        if old < new:
            total = new - old
            left = total // 2
            return left, total - left, "pad"
        total = old - new
        left = total // 2
        return left, total - left, "crop"

    px_l, px_r, mode_x = split_plan(x, target_xyz[0])
    py_l, py_r, mode_y = split_plan(y, target_xyz[1])
    pz_l, pz_r, mode_z = split_plan(z, target_xyz[2])

    out_data = data
    if mode_z == "crop":
        out_data = out_data[:, :, pz_l : z - pz_r, :]
    if mode_y == "crop":
        out_data = out_data[:, py_l : y - py_r, :, :]
    if mode_x == "crop":
        out_data = out_data[px_l : x - px_r, :, :, :]

    pad_width = (
        (px_l if mode_x == "pad" else 0, px_r if mode_x == "pad" else 0),
        (py_l if mode_y == "pad" else 0, py_r if mode_y == "pad" else 0),
        (pz_l if mode_z == "pad" else 0, pz_r if mode_z == "pad" else 0),
        (0, 0),
    )
    if any(width != (0, 0) for width in pad_width):
        out_data = np.pad(out_data, pad_width=pad_width, mode="constant", constant_values=fill_value)

    pad_left = np.array(
        [
            px_l if mode_x == "pad" else 0,
            py_l if mode_y == "pad" else 0,
            pz_l if mode_z == "pad" else 0,
        ],
        dtype=float,
    )
    crop_left = np.array(
        [
            px_l if mode_x == "crop" else 0,
            py_l if mode_y == "crop" else 0,
            pz_l if mode_z == "crop" else 0,
        ],
        dtype=float,
    )
    new_translation = old_translation + linear @ crop_left - linear @ pad_left
    affine[:3, 3] = new_translation

    if original_ndim == 3:
        out_data = out_data[..., 0]

    header = img.header.copy()
    qcode = int(header.get("qform_code", 1)) or 1
    scode = int(header.get("sform_code", 1)) or 1
    out_img = nib.Nifti1Image(out_data, affine, header=header)
    out_img.set_qform(affine, code=qcode)
    out_img.set_sform(affine, code=scode)

    info = {
        "old_shape": (x, y, z, t),
        "new_shape": out_img.shape if len(out_img.shape) == 4 else out_img.shape + (1,),
        "plan": {
            "x": (px_l, px_r, mode_x),
            "y": (py_l, py_r, mode_y),
            "z": (pz_l, pz_r, mode_z),
        },
        "affine_old_t": tuple(old_translation.tolist()),
        "affine_new_t": tuple(new_translation.tolist()),
    }
    return out_img, info


def best_match_by_subject(given_name: str, target_dir: Path) -> Optional[Path]:
    match = re.search(r"(sub-\d{3}S\d{4})", given_name, flags=re.IGNORECASE)
    if match is None:
        return None
    subject_id = match.group(1)
    candidates = [path for path in target_dir.rglob(f"*{subject_id}*") if is_nifti_file(path)]
    if not candidates:
        return None

    def score(path: Path) -> Tuple[int, int, int, str]:
        name = path.name.lower()
        zscore_bonus = 2 if "zscore" in name else 0
        bold_bonus = 1 if "bold_mc" in name else 0
        return (zscore_bonus, bold_bonus, -len(str(path)), str(path))

    return max(candidates, key=score)


def make_output_image(z_data: np.ndarray, template_img: nib.Nifti1Image) -> nib.Nifti1Image:
    header = template_img.header.copy()
    qcode = int(header.get("qform_code", 1)) or 1
    scode = int(header.get("sform_code", 1)) or 1
    out_img = nib.Nifti1Image(z_data.astype(np.float32), template_img.affine, header=header)
    out_img.set_data_dtype(np.float32)
    out_img.set_qform(template_img.affine, code=qcode)
    out_img.set_sform(template_img.affine, code=scode)
    return out_img


def global_zscore(arr: np.ndarray, background_mask: np.ndarray) -> Optional[np.ndarray]:
    foreground_mask = ~background_mask
    if foreground_mask.shape != arr.shape[:3]:
        raise ValueError(f"mask shape {foreground_mask.shape} does not match image shape {arr.shape[:3]}")
    if not np.any(foreground_mask):
        return None

    vals = arr[foreground_mask, :]
    if vals.size == 0:
        return None

    mu = float(vals.mean())
    std = float(vals.std())
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (arr - mu) / (std + 1e-6)
    z[background_mask, :] = 0.0
    return z.astype(np.float32, copy=False)


def load_4d_padded(path: Path, target_shape: Tuple[int, int, int]) -> Tuple[Optional[nib.Nifti1Image], Optional[Dict[str, Any]]]:
    img = nib.load(str(path))
    if len(img.shape) != 4:
        return None, None
    return pad_crop_keep_world(img, target_xyz=target_shape, fill_value=0.0)


def resolve_mni_mask(mask_arg: str) -> Path:
    mask_path = Path(mask_arg)
    if mask_path.exists():
        return mask_path
    script_relative = Path(__file__).resolve().parent / mask_arg
    if script_relative.exists():
        return script_relative
    return mask_path


def resample_mni_background(mask_img: nib.Nifti1Image, target_img: nib.Nifti1Image) -> np.ndarray:
    if resample_to_img is None:
        raise RuntimeError("nilearn is required for MNI mask resampling but is not installed")
    try:
        mask_res = resample_to_img(
            mask_img,
            target_img,
            interpolation="nearest",
            force_resample=True,
            copy_header=True,
        )
    except TypeError:
        mask_res = resample_to_img(mask_img, target_img, interpolation="nearest")
    mask_arr = np.asarray(mask_res.dataobj)
    if mask_arr.ndim == 4:
        mask_arr = mask_arr[..., 0]
    return mask_arr == 0


def process_one_file(
    in_file: Path,
    out_file: Path,
    space: str,
    group: str,
    args: argparse.Namespace,
    mni_mask_img: Optional[nib.Nifti1Image] = None,
) -> str:
    new_img, info = load_4d_padded(in_file, args.target_shape)
    if new_img is None:
        return "skipped_non4d"

    arr = np.asarray(new_img.dataobj, dtype=np.float32)
    if arr.ndim != 4:
        return "skipped_non4d"

    if space == "native":
        background_mask = (arr == 0).all(axis=3)
    elif space == "t1":
        native_dir = args.data_root / SPACE_DIRS["native"] / group
        native_match = best_match_by_subject(in_file.name, native_dir)
        if native_match is None:
            print(f"[WARN] T1 mask source not found for {in_file.name} in {native_dir}")
            return "skipped_no_mask"
        native_img, _ = load_4d_padded(native_match, args.target_shape)
        if native_img is None:
            print(f"[WARN] T1 mask source is not 4D, skipping: {native_match}")
            return "skipped_no_mask"
        native_arr = np.asarray(native_img.dataobj, dtype=np.float32)
        background_mask = (native_arr == 0).all(axis=3)
    elif space == "mni":
        if mni_mask_img is None:
            print(f"[WARN] MNI mask unavailable, skipping: {in_file}")
            return "skipped_no_mask"
        background_mask = resample_mni_background(mni_mask_img, new_img)
    else:
        raise ValueError(f"Unsupported space: {space}")

    z_data = global_zscore(arr, background_mask)
    if z_data is None:
        print(f"[WARN] Empty foreground after masking, skipping: {in_file}")
        return "skipped_empty_foreground"

    print(
        f"[AFT] {space}/{group} {in_file.name}: "
        f"{info['old_shape']} -> {new_img.shape}, plan={info['plan']}"
    )
    if args.dry_run:
        print(f"[DRY-RUN] Would save {out_file}")
        return "dry_run"

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_img = make_output_image(z_data, new_img)
    nib.save(out_img, str(out_file))
    print(f"[SAVE] {out_file}")
    return "processed"


def increment(counts: Counts, status: str) -> None:
    if not hasattr(counts, status):
        raise ValueError(f"Unknown status: {status}")
    setattr(counts, status, getattr(counts, status) + 1)


def process_space_group(
    space: str,
    group: str,
    args: argparse.Namespace,
    mni_mask_img: Optional[nib.Nifti1Image] = None,
) -> Counts:
    counts = Counts()
    input_dir = args.data_root / SPACE_DIRS[space] / group
    if not input_dir.is_dir():
        print(f"[WARN] Missing input dir, skipping {space}/{group}: {input_dir}")
        counts.skipped_input_dir += 1
        return counts

    files = iter_nifti_files(input_dir)
    print(f"[INFO] {space}/{group}: found {len(files)} NIfTI file(s) in {input_dir}")
    for in_file in progress(files, desc=f"{space}/{group}"):
        counts.seen += 1
        out_file = output_path(args.out_root, space, group, in_file)
        if out_file.exists() and not args.overwrite:
            print(f"[SKIP] Exists: {out_file}")
            counts.skipped_existing += 1
            continue

        try:
            status = process_one_file(in_file, out_file, space, group, args, mni_mask_img)
            increment(counts, status)
        except Exception as exc:
            counts.failed += 1
            print(f"[ERROR] Failed {in_file}: {exc}", file=sys.stderr)

    return counts


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Process ADNI native/T1/MNI 4D NIfTIs with sample-wise global z-score."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/mnt/dataset4/wangmo/iclr2026/ADNI(all)"),
        help="ADNI root containing nativespace, t1space, and mnispace group folders.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("/mnt/dataset4/DATASETS/fsl_fmri/global/ADNI(all)"),
        help="Output root for processed full 4D NIfTI files.",
    )
    parser.add_argument(
        "--spaces",
        choices=["native", "t1", "mni", "all"],
        default="all",
        help="Space to process.",
    )
    parser.add_argument(
        "--groups",
        type=parse_groups,
        default=parse_groups("ad,mci,cn"),
        help="Comma-separated group list.",
    )
    parser.add_argument(
        "--target-shape",
        type=parse_target_shape,
        default=(96, 96, 96),
        help="Spatial target shape formatted as X,Y,Z.",
    )
    parser.add_argument(
        "--mni-mask",
        default="MNI152_T1_2mm_Brain_Mask.nii.gz",
        help="MNI brain mask path. Relative paths are also checked next to this script.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned work without writing outputs.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    warnings.filterwarnings("default")
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    data_root = args.data_root.resolve()
    out_root = args.out_root.resolve()
    if out_root == data_root or data_root in out_root.parents:
        print(
            f"[ERROR] --out-root must not be the data root or inside it: {args.out_root}",
            file=sys.stderr,
        )
        return 2

    spaces = selected_spaces(args.spaces)
    mni_mask_img = None
    if "mni" in spaces:
        mask_path = resolve_mni_mask(args.mni_mask)
        if not mask_path.exists():
            print(f"[WARN] MNI mask not found; MNI files will be skipped: {mask_path}", file=sys.stderr)
        else:
            mni_mask_img = nib.load(str(mask_path))
            print(f"[INFO] Loaded MNI mask: {mask_path}")

    total = Counts()
    for space in spaces:
        for group in args.groups:
            counts = process_space_group(space, group, args, mni_mask_img=mni_mask_img)
            total.add(counts)

    print(
        "[SUMMARY] "
        f"seen={total.seen}, processed={total.processed}, dry_run={total.dry_run}, "
        f"skipped_existing={total.skipped_existing}, skipped_input_dir={total.skipped_input_dir}, "
        f"skipped_non4d={total.skipped_non4d}, skipped_no_mask={total.skipped_no_mask}, "
        f"skipped_empty_foreground={total.skipped_empty_foreground}, failed={total.failed}"
    )
    return 1 if total.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
