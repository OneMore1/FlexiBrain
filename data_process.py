#!/usr/bin/env python3
"""Generic three-space 4D voxel preprocessing with sample-wise global z-score."""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np

try:
    from nilearn.image import resample_to_img
except ImportError:
    resample_to_img = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


SPACE_ORDER = ("native", "t1", "mni")


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


def parse_csv(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def selected_spaces(value: str) -> List[str]:
    if value == "all":
        return list(SPACE_ORDER)
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


def clean_output_name(in_file: Path) -> str:
    return f"{nifti_stem(in_file)}_global_zscore.nii.gz"


def iter_nifti_files(input_dir: Path) -> List[Path]:
    if not input_dir.is_dir():
        return []
    return sorted((path for path in input_dir.iterdir() if is_nifti_file(path)), key=lambda p: p.name)


def progress(items: Sequence[Path], desc: str) -> Iterable[Path]:
    if tqdm is None:
        return items
    return tqdm(items, desc=desc)


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


def extract_match_key(path: Path) -> str:
    stem = nifti_stem(path)
    bids_match = re.search(r"(sub-[A-Za-z0-9]+)", stem, flags=re.IGNORECASE)
    if bids_match:
        return bids_match.group(1).lower()

    for token in ("_space-", "_desc-", "_task-", "_run-", "_bold", "_fmri"):
        idx = stem.lower().find(token)
        if idx > 0:
            return stem[:idx].lower()
    return stem.lower()


def best_matching_file(source_file: Path, target_dir: Path) -> Optional[Path]:
    if not target_dir.is_dir():
        return None

    key = extract_match_key(source_file)
    candidates = [path for path in target_dir.rglob("*") if is_nifti_file(path)]
    if not candidates:
        return None

    source_stem = nifti_stem(source_file).lower()

    def score(path: Path) -> Tuple[int, int, int, str]:
        stem = nifti_stem(path).lower()
        exact = 3 if stem == source_stem else 0
        key_match = 2 if key and key in stem else 0
        short_path = -len(str(path))
        return (exact, key_match, short_path, str(path))

    best = max(candidates, key=score)
    best_score = score(best)
    if best_score[0] == 0 and best_score[1] == 0:
        return None
    return best


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


def resolve_mask_path(mask_arg: Optional[str], script_dir: Path) -> Optional[Path]:
    if not mask_arg:
        return None
    mask_path = Path(mask_arg)
    if mask_path.exists():
        return mask_path
    script_relative = script_dir / mask_arg
    if script_relative.exists():
        return script_relative
    return mask_path


def resample_template_background(mask_img: nib.Nifti1Image, target_img: nib.Nifti1Image) -> np.ndarray:
    if resample_to_img is None:
        raise RuntimeError("nilearn is required for template mask resampling but is not installed")
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


def space_subdir(args: argparse.Namespace, space: str) -> str:
    return {
        "native": args.native_subdir,
        "t1": args.t1_subdir,
        "mni": args.mni_subdir,
    }[space]


def input_dir_for(args: argparse.Namespace, space: str, group: Optional[str]) -> Path:
    base = args.input_root / space_subdir(args, space)
    return base / group if group else base


def output_dir_for(args: argparse.Namespace, space: str, group: Optional[str]) -> Path:
    base = args.output_root / space_subdir(args, space)
    return base / group if group else base


def output_path_for(args: argparse.Namespace, space: str, group: Optional[str], in_file: Path) -> Path:
    return output_dir_for(args, space, group) / clean_output_name(in_file)


def discover_groups(args: argparse.Namespace, space: str) -> List[Optional[str]]:
    if args.groups is not None:
        return args.groups

    base = args.input_root / space_subdir(args, space)
    if not base.is_dir():
        return [None]

    groups: List[Optional[str]] = []
    if iter_nifti_files(base):
        groups.append(None)
    groups.extend(sorted(path.name for path in base.iterdir() if path.is_dir() and iter_nifti_files(path)))
    return groups or [None]


def process_one_file(
    in_file: Path,
    out_file: Path,
    space: str,
    group: Optional[str],
    args: argparse.Namespace,
    template_mask_img: Optional[nib.Nifti1Image] = None,
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
        native_dir = input_dir_for(args, "native", group)
        native_match = best_matching_file(in_file, native_dir)
        if native_match is None:
            print(f"[WARN] Native-space mask source not found for {in_file.name} in {native_dir}")
            return "skipped_no_mask"
        native_img, _ = load_4d_padded(native_match, args.target_shape)
        if native_img is None:
            print(f"[WARN] Native-space mask source is not 4D, skipping: {native_match}")
            return "skipped_no_mask"
        native_arr = np.asarray(native_img.dataobj, dtype=np.float32)
        background_mask = (native_arr == 0).all(axis=3)
    elif space == "mni":
        if template_mask_img is None:
            print(f"[WARN] Template mask unavailable, skipping: {in_file}")
            return "skipped_no_mask"
        background_mask = resample_template_background(template_mask_img, new_img)
    else:
        raise ValueError(f"Unsupported space: {space}")

    z_data = global_zscore(arr, background_mask)
    if z_data is None:
        print(f"[WARN] Empty foreground after masking, skipping: {in_file}")
        return "skipped_empty_foreground"

    group_label = group if group is not None else "ungrouped"
    print(
        f"[AFT] {space}/{group_label} {in_file.name}: "
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
    group: Optional[str],
    args: argparse.Namespace,
    template_mask_img: Optional[nib.Nifti1Image] = None,
) -> Counts:
    counts = Counts()
    input_dir = input_dir_for(args, space, group)
    group_label = group if group is not None else "ungrouped"

    if not input_dir.is_dir():
        print(f"[WARN] Missing input dir, skipping {space}/{group_label}: {input_dir}")
        counts.skipped_input_dir += 1
        return counts

    files = iter_nifti_files(input_dir)
    print(f"[INFO] {space}/{group_label}: found {len(files)} NIfTI file(s) in {input_dir}")
    for in_file in progress(files, desc=f"{space}/{group_label}"):
        counts.seen += 1
        out_file = output_path_for(args, space, group, in_file)
        if out_file.exists() and not args.overwrite:
            print(f"[SKIP] Exists: {out_file}")
            counts.skipped_existing += 1
            continue

        try:
            status = process_one_file(in_file, out_file, space, group, args, template_mask_img)
            increment(counts, status)
        except Exception as exc:
            counts.failed += 1
            print(f"[ERROR] Failed {in_file}: {exc}", file=sys.stderr)

    return counts


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Process native/T1/MNI 4D voxel NIfTIs with sample-wise global z-score."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("."),
        help="Root containing the native, T1, and MNI input subdirectories.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("global_zscore_outputs"),
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
        type=parse_csv,
        default=None,
        help="Optional comma-separated group/subfolder list. If omitted, groups are auto-discovered.",
    )
    parser.add_argument("--native-subdir", default="nativespace", help="Native-space subdirectory under input/output roots.")
    parser.add_argument("--t1-subdir", default="t1space", help="T1-space subdirectory under input/output roots.")
    parser.add_argument("--mni-subdir", default="mnispace", help="MNI/template-space subdirectory under input/output roots.")
    parser.add_argument(
        "--target-shape",
        type=parse_target_shape,
        default=(96, 96, 96),
        help="Spatial target shape formatted as X,Y,Z.",
    )
    parser.add_argument(
        "--template-mask",
        default="MNI152_T1_2mm_Brain_Mask.nii.gz",
        help="Brain mask for MNI/template-space inputs. Relative paths are also checked next to this script.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned work without writing outputs.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    warnings.filterwarnings("default")
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    for space in SPACE_ORDER:
        input_space_dir = (input_root / space_subdir(args, space)).resolve()
        output_space_dir = (output_root / space_subdir(args, space)).resolve()
        if output_space_dir == input_space_dir or input_space_dir in output_space_dir.parents:
            print(
                "[ERROR] Output space directory must not be the same as or inside "
                f"the input space directory: {output_space_dir}",
                file=sys.stderr,
            )
            return 2

    spaces = selected_spaces(args.spaces)
    template_mask_img = None
    if "mni" in spaces:
        mask_path = resolve_mask_path(args.template_mask, Path(__file__).resolve().parent)
        if mask_path is None or not mask_path.exists():
            print(f"[WARN] Template mask not found; MNI/template files will be skipped: {args.template_mask}", file=sys.stderr)
        else:
            template_mask_img = nib.load(str(mask_path))
            print(f"[INFO] Loaded template mask: {mask_path}")

    total = Counts()
    for space in spaces:
        for group in discover_groups(args, space):
            counts = process_space_group(space, group, args, template_mask_img=template_mask_img)
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
