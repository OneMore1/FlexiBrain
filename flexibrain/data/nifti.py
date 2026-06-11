from __future__ import annotations
import os
import glob
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset


class fMRIDataset(Dataset):
    def __init__(self, 
                 data_root, datasets, split_suffixes, crop_length=40, downstream=False):

        self.file_paths = []
        self.crop_length = crop_length
        self.downstream = downstream

        for dataset_name in datasets:
            for suffix in split_suffixes:
                folder_name = f"{dataset_name}_{suffix}"
                folder_path = os.path.join(data_root, folder_name)
                if not os.path.exists(folder_path):
                    print(f"Warning: Folder not found: {folder_path}")
                    continue

                for root, dirs, files in os.walk(folder_path):
                    npz_files = glob.glob(os.path.join(root, "*.npz"))
                    if len(npz_files) > 1:
                        sample_size = max(1, int(len(npz_files))) 
                        npz_files = random.sample(npz_files, sample_size)
                    self.file_paths.extend(npz_files)

        print(f"Dataset loaded. Total files found: {len(self.file_paths)}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        try:
            with np.load(file_path) as data_file:
                key = list(data_file.keys())[0]
                fmri_data = data_file[key] 
                fmri_data = fmri_data.astype(np.float32)
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return None

        total_time_frames = fmri_data.shape[-1]
        if total_time_frames > self.crop_length:
            start_idx = np.random.randint(0, total_time_frames - self.crop_length + 1)
            end_idx = start_idx + self.crop_length
            cropped_data = fmri_data[..., start_idx:end_idx]
        else:
            cropped_data = fmri_data[..., :self.crop_length]

        data_tensor = torch.from_numpy(cropped_data)
        data_tensor = data_tensor.permute(3, 0, 1, 2)

        return data_tensor


def _read_list_files(txt_files: Union[str, Path, Sequence[Union[str, Path]]]) -> List[Path]:
    """Read one or many .txt files and collect absolute paths listed in them.

    Each line should contain a path to a .nii or .nii.gz file. Empty lines and lines
    starting with '#' are ignored. Paths are expanded and normalized to absolute Paths.
    """
    if isinstance(txt_files, (str, Path)):
        txt_files = [txt_files]
    paths: List[Path] = []
    for f in txt_files:  # type: ignore[assignment]
        f = Path(f)
        if not f.exists():
            raise FileNotFoundError(f"List file not found: {f}")
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = Path(os.path.expanduser(line)).resolve()
            # allow relative paths inside list files (relative to the list file dir)
            if not p.exists():
                p = (f.parent / line).resolve()
            if not p.exists():
                raise FileNotFoundError(f"Path from list file does not exist: {line} (resolved: {p})")
            if p.suffix not in {".nii", ".gz"} and not str(p).endswith(".nii.gz"):
                raise ValueError(f"Not a NIfTI file: {p}")
            paths.append(p)
    # deduplicate while preserving order
    seen = set()
    deduped = []
    for p in paths:
        if p not in seen:
            deduped.append(p)
            seen.add(p)
    return deduped


def _space_time_units_to_mm_s(header: nib.nifti1.Nifti1Header) -> Tuple[Tuple[float, float, float], float]:
    """Return (vx, vy, vz) in millimeters and TR in seconds from a NIfTI header.

    Uses header.get_zooms() and header.get_xyzt_units(). Safely handles cases with
    missing time dimension or unusual units.
    """
    zooms = header.get_zooms()
    # space-time units, e.g. ("mm", "sec")
    space_u, time_u = header.get_xyzt_units()

    # Spatial voxel sizes
    vx, vy, vz = (zooms + (1.0, 1.0, 1.0, 1.0))[:3]
    # Convert to mm if needed
    if space_u == "m":
        vx, vy, vz = vx * 1000.0, vy * 1000.0, vz * 1000.0
    elif space_u in ("mm", None, "unknown"):
        pass
    else:
        # Fallback: assume values already in mm
        pass

    # Temporal resolution (TR)
    tr = 0.0
    if len(zooms) >= 4:
        tr = float(zooms[3])
        if time_u == "msec":
            tr = tr / 1000.0
        elif time_u in ("usec", "microsec"):
            tr = tr / 1e6
        elif time_u in ("sec", None, "unknown"):
            pass
        else:
            # Unknown -> leave as-is
            pass
    return (float(vx), float(vy), float(vz)), float(tr)


def _load_nifti(path: Union[str, Path], mmap: bool = True) -> Tuple[np.ndarray, np.ndarray, nib.nifti1.Nifti1Header]:
    try:
        img = nib.load(str(path), mmap=mmap)
        data = img.get_fdata(dtype=np.float32)
        affine = img.affine.copy()
        header = img.header.copy()
        return data, affine, header
    except Exception as e:
        # Return None to signal invalid file
        return None, None, None


class NiftiTxtDataset(Dataset):
    """Dataset that loads NIfTI volumes listed in one or more .txt files.

    Each item returns a dict with:
      - 'data': np.ndarray (from get_fdata())
      - 'affine': np.ndarray (4x4)
      - 'header': nibabel header
      - 'voxel': (vx, vy, vz) in millimeters
      - 'tr': float, seconds (0.0 if not present)
      - 'path': pathlib.Path to the NIfTI file
      - 'subject_idx': integer index inside this dataset
      - 'T_selected': int, number of time frames selected based on T_prime and tau_seconds

    Parameters
    ----------
    txt_files: str | Path | Sequence[str|Path]
        One or more text files containing absolute (or relative) paths to NIfTI files.
    transform: Optional[callable]
        Optional transform applied to the sample dict (after loading).
    return_torch: bool
        If True, converts 'data' and 'affine' to torch tensors.
    memory_map: bool
        If True, enables nibabel's memory mapping. Disable to force full load into RAM.
    cache_meta: bool
        If True, caches voxel/TR in memory to avoid recomputing for repeated access.
    T_prime: Optional[int]
        Target number of time patches after TAPE (Time-to-space patch embedding).
        If provided, dataset will automatically select appropriate time frames to ensure
        all samples have the same T_prime after TAPE processing.
        Formula: T_selected = T_prime * tau_seconds / TR
    tau_seconds: float
        Time window in seconds for TAPE kernel (default: 6.0).
        Used to calculate T_selected when T_prime is specified.
    """

    def __init__(
        self,
        txt_files: Union[str, Path, Sequence[Union[str, Path]]],
        transform: Optional[callable] = None,
        return_torch: bool = False,
        memory_map: bool = True,
        cache_meta: bool = True,
        T_prime: Optional[int] = None,
        tau_seconds: float = 6.0,
        default_tr: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.paths: List[Path] = _read_list_files(txt_files)
        if len(self.paths) == 0:
            raise ValueError("No NIfTI paths found in the provided list files.")
        self.transform = transform
        self.return_torch = bool(return_torch)
        self.memory_map = bool(memory_map)
        self.cache_meta = bool(cache_meta)
        self.T_prime = T_prime
        self.tau_seconds = float(tau_seconds)
        self.default_tr = float(default_tr) if default_tr is not None else None
        if self.default_tr is not None and self.default_tr <= 0:
            raise ValueError("default_tr must be positive when provided")
        self._meta_cache: Dict[int, Tuple[Tuple[float, float, float], float]] = {}

    def __len__(self) -> int:
        return len(self.paths)

    def _get_meta(self, idx: int, header: Optional[nib.nifti1.Nifti1Header] = None) -> Tuple[Tuple[float, float, float], float]:
        if self.cache_meta and idx in self._meta_cache:
            return self._meta_cache[idx]
        if header is None:
            _, _, header = _load_nifti(self.paths[idx], mmap=self.memory_map)
        voxel, tr = _space_time_units_to_mm_s(header)
        voxel = tuple(float(v) for v in voxel)
        if any((not np.isfinite(v)) or v <= 0 for v in voxel):
            raise ValueError(f"Invalid voxel spacing for {self.paths[idx]}: {voxel}")
        tr = float(tr)
        if (not np.isfinite(tr)) or tr <= 0:
            if self.default_tr is None:
                raise ValueError(
                    f"Invalid or missing TR for {self.paths[idx]}: {tr}. "
                    "Set data.default_tr or pass --default-tr to use an explicit fallback."
                )
            tr = self.default_tr
        if self.cache_meta:
            self._meta_cache[idx] = (voxel, tr)
        return voxel, tr

    def _calculate_T_selected(self, tr: float, T_total: int) -> int:
        """
        Calculate the number of time frames to select based on T_prime and tau_seconds.

        Formula:
            kt = round(tau_seconds / tr)  # kernel size in time dimension
            T_selected = T_prime * kt

        This ensures that after TAPE (Time-to-space patch embedding), all samples
        will have the same number of time patches (T_prime).

        Args:
            tr: Temporal resolution (TR) in seconds
            T_total: Total number of time frames available in the data

        Returns:
            T_selected: Number of time frames to use (min with T_total)
        """
        if self.T_prime is None:
            return T_total
        if tr <= 0:
            raise ValueError("TR must be positive when T_prime is set")

        # Calculate kernel size in time dimension
        kt = max(1, round(self.tau_seconds / tr))

        # Calculate required time frames to get T_prime patches
        T_selected = self.T_prime * kt

        # Ensure we don't exceed available data
        T_selected = min(T_selected, T_total)

        return T_selected

    def __getitem__(self, idx: int) -> Dict:
        # Try to load file, skip to next valid file if current is invalid
        attempt = 0
        max_attempts = len(self.paths)

        while attempt < max_attempts:
            current_idx = (idx + attempt) % len(self.paths)
            p = self.paths[current_idx]
            data, affine, header = _load_nifti(p, mmap=self.memory_map)

            # If file is valid, process it
            if data is not None:
                voxel, tr = self._get_meta(current_idx, header)

                # # 检测到tr=2或者1.96
                # if not (np.isclose(tr, 2.0, atol=1e-2) or np.isclose(tr, 1.96, atol=1e-2)):
                #     attempt += 1
                #     continue
                # print(f"TR is {tr} for {p}")

                # Calculate T_selected based on T_prime and tau_seconds
                T_total = data.shape[3] if len(data.shape) >= 4 else 1
                T_selected = self._calculate_T_selected(tr, T_total)

                # Slice data to T_selected frames
                if len(data.shape) >= 4 and T_selected < T_total:
                    data = data[..., :T_selected]
                
                sample = {
                    "data": torch.from_numpy(data) if self.return_torch else data,
                    "affine": torch.from_numpy(affine) if self.return_torch else affine,
                    "header": header,
                    "voxel": voxel,
                    "tr": tr,
                    "path": p,
                    "subject_idx": current_idx,
                    "T_selected": T_selected,
                    "T_prime": self.T_prime,
                    "tau_seconds": self.tau_seconds,
                }
                if self.transform is not None:
                    sample = self.transform(sample)
                return sample

            # Try next file if current one is invalid
            attempt += 1

        # If all files are invalid, raise error
        raise RuntimeError(f"Could not find any valid file starting from index {idx}")

    def meta_dict(self) -> Dict[int, Dict[str, Union[Tuple[float, float, float], float]]]:
        """Return {subject_idx: {"voxel": (vx,vy,vz), "tr": tr}} for the whole dataset."""
        meta: Dict[int, Dict[str, Union[Tuple[float, float, float], float]]] = {}
        for i, p in enumerate(self.paths):
            if self.cache_meta and i in self._meta_cache:
                voxel, tr = self._meta_cache[i]
            else:
                # read header cheaply without loading full data
                img = nib.load(str(p), mmap=True)
                voxel, tr = _space_time_units_to_mm_s(img.header)
                if self.cache_meta:
                    self._meta_cache[i] = (voxel, tr)
            meta[i] = {"voxel": voxel, "tr": tr}
        return meta




def build_train_val_from_lists(
    train_txts: Union[str, Path, Sequence[Union[str, Path]]],
    val_txts: Union[str, Path, Sequence[Union[str, Path]]],
    *,
    return_torch: bool = False,
    memory_map: bool = True,
    T_prime: Optional[int] = None,
    tau_seconds: float = 6.0,
) -> Tuple[NiftiTxtDataset, NiftiTxtDataset, Dict[str, Dict[int, Dict[str, Union[Tuple[float, float, float], float]]]]]:
    """Convenience helper to create train/val datasets and collect their meta dicts.

    Parameters
    ----------
    train_txts, val_txts: str | Path | Sequence[str|Path]
        Text files containing paths to NIfTI files
    return_torch: bool
        If True, converts data and affine to torch tensors
    memory_map: bool
        If True, enables nibabel's memory mapping
    T_prime: Optional[int]
        Target number of time patches after TAPE. If provided, dataset will automatically
        select appropriate time frames to ensure all samples have the same T_prime.
    tau_seconds: float
        Time window in seconds for TAPE kernel (default: 6.0)

    Returns
    -------
    train_set, val_set, meta_all
      where meta_all = {"train": {...}, "val": {...}}
    """
    train_set = NiftiTxtDataset(
        train_txts,
        return_torch=return_torch,
        memory_map=memory_map,
        T_prime=T_prime,
        tau_seconds=tau_seconds,
    )
    val_set = NiftiTxtDataset(
        val_txts,
        return_torch=return_torch,
        memory_map=memory_map,
        T_prime=T_prime,
        tau_seconds=tau_seconds,
    )
    meta_all = {"train": train_set.meta_dict(), "val": val_set.meta_dict()}
    return train_set, val_set, meta_all


