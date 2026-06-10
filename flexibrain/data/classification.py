import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import torch
import re
import ast
from pathlib import Path
import pandas as pd

from flexibrain.data.nifti import NiftiTxtDataset, _read_list_files, _load_nifti, _space_time_units_to_mm_s

# class ClassificationDataset(NiftiTxtDataset):
#     """
#     Classification dataset that extends NiftiTxtDataset.
    
#     Extracts binary labels from filenames:
#     - Files containing 'control' -> label 0
#     - Files containing 'patient' -> label 1
#     """
    
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.labels = self._extract_labels()
    
#     def _extract_labels(self) -> List[int]:
#         """Extract binary labels from file paths."""
#         labels = []
#         for path in self.paths:
#             path_str = str(path).lower()
#             if 'cn' in path_str:
#                 labels.append(0)
#             elif 'ad' in path_str:
#                 labels.append(1)
#             else:
#                 raise ValueError(
#                     f"Cannot determine label for {path}. "
#                     f"Filename must contain 'control' or 'patient'."
#                 )
#         return labels
    
#     def __getitem__(self, idx: int) -> Dict:
#         sample = super().__getitem__(idx)
#         sample['label'] = self.labels[idx]
#         return sample
    



class ClassificationDataset(NiftiTxtDataset):

    _seven_digits = re.compile(r'(\d{4,8})(?!\d)')

    def __init__(
        self,
        *args,
        csv_path: Union[str, Path],
        id_column: str = 'Subject',
        label_column: str = 'Group_idx',
        label_mode: str = 'multiclass',
        path_id_mode: str = 'auto',
        normal_label: int = 2,
        exclude_labels: Optional[List[int]] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.csv_path = Path(csv_path)
        self.id_column = id_column
        self.label_column = label_column
        self.label_mode = str(label_mode)
        self.path_id_mode = str(path_id_mode)
        self.normal_label = int(normal_label)
        self.exclude_labels = set(int(x) for x in (exclude_labels or []))

        self._df = self._load_csv(self.csv_path, self.id_column, self.label_column)
        self._id_to_label = self._build_id_to_label(self._df, self.id_column, self.label_column)

        self.labels = self._extract_labels()
        self.valid_indices = [i for i, label in enumerate(self.labels) if label is not None]

    @staticmethod
    def _normalize_id(x: Any) -> str:
        s = str(x).strip()
        if '_' in s:
            return s.upper()
        s = s.lstrip('0')
        return s if s != '' else '0'

    @classmethod
    def _load_csv(cls, csv_path: Path, id_column: str, label_column: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        df = df.copy()
        df['_norm_id'] = df[id_column].apply(cls._normalize_id)
        return df

    @staticmethod
    def _parse_label(val: Any) -> Union[int, str, np.ndarray, None]:

        if pd.isna(val):
            return None

        if isinstance(val, (list, tuple, np.ndarray)):
            return np.asarray(val, dtype=int)
        
        try:
            return int(float(val))
        except Exception:
            pass

        if isinstance(val, str):
            s = val.strip()
            try:
                lit = ast.literal_eval(s)
                if isinstance(lit, (list, tuple, np.ndarray)):
                    return np.asarray(lit, dtype=int)
                if isinstance(lit, (int, float)):
                    return int(lit)
            except Exception:
                tokens = [t for t in re.split(r'[,\s]+', s) if t]
                if all(t.isdigit() for t in tokens) and len(tokens) > 1:
                    return np.asarray([int(t) for t in tokens], dtype=int)
            return s

        raise ValueError(f"无法解析标签值: {val!r}")

    @classmethod
    def _build_id_to_label(cls, df: pd.DataFrame, id_column: str, label_column: str):
        mapping = {}
        for _, row in df.iterrows():
            key = row['_norm_id']
            lbl = cls._parse_label(row[label_column])
            if lbl is None:
                continue
            if key in mapping:
                a, b = np.asarray(mapping[key]), np.asarray(lbl)
            else:
                mapping[key] = lbl
        return mapping

    def _extract_path_id(self, name: str) -> str:
        mode = self.path_id_mode.lower()
        if mode == 'auto':
            if 'ADNI_' in name or re.search(r'\d{3}_S_\d{4}', name) or re.search(r'sub-\d+', name, flags=re.IGNORECASE):
                mode = 'adni'
            elif 'ADHD_' in name:
                mode = 'adhd'
            else:
                mode = 'digits'

        if mode == 'adni':
            match = re.search(r'(\d{3}_S_\d{4})', name)
            if match:
                return match.group(1).upper()
            match = re.search(r'sub-(\d+)', name, flags=re.IGNORECASE)
            if match:
                return self._normalize_id(match.group(1))
            raise ValueError(f"Cannot extract ADNI subject id from filename: {name}")

        if mode == 'adhd':
            match = re.search(r'ADHD_[^_]+_(\d+)_', name)
            if not match:
                raise ValueError(f"Cannot extract ADHD subject id from filename: {name}")
            return self._normalize_id(match.group(1))

        matches = self._seven_digits.findall(name)
        if not matches:
            raise ValueError(f"Cannot extract subject id from filename: {name}")
        return self._normalize_id(matches[-1])

    def _extract_labels(self) -> List[Union[int, str, np.ndarray]]:

        labels: List[Union[int, str, np.ndarray]] = []
        for path in self.paths:
            name = Path(path).name  
            norm_id = self._extract_path_id(name)

            label = self._id_to_label.get(norm_id)
            if label is None:
                labels.append(None)
                continue
            label = self._convert_label(label)

            labels.append(label)
        return labels

    def _convert_label(self, label: Union[int, str, np.ndarray]) -> Union[int, np.ndarray]:
        if self.label_mode in ('multiclass', 'raw', None):
            if not isinstance(label, np.ndarray) and int(label) in self.exclude_labels:
                return None
            return label
        if self.label_mode == 'binary_control_vs_disease':
            if isinstance(label, np.ndarray):
                return np.asarray([0 if int(x) == self.normal_label else 1 for x in label], dtype=int)
            return 0 if int(label) == self.normal_label else 1
        if self.label_mode == 'binary_pd_vs_prodromal':
            if isinstance(label, np.ndarray):
                converted = []
                for x in label:
                    xi = int(x)
                    if xi in self.exclude_labels:
                        converted.append(-1)
                    else:
                        converted.append(1 if xi == 1 else 0)
                return np.asarray(converted, dtype=int)
            label_i = int(label)
            if label_i in self.exclude_labels:
                return None
            # PPMI Group_idx: 0=Prodromal, 1=PD, 2=Control.
            return 1 if label_i == 1 else 0
        if self.label_mode == 'binary_gender':
            def to_gender(v: Any) -> int:
                s = str(v).strip().upper()
                if s in {'M', 'MALE', '1'}:
                    return 1
                if s in {'F', 'FEMALE', '0'}:
                    return 0
                raise ValueError(f"Unknown gender label: {v!r}")

            if isinstance(label, np.ndarray):
                return np.asarray([to_gender(x) for x in label], dtype=int)
            return to_gender(label)
        if self.label_mode == 'binary_gender_abide':
            def to_gender_abide(v: Any) -> int:
                s = str(v).strip().upper()
                if s in {'0', 'M', 'MALE'}:
                    return 1
                if s in {'1', 'F', 'FEMALE'}:
                    return 0
                raise ValueError(f"Unknown ABIDE gender label: {v!r}")

            if isinstance(label, np.ndarray):
                return np.asarray([to_gender_abide(x) for x in label], dtype=int)
            return to_gender_abide(label)
        raise ValueError(f"Unknown label_mode: {self.label_mode}")

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Dict:
        raw_idx = self.valid_indices[idx]
        sample = super().__getitem__(raw_idx)
        sample['label'] = self.labels[raw_idx]
        return sample


def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function to handle:
    1. nibabel headers and other non-tensor objects
    2. Variable-length time dimensions (due to different TR values)

    For variable-length data, we pad to the maximum length in the batch.
    """
    # Separate tensor/array fields from non-collatable fields
    tensor_fields = ['data', 'affine']
    scalar_fields = ['tr', 'subject_idx', 'T_selected', 'T_prime', 'tau_seconds', 'label']
    tuple_fields = ['voxel']
    object_fields = ['header', 'path']  # These won't be collated

    collated = {}

    # Handle tensor/array fields with padding for variable-length data
    for field in tensor_fields:
        if field in batch[0]:
            values = [item[field] for item in batch]

            if field == 'data':
                # Data has variable time dimension due to different TR values
                # Pad all to the maximum time length
                max_t = max(v.shape[-1] if len(v.shape) >= 4 else 1 for v in values)

                padded_values = []
                for v in values:
                    if len(v.shape) >= 4 and v.shape[-1] < max_t:
                        # Pad in time dimension (last dimension)
                        pad_amount = max_t - v.shape[-1]
                        if isinstance(v, torch.Tensor):
                            v = torch.nn.functional.pad(v, (0, pad_amount), mode='constant', value=0)
                        else:
                            v = np.pad(v, ((0, 0), (0, 0), (0, 0), (0, pad_amount)), mode='constant', constant_values=0)
                    padded_values.append(v)

                # Convert to tensor and stack
                if isinstance(padded_values[0], torch.Tensor):
                    collated[field] = torch.stack(padded_values)
                else:
                    collated[field] = torch.from_numpy(np.stack(padded_values))
            else:
                # Affine matrices should all be the same size (4x4)
                if isinstance(values[0], torch.Tensor):
                    collated[field] = torch.stack(values)
                else:
                    collated[field] = torch.from_numpy(np.stack(values))

    # Handle scalar fields
    for field in scalar_fields:
        if field in batch[0]:
            values = [item[field] for item in batch]
            if isinstance(values[0], (int, float)):
                collated[field] = torch.tensor(values)
            else:
                collated[field] = values

    # Handle tuple fields (like voxel sizes)
    for field in tuple_fields:
        if field in batch[0]:
            collated[field] = [item[field] for item in batch]

    # Handle object fields (keep as lists)
    for field in object_fields:
        if field in batch[0]:
            collated[field] = [item[field] for item in batch]

    return collated

def prepare_batch_data(batch: Dict, device: torch.device) -> Tuple[torch.Tensor, Dict, np.ndarray, torch.Tensor, Optional[torch.Tensor]]:
    """Prepare batch data for model forward pass.

    Returns:
        x: Input tensor (B, 96, 96, 96, T_max)
        meta: Dict {batch_index: {"voxel": (vx, vy, vz), "tr": float}}
        orig_Ts: Array of original time steps
        labels: Classification labels
        affines: Affine matrices or None
    """
    # Data is already padded and stacked by custom_collate_fn
    x = batch['data'].to(device, dtype=torch.float32)

    # Build meta dict - use batch index as key
    batch_size = x.shape[0]
    voxels = batch['voxel']
    trs = batch['tr']

    # Convert trs to numpy if needed
    if isinstance(trs, torch.Tensor):
        trs = trs.cpu().numpy()
    elif isinstance(trs, list):
        trs = np.array(trs)

    meta = {}
    for i in range(batch_size):
        # Get voxel
        if isinstance(voxels, list):
            voxel = voxels[i] if isinstance(voxels[i], tuple) else tuple(voxels[i])
        else:
            print("voxels is empty")
            voxel = (2.0, 2.0, 2.0)  # Default voxel size

        # Get TR
        if isinstance(trs, np.ndarray):
            tr = float(trs[i])
        elif isinstance(trs, list):
            tr = float(trs[i])
        else:
            print("trs is empty")
            tr = 2.0  # Default TR

        meta[i] = {"voxel": voxel, "tr": tr}

    # Get original time steps (T_selected from dataset)
    orig_Ts = batch.get('T_selected', x.shape[-1])
    if isinstance(orig_Ts, torch.Tensor):
        orig_Ts = orig_Ts.cpu().numpy()
    elif isinstance(orig_Ts, list):
        orig_Ts = np.array(orig_Ts)

    # Handle labels
    labels = batch['label']
    if isinstance(labels, torch.Tensor):
        labels = labels.to(device, dtype=torch.long)
    else:
        labels = torch.tensor(labels, dtype=torch.long, device=device)

    # Get affines if available
    affines = batch['affine'].to(device, dtype=torch.float32) if 'affine' in batch else None

    return x, meta, orig_Ts, labels, affines
