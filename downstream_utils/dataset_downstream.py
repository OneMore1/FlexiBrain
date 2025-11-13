import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import torch
import sys
import os
import re
import ast
from pathlib import Path
import pandas as pd

# Add parent directory to path to import dataset
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import NiftiTxtDataset, _read_list_files, _load_nifti, _space_time_units_to_mm_s

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
#             if 'control' in path_str:
#                 labels.append(0)
#             elif 'patient' in path_str:
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
    """
    从文件名中提取 7 位数字 ID -> 去零归一化 -> 在 CSV 中按 SUB_ID 匹配 -> 取 age_group 为标签。
    - age_group 若为标量，返回 int；
    - age_group 若为 one-hot 字符串/序列，返回 np.ndarray[int]。
    """

    _seven_digits = re.compile(r'(\d{6})(?!\d)')  # 匹配 7 位数字（优先取最后一次出现）

    def __init__(
        self,
        *args,
        csv_path: Union[str, Path],
        id_column: str = 'Subject',
        label_column: str = 'Group_idx',
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.csv_path = Path(csv_path)
        self.id_column = id_column
        self.label_column = label_column

        self._df = self._load_csv(self.csv_path, self.id_column, self.label_column)
        self._id_to_label = self._build_id_to_label(self._df, self.id_column, self.label_column)

        self.labels = self._extract_labels()

    @staticmethod
    def _normalize_id(x: Any) -> str:
        """将 ID 转为字符串并去掉前导 0（空则用 '0'）。"""
        s = str(x).strip().lstrip('0')
        return s if s != '' else '0'

    @classmethod
    def _load_csv(cls, csv_path: Path, id_column: str, label_column: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        if id_column not in df.columns or label_column not in df.columns:
            raise ValueError(f"CSV 缺少必要列：{id_column}, {label_column}")
        df = df.copy()
        df['_norm_id'] = df[id_column].apply(cls._normalize_id)
        return df

    @staticmethod
    def _parse_label(val: Any) -> Union[int, np.ndarray]:
        """
        支持以下形式：
        - 标量：0/1/2… 或 '0'/'1'/'2' → int
        - 字符串 one-hot：'[0,1,0]'、'0,1,0'、'0 1 0' → np.ndarray[int]
        - 已是 list/tuple/np.ndarray → np.ndarray[int]
        """
        # list/array 直接转
        if isinstance(val, (list, tuple, np.ndarray)):
            return np.asarray(val, dtype=int)

        # 先尝试标量数字
        try:
            return int(float(val))
        except Exception:
            pass

        # 再尝试字符串形式的列表
        if isinstance(val, str):
            s = val.strip()
            # 先 literal_eval
            try:
                lit = ast.literal_eval(s)
                if isinstance(lit, (list, tuple, np.ndarray)):
                    return np.asarray(lit, dtype=int)
                if isinstance(lit, (int, float)):
                    return int(lit)
            except Exception:
                # 退化为手动分割
                tokens = [t for t in re.split(r'[,\s]+', s) if t]
                if all(t.isdigit() for t in tokens) and len(tokens) > 1:
                    return np.asarray([int(t) for t in tokens], dtype=int)

        raise ValueError(f"无法解析标签值: {val!r}")

    @classmethod
    def _build_id_to_label(cls, df: pd.DataFrame, id_column: str, label_column: str):
        """构建 norm_id -> label 的映射，并在有冲突时报错。"""
        mapping = {}
        for _, row in df.iterrows():
            key = row['_norm_id']
            lbl = cls._parse_label(row[label_column])
            if key in mapping:
                a, b = np.asarray(mapping[key]), np.asarray(lbl)
                if a.shape != b.shape or not np.array_equal(a, b):
                    raise ValueError(f"CSV 中 SUB_ID={key} 存在冲突标签: {mapping[key]} vs {lbl}")
            else:
                mapping[key] = lbl
        return mapping

    def _extract_labels(self) -> List[Union[int, np.ndarray]]:
        """从文件名中取 7 位数字，归一化后到 CSV 查 age_group。"""
        labels: List[Union[int, np.ndarray]] = []
        for path in self.paths:
            name = Path(path).name  # 仅文件名（不含目录）
            match = None
            # 若出现多次，取“最后一次”出现的 7 位数字更稳妥
            for m in self._seven_digits.finditer(name):
                match = m
            if not match:
                raise ValueError(
                    f"文件名中找不到 7 位数字 ID：{name}。"
                    "应包含形如 0000123 的片段。"
                )
            raw_seven = match.group(1)
            norm_id = self._normalize_id(raw_seven)

            try:
                label = self._id_to_label[norm_id]
            except KeyError:
                raise KeyError(
                    f"CSV（{self.csv_path}）中找不到 {self.id_column}（归一化）== {norm_id} 的行 "
                    f"(来源于文件名片段 {raw_seven})。"
                )
            labels.append(label)
        return labels

    def __getitem__(self, idx: int) -> Dict:
        sample = super().__getitem__(idx)
        sample['label'] = self.labels[idx]
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
                            v = np.pad(v, ((0, 0), (0, 0), (0, 0), (0, pad_amount)), mode='constant', value=0)
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