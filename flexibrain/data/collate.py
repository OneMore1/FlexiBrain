from typing import Dict, List, Any, Tuple, Optional
import torch
import numpy as np


def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function to handle:
    1. nibabel headers and other non-tensor objects
    2. Variable-length time dimensions (due to different TR values)

    For variable-length data, we pad to the maximum length in the batch.
    """
    # Separate tensor/array fields from non-collatable fields
    tensor_fields = ['data', 'affine']
    scalar_fields = ['tr', 'subject_idx', 'T_selected', 'T_prime', 'tau_seconds']
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


def prepare_batch_data(batch: Dict, device: torch.device) -> Tuple[torch.Tensor, Dict, np.ndarray, Optional[torch.Tensor]]:
    """
    Prepare batch data for model forward pass.

    Returns:
        x: Input tensor (B, 96, 96, 96, T_max)
        meta: Dict {subject_idx: {"voxel": (vx, vy, vz), "tr": float}}
        orig_Ts: Array of original time steps
        affines: Affine matrices or None
    """
    # Move data to device
    x = batch['data'].to(device, dtype=torch.float32)

    # Build meta dict: {batch_index: {"voxel": (vx, vy, vz), "tr": float}}
    subject_idxs = batch['subject_idx'].cpu().numpy()
    voxels = batch['voxel']  # List of tuples or tensor
    trs = batch['tr'].cpu().numpy() if isinstance(batch['tr'], torch.Tensor) else batch['tr']

    meta = {}
    for i, subject_idx in enumerate(subject_idxs):
        # Handle voxel format
        if isinstance(voxels, (list, tuple)):
            voxel = voxels[i]
        else:
            voxel = tuple(voxels[i].cpu().numpy()) if isinstance(voxels[i], torch.Tensor) else voxels[i]

        tr = float(trs[i])
        # Use batch index (i) as key, not subject_idx
        meta[i] = {"voxel": voxel, "tr": tr}

    # Get original time steps (number of frames, not TR)
    # T_selected is the actual number of time frames selected by the dataset
    # Do NOT use 'tr' (time resolution in seconds) as it will cause incorrect T_pad calculation
    if 'T_selected' in batch:
        orig_Ts = batch['T_selected'].cpu().numpy() if isinstance(batch['T_selected'], torch.Tensor) else batch['T_selected']
    else:
        # Fallback: use actual data time dimension if T_selected is not available
        orig_Ts = np.array([x.shape[-1] for x in batch['data']])

    # Get affines if available
    affines = batch['affine'].to(device, dtype=torch.float32) if 'affine' in batch else None

    return x, meta, orig_Ts, affines
