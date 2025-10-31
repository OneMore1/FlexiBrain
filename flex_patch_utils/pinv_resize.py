# Copyright (c) 2025
# Utilities to build resize operators and their pseudoinverses.

from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from functools import lru_cache

@torch.no_grad()
def _resize_2d(x: Tensor, shape: Tuple[int, int],
               interpolation: str = "bicubic",
               antialias: bool = True) -> Tensor:
    """
    Resize a 2D tensor x[h0,w0] -> shape[h,w] using torch interpolate.
    Matches the "wrap with [None,None,...]" trick from your flex_patch_embed.py.
    """
    x_resized = F.interpolate(
        x[None, None, ...],
        shape,
        mode=interpolation,
        antialias=antialias,
    )
    return x_resized[0, 0, ...]


@lru_cache(maxsize=256)
def _calculate_pinv_2d(old_shape: Tuple[int, int],
                       new_shape: Tuple[int, int],
                       interpolation: str = "bicubic",
                       antialias: bool = True,
                       device: torch.device = torch.device("cpu"),
                       dtype: torch.dtype = torch.float32) -> Tensor:
    """
    Build the (flattened) resize matrix R s.t. vec(new) = R @ vec(old),
    then return pinv(R). This mirrors your flex_patch_embed.py approach.

    Args:
        old_shape: (h0, w0)
        new_shape: (h, w)
    Returns:
        pinv(R): Tensor of shape [(h*w), (h0*w0)]
    """
    # Construct R by sending basis vectors through the geometric resize op.
    mat = []
    h0, w0 = int(old_shape[0]), int(old_shape[1])
    for i in range(int(np.prod(old_shape))):
        basis = torch.zeros((h0, w0), dtype=dtype, device=device)
        idx = np.unravel_index(i, (h0, w0))
        basis[idx] = 1.0
        mat.append(_resize_2d(basis, new_shape, interpolation, antialias).reshape(-1))
    resize_matrix = torch.stack(mat)  # [(h*w), (h0*w0)]
    pinv = torch.linalg.pinv(resize_matrix)
    return pinv  # [(h*w), (h0*w0)]
