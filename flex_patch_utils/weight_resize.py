# Weight (kernel) resizing using pseudoinverse-based geometric operators.
from typing import Tuple
import torch
from torch import Tensor
from einops import rearrange

from .pinv_resize import _calculate_pinv_2d

# ------------- 1D (e.g., time) -----------------

def resize_conv1d_weight_with_pinv(
    w_star: Tensor, k_new: int,
    interpolation: str = "bicubic",
    antialias: bool = True,
) -> Tensor:
    """
    Resample a Conv1d kernel from K_old -> k_new using pinv of a 2D operator
    on a degenerate dimension (1,K). This keeps the math aligned with the 2D codepath.

    Args:
        w_star: [Out, In, K_old]
        k_new:  new kernel length
    Returns:
        w_new:  [Out, In, k_new]
    """
    Out, In, K_old = w_star.shape
    if k_new == K_old:
        return w_star

    dev, dt = w_star.device, w_star.dtype
    requires_grad = w_star.requires_grad

    # Build pinv((1,K_old)->(1,k_new)) - 这个操作不需要梯度
    with torch.no_grad():
        pinv = _calculate_pinv_2d(
            (1, int(K_old)), (1, int(k_new)),
            interpolation=interpolation,
            antialias=antialias,
            device=dev, dtype=dt
        )  # [(1*k_new), (1*K_old)] == [k_new, K_old]

    W = w_star.reshape(Out * In, K_old)      # [(Out*In), K_old]
    W_new = (pinv @ W.T).T                   # [(Out*In), k_new]
    W_new = W_new.reshape(Out, In, k_new)

    # 恢复requires_grad状态
    if requires_grad:
        W_new = W_new.requires_grad_(True)

    return W_new


def pi_resize_weight_1d(
    w_star: Tensor, k_new: int,
    interpolation: str = "bicubic",
    antialias: bool = True,
) -> Tensor:
    """
    Alias kept for timetospace: same signature as your current helper.
    """
    return resize_conv1d_weight_with_pinv(
        w_star, k_new, interpolation=interpolation, antialias=antialias
    )

# ------------- 3D separable (x,y,z) ------------

def resize_conv3d_weight_separable_with_pinv(
    w_star: Tensor, kx: int, ky: int, kz: int,
    interpolation: str = "bicubic",
    antialias: bool = True,
) -> Tensor:
    """
    Separable 3-axis resize using three 1D pinv operators.
    Mirrors your existing axis-by-axis path; only changes how we form each 1D pinv.

    Args:
        w_star: [Out, In, Kx0, Ky0, Kz0]
        kx, ky, kz: target sizes
    Returns:
        w_new:  [Out, In, kx, ky, kz]
    """
    Out, In, Kx0, Ky0, Kz0 = w_star.shape
    if (kx, ky, kz) == (Kx0, Ky0, Kz0):
        return w_star

    dev, dt = w_star.device, w_star.dtype
    requires_grad = w_star.requires_grad
    W = w_star

    # x-axis
    with torch.no_grad():
        Rx_p = _calculate_pinv_2d(
            (int(Kx0), 1), (int(kx), 1),
            device=dev, dtype=dt,
            interpolation=interpolation, antialias=antialias
        )  # [kx, Kx0]
    W = W.permute(2, 0, 1, 3, 4).reshape(Kx0, -1)           # [Kx0, Out*In*Ky0*Kz0]
    W = (Rx_p @ W).reshape(kx, Out, In, Ky0, Kz0).permute(1, 2, 0, 3, 4)

    # y-axis
    with torch.no_grad():
        Ry_p = _calculate_pinv_2d(
            (int(Ky0), 1), (int(ky), 1),
            device=dev, dtype=dt,
            interpolation=interpolation, antialias=antialias
        )  # [ky, Ky0]
    W = W.permute(3, 0, 1, 2, 4).reshape(Ky0, -1)
    W = (Ry_p @ W).reshape(ky, Out, In, kx, Kz0).permute(1, 2, 3, 0, 4)

    # z-axis
    with torch.no_grad():
        Rz_p = _calculate_pinv_2d(
            (int(Kz0), 1), (int(kz), 1),
            device=dev, dtype=dt,
            interpolation=interpolation, antialias=antialias
        )  # [kz, Kz0]
    W = W.permute(4, 0, 1, 2, 3).reshape(Kz0, -1)
    W = (Rz_p @ W).reshape(kz, Out, In, kx, ky).permute(1, 2, 3, 4, 0)

    # 恢复requires_grad状态
    if requires_grad:
        W = W.requires_grad_(True)

    return W

def pi_resize_weight_3d(
    w_star: Tensor, kx: int, ky: int, kz: int,
    interpolation: str = "bicubic",
    antialias: bool = True,
) -> Tensor:
    """
    Alias kept for timetospace: same signature as your current helper.
    """
    return resize_conv3d_weight_separable_with_pinv(
        w_star, kx, ky, kz, interpolation=interpolation, antialias=antialias
    )
