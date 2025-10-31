import numpy as np

def meta_to_matrix(meta: dict, B: int) -> np.ndarray:
    """将 meta: Dict[int, Dict] -> (B,4) 矩阵，列顺序 [rx, ry, rt, tr]"""
    out = np.empty((B, 4), dtype=np.float32)
    for i in range(B):
        if i not in meta:
            raise KeyError(f"元信息缺失: subject {i}")

        m = meta[i]
        # 兼容不同命名: voxel / voxel_size / spacing
        voxel = m.get("voxel", m.get("voxel_size", m.get("spacing")))
        if voxel is None or len(voxel) < 3:
            raise ValueError(f"voxel 缺失或长度不足(需要3): subject {i}, got={voxel}")

        # 允许 rt 单独给出；否则用 voxel[2]
        rx = float(voxel[0])
        ry = float(voxel[1])
        rt = float(m.get("rt", voxel[2]))
        tr = float(m["tr"])  # tr 必须存在

        out[i] = (rx, ry, rt, tr)
    return out

