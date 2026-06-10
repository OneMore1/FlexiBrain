# Third-Party References

Flexibrain keeps only the source fragments needed for the Mamba-JEPA pretrain/downstream path.

- Brain-Harmony / BrainHarmonix: https://github.com/hzlab/Brain-Harmony. The Transformer head block in `flexibrain/models/transformer_block.py` is derived from the official Brain-Harmony `libs/flex_transformer.py` design. The official README marks Brain-Harmony as CC BY-NC-SA 4.0. No standalone Brain-Harmony project directory is vendored.
- 3D Mamba MAE: referenced for the Mamba block factory design. The original project license file is preserved in `licenses/mamba_mae_LICENSE_CC_BY_NC_4.0.txt`.
- Mamba / mamba_ssm: the minimal Python source needed by the custom Mamba block is included under `mamba_ssm/`; the Apache-2.0 license is preserved in `licenses/mamba2_LICENSE_Apache_2.0.txt`.
- causal-conv1d: used as a CUDA extension dependency and installed via requirements; BSD-3-Clause license is preserved in `licenses/causal_conv1d_LICENSE_BSD_3_Clause.txt`.

Binary build artifacts (`*.so`, `*.whl`), checkpoints, logs, and datasets are not included in this repository.
