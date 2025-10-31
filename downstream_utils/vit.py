import sys
import os
import torch
import torch.nn as nn


# Add Brain-Harmony to path
brain_harmony_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Brain-Harmony')
sys.path.insert(0, brain_harmony_path)

from libs.flex_transformer import Block
from mamba_mae.models_vit_jepa import VolumeVitJEPA


class VolumeVitJEPAClassifierCLS(nn.Module):
    """
    使用 VolumeVitJEPA 的 context encoder 作为骨干提取变长 token，
    在其输出序列前拼接一个可学习 CLS token，
    经过若干层 Transformer Block（复用你工程里的 Block）后，
    取 CLS 向量接 MLP 分类头。
    """
    def __init__(
        self,
        backbone: VolumeVitJEPA,
        num_classes: int,
        head_depth: int = 2,
        head_num_heads: int = 8,
        head_mlp_ratio: float = 4.0,
        head_qkv_bias: bool = True,
        head_attn_drop: float = 0.0,
        head_proj_drop: float = 0.0,
        head_drop_path: float = 0.0,
        head_norm_epsilon: float = 1e-5,
        mlp_hidden: int = 1024,          # 分类 MLP 隐藏维度
        mlp_depth: int = 2,              # 分类 MLP 层数
        mlp_dropout: float = 0.1,
        freeze_backbone: bool = False,   # 是否冻结骨干
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.backbone = backbone
        self.embed_dim = backbone.embed_dim

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        else:
            # 在下游任务中，需要解冻所有参数（包括target_blocks）
            # 因为target_blocks在预训练时被冻结，但下游任务需要微调
            for p in self.backbone.parameters():
                p.requires_grad = True

        # 确保 dtype 与设备与骨干一致
        if dtype is None:
            # 尽量与骨干保持一致（VolumeVitJEPA 默认为 fp16）
            dtype = next((p.dtype for p in self.backbone.parameters() if p is not None), torch.float16)
        if device is None:
            device = next(self.backbone.parameters()).device

        factory_kwargs = dict(device=device, dtype=dtype)

        # 可学习 CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim, **factory_kwargs))
        nn.init.normal_(self.cls_token, std=0.02)

        # 轻量 Transformer 头（对 [CLS]+tokens 做聚合）
        dpr = [x.item() for x in torch.linspace(0, head_drop_path, head_depth)] if head_depth > 0 else []
        # 这里复用你工程里的 Block 定义，保证与主干的注意力掩码接口一致
        def _norm_layer_with_dtype(dim):
            ln = (nn.LayerNorm)(dim, eps=head_norm_epsilon, **factory_kwargs)
            return ln

        self.head_blocks = nn.ModuleList([
            Block(
                dim=self.embed_dim,
                num_heads=head_num_heads,
                mlp_ratio=head_mlp_ratio,
                qkv_bias=head_qkv_bias,
                attn_drop=head_attn_drop,
                drop=head_proj_drop,
                drop_path=dpr[i] if head_depth > 0 else 0.0,
                norm_layer=_norm_layer_with_dtype,
            ) for i in range(head_depth)
        ])

        self.head_norm = (nn.LayerNorm)(self.embed_dim, eps=head_norm_epsilon, **factory_kwargs)

        # 分类 MLP 头
        mlp_layers = []
        in_dim = self.embed_dim
        for _ in range(max(mlp_depth - 1, 0)):
            mlp_layers += [nn.Linear(in_dim, mlp_hidden, **factory_kwargs), nn.GELU(), nn.Dropout(mlp_dropout)]
            in_dim = mlp_hidden
        mlp_layers += [nn.Linear(in_dim, num_classes, **factory_kwargs)]
        self.classifier = nn.Sequential(*mlp_layers)

    def _encode_backbone(self, x, meta=None, orig_Ts=None, affines=None):
        """
        仅使用骨干的 context encoder（不做mask/不走 predictor/target）
        返回:
          feat:      [B, L, D]
          attn_pad:  [B, L]  (True=padding)
        """
        xf, attn_pad, lengths, _ = self.backbone.patch_embed(x, meta, orig_Ts, affines)  # [B,L,D], [B,L]

        # 直接跑 context encoder 全可见序列
        attn_mask_flash_attn = ~attn_pad
        feat = xf
        for blk in self.backbone.context_blocks:
            feat = blk(feat, attention_mask=attn_mask_flash_attn)
        if self.backbone.norm_f is not None:
            feat = self.backbone.norm_f(feat)
        return feat, attn_pad

    def forward(self, x, meta=None, orig_Ts=None, affines=None):
        """
        x: 与 VolumeVitJEPA.patch_embed 输入保持一致
        返回: logits [B, num_classes]
        """
        feat, attn_pad = self._encode_backbone(x, meta=meta, orig_Ts=orig_Ts, affines=affines)  # [B,L,D], [B,L]
        B, L, D = feat.shape
        device = feat.device

        # 拼接 CLS（有效位），并扩展注意力 mask
        cls_tok = self.cls_token.to(dtype=feat.dtype)              # [1,1,D]
        cls_tok = cls_tok.expand(B, -1, -1)                        # [B,1,D]
        x_cat = torch.cat([cls_tok, feat], dim=1)                  # [B,1+L,D]

        cls_pad = torch.zeros(B, 1, dtype=torch.bool, device=device)  # False=有效
        attn_cat = torch.cat([cls_pad, attn_pad], dim=1)           # [B,1+L]

        # 轻量 Transformer 聚合
        # 注意：flash_attn中的_get_unpad_data期望mask中1表示有效，0表示填充
        # 但我们的attn_cat中False表示有效，True表示填充，所以需要反转
        attn_cat_for_flash = ~attn_cat  # 反转：True=有效, False=填充

        h = x_cat
        for blk in self.head_blocks:
            h = blk(h, attention_mask=attn_cat_for_flash)
        h = self.head_norm(h)

        cls_feat = h[:, 0, :]                                     # [B,D]
        logits = self.classifier(cls_feat)                         # [B,C]
        return logits
    

class VolumeVitJEPAClassifierAvgPool(nn.Module):
    """
    使用 VolumeVitJEPA 的 context encoder 提取特征，
    对所有有效 token 进行平均池化，
    然后通过 MLP 分类头进行分类。

    这是一个更简单的架构，不使用 Transformer 聚合。
    """
    def __init__(
        self,
        backbone: VolumeVitJEPA,
        num_classes: int,
        mlp_hidden: int = 1024,
        mlp_depth: int = 2,
        mlp_dropout: float = 0.1,
        freeze_backbone: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.backbone = backbone
        self.embed_dim = backbone.embed_dim

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        else:
            # 在下游任务中，需要解冻所有参数
            for p in self.backbone.parameters():
                p.requires_grad = True

        # 确保 dtype 与设备与骨干一致
        if dtype is None:
            dtype = next((p.dtype for p in self.backbone.parameters() if p is not None), torch.float16)
        if device is None:
            device = next(self.backbone.parameters()).device

        factory_kwargs = dict(device=device, dtype=dtype)

        # 分类 MLP 头
        mlp_layers = []
        in_dim = self.embed_dim
        for _ in range(max(mlp_depth - 1, 0)):
            mlp_layers += [nn.Linear(in_dim, mlp_hidden, **factory_kwargs), nn.GELU(), nn.Dropout(mlp_dropout)]
            in_dim = mlp_hidden
        mlp_layers += [nn.Linear(in_dim, num_classes, **factory_kwargs)]
        self.classifier = nn.Sequential(*mlp_layers)

    def _encode_backbone(self, x, meta=None, orig_Ts=None, affines=None):
        """
        仅使用骨干的 context encoder
        返回:
          feat:      [B, L, D]
          attn_pad:  [B, L]  (True=padding)
        """
        xf, attn_pad, lengths = self.backbone.patch_embed(x, meta, orig_Ts, affines)

        attn_mask_flash_attn = ~attn_pad
        # 直接跑 context encoder 全可见序列
        feat = xf
        for blk in self.backbone.context_blocks:
            feat = blk(feat, attention_mask=attn_mask_flash_attn)
        if self.backbone.norm_f is not None:
            feat = self.backbone.norm_f(feat)
        return feat, attn_pad

    def forward(self, x, meta=None, orig_Ts=None, affines=None):
        """
        x: 与 VolumeVitJEPA.patch_embed 输入保持一致
        返回: logits [B, num_classes]
        """
        feat, attn_pad = self._encode_backbone(x, meta=meta, orig_Ts=orig_Ts, affines=affines)

        # 平均池化：对所有有效 token 进行平均
        # attn_pad: True=padding, False=valid
        # 创建有效位掩码
        valid_mask = ~attn_pad  # [B, L] -> True=valid

        # 计算每个样本的有效 token 数
        valid_counts = valid_mask.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]

        # 对有效位置求和
        feat_masked = feat * valid_mask.unsqueeze(-1).float()  # [B, L, D]
        feat_sum = feat_masked.sum(dim=1)  # [B, D]

        # 平均
        feat_avg = feat_sum / valid_counts.float()  # [B, D]

        # 分类
        logits = self.classifier(feat_avg)  # [B, num_classes]
        return logits


if __name__ == "__main__":
    # 假定你已有预训练骨干
    backbone = VolumeVitJEPA(embed_dim=1024, depth=24, num_heads=8)

    model = VolumeVitJEPAClassifierCLS(
        backbone=backbone,
        num_classes=10,
        head_depth=2,
        head_num_heads=8,
        head_mlp_ratio=4.0,
        freeze_backbone=False,   # 端到端微调；若只训头部可设 True
    ).to(next(backbone.parameters()).device)

    # 伪数据
    B = 2
    x = torch.randn(B, 1, 32, 64, 64, 64, device=next(backbone.parameters()).device, dtype=next(backbone.parameters()).dtype)  # 形状按你的 patch_embed 接口替换
    meta = None; orig_Ts = None; affines = None

    logits = model(x, meta=meta, orig_Ts=orig_Ts, affines=affines)  # [B, num_classes]
    print(logits.shape)