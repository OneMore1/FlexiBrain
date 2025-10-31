import torch
import torch.nn as nn
import sys
import os
from timm.models.vision_transformer import DropPath

# Add Brain-Harmony to path
brain_harmony_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Brain-Harmony')
sys.path.insert(0, brain_harmony_path)

from libs.flex_transformer import Block
from mamba_mae.models_vim_mae import VolumeMambaJEPA
from flex_patch_utils.utils import meta_to_matrix


# ---------- 下游分类器（CLS + 轻量 Transformer 聚合头） ----------
class MambaJEPAClassifier(nn.Module):
    """
    只用 VolumeMambaJEPA 的 context encoder (patch_embed + blocks + norm_f) 提特征，
    前拼 CLS，过若干轻量 TransformerBlock，取 CLS 做 MLP 分类。
    """
    def __init__(
        self,
        backbone: 'VolumeMambaJEPA',
        num_classes: int,
        head_depth: int = 2,
        head_num_heads: int = 8,
        head_mlp_ratio: float = 4.0,
        head_qkv_bias: bool = True,
        head_attn_drop: float = 0.0,
        head_proj_drop: float = 0.0,
        head_drop_path: float = 0.0,
        head_norm_epsilon: float = 1e-5,
        # 分类头
        mlp_hidden: int = 1024,
        mlp_depth: int = 2,
        mlp_dropout: float = 0.1,
        # 训练策略
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
            # 在下游任务中，需要解冻所有参数（包括target_blocks）
            # 因为target_blocks在预训练时被冻结，但下游任务需要微调
            for p in self.backbone.parameters():
                p.requires_grad = True

        if dtype is None:
            dtype = next(self.backbone.parameters()).dtype
        if device is None:
            device = next(self.backbone.parameters()).device
        factory = dict(device=device, dtype=dtype)

        # CLS
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim, **factory))
        nn.init.normal_(self.cls_token, std=0.02)

        # 轻量 Transformer 聚合头
        dpr = [x.item() for x in torch.linspace(0, head_drop_path, head_depth)] if head_depth > 0 else []
        # 这里复用你工程里的 Block 定义，保证与主干的注意力掩码接口一致
        def _norm_layer_with_dtype(dim):
            # nn.LayerNorm不接受device和dtype参数，需要分开处理
            ln = nn.LayerNorm(dim, eps=head_norm_epsilon)
            if device is not None:
                ln = ln.to(device=device)
            if dtype is not None:
                ln = ln.to(dtype=dtype)
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
        self.head_norm = nn.LayerNorm(self.embed_dim, eps=head_norm_epsilon, **factory)

        # 分类 MLP
        layers = []
        in_dim = self.embed_dim
        for _ in range(max(mlp_depth - 1, 0)):
            layers += [nn.Linear(in_dim, mlp_hidden, **factory), nn.GELU(), nn.Dropout(mlp_dropout)]
            in_dim = mlp_hidden
        layers += [nn.Linear(in_dim, num_classes, **factory)]
        self.classifier = nn.Sequential(*layers)

    @torch.no_grad()
    def _encode_backbone_nograd(self, x, meta=None, orig_Ts=None, affines=None, inference_params=None):
        # 与你的 backbone 一致：不做遮挡，直接跑 context encoder
        xf, attn_pad, _, _ = self.backbone.patch_embed(x, meta, orig_Ts, affines)
        feat = self.backbone._run_blocks(
            xf, attn_pad,
            blocks=self.backbone.blocks,
            norm_layer=self.backbone.norm_f,
            inference_params=inference_params
        )
        return feat, attn_pad

    def _encode_backbone(self, x, meta=None, orig_Ts=None, affines=None, inference_params=None, explain_mode=False):
        # 在explain_mode下，patch_embed不删除背景
        if explain_mode:
            xf, attn_pad, _, _ = self.backbone.patch_embed(x, meta, orig_Ts, affines, explain_mode=True)
        else:
            xf, attn_pad, _, _ = self.backbone.patch_embed(x, meta, orig_Ts, affines)

        feat = self.backbone._run_blocks(
            xf, attn_pad,
            blocks=self.backbone.blocks,
            norm_layer=self.backbone.norm_f,
            inference_params=inference_params
        )

        return feat, attn_pad

    def forward_from_tokens(self, tokens, attn_pad, inference_params=None):
        """
        从已经提取的token开始进行前向传播

        Args:
            tokens: [B, L, D] backbone特征token
            attn_pad: [B, L] attention mask (True=padding, False=valid)
            inference_params: 推理参数

        Returns:
            logits: [B, num_classes]
        """
        B, L, D = tokens.shape
        device = tokens.device

        # 拼 CLS，并扩展 mask（CLS 为有效位 False）
        cls_tok = self.cls_token.to(dtype=tokens.dtype).expand(B, -1, -1)  # [B,1,D]
        x_cat = torch.cat([cls_tok, tokens], dim=1)                        # [B,1+L,D]

        cls_pad = torch.zeros(B, 1, dtype=torch.bool, device=device)     # False=valid
        attn_cat = torch.cat([cls_pad, attn_pad], dim=1)                 # [B,1+L]

        # 轻量 Transformer 聚合
        # 注意：flash_attn中的_get_unpad_data期望mask中1表示有效，0表示填充
        # 但我们的attn_cat中False表示有效，True表示填充，所以需要反转
        attn_cat_for_flash = ~attn_cat  # 反转：True=有效, False=填充

        h = x_cat
        for blk in self.head_blocks:
            h = blk(h, attention_mask=attn_cat_for_flash)
        h = self.head_norm(h)

        cls_feat = h[:, 0, :]                                            # [B,D]
        logits = self.classifier(cls_feat)                                # [B,C]
        return logits

    def forward(self, x, meta=None, orig_Ts=None, affines=None, inference_params=None, explain_mode=False):
        """
        返回: logits [B, num_classes]
        """
        feat, attn_pad = self._encode_backbone(x, meta=meta, orig_Ts=orig_Ts, affines=affines,
                                             inference_params=inference_params, explain_mode=explain_mode)
        B, L, D = feat.shape
        device = feat.device

        # cond_vec = torch.from_numpy(meta_to_matrix(meta, B)).to(device)
        # cond_vec = self.backbone.reso_prior_e(cond_vec) # [B,5]
        # feat = self.backbone.CondLoRA(feat, cond_vec, attn_mask=attn_pad)
        # feat, moe_aux = self.backbone.moe(feat, attn_mask=attn_pad, cond_vec=None)
        # 拼 CLS，并扩展 mask（CLS 为有效位 False）
        cls_tok = self.cls_token.to(dtype=feat.dtype).expand(B, -1, -1)  # [B,1,D]
        x_cat = torch.cat([cls_tok, feat], dim=1)                        # [B,1+L,D]

        cls_pad = torch.zeros(B, 1, dtype=torch.bool, device=device)     # False=valid
        attn_cat = torch.cat([cls_pad, attn_pad], dim=1)                 # [B,1+L]

        # 轻量 Transformer 聚合
        # 注意：flash_attn中的_get_unpad_data期望mask中1表示有效，0表示填充
        # 但我们的attn_cat中False表示有效，True表示填充，所以需要反转
        attn_cat_for_flash = ~attn_cat  # 反转：True=有效, False=填充

        h = x_cat
        for blk in self.head_blocks:
            h = blk(h, attention_mask=attn_cat_for_flash)
        h = self.head_norm(h)

        cls_feat = h[:, 0, :]                                            # [B,D]
        logits = self.classifier(cls_feat)                                # [B,C]
        return logits

# ---------- 下游分类器（平均池化 + MLP） ----------
class MambaJEPAClassifierAvgPool(nn.Module):
    """
    使用 VolumeMambaJEPA 的 context encoder 提取特征，
    对所有有效 token 进行平均池化，
    然后通过 MLP 分类头进行分类。

    这是一个更简单的架构，不使用 Transformer 聚合。
    """
    def __init__(
        self,
        backbone: 'VolumeMambaJEPA',
        num_classes: int,
        mlp_hidden: int = 1024,
        mlp_depth: int = 3,
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

        if dtype is None:
            dtype = next(self.backbone.parameters()).dtype
        if device is None:
            device = next(self.backbone.parameters()).device
        factory = dict(device=device, dtype=dtype)

        # 分类 MLP
        layers = []
        in_dim = self.embed_dim
        for _ in range(max(mlp_depth - 1, 0)):
            layers += [nn.Linear(in_dim, mlp_hidden, **factory), nn.GELU(), nn.Dropout(mlp_dropout)]
            in_dim = mlp_hidden
        layers += [nn.Linear(in_dim, num_classes, **factory)]
        self.classifier = nn.Sequential(*layers)

    def _encode_backbone(self, x, meta=None, orig_Ts=None, affines=None, inference_params=None, explain_mode=False):
        # 在explain_mode下，patch_embed不删除背景
        if explain_mode:
            xf, attn_pad, _ = self.backbone.patch_embed(x, meta, orig_Ts, affines, explain_mode=True)
        else:
            xf, attn_pad, _ = self.backbone.patch_embed(x, meta, orig_Ts, affines)

        feat = self.backbone._run_blocks(
            xf, attn_pad,
            blocks=self.backbone.blocks,
            norm_layer=self.backbone.norm_f,
            inference_params=inference_params
        )
        return feat, attn_pad

    def forward_from_tokens(self, tokens, attn_pad, inference_params=None):
        """
        从已经提取的token开始进行前向传播

        Args:
            tokens: [B, L, D] backbone特征token
            attn_pad: [B, L] attention mask (True=padding, False=valid)
            inference_params: 推理参数

        Returns:
            logits: [B, num_classes]
        """
        B, L, D = tokens.shape

        # 平均池化：对所有有效 token 进行平均
        # attn_pad: True=padding, False=valid
        # 创建有效位掩码
        valid_mask = ~attn_pad  # [B, L] -> True=valid

        # 计算每个样本的有效 token 数
        valid_counts = valid_mask.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]

        # 对有效位置求和
        feat_masked = tokens * valid_mask.unsqueeze(-1).float()  # [B, L, D]
        feat_sum = feat_masked.sum(dim=1)  # [B, D]

        # 平均
        feat_avg = feat_sum / valid_counts.float()  # [B, D]

        # 分类
        logits = self.classifier(feat_avg)  # [B, num_classes]
        return logits

    def forward(self, x, meta=None, orig_Ts=None, affines=None, inference_params=None, explain_mode=True):
        """
        返回: logits [B, num_classes]
        """
        feat, attn_pad = self._encode_backbone(x, meta=meta, orig_Ts=orig_Ts, affines=affines,
                                             inference_params=inference_params, explain_mode=explain_mode)
        B, L, D = feat.shape

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
    # 假设 VolumeMambaJEPA 已按你的代码定义
    backbone = VolumeMambaJEPA(
        embed_dim=1024,
        depth=24,
        predictor_depth=2,
        mixer_type="mamba",
        bimamba_type="v2",
        fused_add_norm=True,
        residual_in_fp32=False,
        dtype=torch.float16,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    # backbone.load_state_dict(torch.load("mamba_jepa_pretrain.pt"), strict=True)

    model = MambaJEPAClassifier(
        backbone=backbone,
        num_classes=10,
        head_depth=2,
        head_num_heads=8,
        head_mlp_ratio=4.0,
        head_proj_drop=0.1,
        head_drop_path=0.05,
        freeze_backbone=False,
    ).to(next(backbone.parameters()).device)

    # 伪数据（形状按你的 STAPE4D_TimeToSpace 接口替换）
    B = 2
    x = torch.randn(B, 1, 32, 64, 64, 64,
                    device=next(backbone.parameters()).device,
                    dtype=next(backbone.parameters()).dtype)
    meta = orig_Ts = affines = None

    with torch.cuda.amp.autocast(enabled=(x.dtype in (torch.float16, torch.bfloat16))):
        logits = model(x, meta=meta, orig_Ts=orig_Ts, affines=affines)
    print("logits:", logits.shape)  # [B, num_classes]