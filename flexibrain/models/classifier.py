import torch
import torch.nn as nn
from flexibrain.models.transformer_block import Block

from flexibrain.models.mamba_jepa import VolumeMambaJEPA


class MambaJEPAClassifier(nn.Module):

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
            for p in self.backbone.parameters():
                p.requires_grad = True

        if dtype is None:
            dtype = next(self.backbone.parameters()).dtype
        if device is None:
            device = next(self.backbone.parameters()).device
        factory = dict(device=device, dtype=dtype)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim, **factory))
        nn.init.normal_(self.cls_token, std=0.02)

        dpr = [x.item() for x in torch.linspace(0, head_drop_path, head_depth)] if head_depth > 0 else []
        def _norm_layer_with_dtype(dim):
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

        layers = []
        in_dim = self.embed_dim
        for _ in range(max(mlp_depth - 1, 0)):
            layers += [nn.Linear(in_dim, mlp_hidden, **factory), nn.GELU(), nn.Dropout(mlp_dropout)]
            in_dim = mlp_hidden
        layers += [nn.Linear(in_dim, num_classes, **factory)]
        self.classifier = nn.Sequential(*layers)

    @torch.no_grad()
    def _encode_backbone_nograd(self, x, meta=None, orig_Ts=None, affines=None, inference_params=None):
        xf, attn_pad, _, _ = self.backbone.patch_embed(x, meta, orig_Ts, affines)
        feat = self.backbone._run_blocks(
            xf, attn_pad,
            blocks=self.backbone.blocks,
            norm_layer=self.backbone.norm_f,
            inference_params=inference_params
        )
        return feat, attn_pad

    def _encode_backbone(self, x, meta=None, orig_Ts=None, affines=None, inference_params=None):
        xf, attn_pad, _, _ = self.backbone.patch_embed(x, meta, orig_Ts, affines)

        feat = self.backbone._run_blocks(
            xf, attn_pad,
            blocks=self.backbone.blocks,
            norm_layer=self.backbone.norm_f,
            inference_params=inference_params
        )

        return feat, attn_pad

    def forward_from_tokens(self, tokens, attn_pad, inference_params=None):

        B, L, D = tokens.shape
        device = tokens.device

        cls_tok = self.cls_token.to(dtype=tokens.dtype).expand(B, -1, -1)  # [B,1,D]
        x_cat = torch.cat([cls_tok, tokens], dim=1)                        # [B,1+L,D]

        cls_pad = torch.zeros(B, 1, dtype=torch.bool, device=device)     # False=valid
        attn_cat = torch.cat([cls_pad, attn_pad], dim=1)                 # [B,1+L]

        attn_cat_for_flash = ~attn_cat  

        h = x_cat
        for blk in self.head_blocks:
            h = blk(h, attention_mask=attn_cat_for_flash)
        h = self.head_norm(h)

        cls_feat = h[:, 0, :]                                            # [B,D]
        logits = self.classifier(cls_feat)                                # [B,C]
        return logits

    def forward(self, x, meta=None, orig_Ts=None, affines=None, inference_params=None):

        feat, attn_pad = self._encode_backbone(x, meta=meta, orig_Ts=orig_Ts, affines=affines,
                                             inference_params=inference_params)
        B, L, D = feat.shape
        device = feat.device

        cls_tok = self.cls_token.to(dtype=feat.dtype).expand(B, -1, -1)  # [B,1,D]
        x_cat = torch.cat([cls_tok, feat], dim=1)                        # [B,1+L,D]

        cls_pad = torch.zeros(B, 1, dtype=torch.bool, device=device)     # False=valid
        attn_cat = torch.cat([cls_pad, attn_pad], dim=1)                 # [B,1+L]

        attn_cat_for_flash = ~attn_cat  

        h = x_cat
        for blk in self.head_blocks:
            h = blk(h, attention_mask=attn_cat_for_flash)
        h = self.head_norm(h)

        cls_feat = h[:, 0, :]                                            # [B,D]
        logits = self.classifier(cls_feat)                                # [B,C]
        return logits

class MambaJEPAClassifierAvgPool(nn.Module):

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
            for p in self.backbone.parameters():
                p.requires_grad = True

        if dtype is None:
            dtype = next(self.backbone.parameters()).dtype
        if device is None:
            device = next(self.backbone.parameters()).device
        factory = dict(device=device, dtype=dtype)

        layers = []
        in_dim = self.embed_dim
        for _ in range(max(mlp_depth - 1, 0)):
            layers += [nn.Linear(in_dim, mlp_hidden, **factory), nn.GELU(), nn.Dropout(mlp_dropout)]
            in_dim = mlp_hidden
        layers += [nn.Linear(in_dim, num_classes, **factory)]
        self.classifier = nn.Sequential(*layers)

    def _encode_backbone(self, x, meta=None, orig_Ts=None, affines=None, inference_params=None):
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
        Args:
            tokens: [B, L, D] backbone token
            attn_pad: [B, L] attention mask (True=padding, False=valid)
            inference_params

        Returns:
            logits: [B, num_classes]
        """
        B, L, D = tokens.shape

        valid_mask = ~attn_pad  # [B, L] -> True=valid

        valid_counts = valid_mask.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]

        feat_masked = tokens * valid_mask.unsqueeze(-1).float()  # [B, L, D]
        feat_sum = feat_masked.sum(dim=1)  # [B, D]

        feat_avg = feat_sum / valid_counts.float()  # [B, D]

        logits = self.classifier(feat_avg)  # [B, num_classes]
        return logits

    def forward(self, x, meta=None, orig_Ts=None, affines=None, inference_params=None):

        feat, attn_pad = self._encode_backbone(x, meta=meta, orig_Ts=orig_Ts, affines=affines,
                                             inference_params=inference_params)
        B, L, D = feat.shape

        valid_mask = ~attn_pad  # [B, L] -> True=valid

        valid_counts = valid_mask.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]

        feat_masked = feat * valid_mask.unsqueeze(-1).float()  # [B, L, D]
        feat_sum = feat_masked.sum(dim=1)  # [B, D]

        feat_avg = feat_sum / valid_counts.float()  # [B, D]

        logits = self.classifier(feat_avg)  # [B, num_classes]
        return logits
