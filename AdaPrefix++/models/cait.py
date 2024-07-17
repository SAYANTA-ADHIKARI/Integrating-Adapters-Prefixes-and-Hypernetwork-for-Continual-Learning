""" Class-Attention in Image Transformers (CaiT)

Paper: 'Going deeper with Image Transformers' - https://arxiv.org/abs/2103.17239

Original code and weights from https://github.com/facebookresearch/deit, copyright below

Modifications and additions for timm hacked together by / Copyright 2021, Ross Wightman
"""

# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
from functools import partial
from argparse import Namespace
from typing import Callable
import torch
import torch.nn as nn

from timm.layers import PatchEmbed, Mlp, DropPath, trunc_normal_
from .peft.adapter import Adapter
from .peft.prefix import Prefix


class ClassAttn(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to do CA

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        has_prefix: bool = False,
        args: Namespace = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.has_prefix = has_prefix
        if has_prefix:
            self.prefix = Prefix(
                prefix_length=args.prefix_length,
                dim=dim,
                num_head=num_heads,
                args=args,
            )

    def forward(self, x, prefix=True):
        B, N, C = x.shape
        q = (
            self.q(x[:, 0])
            .unsqueeze(1)
            .reshape(B, 1, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        if self.has_prefix and prefix:
            k, v = self.prefix([k, v])
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_cls = attn @ v

        x_cls = x_cls.transpose(1, 2).reshape(B, 1, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)

        return x_cls


class LayerScaleBlockClassAttn(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to add CA and LayerScale
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        attn_block: Callable = ClassAttn,
        mlp_block: Callable = Mlp,
        init_values: float = 1e-4,
        has_prefix: bool = False,
        has_adapter: bool = False,
        args: Namespace = None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_block(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            has_prefix=has_prefix,
            args=args,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = mlp_block(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.gamma_1 = nn.Parameter(init_values * torch.ones(dim))
        self.gamma_2 = nn.Parameter(init_values * torch.ones(dim))
        self.has_adapter = has_adapter
        if has_adapter:
            self.adapter = Adapter(
                embed_dim=dim, bottleneck=dim // args.reduction_factor, args=args
            )

    def forward(self, x, x_cls, prefix=True, adapter=True):
        u = torch.cat((x_cls, x), dim=1)
        x_cls = x_cls + self.drop_path(
            self.gamma_1 * self.attn(self.norm1(u), prefix=prefix)
        )
        if self.has_adapter and adapter:
            res = self.adapter(self.mlp(self.norm2(x_cls)))
        else:
            res = self.mlp(self.norm2(x_cls))
        x_cls = x_cls + self.drop_path(self.gamma_2 * res)
        return x_cls


class TalkingHeadAttn(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to add Talking Heads Attention (https://arxiv.org/pdf/2003.02436v1.pdf)
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        has_prefix: bool = False,
        args: Namespace = None,
    ):
        super().__init__()

        self.num_heads = num_heads

        head_dim = dim // num_heads

        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)

        self.proj_l = nn.Linear(num_heads, num_heads)
        self.proj_w = nn.Linear(num_heads, num_heads)

        self.proj_drop = nn.Dropout(proj_drop)
        self.has_prefix = has_prefix
        if has_prefix:
            self.prefix = Prefix(
                prefix_length=args.prefix_length,
                dim=dim,
                num_head=num_heads,
                args=args,
            )

    def forward(self, x, prefix=True):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        if self.has_prefix and prefix:
            k, v = self.prefix([k, v])

        attn = q @ k.transpose(-2, -1)

        attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        attn = attn.softmax(dim=-1)

        attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScaleBlock(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to add layerScale
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: int = 4.0,
        qkv_bias: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        attn_block: Callable = TalkingHeadAttn,
        mlp_block: Callable = Mlp,
        init_values: int = 1e-4,
        has_adapter: bool = False,
        has_prefix: bool = False,
        args: Namespace = None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_block(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            has_prefix=has_prefix,
            args=args,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = mlp_block(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.gamma_1 = nn.Parameter(init_values * torch.ones(dim))
        self.gamma_2 = nn.Parameter(init_values * torch.ones(dim))
        self.has_adapter = has_adapter
        if has_adapter:
            self.adapter = Adapter(
                embed_dim=dim, bottleneck=dim // args.reduction_factor, args=args
            )

    def forward(self, x, adapter=True, prefix=True):
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), prefix=prefix))
        if self.has_adapter and adapter:
            res = self.adapter(self.mlp(self.norm2(x)))
        else:
            res = self.mlp(self.norm2(x))
        x = x + self.drop_path(self.gamma_2 * res)
        return x


class Cait(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to adapt to our cait models
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        global_pool="token",
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        pos_drop_rate=0.0,
        proj_drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        block_layers=LayerScaleBlock,
        block_layers_token=LayerScaleBlockClassAttn,
        patch_layer=PatchEmbed,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        act_layer=nn.GELU,
        attn_block=TalkingHeadAttn,
        mlp_block=Mlp,
        init_values=1e-4,
        attn_block_token_only=ClassAttn,
        mlp_block_token_only=Mlp,
        depth_token_only=2,
        mlp_ratio_token_only=4.0,
        has_output_adapter=False,
        has_prefix=False,
        leave_out=[],
        args=None,
    ):
        super().__init__()
        assert global_pool in ("", "token", "avg")

        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim
        self.grad_checkpointing = False
        self.has_output_adapter = has_output_adapter

        self.patch_embed = patch_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=pos_drop_rate)

        dpr = [drop_path_rate for i in range(depth)]
        self.blocks = nn.ModuleList(
            [
                block_layers(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    attn_block=attn_block,
                    mlp_block=mlp_block,
                    init_values=init_values,
                    has_adapter=has_output_adapter and (i not in leave_out),
                    has_prefix=has_prefix and (i not in leave_out),
                    args=args,
                )
                for i in range(depth)
            ]
        )

        self.blocks_token_only = nn.ModuleList(
            [
                block_layers_token(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio_token_only,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    attn_block=attn_block_token_only,
                    mlp_block=mlp_block_token_only,
                    init_values=init_values,
                    has_adapter=has_output_adapter and (i not in leave_out),
                    has_prefix=has_prefix and (i not in leave_out),
                    args=args,
                )
                for i in range(depth_token_only)
            ]
        )

        self.norm = norm_layer(embed_dim)

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        def _matcher(name):
            if any(
                [name.startswith(n) for n in ("cls_token", "pos_embed", "patch_embed")]
            ):
                return 0
            elif name.startswith("blocks."):
                return int(name.split(".")[1]) + 1
            elif name.startswith("blocks_token_only."):
                # overlap token only blocks with last blocks
                to_offset = len(self.blocks) - len(self.blocks_token_only) + 1
                return int(name.split(".")[1]) + to_offset
            elif name.startswith("norm."):
                return len(self.blocks)
            else:
                return float("inf")

        return _matcher

    def forward_features(self, x, prefix=True, adapter=True):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for i, blk in enumerate(self.blocks):
            x = blk(x, prefix=prefix, adapter=adapter)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        for i, blk in enumerate(self.blocks_token_only):
            cls_tokens = blk(x, cls_tokens, prefix=prefix, adapter=adapter)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, 1:].mean(dim=1) if self.global_pool == "avg" else x[:, 0]
        return x if pre_logits else self.head(x)

    def forward(self, x, prefix=True, adapter=True):
        x = self.forward_features(x, prefix=prefix, adapter=adapter)
        x = self.forward_head(x, pre_logits=True)
        return x
