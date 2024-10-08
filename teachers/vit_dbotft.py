##################################################
# This file is mostly re-used from:
# https://github.com/liuxingbin/dbot/blob/main/models_vit.py
##################################################

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """Vision Transformer with support for global average pooling"""

    def __init__(self, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        # We use dBOT-ft-in1k,
        # which takes GAP features from the last layer of the transformer
        # and applies a layer norm
        norm_layer = kwargs["norm_layer"]
        embed_dim = kwargs["embed_dim"]
        self.fc_norm = norm_layer(embed_dim)
        del self.norm  # remove the original norm

    def forward_features_orig(self, x):
        # Original forward_features method
        # kept for reference

        B = x.shape[0]
        x = self.patch_embed(x)

        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        # global pool or cls token
        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        gp = x[:, 1:, :].mean(dim=1)  # global pool without cls token
        gp = self.fc_norm(gp)

        return {
            "x_norm_clstoken": gp,
            "x_norm_patchtokens": x[
                :, 1:
            ],  # this is actually pre-norm, because norm is deleted
            "x_prenorm": x,
            "x_prenorm_clstoken": x[:, 0],
            "x_prenorm_patchtokens": x[:, 1:],
        }


def dbotft_vitbase(**kwargs):
    patch_size = kwargs.pop("patch_size", 16)
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def dbotft_vitlarge(**kwargs):
    patch_size = kwargs.pop("patch_size", 16)
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def dbotft_vithuge(**kwargs):
    patch_size = kwargs.pop("patch_size", 14)
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model
