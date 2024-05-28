# adapted from
#     https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
#     https://github.com/facebookresearch/dino/blob/main/vision_transformer.py

import torch
import torch.nn as nn
from .vit import VisionTransformer
from functools import partial
from typing import Union, Tuple, List

def vit_tiny(
        image_size: Union[int, Tuple[int, int]],
        input_channels: int,
        patch_size: Union[int, Tuple[int, int]] = 16,
        **kwargs):
    return VisionTransformer(
                image_size = image_size,
                input_channels = input_channels,
                patch_size=patch_size, 
                embed_dim=192, 
                depth=12, 
                num_heads=3, 
                mlp_ratio=4,
                qkv_bias=True, 
                norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                **kwargs
            )

def vit_small(
        image_size: Union[int, Tuple[int, int]],
        input_channels: int,
        patch_size: Union[int, Tuple[int, int]] = 16,
        **kwargs):
    return VisionTransformer(
                image_size = image_size,
                input_channels = input_channels,
                patch_size=patch_size, 
                embed_dim=384, 
                depth=12, 
                num_heads=6, 
                mlp_ratio=4,
                qkv_bias=True, 
                norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                **kwargs
            )

def vit_base(
        image_size: Union[int, Tuple[int, int]],
        input_channels: int,
        patch_size: Union[int, Tuple[int, int]] = 16,
        **kwargs):
    return VisionTransformer(
                image_size = image_size,
                input_channels = input_channels,
                patch_size=patch_size, 
                embed_dim=768, 
                depth=12, 
                num_heads=12, 
                mlp_ratio=4,
                qkv_bias=True, 
                norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                **kwargs
            )

def vit_backbone(
        arch: str,
        image_size: Union[int, Tuple[int, int]],
        input_channels: int,
        patch_size: Union[int, Tuple[int, int]] = 16,
    ):
    if arch.lower() == "vit_tiny":
        return vit_tiny(image_size, input_channels, patch_size)
    elif arch.lower() == "vit_small":
        return vit_small(image_size, input_channels, patch_size)
    elif arch.lower() == "vit_base":
        return vit_base(image_size, input_channels, patch_size)
