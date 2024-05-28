# adapted from
#     https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
#     https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
import math
import torch
import torch.nn as nn
from .patch_embedding_layer import patch_embedding
from typing import Union, Tuple

class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0,\
            f'dim={dim} is not divisible by num_heads={num_heads}'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  
    random_tensor = torch.bernoulli(torch.empty(shape).fill_(keep_prob)).to(x.device)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    def __init__(
            self,
            drop_prob: float = 0.,
            scale_by_keep: bool = True
        ):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

class MLP(nn.Module):
    def __init__(
            self, 
            input_dim: int, 
            hidden_dim: int = None, 
            output_dim: int = None, 
            activation = nn.GELU, 
            mlp_drop=0.
        ):
        super().__init__()
        output_dim = output_dim if output_dim is not None else input_dim
        self.mlp_layers = nn.Sequential(
                            nn.Linear(input_dim, hidden_dim),
                            activation(),
                            nn.Linear(hidden_dim, output_dim)
                        )
        self.dropout = nn.Dropout(mlp_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp_layers(x)
        x = self.dropout(x)
        return x
        
class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path_rate: float = 0.,
            activation: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
        ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.drop_path1 = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = MLP(
            input_dim=dim,
            hidden_dim=int(dim * mlp_ratio),
            activation=activation,
            mlp_drop=proj_drop,
        )
        self.drop_path2 = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def get_attn(self, x: torch.Tensor) -> torch.Tensor:
        _, attn = self.attn(self.norm1(x))
        return attn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.attn(self.norm1(x))
        x = y + self.drop_path1(y)
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x

class VisionTransformer(nn.Module):
    def __init__(
            self,
            image_size: Union[int, Tuple[int, int]],
            input_channels: int,
            patch_size: Union[int, Tuple[int, int]] = 16,
            num_classes: int = 0, 
            embed_dim: int = 768,
            depth: int = 12,
            drop_path_rate: float = 0.1,
            num_heads: int = 12,
            mlp_ratio: int = 4.,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            pos_drop_rate: float = 0.,
            head_drop_rate: float = 0.,
            class_token: bool = True,
            use_fc_norm: bool = False,
            norm_layer: nn.Module = nn.LayerNorm
        ):
        super().__init__()
        
        assert isinstance(image_size, (tuple, int))
        self.image_size = image_size \
                            if isinstance(image_size, tuple) \
                                else (image_size, image_size)
        
        assert isinstance(patch_size, (tuple, int))
        self.patch_size = patch_size \
                            if isinstance(patch_size, tuple) \
                                else (patch_size, patch_size)
        self.patch_embed = patch_embedding(
                    self.image_size, self.patch_size,
                    input_channels, embed_dim,
                )
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) \
                                                    if class_token else None
        self.num_prefix_tokens = 1 if class_token else 0
        embed_len = self.patch_embed.num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)
        self.pos_drop = nn.Dropout(p=pos_drop_rate)

        self.norm_layer = norm_layer(embed_dim) \
                                if norm_layer is not None else nn.Identity()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.Sequential(*[
                            Block(
                                dim=embed_dim,
                                num_heads=num_heads,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias,
                                #qk_scale=qk_scale,
                                proj_drop=proj_drop,
                                attn_drop=attn_drop, 
                                drop_path_rate=dpr[i], 
                                norm_layer=norm_layer)
                                for i in range(depth)
                            ])
        self.norm = norm_layer(embed_dim)

        self.embed_dim = embed_dim
        
        # Classifier Head
        self.fc_norm = norm_layer(self.embed_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(head_drop_rate)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        nn.init.trunc_normal_(self.pos_embed, std=.02)
        if class_token:
            nn.init.normal_(self.cls_token, std=1e-6)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size[1]
        h0 = h // self.patch_embed.patch_size[0]
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
        
    def forward(self, x: torch.Tensor):
        
        B, C, W, H = x.shape
        x = self.patch_embed(x) # patch linear embedding
        
        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1) if self.cls_token is not None else []          
        x = torch.cat((cls_tokens, x), dim=1)
        
        # add positional encoding to each token
        # x = x + self.pos_embed # in timm implementation
        x = x + self.interpolate_pos_encoding(x, W, H)
        x = self.pos_drop(x)
        
        x = self.blocks(x)
        x = self.norm_layer(x)

        x = x[:, 0]  # class token

        if self.head is not None:
            x = self.fc_norm(x)
            x = self.head_drop(x)
            x = self.head(x)

        return x
