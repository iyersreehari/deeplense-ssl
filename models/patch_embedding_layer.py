import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

class patch_embedding(nn.Module):
    def __init__(
            self,
            image_size: Tuple[int, int],
            patch_size: Tuple[int, int],
            input_channels: int,
            patch_embedding_dim: int,
            padding: bool = False,
            flatten: bool = True, # converts BCHW to B(C*H)W
            bias: bool = True
        ):
        super().__init__()
        assert isinstance(image_size, tuple),\
            'image_size must be a tuple describing image dimension'
        assert isinstance(patch_size, tuple),\
            'patch_size must be a tuple describing patch dimension'
        self.padding = padding
        self.flatten = flatten
        self.patch_size = patch_size
        if not padding:
            assert image_size[0]%patch_size[0] == 0, \
                f'image height {image_size[0]} is not divisible by patch height {patch_size[0]}'
            assert image_size[1]%patch_size[1] == 0, \
                f'image width {image_size[1]} is not divisible by patch width {patch_size[1]}'
            self.num_patches = (image_size[0]//patch_size[0])*(image_size[1]//patch_size[1])
        else:
            self.pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
            self.pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
            self.num_patches = ((image_size[0]+self.pad_h)//patch_size[0])*((image_size[1]+self.pad_w)//patch_size[1])
        self.proj = nn.Conv2d(input_channels, patch_embedding_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
    def forward(self, x):
        if self.padding:
            x = F.pad(x, (0, self.pad_w, 0, self.pad_h))
        x = self.proj(x)
        if self.flatten: # converts BCHW to B(C*H)W
            x = x.flatten(2).transpose(1, 2)
        return x
