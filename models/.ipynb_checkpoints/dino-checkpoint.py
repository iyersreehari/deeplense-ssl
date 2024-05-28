# adapted from
#    https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
import torch
import torch.nn as nn
from typing import Union, Tuple, List

class DINOHead(nn.Module):
    def __init__(
            self, 
            input_dim, 
            output_dim, 
            hidden_dim=2048,
            bottleneck_dim=256,
            use_bn=False, 
            norm_last_layer=True, 
            nlayers=3,  
        ):
        super().__init__()
        
        nlayers = max(nlayers, 1)
        layer_dim = []
        layer_dim.append(input_dim)
        for i in range(nlayers-1):
            layer_dim.append(hidden_dim)
        layer_dim.append(bottleneck_dim)

        mlp = []
        for i in range(len(layer_dim)-1):
            mlp.append(nn.Linear(layer_dim[i], layer_dim[i+1]))
            if i < len(layer_dim) - 2:
                if use_bn:
                    mlp.append(nn.BatchNorm1d(layer_dim[i+1]))
                mlp.append(nn.GELU())
        self.mlp = nn.Sequential(*mlp)
            
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, output_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

class MultiCropWrapper(nn.Module):
    def __init__(
            self, 
            backbone, 
            head
        ):
        super().__init__()
        backbone.head = nn.Identity()  # deactivate original head
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        # convert to list
        if not isinstance(x, list):
            x = [x]
        crop_dims = torch.tensor([inp.shape[-1] for inp in x])
        crop_dim_counts = torch.unique_consecutive(crop_dims, return_counts=True)[1]
        crops_idx = torch.cumsum(crop_dim_counts, 0)

        start_idx = 0
        output = torch.empty(0).to(x[0].device)
        
        for end_idx in crops_idx:
            # concatenate similar shaped crops along the batch dim
            x_cat = torch.cat(x[start_idx: end_idx])
            _out = self.backbone(x_cat)
            # The output is a tuple with XCiT model. See:
            # https://github.com/facebookresearch/xcit/blob/master/xcit.py#L404-L405
            if isinstance(_out, tuple):
                _out = _out[0]
            # accumulate outputs
            output = torch.cat((output, _out))
            start_idx = end_idx
        # Run the head forward on the concatenated features.
        return self.head(output)