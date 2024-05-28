import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------
# BYOL Loss
class BYOLLoss(nn.Module):
    '''
    Implements the BYOL loss function
    as mentioned in the original work
    - https://arxiv.org/pdf/2006.07733.pdf
    
    params:
        online_pred1: (torch.Tensor)
            prediction due to online 
            network on view1
        online_pred2: (torch.Tensor)
            prediction due to online 
            network on view2
        target_proj1: (torch.Tensor)
            output of the target 
            projection network on view1
        target_proj2: (torch.Tensor)
            output of the target 
            projection network on view2
            
    return:
        BYOL loss 
    '''
    
    def __init__(self):
        super(BYOLLoss, self).__init__()
        
    def forward(online_pred1: torch.Tensor,\
              target_proj1: torch.Tensor,\
              online_pred2: torch.Tensor,\
              target_proj2: torch.Tensor) -> torch.FloatTensor:
        norm_online_pred1 = F.normalize(online_pred1, dim=-1, p=2)
        norm_online_pred2 = F.normalize(online_pred2, dim=-1, p=2)
        norm_target_proj1 = F.normalize(target_proj1, dim=-1, p=2)
        norm_target_proj2 = F.normalize(target_proj2, dim=-1, p=2)
        
        return (4 - 2*(norm_online_pred1*norm_target_proj1).sum(dim=-1)\
                  - 2*(norm_online_pred2*norm_target_proj2).sum(dim=-1))\
                .sum(dim=-1)
# --------------------------------------------------------------- 