import torch.nn as nn
from torch.distributions.beta import Beta


class CustomCutMix(nn.Module):
    def __init__(self, alpha = 0.5):
        super(Model, self).__init__()
        self.distrib = Beta(torch.tensor(alpha), torch.tensor(alpha))
    
    
    def forward(self, *blocks, mix_box):
        bs = blocks[0].shape[0]
        shuffle = torch.randperm(bs)
        for block in blocks:
            temp_block = block[shuffle]
            blocks[..., y1:y2, x1:x2] = temp_block[..., y1:y2, x1:x2]
        
        return blocks
    
    
    def create_box(self, W, H):
        lam = self.distrib.sample((1,))
        
        cut_rat = torch.sqrt(1. - lam)
        cut_w = torch.round(W * cut_rat).type(torch.long)
        cut_h = torch.round(H * cut_rat).type(torch.long)
        # uniform
        cx = torch.randint(0, W, (1,))
        cy = torch.randint(0, H, (1,))
        x1 = torch.clamp(cx - cut_w // 2, 0, W)
        y1 = torch.clamp(cy - cut_h // 2, 0, H)
        x2 = torch.clamp(cx + cut_w // 2, 0, W)
        y2 = torch.clamp(cy + cut_h // 2, 0, H)
        return x1, y1, x2, y2
