import torch.nn as nn
from torch.nn.utils import spectral_norm
from .spade import SPADE


class SPADEBlk(nn.Module):
    def __init__(self, x_in_features, x_out_features, num_embeddings, embedding_dim, norm_type, learned_shortcut=False):
        super(SPADEBlk, self).__init__()
        self.spade = SPADE(x_in_features, num_embeddings, norm_type, embedding_dim=embedding_dim)
        if not learned_shortcut:
            self.activ = nn.LeakyReLU(inplace=True)
            self.conv = spectral_norm(nn.Conv2d(x_in_features, x_out_features, kernel_size=3, padding=1, bias=True))
        else:
            # As it is done in the SPADE repo. This differs from paper where ...
            self.activ = nn.Identity()
            self.conv = spectral_norm(nn.Conv2d(x_in_features, x_out_features, kernel_size=1, padding=0, bias=False))
        

    def forward(self, x, seg):
        x = self.spade(x, seg)
        x = self.activ(x)
        x = self.conv(x)
        return x


class SPADEResBlk(nn.Module):
    def __init__(self, x_in_features, x_out_features, num_embeddings, embedding_dim, norm_type):
        super(SPADEResBlk, self).__init__()
        # in the paper it is suggested to add it only when x_in_features != x_out_features
        self.learned_shortcut = (x_in_features != x_out_features)
        
        self.blk_1 = SPADEBlk(x_in_features, x_out_features, num_embeddings, embedding_dim, norm_type)
        self.blk_2 = SPADEBlk(x_out_features, x_out_features, num_embeddings, embedding_dim, norm_type)
        if self.learned_shortcut:
            self.blk_skip = SPADEBlk(x_in_features, x_out_features, num_embeddings, embedding_dim, norm_type, learned_shortcut=True)
        

    def forward(self, x, seg):
        y = self.blk_1(x, seg)
        y = self.blk_2(y, seg)
        shortcut = self.blk_skip(x, seg) if self.learned_shortcut else x
        return y + shortcut
