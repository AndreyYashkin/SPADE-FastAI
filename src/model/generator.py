import torch
import torch.nn as nn
from torch.nn.functional import one_hot
from torch.nn.utils import spectral_norm
from enum import IntEnum
from .spade_resblk import SPADEResBlk


class GenerationSize(IntEnum):
    s_128x128 = 128
    s_256x256 = 256
    s_512x512 = 512


class Generator(nn.Module):
    def __init__(self, size:GenerationSize, nz, nf, num_embeddings, norm_type, embedding_dim=None):
        super(Generator, self).__init__()
        self.nf = nf
        self.num_embeddings = num_embeddings
        self.pre = nn.Linear(nz, 16*nf*4*4)
        layers = [
            SPADEResBlk(16*nf, 16*nf, num_embeddings, embedding_dim, norm_type),
            nn.Upsample(scale_factor=2),
            SPADEResBlk(16*nf, 16*nf, num_embeddings, embedding_dim, norm_type)
            ]
        if size > GenerationSize.s_128x128:
            layers += [nn.Upsample(scale_factor=2)]
        layers += [
            SPADEResBlk(16*nf, 16*nf, num_embeddings, embedding_dim, norm_type),
            nn.Upsample(scale_factor=2),
            SPADEResBlk(16*nf, 8*nf, num_embeddings, embedding_dim, norm_type),
            nn.Upsample(scale_factor=2),
            SPADEResBlk(8*nf, 4*nf, num_embeddings, embedding_dim, norm_type),
            nn.Upsample(scale_factor=2),
            SPADEResBlk(4*nf, 2*nf, num_embeddings, embedding_dim, norm_type),
            nn.Upsample(scale_factor=2),
            SPADEResBlk(2*nf, nf, num_embeddings, embedding_dim, norm_type),
            ]
        if size == GenerationSize.s_512x512:
            layers += [
                nn.Upsample(scale_factor=2),
                SPADEResBlk(nf, nf//2, num_embeddings, embedding_dim, norm_type)
                ]
            nf //= 2
        layers += [
            nn.LeakyReLU(),
            spectral_norm(nn.Conv2d(1*nf, 3, kernel_size=3, padding=1)),
            nn.Tanh()
            ]
            
        self.main = nn.Sequential(*layers)

        for layer in self.main:
            # SPADEResBlks need also the segmenation mask as input
            if type(layer) == SPADEResBlk:
                layer.register_forward_pre_hook(self.add_segmentation_hook)
        
    
    def forward(self, noise, seg):
        x = self.pre(noise)
        x = x.view(-1, 16*self.nf, 4, 4)
        self.seg = seg.float() # FIXME make dataloader return float segmentation
        x = self.main(x)
        self.seg = None
        return x
    
    
    # add segmentation as a second input parameter
    def add_segmentation_hook(self, module, input):
        return (*input, self.seg)
