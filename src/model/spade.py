import torch
import torch.nn as nn
from torch.nn.functional import interpolate
from torch.nn.utils import spectral_norm


class SPADE(nn.Module):
    # TODO add norm type as param
    def __init__(self, x_features, num_embeddings, norm_type, embedding_dim=None):
        super(SPADE, self).__init__()
        if not embedding_dim:
            embedding_dim = num_embeddings

        # we will calculate affine params, which are spatially-dependent, ourselves from the segmentation mask
        self.norm = norm_type(x_features, affine=False)
        self.seg_feature_extr = nn.Sequential(
            spectral_norm(nn.Conv2d(num_embeddings, embedding_dim, kernel_size=3, padding=1)),
            nn.LeakyReLU()
            )
        self.conv_gamma = spectral_norm(nn.Conv2d(embedding_dim, x_features, kernel_size=3, padding=1))
        self.conv_beta = spectral_norm(nn.Conv2d(embedding_dim, x_features, kernel_size=3, padding=1))

    def forward(self, x, seg):
        x = self.norm(x)

        N, C, H, W = x.size()
        # calculate affine params form the segmentation mask
        seg = interpolate(seg, size=(H,W), mode='nearest')
        seg = self.seg_feature_extr(seg)
        gamma = self.conv_gamma(seg)
        beta = self.conv_beta(seg)
        return (1+gamma)*x + beta # WARNING в статье про 1 сказано не было
