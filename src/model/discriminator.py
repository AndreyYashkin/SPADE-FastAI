import torch
import torch.nn as nn
from torch.nn.functional import one_hot
from torch.nn.utils import spectral_norm
from .spade_resblk import SPADEResBlk


class Discriminator(nn.Module):
    def __init__(self, nf, layers_n, num_embeddings, norm_type):
        super(Discriminator, self).__init__()
        layers = list()
        conv = nn.Conv2d(3 + num_embeddings, nf, kernel_size=4, stride=2, padding=2)
        conv = spectral_norm(conv)
        layers += [
            conv,
            nn.LeakyReLU(0.2)] # inplace=True

        for i in range(1, layers_n):
            stride = 2 if i != layers_n-1 else 1
            nf_new = min(nf * 2, 512)
            conv = nn.Conv2d(nf, nf_new, kernel_size=4, stride=stride, padding=2)
            conv = spectral_norm(conv)
            layers += [
                conv,
                # Using BatchNorm2d and track_running_stats is true leads to train failure.
                # Possibly if GANTrainer is modified to send concatenated batch of real and fake images as it is done
                # in the orig SPADE realisation training without track_running_stats=False will be possible.
                norm_type(nf_new, track_running_stats=False),
                nn.LeakyReLU(0.2, inplace=True)]
            nf = nf_new
        conv = nn.Conv2d(nf, 1, kernel_size=4, stride=1, padding=2)
        conv = spectral_norm(conv)
        layers += [conv]
        self.body = nn.Sequential(*layers)

    def forward(self, img, seg):
        img.__class__ = torch.Tensor
        x = torch.cat([img, seg], 1)
        return self.body(x)


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, Arch, scale_n):
        super(MultiScaleDiscriminator, self).__init__()
        l = [Arch() for i in range(scale_n)]
        self.dis_l = torch.nn.ModuleList(l)

    def downsample(self, x):
        return nn.functional.avg_pool2d(x, kernel_size=3,
                                        stride=2, padding=[1, 1],
                                        count_include_pad=False)

    def forward(self, img, seg):
        seg = seg.float() # FIXME make dataloader return float segmentation
        outs = []
        for dis in self.dis_l:
            out = dis(img, seg)
            out = torch.flatten(out, start_dim=1)
            outs.append(out)
            img = self.downsample(img)
            seg = self.downsample(seg)
        return torch.cat(outs, dim=1)


'''
class UNetDiscriminator(nn.Module):
    def __init__(self, nf, layers_n, num_embeddings, norm_layer=nn.BatchNorm2d):
        super(UNetDiscriminator, self).__init__()
        self.class_n = class_n
        self.encoder = Discriminator(class_n, layers_n, nf, padding_same=True)
        self.middle =  nn.Sequential(
            spectral_norm(nn.Conv2d(1, nf, kernel_size=4, stride=1, padding=1)),
            nn.LeakyReLU(0.2))
        self.skips = list()
        layers = list()
        for layer in reversed(self.encoder.body):
            if type(layer) == nn.Conv2d:
                if layer.stride == 2:
                    new_nf = layer.in_channels * 2
                    spade_resblk = SPADEResBlk(layer.in_channels + nf, new_nf, num_embeddings)
                    nf = new_nf

                    layer.register_forward_pre_hook(self.store_skip_hook)
                    spade_resblk.register_forward_pre_hook(self.get_skip_hook)

                    layers += [
                        spade_resblk,
                        nn.Upsample(scale_factor=2)
                    ]

        layers += [spectral_norm(nn.Conv2d(nf, 1, kernel_size=3, padding=1))]
        self.decoder = nn.Sequential(*layers)


    def forward(self, img, seg):
        x = self.encoder(img, seg)
        self.seg = seg
        y = self.decoder(x)
        self.seg = None
        return x, y

    def store_skip_hook(self, module, input):
        self.skips.append(input)

    def get_skip_hook(self, module, input):
        return torch.cat([input, self.skips.pop()], dim=1)
'''
