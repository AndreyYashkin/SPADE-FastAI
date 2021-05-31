import torch
import torch.nn as nn
import torch.nn.functional as F


def gen_hinge_loss(fake_pred, output, *target_args):
    return -fake_pred.mean()


def dis_hinge_loss(real_pred, fake_pred):
    return F.relu(1 - real_pred).mean() + F.relu(1 + fake_pred).mean()


class GenLoss(nn.Module):
    def __init__(self, gen_loss, perceptual_loss, l):
        super().__init__()
        self.gen_loss = gen_loss
        self.perceptual_loss = perceptual_loss
        self.l = l

    def forward(self, fake_pred, output, *target_args):
        #print(target_args)
        return self.gen_loss(fake_pred, output, *target_args) + self.l * self.perceptual_loss(target_args[0], output)


'''
def unet_gen_loss_func(fake_pred, output, *target_args, orig_gen_loss_func=gen_hinge_loss):
    return orig_gen_loss_func(fake_pred[0], output, target) + orig_gen_loss_func(fake_pred[1], output, target)


def unet_dis_loss_func(real_pred, fake_pred, orig_dis_loss_func=dis_hinge_loss):
    return orig_dis_loss_func(real_pred[0], fake_pred[0]) + orig_dis_loss_func(real_pred[1], fake_pred[1])


def unet_consistancy_loss(dis, real, fake, seg, cutmix):
    comb_img = torch.cat([real, fake])
    comb_seg = torch.cat([seg, seg])
    mix_box = cutmix.create_box(*real.shape[-2:])
    consistancy = F.mse_loss(cutmix(dis(comb_img, comb_seg)[1], mix_box=mix_box),
                             dis(cutmix(comb_img, comb_seg, mix_box=mix_box))[1])
    return consistancy
'''
