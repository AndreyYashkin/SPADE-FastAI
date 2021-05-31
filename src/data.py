from functools import partial
from fastai.vision.all import *
from fastai.vision.gan import generate_noise
from torch.nn.functional import one_hot

d_mean = [0.5, 0.5, 0.5]
d_std = [0.5, 0.5, 0.5]
gan_stats = (d_mean, d_std)

#'''
class ToOneHotTransform(Transform):
    def __init__(self, class_n, unknowkn=False):
        self.class_n = class_n+1 if unknowkn else class_n
    def encodes(self, seg: TensorMask):
        return one_hot(seg.long(), self.class_n).permute(0, 3, 1, 2)#.float() # FIXME why don't .float() work here and long is returned?
    def decodes(self, seg: TensorMask):
        return torch.argmax(seg, dim=1)
#'''
@typedispatch
def show_batch(xx:tuple, y:TensorImage, samples, ctxs=None, max_n=10, nrows=None, ncols=None, figsize=None, **kwargs):
    if ctxs is None: ctxs = get_grid(min(len(samples), max_n), nrows=nrows, ncols=ncols, add_vert=1, figsize=figsize, double=True)
    for i in range(2):
        ctxs[i::2] = [b.show(ctx=c, **kwargs) for b,c,_ in zip(samples.itemgot(i+1),ctxs[i::2],range(max_n))]
    return ctxs

@typedispatch
def show_results(xx:tuple, y:TensorImage, samples, outs, ctxs=None, max_n=10, nrows=None, ncols=None, figsize=None, **kwargs):
    if ctxs is None: ctxs = get_grid(min(len(samples), max_n), nrows=nrows, ncols=ncols, add_vert=1, figsize=figsize, double=True)
    ctxs[0::2] = [b.show(ctx=c, **kwargs) for b,c,_ in zip(samples.itemgot(1),ctxs[0::2],range(max_n))]
    ctxs[1::2] = [b.show(ctx=c, **kwargs) for b,c,_ in zip(outs.itemgot(0),ctxs[1::2],range(max_n))]
    return ctxs

class SpadeDataLoaders(DataLoaders):
    @classmethod
    @delegates(DataLoaders.from_dblock)
    def from_mask_path_func(cls, path, img_to_mask_path, nz, item_tfms=None, batch_tfms=None, **kwargs):
        dblock = DataBlock((TransformBlock, MaskBlock, ImageBlock),
                   get_items=get_image_files,
                   get_x=[partial(generate_noise, size=nz), img_to_mask_path],
                   get_y=[lambda path: path],
                   splitter=IndexSplitter(valid_idx=[]), # No valid ds is needed for GAN
                   item_tfms=item_tfms,
                   batch_tfms=batch_tfms)
        res = cls.from_dblock(dblock, path, path=path, **kwargs)
        return res
