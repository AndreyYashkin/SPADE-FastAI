from fastai.callback.core import Callback
from fastai.vision.gan import  GANTrainer
#from .model.gan_loss import unet_consistancy_loss
#from .model.cutmix import CustomCutMix
from fastai.callback.schedule import _Annealer, sched_exp


def add_mask(gb, proto_cb):
    return (proto_cb[0], gb[1])


# TODO find which imports are not needed
# old
import tempfile
import os
import shutil
import matplotlib.pyplot as plt
# new
import matplotlib.animation as animation
from torchvision.utils import save_image
from fastcore.basics import store_attr



def SchedExpPart(start, end, start_part, end_part):
    "`Schedule` that can be splited on several fits "
    sched = lambda start, end, pos: sched_exp(start, end, start_part+(end_part-start_part)*pos)
    return _Annealer(sched, start, end)


class GANEvolutionRecorder(Callback):
    "`Callback` that records the evolution of the generator during the training process."
    run_after = GANTrainer
    def __init__(self, test_batch=None, storege_path=None, show_after_fit=False, figsize=None, plot_fn=None):
        store_attr('storege_path,show_after_fit,figsize,plot_fn')
        if test_batch is not None:
            self.test_batch = test_batch[:-1] # HACK this may not work in data another setup. Needs to be generalised?
        self.temp_dir = False if self.storege_path else True
        if self.storege_path:
          os.makedirs(self.storege_path, exist_ok=True)
        else:
          self.storege_path = tempfile.mkdtemp()
        self.records = 0

    def __del__(self):
      if self.temp_dir:
        shutil.rmtree(self.storege_path)

    def file_path(self, record):
      "Get path to save the generator output."
      fn = 'record_{}.png'.format(record)
      return os.path.join(self.storege_path , fn)

    def before_fit(self):
      "Set the batch for tracking the generator evolution if it is not provided."
      if self.test_batch is None:
        if self.dls.valid.n > 0:
          batch = self.dls.valid.one_batch()
        else:
          batch = self.dls.train.one_batch()
        self.test_batch = batch[:-1] # HACK

    def after_epoch(self):
      "Save the output of current generator."
      # FIXME avoid running this code after Learner.show_results and Learner.get_preds
      output = self.generator(*self.test_batch)
      path = self.file_path(self.records)
      if self.plot_fn:
        self.plot_fn(path, self.test_batch, output)
      else:
        save_image(output, path)
      self.records += 1

    def after_fit(self):
      "Plot the evolution of the generator if needed."
      if self.show_after_fit:
        self.plot_animation()

    def create_animation(self, figsize=None):
      "Create matplotlib.animation representing the evolution of the generator."
      # WARNING this will probably crash your notebook if you have to many images or they are too big
      if figsize is None:
        figsize = self.figsize
      fig = plt.figure(figsize=figsize)
      fig.subplots_adjust(left=0.0,
                        bottom=0.0,
                        right=1.0,
                        top=1.0)
      plt.axis("off")
      ims = []
      for i in range(self.records):
        arr = plt.imread(self.file_path(i))
        im = plt.imshow(arr, animated=True)
        ims.append([im])
      plt.close()
      return animation.ArtistAnimation(fig, ims)

    def plot_animation(self, figsize=None):
      "Embed the evolution animation in Jupyter notebook."
      ani = self.create_animation(figsize)
      try:
        from IPython.display import display, HTML
      except:
        warn(f"Cannot import display, HTML from IPython.display.")
        return
      display(HTML(ani.to_jshtml()))


class AddPerceptualLoss(Callback):
    def __init__(self, perceptual_loss):
        super(Callback, self).__init__()
        self.perceptual_loss = perceptual_loss

    def after_loss(self):
        if self.gen_mode:
            real = self.yb[0]
            fake = self.pred
            loss = self.perceptual_loss(real, fake)
            self.learn.loss_grad += loss
            self.learn.loss += loss.clone() # TODO what does it do?

'''
class ConsistencyRegularization(Callback):
    def __init__(self, cutmix_inst:CustomCutMix, policy_fn):
        super(Callback, self).__init__()
        self.cutmix_inst = cutmix_inst
        self.policy_fn = policy_fn
        self.seen_epoch = 0


    def before_epoch(self):
        self.consistancy_mul = policy_fn(self.seen_epoch)


    def after_loss(self):
        #self.learn. ... trainer.gen_mode
        if not self.gen_mode and self.consistancy_mul > 0:
            # getattr(self.learn, 'custom_cut_mix')
            # self=learn
            # self.loss_func(self.pred, *self.yb)
            # TODO можно ли learn -> self???
            dis = self.model
            real = self.yb
            fake = self.pred
            seg = self.xb[1]
            consistancy += self.consistancy_mul * unet_consistancy_loss(dis, real, fake, seg, self.cutmix_inst)
            self.loss_grad += consistancy
            self.loss += consistancy.clone() # TODO what does it do?


    def after_epoch(self):
        self.seen_epoch += 1
'''
