from functools import partial
import torch
import torch.nn as nn
import torchvision.models as models


class PerceptualLoss(nn.Module):
    def __init__(self, features_model, layers, loss_fn): # TODO check that there is no problems with layers order
        super().__init__()
        l = list(features_model.children())[:max(layers.keys())+1]
        self.features_model = nn.Sequential(*l)
        for p in self.features_model.parameters():
            p.requires_grad = False
        self.features_model.eval()
        self.weights = layers.values()
        self.loss_fn = loss_fn
        self.features = list()
        for id in layers.keys():
            layer = self.features_model[id]
            hook_fn = partial(self.save_features_hook)
            layer.register_forward_hook(hook_fn)

    def forward(self, predict, target):
        predict_features = self.get_features(predict)
        target_features = self.get_features(target)
        loss = torch.zeros(1, device=predict.device)
        for p_f, t_f, w in zip(predict_features, target_features, self.weights):
            loss += self.loss_fn(p_f, t_f) * w
        return loss

    def get_features(self, x):
        self.features_model(x)
        features = self.features
        self.features = list()
        return features

    def save_features_hook(self, module, input, output):
        self.features.append(output)


    @classmethod
    def from_VGG19(cls, layers={1:1.0/32, 6:1.0/16, 11:1.0/8, 20:1.0/4, 29:1.0}, loss_fn=nn.L1Loss()):
        model = models.vgg19(pretrained=True).features
        return cls(model, layers, loss_fn)
