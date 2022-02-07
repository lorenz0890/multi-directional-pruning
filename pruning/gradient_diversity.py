import torch
from abc import ABC

class GradientDiversity(ABC):
    def __init__(self, lb):
        self.lb = lb
        self.global_gd = 0.0
        self.layer_gd = {}
        self.accum_g = {}

    def update_lgd(self):
        for n in self.accum_g :
            self.layer_gd[n] = self.lb / torch.pow(torch.norm(self.accum_g [n]), 2)


    def update_ggd(self):
        self.global_gd = 0
        for n in self.layer_gd:
            self.global_gd += self.layer_gd[n]

    def update_gd(self, idx):
        if idx % self.lb == 0:
            self.update_lgd()
            self.update_ggd()
            self.reset_accum_grads()

    def norm_grads(self, model):
        for i, (n, p) in enumerate(model.named_parameters()):
            if p.grad is not None:
                p.grad.data = p.grad.data / torch.norm(p.grad.data) # replace by grad clipping?

    def accum_grads(self, model):
        for i, (n, p) in enumerate(model.named_parameters()):
            if not 'bias' in n:
                if not n in self.accum_g:
                    self.accum_g[n] = p.grad
                else:
                    self.accum_g[n] += p.grad

    def reset_accum_grads(self):
        self.accum_g = {}