import torch

from pruning import GradientDiversity

class GradientDiversityTopKGradients(GradientDiversity):
    def __init__(self, lb, k):
        super().__init__(lb)
        self.delete_g = []
        self.k = k

    def select_delete_grads(self, idx):
        if idx % self.lb == 0:
            self.delete_g = []
            lgd_cpy = self.layer_gd.copy()
            for i in range(0, self.k):
                min_lgd = min(self.layer_gd, key=lgd_cpy.get)
                self.delete_g.append(min_lgd)
                lgd_cpy[min_lgd] = float('inf')
            print(self.delete_g, flush=True)

    def delete_selected_grads(self, model):
        for n in self.delete_g:
            model_state = model.state_dict(keep_vars=True)
            if model_state[n].grad is not None:
                model_state[n].grad.data = model_state[n].grad.data * 0