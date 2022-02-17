import torch
from torch.distributions import Categorical

from pruning import GradientDiversity

class MonteCarloGDTopKGradients(GradientDiversity):
    def __init__(self, lb, k, se, model):
        super().__init__(lb)
        self.delete_g = []
        self.k = k
        self.se = se
        self.observations = {}
        self.probabilities = {}
        self.names = []
        self.epoch = 0
        ctr = 0
        for n, p in model.named_parameters():
            if 'bias' not in n:
                ctr += 1
                self.names .append(n)
        for n in self.names :
            if 'bias' not in n:
                self.probabilities[n] = 1 / ctr
        self.cat = Categorical(probs=torch.tensor(list(self.probabilities.values())),
                          logits=None,
                          validate_args=None)
        self.observations_ctr = 0

    def __update_probabilities(self):
        # Get observations
        for n in self.delete_g:
            if n not in self.observations:
                self.observations[n] = 1
            else:
                self.observations[n] += 1
        obs_total = sum(self.observations.values())

        # Update Probabilities posterior = (prior + likelihood)/n
        # https://stats.stackexchange.com/questions/411608/bayesian-updating-update-probability-that-measurement-is-real
        self.observations_ctr += 1
        prior = self.probabilities.copy()
        likelihood = self.observations.copy()
        for n in self.probabilities:
            if n in self.observations:
                likelihood[n] = self.observations[n] / obs_total
            else:
                likelihood[n] = 0.0
        posterior = self.probabilities.copy()
        for n in likelihood:
            posterior[n] = prior[n] + (likelihood[n] - prior[n]) / self.observations_ctr
        self.probabilities = posterior
        self.cat.probs = torch.Tensor(list(self.probabilities.values()))

    def __select_delete_grads_probabilistic(self):
        self.del_g = list(set([self.names[self.cat.sample()] for i in range(self.k)]))

    def update_epoch(self, epoch):
        self.epoch = epoch

    def update_k(self, k):
        self.k = k

    def accum_grads(self, model):
        if (self.epoch+1) % self.se == 0:
            for i, (n, p) in enumerate(model.named_parameters()):
                if not 'bias' in n:
                    if not n in self.accum_g:
                        self.accum_g[n] = p.grad
                    else:
                        self.accum_g[n] += p.grad

    def select_delete_grads(self, idx, epoch):
        if (epoch+1) % self.se == 0:
            if (idx+1) % self.lb == 0:
                    self.delete_g = []
                    lgd_cpy = self.layer_gd.copy()
                    for i in range(0, self.k):
                        min_lgd = min(self.layer_gd, key=lgd_cpy.get)
                        self.delete_g.append(min_lgd)
                        lgd_cpy[min_lgd] = float('inf')
                    self.__update_probabilities()
        else:
            if (idx+1) % self.lb == 0:
                self.__select_delete_grads_probabilistic()
                print(self.delete_g, flush=True)

    def delete_selected_grads(self, model):
        for n in self.delete_g:
            model_state = model.state_dict(keep_vars=True)
            if model_state[n].grad is not None:
                model_state[n].grad.data = model_state[n].grad.data * 0