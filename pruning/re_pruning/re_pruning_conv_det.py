import numpy as np
import torch

from pruning import RePruning


class RePruningConvDet(RePruning):
    def __init__(self, softness, magnitude_threshold, metric_threshold, lr, attempts, lb, scale):
        super().__init__()
        self.masks = {}
        self.strength = softness
        self.magnitude_threshold = magnitude_threshold
        self.metric_threshold = metric_threshold
        self.attempts = attempts
        self.lr = lr
        self.lb = lb
        self.scale = scale
        self.metrics = {}
    def compute_mask(self, model, acm_g, batch_idx):
        if batch_idx % self.lb == 0:
            sz = 5 #TODO make adjustable or add auto-adjust - maybe random with lower, upper bounds?
            self.metrics = {}
            for attempt in range(self.attempts):
                d_min, g_min, n_min, k_min, j_min, sz_k_min, sz_j_min = float('inf'), None, 0, 0, 0, sz, sz
                for i, (n, p) in enumerate(model.named_parameters()):
                    if p.grad is not None and not 'bias' in n and ('conv' in n or 'features' in n):
                        if len(p.shape) == 4:
                            #print(n, flush=True)
                            W = p
                            G = acm_g[n]
                            W0 = W - self.lr * G
                            sz_k = np.random.randint(int(p.shape[0] * self.scale)+1, p.shape[0]+1)
                            sz_j = np.random.randint(int(p.shape[1] * self.scale)+1, p.shape[1]+1)
                            for k in range(0, p.shape[0]-sz_k, sz_k):
                                for j in range(0, p.shape[1]-sz_j, sz_j):
                                    if not "{}{}{}{}{}".format(n, k, j, sz_k, sz_j) in self.metrics:
                                        metric = torch.norm(W[k:k+sz_k, j+sz_j, :, :]) / torch.norm(W0[k+sz_k, j+sz_j, :, :])
                                        self.metrics["{}{}{}{}{}".format(n, k, j, sz_k, sz_j)] = torch.abs(1 - metric)
                                    if self.metrics["{}{}{}{}{}".format(n, k, j, sz_k, sz_j)] < d_min and self.metrics["{}{}{}{}{}".format(n, k, j, sz_k, sz_j)] <= self.metric_threshold:
                                        d_min = self.metrics["{}{}{}{}{}".format(n, k, j, sz_k, sz_j)]
                                        g_min = p.grad
                                        n_min = n
                                        k_min = k
                                        j_min = j
                                        #sz_min = sz
                                        sz_k_min = sz_k
                                        sz_j_min = sz_j


                if g_min is not None:
                    self.metrics["{}{}{}{}{}".format(n_min, k_min, j_min, sz_k_min, sz_j_min)] = float('inf')
                    for i, (n, p) in enumerate(model.named_parameters()):
                        if p.grad is not None and n == n_min:
                            if len(p.shape) == 4:
                                mask = torch.ones_like(p.data)
                                mask[k_min:k_min+sz_k_min][j_min:j_min+sz_j_min] = mask[k_min:k_min+sz_k_min][j_min:j_min+sz_j_min] * 0.0
                                if n in self.masks:
                                    self.masks[n] = self.masks[n] * mask *  torch.where(torch.abs(p.data) > self.magnitude_threshold, torch.ones_like(p.data),
                                              torch.zeros_like(p.data))
                                else:
                                    self.masks[n] = mask *  torch.where(torch.abs(p.data) > self.magnitude_threshold, torch.ones_like(p.data),
                                              torch.zeros_like(p.data))

    def apply_mask(self, model):
        for i, (n, p) in enumerate(model.named_parameters()):
            if p.grad is not None and not 'bias' in n and ('conv' in n or 'features' in n):
                if n in self.masks:
                    p.data = p.data * (self.masks[n] + (torch.ones_like(p.data) - self.masks[n]) * self.strength)
                    if p.data.grad is not None:
                        p.data.grad = p.data.grad * (self.masks[n] + (torch.ones_like(p.data) - self.masks[n]) * self.strength)

    def apply_threshold(self, model):
        for i, (n, p) in enumerate(model.named_parameters()):
            if p.grad is not None and not 'bias' in n and ('conv' in n or 'features' in n):
                p.data = torch.where(torch.abs(p.data) > self.magnitude_threshold, p.data, torch.zeros_like(p.data))
                if p.data.grad is not None:
                    p.data.grad = torch.where(torch.abs(p.data.grad) > self.magnitude_threshold, p.data,
                                              torch.zeros_like(p.data.grad))

