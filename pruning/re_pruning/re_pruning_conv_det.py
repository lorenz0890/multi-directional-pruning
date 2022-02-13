import numpy as np
import torch

from pruning import RePruning


class RePruningConvDet(RePruning):
    def __init__(self, softness, magnitude_threshold, metric_threshold, lr, sample, lb, scale, prune):
        super().__init__()
        self.masks = {}
        self.strength = softness
        self.magnitude_threshold = magnitude_threshold
        self.metric_threshold = metric_threshold
        self.sample = sample
        self.prune = prune
        self.lr = lr
        self.lb = lb
        self.scale = scale
        self.metrics = {}
    def compute_mask(self, model, acm_g, batch_idx):
        with torch.no_grad():
            if batch_idx % self.lb == 0:
                self.metrics = {}
                for sample in range(self.sample):
                    for i, (n, p) in enumerate(model.named_parameters()):
                        if p.grad is not None and (not 'bias' in n) and ('conv' in n or 'features' in n):
                            if len(p.shape) == 4:
                                W = p
                                G = acm_g[n]
                                W0 = W - self.lr * G
                                sz_k = np.random.randint(int(p.shape[0] * self.scale)+1, p.shape[0]+1) #TODO ensure we can reach every filter
                                sz_j = np.random.randint(int(p.shape[1] * self.scale)+1, p.shape[1]+1)
                                for k in range(0, p.shape[0]-sz_k, sz_k):
                                    if p.shape[1] == 1:
                                        j = 0
                                        if not "{}:{}:{}:{}:{}".format(n, k, j, sz_k, sz_j) in self.metrics:
                                            norm_w = torch.norm(W[k:k+sz_k, j, :, :])
                                            if norm_w != 0:
                                                norm_w0 = torch.norm(W0[k:k + sz_k, j, :, :])
                                                metric = norm_w / norm_w0
                                                self.metrics["{}:{}:{}:{}:{}".format(n, k, j, sz_k, sz_j)] = torch.abs(1 - metric).item()
                                            else:
                                                self.metrics["{}:{}:{}:{}:{}".format(n, k, j, sz_k, sz_j)] = float('inf')
                                    else:
                                        for j in range(0, p.shape[1]-sz_j, sz_j):
                                            if not "{}:{}:{}:{}:{}".format(n, k, j, sz_k, sz_j) in self.metrics:
                                                norm_w = torch.norm(W[k:k+sz_k, j+sz_j, :, :])
                                                if norm_w != 0:
                                                    norm_w0 = torch.norm(W0[k:k + sz_k, j:j + sz_j, :, :])
                                                    metric = norm_w / norm_w0
                                                    self.metrics["{}:{}:{}:{}:{}".format(n, k, j, sz_k, sz_j)] = torch.abs(1 - metric).item()
                                                else:
                                                    self.metrics["{}:{}:{}:{}:{}".format(n, k, j, sz_k, sz_j)] = float('inf')

                for a in range(0, self.prune):
                    if len(self.metrics) > 0:
                        best_key = min(self.metrics, key=self.metrics.get)
                        if self.metrics[best_key] == float("inf") or self.metrics[best_key] > self.metric_threshold:
                            break
                        else:
                            del self.metrics[best_key]
                        idx = best_key.split(':')
                        mask = torch.ones_like(model.state_dict()[idx[0]].data)
                        mask[int(idx[1]):int(idx[1]) + int(idx[3])][int(idx[2]):int(idx[2]) + int(idx[4])] = mask[int(idx[1]):int(idx[1]) + int(idx[3])][int(idx[2]):int(idx[2]) + int(idx[4])] * 0.0
                        if idx[0] in self.masks:
                            self.masks[idx[0]] = self.masks[idx[0]] * mask * torch.where(
                                torch.abs(model.state_dict()[idx[0]].data) > self.magnitude_threshold, torch.ones_like(model.state_dict()[idx[0]].data),
                                torch.zeros_like(model.state_dict()[idx[0]].data))
                        else:
                            self.masks[idx[0]] = mask * torch.where(torch.abs(model.state_dict()[idx[0]].data) > self.magnitude_threshold,
                                                               torch.ones_like(model.state_dict()[idx[0]].data),
                                                               torch.zeros_like(model.state_dict()[idx[0]].data))



    def apply_mask(self, model):
        with torch.no_grad():
            for i, (n, p) in enumerate(model.named_parameters()):
                if p.grad is not None and not 'bias' in n and ('conv' in n or 'features' in n):
                    if n in self.masks:
                        p.data = p.data * (self.masks[n] + (torch.ones_like(p.data) - self.masks[n]) * self.strength)
                        if p.data.grad is not None:
                            p.data.grad = p.data.grad * self.masks[n]#(self.masks[n] + (torch.ones_like(p.data) - self.masks[n]) * self.strength)

    def apply_threshold(self, model):
        with torch.no_grad():
            for i, (n, p) in enumerate(model.named_parameters()):
                if p.grad is not None and not 'bias' in n and ('conv' in n or 'features' in n):
                    p.data = torch.where(torch.abs(p.data) > self.magnitude_threshold, p.data, torch.zeros_like(p.data))
                    if p.data.grad is not None:
                        p.data.grad = torch.where(torch.abs(p.data.grad) > self.magnitude_threshold, p.data,
                                                  torch.zeros_like(p.data.grad))

