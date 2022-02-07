import torch

from pruning import RePruning


class RePruningConvDet(RePruning):
    def __init__(self, strength, magnitude_threshold, metric_threshold, lr, attempts, lb):
        super().__init__()
        self.masks = {}
        self.strength = strength
        self.magnitude_threshold = magnitude_threshold
        self.metric_threshold = metric_threshold
        self.attempts = attempts
        self.lr = lr
        self.lb = lb

    def compute_mask(self, model, acm_g, batch_idx):
        if batch_idx % self.lb == 0:
            metrics = {}
            for attempt in range(self.attempts):
                d_min, g_min, n_min, k_min, j_min = float('inf'), None, 0, 0, 0
                for i, (n, p) in enumerate(model.named_parameters()):
                    if p.grad is not None and not 'bias' in n and 'conv' in n:
                        if len(p.shape) == 4:
                            #sparsity = 1 - (torch.count_nonzero(p) / torch.numel(p)).numpy()
                            #if sparsity < 0.25 or True:
                            #print(n, p.shape, flush=True)
                            for k in range(p.shape[0]):
                                for j in range(p.shape[1]):
                                    if not "{}{}{}".format(n, k, j) in metrics:
                                        W = p
                                        G = acm_g[n]
                                        metric = torch.norm(W[k, j, :, :]) / torch.norm((W - self.lr * G)[k, j, :, :])
                                        #print(metric, flush=True)
                                        metrics["{}{}{}".format(n, k, j)] = torch.abs(1 - metric)
                                    if self.metric_threshold < metrics["{}{}{}".format(n, k, j)] < d_min:
                                        #print('x')
                                        d_min = metrics["{}{}{}".format(n, k, j)]
                                        g_min = p.grad
                                        n_min = n
                                        k_min = k
                                        j_min = j

                if g_min is not None:
                    metrics["{}{}{}".format(n_min, k_min, j_min)] = float('inf')
                    for i, (n, p) in enumerate(model.named_parameters()):
                        if p.grad is not None and n == n_min:
                            if len(p.shape) == 4:
                                mask = torch.ones_like(p.data)
                                mask[k_min][j_min] = mask[k_min][j_min] * 0.0
                                if n in self.masks:
                                    self.masks[n] = self.masks[n] * mask
                                else:
                                    self.masks[n] = mask

    def apply_mask(self, model):
        for i, (n, p) in enumerate(model.named_parameters()):
            if n in self.masks:
                p.data = p.data * (self.masks[n] + (torch.ones_like(p.data) - self.masks[n]) * self.strength)
                if p.data.grad is not None:
                    p.data.grad = p.data.grad * (self.masks[n] + (torch.ones_like(p.data) - self.masks[n]) * self.strength)

    def apply_threshold(self, model):
        for i, (n, p) in enumerate(model.named_parameters()):
            p.data = torch.where(torch.abs(p.data) > self.magnitude_threshold, p.data, torch.zeros_like(p.data))
            if p.data.grad is not None:
                p.data.grad = torch.where(torch.abs(p.data.grad) > self.magnitude_threshold, p.data,
                                          torch.zeros_like(p.data.grad))

