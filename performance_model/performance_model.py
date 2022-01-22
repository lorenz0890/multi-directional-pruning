import torch
from thop import profile

class PerformanceModel:
    def __init__(self, model, train_loader):

        in_shape = next(enumerate(train_loader))[1][0].shape
        self.macs = profile(model,
                            inputs=(torch.rand(in_shape[0],
                                               in_shape[1],
                                               in_shape[2],
                                               in_shape[3]),),
                            ret_layer_info=True)[2]

        self.macs = profile(model,
                            inputs=(torch.rand(in_shape[0],
                                               in_shape[1],
                                               in_shape[2],
                                               in_shape[3]),),
                            ret_layer_info=True)[2]
        self.flops_accumulated = 0.0
        self.flops_accumulated_base = 0.0

    def eval(self, model, ac = 1):
        stats_batch = self.__current_flops(model, ac)
        self.flops_accumulated += stats_batch[0]
        self.flops_accumulated_base += stats_batch[1]

    def __current_flops(self, model, ac):
        flops_i, flops_i_base, density_i = 0, 0, 0
        ctr = 0
        for name, param in model.named_parameters():
            # https://openai.com/blog/ai-and-compute/
            prefix = name.split('.')[0]
            postfix = name.split('.')[1]
            if postfix != 'bias':
                density = (torch.count_nonzero(param) / torch.numel(param)).numpy()
                density_g = 1.0
                if param.grad is not None:
                    density_g = (torch.count_nonzero(param.grad) / torch.numel(param.grad)).numpy()
                    macs_i_n = self.macs[prefix][0] * 2 * density  # fwd
                    flops_i += macs_i_n + 2 * macs_i_n * density_g / ac # bwd = 2*fwd*g_sparsity
                else:
                    flops_i += self.macs[prefix][0] * 2 * density

                density_i += density
                if param.grad is not None:
                    flops_i_base += self.macs[prefix][0] * 2 * 3
                else:
                    flops_i_base += self.macs[prefix][0] * 2
                ctr += 1

        return flops_i * 1e-9, flops_i_base * 1e-9, 1 - density_i / ctr