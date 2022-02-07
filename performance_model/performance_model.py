import gc

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
        self.flops_accumulated = 1e-38#0.0
        self.flops_accumulated_base = 1e-38#0.0
        self.flops_current = 1e-38#0.0
        self.flops_current_base = 1e-38#0.0

        self.flops_accumulated_fwd = 1e-38  # 0.0
        self.flops_accumulated_base_fwd = 1e-38  # 0.0
        self.flops_current_fwd = 1e-38  # 0.0
        self.flops_current_base_fwd = 1e-38  # 0.0

        self.flops_accumulated_bwd = 1e-38  # 0.0
        self.flops_accumulated_base_bwd = 1e-38  # 0.0
        self.flops_current_bwd = 1e-38  # 0.0
        self.flops_current_base_bwd = 1e-38  # 0.0

        self.sparsity_current = 0.0

    def eval(self, model, ac = 1):
        stats_batch = self.__current_flops(model, ac)
        self.flops_current = stats_batch[0]
        self.flops_current_base = stats_batch[1]
        self.flops_accumulated += stats_batch[0]
        self.flops_accumulated_base += stats_batch[1]

        self.sparsity_current = stats_batch[2]

        self.flops_current_fwd = stats_batch[3]
        self.flops_current_base_fwd = stats_batch[4]
        self.flops_accumulated_fwd += stats_batch[3]
        self.flops_accumulated_base_fwd += stats_batch[4]

        self.flops_current_bwd = stats_batch[5]
        self.flops_current_base_bwd = stats_batch[6]
        self.flops_accumulated_bwd += stats_batch[5]
        self.flops_accumulated_base_bwd += stats_batch[6]

    def __current_flops(self, model, ac):
        flops_i, flops_i_base, density_i = 0, 0, 0
        flops_i_fwd, flops_i_base_fwd = 0, 0
        flops_i_bwd, flops_i_base_bwd = 0, 0
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
                    flops_i_n_fwd = self.macs[prefix][0] * 2 * density  # fwd
                    flops_i_n_bwd = 2 * flops_i_n_fwd * density_g / ac# bwd = 2*fwd*g_sparsity/ac
                    flops_i_fwd += flops_i_n_fwd
                    flops_i_bwd += flops_i_n_bwd
                    flops_i += flops_i_n_fwd + flops_i_n_bwd
                else:
                    flops_i_fwd += self.macs[prefix][0] * 2 * density
                    flops_i += self.macs[prefix][0] * 2 * density

                density_i += density

                if param.grad is not None:
                    flops_i_n_fwd = self.macs[prefix][0] * 2 # fwd,bwd updaten
                    flops_i_n_bwd = 2 * flops_i_n_fwd
                    flops_i_base_fwd += flops_i_n_fwd
                    flops_i_base_bwd += flops_i_n_bwd
                    flops_i_base += flops_i_n_fwd + flops_i_n_bwd
                else:
                    flops_i_n_fwd = self.macs[prefix][0] * 2
                    flops_i_base += flops_i_n_fwd
                ctr += 1

        return flops_i * 1e-9, flops_i_base * 1e-9, 1 - density_i / ctr, flops_i_fwd, flops_i_base_fwd, flops_i_bwd, flops_i_base_bwd

    def print_memstats(self, batch_idx, interval):
        if batch_idx % interval == 0:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print('Using device:', device)
            print('Batch index:', batch_idx)

            # torch.cuda.empty_cache()
            if device.type == 'cuda':
                print(torch.cuda.get_device_name(0))
                print('Memory Usage:')
                print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
                print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')
                # print(torch.cuda.memory_stats().keys())
                print('Inactive:  ',
                      round(torch.cuda.memory_stats()['inactive_split_bytes.all.current'] / 1024 ** 3, 1),
                      'GB')
                # print(torch.cuda.memory_snapshot())

                ctr = 0
                ctr_mem = 0
                for obj in gc.get_objects():
                    try:
                        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                            ctr += 1
                            ctr_mem += obj.size()
                            # print(type(obj), obj.size())
                    except:
                        pass
                print('Referenced objects:', ctr)
                print('Referenced objects size:', ctr_mem)