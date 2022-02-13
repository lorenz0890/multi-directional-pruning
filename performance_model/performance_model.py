import gc

import numpy as np
import torch
from thop import profile

class PerformanceModel:
    def __init__(self, model, train_loader, config, overhead_gd_top_k=False,
                 overhead_gd_top_k_mc=False, overhead_re_pruning=False, ):

        self.config = config
        self.est_oh_gd_top_k = overhead_gd_top_k
        self.est_oh_gd_top_k_mc = overhead_gd_top_k_mc
        self.est_oh_re_pruning = overhead_re_pruning

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
        self.c_sparsity_current = 0.0
        self.l_sparsity_current = 0.0


    def __estimate_overhead_re_pruning(self, model):
        # https://coek.info/pdf-algorithms-54ed9513cfa623e30da35434ea7edcc833265.html
        # https://mast.queensu.ca/~andrew/notes/pdf/2010a.pdf
        oh_search_space_iteration = 0 #TODO ensure that if multiple methods combined, accum is only accounted for once
        for name, param in model.named_parameters():
            prefix = name.split('.')[0]
            postfix = name.split('.')[1]
            if postfix != 'bias':
                density = (torch.count_nonzero(param) / torch.numel(param)).numpy()
                # RE Pruning
                if 'conv' in name or 'features' in name:
                    oh_w0 = 2 * torch.count_nonzero(param) / self.config.get('SPECIFICATION', 'lb', int)  # Obtaining W0 from accumulated grads very lb batches
                    oh_metric = 2 * (density * (np.prod(list(param.shape)))) / self.config.get('SPECIFICATION', 'lb', int)  # 2 times Frobenius Norm every lb batches
                    oh_search_space_iteration += oh_w0 + oh_metric * self.config.get('SPECIFICATION', 'sample_c', int) * self.config.get('SPECIFICATION', 'scale_c', float)  # metric times attempts times scale
                if 'fc' in name or 'classifier' in name:
                    oh_w0 = 2 * torch.count_nonzero(param) / self.config.get('SPECIFICATION', 'lb',int)
                    oh_metric = 2 * (density * (np.prod(list(param.shape)))) / self.config.get('SPECIFICATION', 'lb',int)
                    oh_search_space_iteration += oh_w0 + oh_metric * self.config.get('SPECIFICATION', 'sample_l', int) * self.config.get('SPECIFICATION', 'scale_l', float)
        return oh_search_space_iteration

    def __estimate_overhead_gd_top_k(self, model):
        # https://coek.info/pdf-algorithms-54ed9513cfa623e30da35434ea7edcc833265.html
        # https://mast.queensu.ca/~andrew/notes/pdf/2010a.pdf
        oh_search_space_iteration = 0, 0
        for name, param in model.named_parameters():
            prefix = name.split('.')[0]
            postfix = name.split('.')[1]
            if postfix != 'bias':
                density = (torch.count_nonzero(param) / torch.numel(param)).numpy()
                if 'conv' in name or 'features' in name or 'fc' in name or 'classifier' in name:
                    oh_metric = (density * (np.prod(list(param.shape)))) / self.config.get('SPECIFICATION', 'lb', int)  # 1 times Frobenius Norm every lb batches
                    oh_search_space_iteration += oh_metric
        return oh_search_space_iteration

    def __estimate_overhead_grad_accumulation(self, model):
        oh_grad_accumulation = 0  # TODO ensure that if multiple methods combined, accum is only accounted for once
        for name, param in model.named_parameters():
            prefix = name.split('.')[0]
            postfix = name.split('.')[1]
            if postfix != 'bias':
                if 'conv' in name or 'features' in name or 'fc' in name or 'classifier' in name:
                    oh_grad_accumulation += torch.count_nonzero(param)  # accumulation of gradients every batch
        return oh_grad_accumulation

    def __estimate_overhead(self, model):
        #Assumptions: regularized baseline with gradient normalzation
        oh_grad_accumulation, oh_re_pruning, oh_gd_top_k, oh_gd_top_k_mc = 0, 0, 0, 0
        if self.est_oh_re_pruning or self.est_oh_gd_top_k or self.est_oh_gd_top_k_mc:
            oh_grad_accumulation = self.__estimate_overhead_grad_accumulation(model)
        if self.est_oh_gd_top_k:
            oh_gd_top_k = self.__estimate_overhead_gd_top_k(model)
        if self.est_oh_re_pruning:
            oh_re_pruning = self.__estimate_overhead_re_pruning(model)
        if self.est_oh_gd_top_k_mc:
            oh_gd_top_k_mc = self.__estimate_overhead_gd_top_k(model)
        return (oh_gd_top_k_mc + oh_gd_top_k + oh_re_pruning + oh_grad_accumulation).item()* 1e-9 #GFLOPs



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

        self.c_sparsity_current = stats_batch[7]
        self.l_sparsity_current = stats_batch[8]
        self.oh = self.__estimate_overhead(model)

    def __current_flops(self, model, ac):
        flops_i, flops_i_base, density_i, c_density_i, l_density_i = 0, 0, 0, 0, 0
        flops_i_fwd, flops_i_base_fwd = 0, 0
        flops_i_bwd, flops_i_base_bwd = 0, 0
        ctr = 0
        ctr_c = 0
        ctr_l = 0
        for name, param in model.named_parameters():
            # https://openai.com/blog/ai-and-compute/
            prefix = name.split('.')[0]
            postfix = name.split('.')[1]
            if postfix != 'bias':
                density = (torch.count_nonzero(param) / torch.numel(param)).numpy()
                c_density = 0.0
                l_density = 0.0
                ctr += 1
                if 'conv' in name or 'features' in name:
                    #print(name, density)
                    c_density = density
                    ctr_c+=1
                if 'fc' in name or 'classifier' in name:
                    l_density = density
                    ctr_l+=1
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
                c_density_i += c_density
                l_density_i += l_density

                if param.grad is not None:
                    flops_i_n_fwd = self.macs[prefix][0] * 2 # fwd,bwd updaten
                    flops_i_n_bwd = 2 * flops_i_n_fwd
                    flops_i_base_fwd += flops_i_n_fwd
                    flops_i_base_bwd += flops_i_n_bwd
                    flops_i_base += flops_i_n_fwd + flops_i_n_bwd
                else:
                    flops_i_n_fwd = self.macs[prefix][0] * 2
                    flops_i_base += flops_i_n_fwd

        return flops_i * 1e-9, flops_i_base * 1e-9, 1 - density_i / ctr, flops_i_fwd, \
               flops_i_base_fwd, flops_i_bwd, flops_i_base_bwd, 1 - c_density_i / ctr_c, 1 - l_density_i / ctr_l

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