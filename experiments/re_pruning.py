from __future__ import print_function
import argparse
import copy

import torch
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import MultiStepLR

from pruning import GradientDiversity, GradientDiversityTopKGradients, RePruningLinearDet
from pruning import RePruningConvDet
from optimizer import PruneAdam
from model import LeNet, AlexNet
from performance_model import PerformanceModel
from utils import regularized_nll_loss, admm_loss, \
    initialize_Z_and_U, update_X, update_Z, update_Z_l1, update_U, \
    print_convergence, print_prune, apply_prune, apply_l1_prune
from torchvision import datasets, transforms
from tqdm import tqdm

class REPruning:
    def __init__(self, model, train_loader, test_loader, config, logger):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.use_cuda = not config.get('OTHER', 'no_cuda', bool) and torch.cuda.is_available()
        self.kwargs = {'num_workers': 1, 'pin_memory': True} if self.use_cuda else {}
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.performance_model = PerformanceModel(model, train_loader, config, overhead_re_pruning=True)
        self.gradient_diversity = GradientDiversityTopKGradients(config.get('SPECIFICATION', 'lb', int), 1) #Only required for G Norm & Accum functionality
        self.conv_pruning = RePruningConvDet(self.config.get('SPECIFICATION', 'softness_c', float),
                                             self.config.get('SPECIFICATION', 'magnitude_t_c', float),
                                             self.config.get('SPECIFICATION', 'metric_t_c', float),
                                             self.config.get('SPECIFICATION', 'lr', float),
                                             self.config.get('SPECIFICATION', 'sample_c', int),
                                             config.get('SPECIFICATION', 'lb', int),
                                             self.config.get('SPECIFICATION', 'scale_c', float),
                                             self.config.get('SPECIFICATION', 'prune_c', int))
        self.linear_pruning = RePruningLinearDet(self.config.get('SPECIFICATION', 'softness_l', float),
                                             self.config.get('SPECIFICATION', 'magnitude_t_l', float),
                                             self.config.get('SPECIFICATION', 'metric_t_l', float),
                                             self.config.get('SPECIFICATION', 'lr', float),
                                             self.config.get('SPECIFICATION', 'sample_l', int),
                                             config.get('SPECIFICATION', 'lb', int),
                                             self.config.get('SPECIFICATION', 'scale_l', float),
                                             self.config.get('SPECIFICATION', 'prune_l', int))
        self.logger = logger
        self.optimizer = SGD(self.model.parameters(), lr=self.config.get('SPECIFICATION', 'lr', float),
                             weight_decay=0.0)
        self.scheduler = MultiStepLR(self.optimizer, milestones=self.config.get('SPECIFICATION', 'steps',
                                                                                lambda a: [int(b) for b in
                                                                                           str(a).split(',')]),
                                                                                gamma=config.get('SPECIFICATION', 'gamma', float))
        # TODO add overhead w/o mask application to performance model
        # TODO add overhead w/ mask application to performance model
        # TODO complete overhead w/ min search (O n log n) performance model
        # TODO encode hyper params in config TODO
        # TODO make logger for metrics OK
        # TODO make some kind of batch mode fro experiments
        # TODO check we do it at right batch (i.e. idx+1 or not)
        # TODO In master thesis correct norm from spectral (i.e. 2 norm) to frobenius
    def dispatch(self):
        torch.manual_seed(self.config.get('OTHER', 'seed', int))
        self.__train()
        self.__test()
        print(self.performance_model.flops_accumulated, self.performance_model.flops_accumulated_base, flush=True)

    def __train(self):
        for epoch in range(self.config.get('SPECIFICATION', 'epochs', int)):
            print('Epoch: {}'.format(epoch + 1))
            self.model.train()
            for batch_idx, (data, target) in enumerate(tqdm(self.train_loader)):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)

                loss = F.nll_loss(output, target, reduction='sum')
                if epoch > self.config.get('SPECIFICATION', 'prune_epochs', int):
                    l1 = sum(p.abs().sum() for p in self.model.parameters())
                    l2 = sum(p.norm() for p in self.model.parameters())
                    loss += self.config.get('SPECIFICATION', 'l1', float) * l1 +self. config.get('SPECIFICATION', 'l1', float) * l2
                loss.backward()

                self.gradient_diversity.norm_grads(self.model)
                if epoch <= self.config.get('SPECIFICATION', 'prune_epochs', int):
                    self.gradient_diversity.accum_grads(self.model)
                    self.conv_pruning.compute_mask(self.model, self.gradient_diversity.accum_g, batch_idx)
                    self.linear_pruning.compute_mask(self.model, self.gradient_diversity.accum_g, batch_idx)
                    self.conv_pruning.apply_mask(self.model)
                    self.linear_pruning.apply_mask(self.model)
                    self.gradient_diversity.update_gd(batch_idx) # clears accumulated grads as side effect
                if epoch > self.config.get('SPECIFICATION', 'prune_epochs', int):
                    self.conv_pruning.apply_threshold(self.model)
                    self.linear_pruning.apply_threshold(self.model)

                self.performance_model.eval(self.model, 1)
                self.optimizer.step()

                if batch_idx % (self.config.get('SPECIFICATION', 'lb', int)) == 0:
                    print('Total SU', self.performance_model.flops_accumulated_base/self.performance_model.flops_accumulated,
                          '\nCurrent SU', self.performance_model.flops_current_base/self.performance_model.flops_current,
                          '\nTotal SU FWD',self.performance_model.flops_accumulated_base_fwd / self.performance_model.flops_accumulated_fwd,
                          '\nCurrent SU FWD', self.performance_model.flops_current_base_fwd / self.performance_model.flops_current_fwd,
                          '\nTotal SU BWD',self.performance_model.flops_accumulated_base_bwd / self.performance_model.flops_accumulated_bwd,
                          '\nCurrent SU BWD',self.performance_model.flops_current_base_bwd / self.performance_model.flops_current_bwd,
                          '\nCurrent Sparsity',self.performance_model.sparsity_current,
                          '\nCurrent Channel Sparsity', self.performance_model.c_sparsity_current,
                          '\nCurrent Linear Sparsity', self.performance_model.l_sparsity_current,
                          '\nCurrent Relative Overhead', self.performance_model.oh / self.performance_model.flops_current,
                          flush=True)
                self.logger.log('total_su', self.performance_model.flops_accumulated_base/self.performance_model.flops_accumulated)

            self.__test()
            self.scheduler.step()

    def __test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))