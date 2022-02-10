from __future__ import print_function
import argparse
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
        self.performance_model = PerformanceModel(model, train_loader) #config.get('SPECIFICATION', 'lb', int)
        self.gradient_diversity = GradientDiversityTopKGradients(config.get('SPECIFICATION', 'lb', int), 1) #Only required for G Norm & Accum functionality
        self.conv_pruning = RePruningConvDet(self.config.get('SPECIFICATION', 'softness_c', float),
                                             self.config.get('SPECIFICATION', 'magnitude_t_c', float),
                                             self.config.get('SPECIFICATION', 'metric_t_c', float),
                                             self.config.get('SPECIFICATION', 'lr', float),
                                             self.config.get('SPECIFICATION', 'attempts_c', int),
                                             config.get('SPECIFICATION', 'lb', int),
                                             self.config.get('SPECIFICATION', 'scale_c', float))
        self.linear_pruning = RePruningLinearDet(self.config.get('SPECIFICATION', 'softness_l', float),
                                             self.config.get('SPECIFICATION', 'magnitude_t_l', float),
                                             self.config.get('SPECIFICATION', 'metric_t_l', float),
                                             self.config.get('SPECIFICATION', 'lr', float),
                                             self.config.get('SPECIFICATION', 'attempts_l', int),
                                             config.get('SPECIFICATION', 'lb', int),
                                                 self.config.get('SPECIFICATION', 'scale_l', float))
        self.logger = logger
        #TODO add overhead to performance model
        #TODO encode hyper params in config OK
        #TODO make logger for metrics OK
        #TODO make some kind of batch mode fro experiments
        #TODO check we do it at right batch (i.e. idx+1 or not)
    def dispatch(self):
        torch.manual_seed(self.config.get('OTHER', 'seed', int))
        optimizer = SGD(self.model.parameters(), lr=self.config.get('SPECIFICATION', 'lr', float), weight_decay=0.0)
        #scheduler = MultiStepLR(optimizer, milestones=[5, 7], gamma=0.1)
        #optimizer = Adam(self.model.parameters(), lr=self.config.get('SPECIFICATION', 'lr', float),
        #                     eps=self.config.get('SPECIFICATION', 'adam_eps', float))

        self.__train(self.config, self.model, self.device, self.train_loader, self.test_loader, optimizer)#, scheduler)
        self.__test(self.config, self.model, self.device, self.test_loader)
        print(self.performance_model.flops_accumulated, self.performance_model.flops_accumulated_base, flush=True)

    def __train(self, config, model, device, train_loader, test_loader, optimizer):#, scheduler):
        scaler = torch.cuda.amp.GradScaler()
        for epoch in range(config.get('SPECIFICATION', 'epochs', int)):
            print('Epoch: {}'.format(epoch + 1))
            model.train()
            for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                with torch.cuda.amp.autocast():
                    loss = F.nll_loss(output, target, reduction='sum') #regularized_nll_loss(config, model, output, target)
                '''
                if epoch > 1:
                    l1_norm = sum(p.abs().sum() for p in model.parameters())
                    loss += 1e-3 * l1_norm
                '''
                loss.backward()
                self.gradient_diversity.norm_grads(model)
                self.gradient_diversity.accum_grads(model)

                if epoch <= self.config.get('SPECIFICATION', 'prune_epochs', int):
                    
                    self.conv_pruning.compute_mask(model, self.gradient_diversity.accum_g, batch_idx)
                    self.linear_pruning.compute_mask(model, self.gradient_diversity.accum_g, batch_idx)
                #if epoch > (self.config.get('SPECIFICATION', 'prune_epochs', int)):
                    self.conv_pruning.apply_mask(model)
                    self.conv_pruning.apply_threshold(model)
                    self.linear_pruning.apply_mask(model)
                    self.linear_pruning.apply_threshold(model)

                self.gradient_diversity.update_gd(batch_idx)

                
                #if epoch > self.config.get('SPECIFICATION', 'prune_epochs', int):
                #self.gradient_diversity.select_delete_grads(batch_idx)
                #self.gradient_diversity.delete_selected_grads(model)

                #if batch_idx % 4 == 0:
                optimizer.step()

                #for p in model.parameters():
                #    p.grad.data = p.grad.data.half().float()
                #    p.data = p.data.half().float()

                self.performance_model.eval(model, 4)

                if batch_idx % (config.get('SPECIFICATION', 'lb', int)) == 0:
                    print('Total SU', self.performance_model.flops_accumulated_base/self.performance_model.flops_accumulated,
                          '\nCurrent SU', self.performance_model.flops_current_base/self.performance_model.flops_current,
                          '\nTotal SU FWD',self.performance_model.flops_accumulated_base_fwd / self.performance_model.flops_accumulated_fwd,
                          '\nCurrent SU FWD', self.performance_model.flops_current_base_fwd / self.performance_model.flops_current_fwd,
                          '\nTotal SU BWD',self.performance_model.flops_accumulated_base_bwd / self.performance_model.flops_accumulated_bwd,
                          '\nCurrent SU BWD',self.performance_model.flops_current_base_bwd / self.performance_model.flops_current_bwd,
                          '\nCurrent Sparsity',self.performance_model.sparsity_current,
                          '\nCurrent Channel Sparsity', self.performance_model.c_sparsity_current,
                          '\nCurrent Linear Sparsity', self.performance_model.l_sparsity_current,
                          flush=True)
                self.logger.log('total_su', self.performance_model.flops_accumulated_base/self.performance_model.flops_accumulated)

            self.__test(config, model, device, test_loader)
            #scheduler.step()

    def __test(self, config, model, device, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))