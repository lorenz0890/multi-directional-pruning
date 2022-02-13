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


class Baseline:
    def __init__(self, model, train_loader, test_loader, config, logger):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.use_cuda = not config.get('OTHER', 'no_cuda', bool) and torch.cuda.is_available()
        self.kwargs = {'num_workers': 1, 'pin_memory': True} if self.use_cuda else {}
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.performance_model = PerformanceModel(model, train_loader)
        self.gradient_diversity = GradientDiversityTopKGradients(1, 1)  # Only required for gradient normalization
        self.logger = logger
        self.optimizer = SGD(self.model.parameters(), lr=self.config.get('SPECIFICATION', 'lr', float), weight_decay=0.0)
        self.scheduler = MultiStepLR(self.optimizer, milestones=self.config.get('SPECIFICATION', 'steps',
                                                                           lambda a: [int(b) for b in str(a).split(',')]),
                                                                           gamma=config.get('SPECIFICATION', 'gamma', float))

    def dispatch(self):
        torch.manual_seed(self.config.get('OTHER', 'seed', int))
        self.__train()
        self.__test()
        self.logger.store()

    def __train(self):
        for epoch in range(self.config.get('SPECIFICATION', 'epochs', int)):
            print('Epoch: {}'.format(epoch + 1))
            self.model.train()
            for batch_idx, (data, target) in enumerate(tqdm(self.train_loader)):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = F.nll_loss(output, target, reduction='sum')
                l1 = sum(p.abs().sum() for p in self.model.parameters())
                l2 = sum(p.norm() for p in self.model.parameters())
                loss += self.config.get('SPECIFICATION', 'l1', float) * l1 +self. config.get('SPECIFICATION', 'l1', float) * l2
                loss.backward()
                self.gradient_diversity.norm_grads(self.model)
                self.optimizer.step()
                self.logger.log('loss', loss.item())
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
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)

        self.logger.log('test_loss', test_loss)
        self.logger.log('test_accuracy', 100. * correct / len(self.test_loader.dataset))
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))