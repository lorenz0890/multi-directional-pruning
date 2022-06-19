from __future__ import print_function

import torch
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

from performance_model import PerformanceModel
from pruning import GradientDiversityTopKGradients


class Baseline:
    def __init__(self, model, train_loader, test_loader, config, logger, visualization=None):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.use_cuda = not config.get('OTHER', 'no_cuda', bool) and torch.cuda.is_available()
        self.kwargs = {'num_workers': 1, 'pin_memory': True} if self.use_cuda else {}
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.gradient_diversity = GradientDiversityTopKGradients(1, 1)  # Only required for gradient normalization
        self.logger = logger
        self.visualization = visualization
        self.optimizer = SGD(self.model.parameters(), lr=self.config.get('SPECIFICATION', 'lr', float),
                             weight_decay=0.0)
        self.scheduler = MultiStepLR(self.optimizer, milestones=self.config.get('SPECIFICATION', 'steps',
                                                                                lambda a: [int(b) for b in
                                                                                           str(a).split(',')]),
                                     gamma=config.get('SPECIFICATION', 'gamma', float))
        self.performance_model = PerformanceModel(model, train_loader, config, logger=logger)

    def dispatch(self):
        self.performance_model.print_cuda_status()
        torch.manual_seed(self.config.get('OTHER', 'seed', int))
        self.__train()
        self.__test()
        self.logger.store()
        if self.config.get('OTHER', 'vis_model', bool): self.visualization.visualize_model(self.model)
        if self.config.get('OTHER', 'vis_log', bool):
            self.visualization.visualize_perfstats(self.logger)
            self.visualization.visualize_key_list(self.logger, ['test_accuracy', 'test_loss', 'train_loss'])
        if self.config.get('OTHER', 'save_model', bool): torch.save(self.model.state_dict(),
                                                                    self.config.get('OTHER', 'out_path', str))

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
                loss += self.config.get('SPECIFICATION', 'l1', float) * l1 + self.config.get('SPECIFICATION', 'l1',
                                                                                             float) * l2
                loss.backward()
                self.gradient_diversity.norm_grads(self.model)
                self.logger.log('train_loss', loss.item())
                self.optimizer.step()
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
        self.logger.log('test_accuracy', correct / len(self.test_loader.dataset))
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))
