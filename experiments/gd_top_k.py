from __future__ import print_function

import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

from performance_model import PerformanceModel
from pruning import GradientDiversityTopKGradients



class GDTopK:
    def __init__(self, model, train_loader, test_loader, config, logger, visualization=None):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.use_cuda = not config.get('OTHER', 'no_cuda', bool) and torch.cuda.is_available()
        self.kwargs = {'num_workers': 1, 'pin_memory': True} if self.use_cuda else {}
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.logger = logger
        self.visualization = visualization
        self.performance_model = PerformanceModel(model, train_loader, config, overhead_gd_top_k=True, logger=logger)
        self.gradient_diversity = GradientDiversityTopKGradients(config.get('SPECIFICATION', 'lb', int),
                                                                 config.get('SPECIFICATION', 'k', int))

    def dispatch(self):
        self.performance_model.print_cuda_status()
        torch.manual_seed(self.config.get('OTHER', 'seed', int))
        optimizer = Adam(self.model.parameters(), lr=self.config.get('SPECIFICATION', 'lr', float),
                         eps=self.config.get('SPECIFICATION', 'adam_eps', float))

        self.__train(self.config, self.model, self.device, self.train_loader, self.test_loader, optimizer)
        self.__test(self.config, self.model, self.device, self.test_loader)
        print(self.performance_model.flops_accumulated, self.performance_model.flops_accumulated_base, flush=True)
        self.logger.store()
        if self.config.get('OTHER', 'vis_model', bool): self.visualization.visualize_model(self.model)
        if self.config.get('OTHER', 'vis_log', bool):
            self.visualization.visualize_perfstats(self.logger)
            self.visualization.visualize_key_list(self.logger, ['test_accuracy', 'test_loss', 'train_loss'])
        if self.config.get('OTHER', 'save_model', bool): torch.save(self.model.state_dict(),
                                                                    self.config.get('OTHER', 'out_path', str))

    def __train(self, config, model, device, train_loader, test_loader, optimizer):
        for epoch in range(config.get('SPECIFICATION', 'epochs', int)):
            print('Epoch: {}'.format(epoch + 1))
            model.train()
            for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target,
                                  reduction='sum')  # regularized_nll_loss(config, model, output, target)
                loss.backward()
                self.gradient_diversity.norm_grads(model)
                self.gradient_diversity.accum_grads(model)
                self.gradient_diversity.update_gd(batch_idx)
                self.gradient_diversity.reset_accum_grads(batch_idx)
                self.gradient_diversity.select_delete_grads(batch_idx)
                self.gradient_diversity.delete_selected_grads(model)
                self.performance_model.eval(model)
                if (batch_idx + 1) % (self.config.get('SPECIFICATION', 'lb', int)) == 0:
                    self.performance_model.print_perf_stats()
                self.performance_model.log_perf_stats()
                self.performance_model.log_layer_sparsity(self.model)
                self.logger.log('train_loss', loss.item())
                optimizer.step()
            self.__test(config, model, device, test_loader)

    def __test(self, config, model, device, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        self.logger.log('test_loss', test_loss)
        self.logger.log('test_accuracy', correct / len(self.test_loader.dataset))
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
