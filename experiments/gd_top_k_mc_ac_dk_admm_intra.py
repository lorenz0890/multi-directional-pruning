from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
from optimizer import PruneAdam
from model import LeNet, AlexNet
from performance_model import PerformanceModel
from pruning import MonteCarloGDTopKGradients
from utils import regularized_nll_loss, admm_loss, \
    initialize_Z_and_U, update_X, update_Z, update_Z_l1, update_U, \
    print_convergence, print_prune, apply_prune, apply_l1_prune
from torchvision import datasets, transforms
from tqdm import tqdm
import random

class MCGDTopKACDKADMMIntra:
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
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.use_cuda = not config.get('OTHER', 'no_cuda', bool) and torch.cuda.is_available()
        self.kwargs = {'num_workers': 1, 'pin_memory': True} if self.use_cuda else {}
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.performance_model = PerformanceModel(model, train_loader, config, overhead_gd_top_k_mc=True, overhead_admm=True, logger=logger)
        self.gradient_diversity = MonteCarloGDTopKGradients(config.get('SPECIFICATION', 'lb', int),
                                                            config.get('SPECIFICATION', 'k', int),
                                                            config.get('SPECIFICATION', 'se', int), model)
        self.ac = config.get('SPECIFICATION', 'ac', int)
        self.loss_log = []
        self.loss_m_log = []
        self.loss_ratio_log = []
        self.k_batch_map = {}
        self.k = config.get('SPECIFICATION', 'k', int)
        self.k_max = config.get('SPECIFICATION', 'k', int)

        self.global_epochs = 0

    def dispatch(self):
        torch.manual_seed(self.config.get('OTHER', 'seed', int))
        optimizer = PruneAdam(self.model.named_parameters(), lr=self.config.get('SPECIFICATION', 'lr', float),
                              eps=self.config.get('SPECIFICATION', 'adam_eps', float))

        for i in range(self.config.get('SPECIFICATION', 'repeat', int)):
            self.__train(self.config, self.model, self.device, self.train_loader, self.test_loader, optimizer)
            mask = apply_l1_prune(self.model, self.device, self.config) if self.config.get('SPECIFICATION', 'l1', bool) else apply_prune(self.model, self.device, self.config)
            print_prune(self.model)
            self.__test(self.config, self.model, self.device, self.test_loader)
            self.__retrain(self.config, self.model, mask, self.device, self.train_loader, self.test_loader, optimizer)
            print(self.performance_model.flops_accumulated, self.performance_model.flops_accumulated_base, flush=True)

        self.logger.store()
        if self.config.get('OTHER', 'vis_model', bool): self.visualization.visualize_model(self.model)
        if self.config.get('OTHER', 'vis_log', bool):
            self.visualization.visualize_perfstats(self.logger)
            self.visualization.visualize_key_list(self.logger, ['top_k', 'accumulation','test_accuracy', 'test_loss', 'train_loss'])
        if self.config.get('OTHER', 'save_model', bool): torch.save(self.model.state_dict(),
                                                                    self.config.get('OTHER', 'out_path', str))

    def __train(self, config, model, device, train_loader, test_loader, optimizer):
        for epoch in range(config.get('SPECIFICATION', 'pre_epochs', int)):
            self.global_epochs += 1
            print('Pre epoch: {}'.format(epoch + 1))
            model.train()
            for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = regularized_nll_loss(config, model, output, target)
                loss = loss / self.ac
                loss.backward()

                self.gradient_diversity.norm_grads(model)
                self.gradient_diversity.update_epoch(self.global_epochs)
                self.gradient_diversity.accum_grads(model)
                self.gradient_diversity.update_gd(batch_idx)
                self.gradient_diversity.reset_accum_grads(batch_idx)
                self.loss_log.append(loss.item())
                loss_m = sum(self.loss_log[-config.get('SPECIFICATION', 'lb', int):]) / len(
                    self.loss_log[-config.get('SPECIFICATION', 'lb', int):])
                self.loss_m_log.append(loss_m)
                self.loss_ratio_log.append(self.loss_log[-1] / loss_m)
                if batch_idx not in self.k_batch_map:
                    self.k_batch_map[batch_idx] = []
                if self.loss_log[-1] > loss_m and self.k > 0:
                    self.k_batch_map[batch_idx].append(max(0, self.k - 1))
                    self.ac = max(1, self.ac * 0.9)
                elif self.k < self.k_max:
                    self.k_batch_map[batch_idx].append(min(self.k_max, self.k + 1))
                    self.ac = min(
                        config.get('SPECIFICATION', 'lb', int) - batch_idx % config.get('SPECIFICATION', 'lb', int),
                        self.ac * 1.1)
                else:
                    self.k_batch_map[batch_idx].append(self.k)
                self.logger.log('accumulation', self.ac)
                self.k = random.choice(self.k_batch_map[batch_idx][-config.get('SPECIFICATION', 'lb', int):])
                self.logger.log('top_k', self.k)
                self.gradient_diversity.update_k(self.k)
                self.gradient_diversity.select_delete_grads(batch_idx, self.global_epochs)
                self.gradient_diversity.delete_selected_grads(model)
                self.performance_model.eval(model, ac=self.ac, epoch=self.global_epochs)
                if (batch_idx + 1) % (self.config.get('SPECIFICATION', 'lb', int)) == 0:
                    self.performance_model.print_perf_stats()
                self.performance_model.log_perf_stats()
                self.performance_model.log_layer_sparsity(self.model)
                self.logger.log('train_loss', loss.item())
                if (batch_idx + 1) % int(self.ac) == 0 or (batch_idx + 1) % len(train_loader) == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            self.__test(config, model, device, test_loader)

        Z, U = initialize_Z_and_U(model)
        for epoch in range(config.get('SPECIFICATION', 'epochs', int)):
            self.global_epochs +=1
            model.train()
            print('Epoch: {}'.format(epoch + 1))
            for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = admm_loss(config, device, model, Z, U, output, target)
                loss = loss / self.ac
                loss.backward()

                self.gradient_diversity.norm_grads(model)
                self.gradient_diversity.update_epoch(self.global_epochs)
                self.gradient_diversity.accum_grads(model)
                self.gradient_diversity.update_gd(batch_idx)
                self.gradient_diversity.reset_accum_grads(batch_idx)
                self.loss_log.append(loss.item())
                loss_m = sum(self.loss_log[-config.get('SPECIFICATION', 'lb', int):]) / len(
                    self.loss_log[-config.get('SPECIFICATION', 'lb', int):])
                self.loss_m_log.append(loss_m)
                self.loss_ratio_log.append(self.loss_log[-1] / loss_m)
                if batch_idx not in self.k_batch_map:
                    self.k_batch_map[batch_idx] = []
                if self.loss_log[-1] > loss_m and self.k > 0:
                    self.k_batch_map[batch_idx].append(max(0, self.k - 1))
                    self.ac = max(1, self.ac * 0.9)
                elif self.k < self.k_max:
                    self.k_batch_map[batch_idx].append(min(self.k_max, self.k + 1))
                    self.ac = min(
                        config.get('SPECIFICATION', 'lb', int) - batch_idx % config.get('SPECIFICATION', 'lb', int),
                        self.ac * 1.1)
                else:
                    self.k_batch_map[batch_idx].append(self.k)
                self.logger.log('accumulation', self.ac)
                self.k = random.choice(self.k_batch_map[batch_idx][-config.get('SPECIFICATION', 'lb', int):])
                self.logger.log('top_k', self.k)
                self.gradient_diversity.update_k(self.k)
                self.gradient_diversity.select_delete_grads(batch_idx, self.global_epochs)
                self.gradient_diversity.delete_selected_grads(model)
                self.performance_model.eval(model, ac=self.ac, epoch=self.global_epochs)
                if (batch_idx + 1) % (self.config.get('SPECIFICATION', 'lb', int)) == 0:
                    self.performance_model.print_perf_stats()
                self.performance_model.log_perf_stats()
                self.performance_model.log_layer_sparsity(self.model)
                self.logger.log('train_loss', loss.item())
                if (batch_idx + 1) % int(self.ac) == 0 or (batch_idx + 1) % len(train_loader) == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            X = update_X(model)
            Z = update_Z_l1(X, U, config) if config.get('SPECIFICATION', 'l1', bool) else update_Z(X, U, config)
            U = update_U(U, X, Z)
            print_convergence(model, X, Z)
            self.__test(config, model, device, test_loader)


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
        self.logger.log('test_loss', test_loss)
        self.logger.log('test_accuracy', correct / len(self.test_loader.dataset))
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


    def __retrain(self, config, model, mask, device, train_loader, test_loader, optimizer):
        for epoch in range(config.get('SPECIFICATION', 'pre_epochs', int)):
            self.global_epochs += 1
            print('Re epoch: {}'.format(epoch + 1))
            model.train()
            for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = F.nll_loss(output, target)
                loss = loss / self.ac
                loss.backward()

                self.gradient_diversity.norm_grads(model)
                self.gradient_diversity.update_epoch(self.global_epochs)
                self.gradient_diversity.accum_grads(model)
                self.gradient_diversity.update_gd(batch_idx)
                self.gradient_diversity.reset_accum_grads(batch_idx)
                self.loss_log.append(loss.item())
                loss_m = sum(self.loss_log[-config.get('SPECIFICATION', 'lb', int):]) / len(
                    self.loss_log[-config.get('SPECIFICATION', 'lb', int):])
                self.loss_m_log.append(loss_m)
                self.loss_ratio_log.append(self.loss_log[-1] / loss_m)
                if batch_idx not in self.k_batch_map:
                    self.k_batch_map[batch_idx] = []
                if self.loss_log[-1] > loss_m and self.k > 0:
                    self.k_batch_map[batch_idx].append(max(0, self.k - 1))
                    self.ac = max(1, self.ac * 0.9)
                elif self.k < self.k_max:
                    self.k_batch_map[batch_idx].append(min(self.k_max, self.k + 1))
                    self.ac = min(
                        config.get('SPECIFICATION', 'lb', int) - batch_idx % config.get('SPECIFICATION', 'lb', int),
                        self.ac * 1.1)
                else:
                    self.k_batch_map[batch_idx].append(self.k)
                self.logger.log('accumulation', self.ac)
                self.k = random.choice(self.k_batch_map[batch_idx][-config.get('SPECIFICATION', 'lb', int):])
                self.logger.log('top_k', self.k)
                self.gradient_diversity.update_k(self.k)
                self.gradient_diversity.select_delete_grads(batch_idx, self.global_epochs)
                self.gradient_diversity.delete_selected_grads(model)
                self.performance_model.eval(model, ac=self.ac, epoch=self.global_epochs)
                if (batch_idx + 1) % (self.config.get('SPECIFICATION', 'lb', int)) == 0:
                    self.performance_model.print_perf_stats()
                self.performance_model.log_perf_stats()
                self.performance_model.log_layer_sparsity(self.model)
                self.logger.log('train_loss', loss.item())
                if (batch_idx + 1) % int(self.ac) == 0 or (batch_idx + 1) % len(train_loader) == 0:
                    optimizer.prune_step(mask)
                    optimizer.zero_grad()

            self.__test(config, model, device, test_loader)