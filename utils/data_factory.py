from torchvision import datasets, transforms
import torch

class DataFactory:
    def __init__(self):
        pass

    def __get_mnist(self, config, kwargs):
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=True, download=True,
                           transform=transforms.Compose([
                               #transforms.RandomHorizontalFlip(),
                               #transforms.RandomVerticalFlip(),
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=config.get('EXPERIMENT', 'train_batch_size', int), shuffle=True, **kwargs)

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=config.get('EXPERIMENT', 'train_batch_size', int), shuffle=True, **kwargs)
        return train_loader, test_loader

    def __get_cifar10(self, config, kwargs):
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('data', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.49139968, 0.48215827, 0.44653124),
                                                      (0.24703233, 0.24348505, 0.26158768))
                             ])), shuffle=True, batch_size=config.get('EXPERIMENT', 'train_batch_size', int), **kwargs)

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('data', train=False, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.49139968, 0.48215827, 0.44653124),
                                                      (0.24703233, 0.24348505, 0.26158768))
                             ])), shuffle=True, batch_size=config.get('EXPERIMENT', 'train_batch_size', int), **kwargs)

        return train_loader, test_loader

    def __get_cifar100(self, config, kwargs):
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('data', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.49139968, 0.48215827, 0.44653124),
                                                      (0.24703233, 0.24348505, 0.26158768))
                             ])), shuffle=True, batch_size=config.get('EXPERIMENT', 'train_batch_size', int), **kwargs)

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('data', train=False, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.49139968, 0.48215827, 0.44653124),
                                                      (0.24703233, 0.24348505, 0.26158768))
                             ])), shuffle=True, batch_size=config.get('EXPERIMENT', 'train_batch_size', int), **kwargs)
        return train_loader, test_loader

    def __get_imagenet(self, config, kwargs):
        train_loader = torch.utils.data.DataLoader(datasets.ImageFolder(root="./data/imagenet/train", transform=transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))
            ])))
        test_loader = torch.utils.data.DataLoader(datasets.ImageFolder(root="./data/imagenet/val", transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])))
        return train_loader, test_loader


    def get_dataset(self, config):
            use_cuda = not config.get('OTHER', 'no_cuda', bool) and torch.cuda.is_available()
            kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
            train_loader, test_loader = None, None
            if config.get('EXPERIMENT', 'dataset', str) == 'mnist':
                train_loader, test_loader = self.__get_mnist(config, kwargs)
            elif config.get('EXPERIMENT', 'dataset', str) == 'cifar10':
                train_loader, test_loader = self.__get_cifar10(config, kwargs)
            elif config.get('EXPERIMENT', 'dataset', str) == 'cifar100':
                train_loader, test_loader = self.__get_cifar100(config, kwargs)
            elif config.get('EXPERIMENT', 'dataset', str) == 'imagenet':
                train_loader, test_loader = self.__get_imagenet(config, kwargs)
            return train_loader, test_loader