from __future__ import print_function
from utils import Parser
from utils import ExperimentFactory
from os import listdir
from os.path import isfile, join

def main():
    config = Parser()
    experiment_factory = ExperimentFactory()

    #config.load('configs/admm_retrain/lenet_mnist.ini')
    #config.load('configs/admm_retrain/x_alexnet_cifar10.ini')
    #config.load('configs/admm_retrain/vgg16_cifar10_2.ini')

    #config.load('configs/admm_intra/lenet_mnist.ini')
    #config.load('configs/admm_intra/x_alexnet_cifar10.ini')

    #config.load('configs/gd_top_k/lenet_mnist.ini')
    #config.load('configs/gd_top_k/x_alexnet_cifar10.ini')
    #config.load('configs/gd_top_k/vgg16_cifar10_2.ini')

    #config.load('configs/gd_top_k_mc/lenet_mnist.ini')

    #config.load('configs/gd_top_k_mc_ac/lenet_mnist.ini')

    #config.load('configs/gd_top_k_mc_ac_dk/lenet_mnist.ini')
    #config.load('configs/gd_top_k_mc_ac_dk/x_alexnet_cifar10.ini')

    #config.load('configs/re_pruning/lenet_mnist.ini')
    config.load('configs/re_pruning/alexnet_cifar10.ini')
    #config.load('configs/re_pruning/vgg16_cifar10_2.ini')
    #config.load('configs/re_pruning/resnet18_cifar10_2.ini')


    experiment = experiment_factory.get_experiment(config)
    experiment.dispatch()

    '''
    #Batch mode
    path = 'configs/baseline/'
    fnames = [f for f in listdir(path) if isfile(join(path, f))]
    for fname in fnames:
        config.load(path+fname)
        experiment = experiment_factory.get_experiment(config)
        experiment.dispatch()
    '''
if __name__ == "__main__":
    main()