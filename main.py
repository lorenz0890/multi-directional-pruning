from __future__ import print_function
from utils import Parser
from utils import ExperimentFactory


def main():
    config = Parser()
    #config.load('configs/admm_retrain/lenet_mnist.ini')
    config.load('configs/admm_retrain/alexnet_cifar10.ini')
    experiment_factory = ExperimentFactory()
    experiment = experiment_factory.get_experiment(config)
    experiment.dispatch()

if __name__ == "__main__":
    main()