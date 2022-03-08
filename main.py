from __future__ import print_function
from utils import Parser
from utils import ExperimentFactory
from os import listdir
from os.path import isfile, join
import timeit

from tqdm import tqdm
from functools import partialmethod
import sys, os

def blockPrint():
    # https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print
    # https://stackoverflow.com/questions/37091673/silence-tqdms-output-while-running-tests-or-running-the-code-via-cron
    sys.stdout = open(os.devnull, 'w')
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

def enablePrint():
    # https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print
    # https://stackoverflow.com/questions/37091673/silence-tqdms-output-while-running-tests-or-running-the-code-via-cron
    sys.stdout = sys.__stdout__
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=False)

def main():
    config = Parser()
    experiment_factory = ExperimentFactory()

    #config.load('configs/ablation_study/admm_retrain/lenet_mnist_4.ini')
    #config.load('configs/ablation_study/admm_retrain/alexnet_cifar10_11.ini')

    #config.load('configs/ablation_study/admm_intra/lenet_mnist_4.ini')
    #config.load('configs/ablation_study/admm_intra/alexnet_cifar10_11.ini')

    #config.load('configs/ablation_study/gd_top_k/lenet_mnist_4.ini')
    #config.load('configs/ablation_study/gd_top_k/alexnet_cifar10_11.ini')

    #config.load('configs/ablation_study/gd_top_k_mc/lenet_mnist_4.ini')
    #config.load('configs/ablation_study/gd_top_k_mc_ac/lenet_mnist_4.ini')

    #config.load('configs/gd_top_k_mc_ac_dk/lenet_mnist_4.ini')
    #config.load('configs/gd_top_k_mc_ac_dk/alexnet_cifar10_11.ini')
    #config.load('configs/gd_top_k_mc_ac_dk/vgg8_cifar10.ini')
    #config.load('configs/gd_top_k_mc_ac_dk/mobilenet_v3_s_cifar10.ini')

    #config.load('configs/ablation_study/gd_top_k_mc_ac_dk_admm_intra/lenet_mnist_4.ini')
    #config.load('configs/gd_top_k_mc_ac_dk_admm_intra/alexnet_cifar10_11.ini')

    #config.load('configs/ablation_study/re_pruning/excluded/lenet_mnist_4.ini')
    #config.load('configs/ablation_study/re_pruning/alexnet_cifar10_11.ini')

    #config.load('configs/ablation_study/re_pruning_ac/finished/lenet_mnist_4.ini')

    #config.load('configs/baseline/vgg8_cifar10.ini')
    #config.load('configs/baseline/vgg8_bn_cifar10.ini')
    #config.load('configs/baseline/wrn16_8_cifar10.ini')
    #config.load('configs/re_pruning_gd_top_k_mc_ac_dk_admm_intra/lenet_mnist_4.ini')
    #config.load('configs/re_pruning_gd_top_k_mc_ac_dk_admm_intra/alexnet_cifar10_11.ini')


    config.load('configs/ablation_study/re_pruning_ac_admm_retrain/lenet_mnist.ini')
    experiment = experiment_factory.get_experiment(config)
    experiment.dispatch()


    #Batch mode
    '''
    paths = [
            #'configs/experiments/gd_top_k_mc_ac_dk/',
            #'configs/experiments/baseline/',
            #'configs/experiments/re_pruning/',
            #'configs/ablation_study/admm_intra/',
            #'configs/ablation_study/admm_retrain/'
            #'configs/ablation_study/gd_top_k/',
            #'configs/ablation_study/gd_top_k_mc/',
            #'configs/ablation_study/gd_top_k_mc_ac/',
            #'configs/ablation_study/gd_top_k_mc_ac_dk/'
            'configs/ablation_study/gd_top_k_mc_ac_dk_admm_retrain/'
            #'configs/ablation_study/alexnet_mixed/'
    ]

    for path in paths:
        enablePrint()
        print('BEGIN:', path, flush=True)
        blockPrint()
        fnames = [f for f in listdir(path) if isfile(join(path, f))]
        failed, succeeded = {}, {}
        fail_ctr, success_ctr, ctr = 0, 0, 0
        for fname in fnames:
            ctr+=1
            enablePrint()
            print('Experiment:', fname, flush=True)
            blockPrint()
            try:
                start = timeit.default_timer()
                config.load(path+fname)
                experiment = experiment_factory.get_experiment(config)
                experiment.dispatch()
                stop = timeit.default_timer()
                success_ctr+=1
                succeeded[fname] = stop - start
            except Exception as e:
                failed[fname] = e
                fail_ctr += 1
            enablePrint()
            print('Success (Current):', (success_ctr) ,'/', ctr, flush=True)
            print('Failed: (Current)', (fail_ctr) ,'/', ctr, flush=True)
            print('Progress:', success_ctr+fail_ctr ,'/', len(fnames), flush=True)
            blockPrint()
        enablePrint()
        print('Success (Total):', (success_ctr) ,'/', len(fnames), flush=True)
        print('Failed (Total):', (fail_ctr) ,'/', len(fnames), flush=True)
        for key in failed:
            print(key, failed[key], flush=True)
        for key in succeeded:
            print(key, succeeded[key], flush=True)
        print('END:', path, flush=True)
        blockPrint()
    '''
if __name__ == "__main__":
    main()