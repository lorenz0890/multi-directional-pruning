from datetime import datetime

import numpy as np
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
import re

sns.set()
from matplotlib import pyplot as plt

class Visualization:
    def __init__(self, config, path='/media/lorenz/Volume/code/msc/pytorch-admm-pruning/logfiles/'):
        self.path = path
        self.logdict = {}
        self.config = config
        self.active = True

    def __make_name(self, tag=None):
        name = ''
        # for section in self.config:
        for element in self.config['EXPERIMENT']:
            name = name + '_' + str(self.config['EXPERIMENT'][element])
        name = name + '_' + datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
        if tag is not None:
            name += '_{}_'.format(tag)
        name += '.png'
        return name

    def __visualize_global_su(self, logger):
        for key in logger.logdict['LOGDATA']:
            if ('total' in key or 'current' in key) and 'su' in key:
                plt.plot(logger.logdict['LOGDATA'][key], label=key.replace('_', ' '))
                plt.legend()
                plt.semilogy()
        plt.title('SU')
        plt.savefig(self.path + self.__make_name('overall_su'))
        plt.clf()

    def __visualize_global_sparsity(self, logger):
        for key in logger.logdict['LOGDATA']:
            if ('total' in key or 'current' in key) and 'sparsity' in key:
                plt.plot(logger.logdict['LOGDATA'][key], label=key.replace('_', ' '))
                plt.legend()
        plt.title('Sparsity')
        plt.savefig(self.path + self.__make_name('total_sp'))
        plt.clf()

    def __visualize_layer_sparsity(self, logger):
        sparsity = []
        labels = []
        sparsity_grad = []
        labels_grad = []
        for key in logger.logdict['LOGDATA']:
            print(key, flush=key)
            if not ('total' in key or 'current' in key) and 'sparsity' in key:
                id = re.search(r'\d+', key.split('_')[-1]).group()
                type = 'layer'
                if 'conv' in key or 'features' in key:
                    type = 'conv'
                if 'fc' in key or 'classifier' in key or 'linear' in key:
                    type = 'linear'
                name = type + id
                if not 'grad' in key:
                    sparsity.append(logger.logdict['LOGDATA'][key])
                    labels.append(name)
                else:
                    sparsity_grad.append(logger.logdict['LOGDATA'][key])
                    labels_grad.append(name)

        ax = plt.subplot()
        print(np.array(sparsity).shape, flush=True)
        cm = ax.pcolormesh(np.array(sparsity), cmap='coolwarm')
        y_pos = range(0, len(labels))
        y_pos = [i + 0.5 for i in y_pos]
        plt.yticks(y_pos, labels, va='center')
        plt.xlabel("iteration")
        plt.ylabel("layer")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="20%", pad=0.3)
        plt.colorbar(cm, cax=cax)
        plt.xlabel('sparsity')
        plt.savefig(self.path + self.__make_name('layer_sp'))
        plt.clf()

        ax = plt.subplot()
        print(np.array(sparsity_grad).shape, flush=True)
        cm = ax.pcolormesh(np.array(sparsity_grad), cmap='coolwarm')
        y_pos = range(0, len(labels_grad))
        y_pos = [i + 0.5 for i in y_pos]
        plt.yticks(y_pos, labels_grad, va='center')
        plt.xlabel("iteration")
        plt.ylabel("layer")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="20%", pad=0.3)
        plt.colorbar(cm, cax=cax)
        plt.xlabel('sparsity grads')
        plt.savefig(self.path + self.__make_name('layer_sp_grads'))
        plt.clf()

    def visualize_key_list(self, logger, key_list):
        for key in key_list:
            if ('total' in key or 'current' in key) and 'su' in key:
                plt.plot(logger.logdict['LOGDATA'][key], label=key.replace('_', ' '))
                plt.legend()
            plt.title(key)
            plt.savefig(self.path + self.__make_name('key'))
            plt.clf()

    def visualize_perfstats(self, logger):
        self.__visualize_global_su(logger)
        self.__visualize_global_sparsity(logger)
        self.__visualize_layer_sparsity(logger)


    def visualize_model(self, model):
        sns.set(rc={'figure.figsize':(11.7,5.27)})
        for k, (n, p) in enumerate(model.named_parameters()):
            if 'bias' not in n:
                print(n, p.shape)
                if len(p.shape) == 2:
                    plt.matshow(p.detach().numpy())
                    plt.grid(None)
                    plt.savefig(self.path+self.__make_name())
                if len(p.shape) == 4:
                    fig, ax = plt.subplots(ncols = p.shape[0], nrows = p.shape[1],
                                     sharex=True, sharey=True, tight_layout=False, squeeze=False)
                    img = p.detach().numpy()
                    for i in range(p.shape[0]):
                        for j in range(p.shape[1]):
                            ax[j][i].autoscale()
                            ax[j][i].matshow(img[i][j])#.imshow(img)
                            ax[j][i].tick_params(axis='both', which='major', labelsize=10)
                            ax[j][i].tick_params(axis='both', which='minor', labelsize=8)
                            ax[j][i].set_xticks([0,1,2,3,4])
                            ax[j][i].set_yticks([0,1,2,3,4])
                            ax[j][i].grid(None)

                    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=0.1)
                    plt.savefig(self.path+self.__make_name())
                    plt.clf()