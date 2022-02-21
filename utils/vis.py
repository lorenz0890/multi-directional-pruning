from datetime import datetime
import seaborn as sns
sns.set()
from matplotlib import pyplot as plt

class Visualization:
    def __init__(self, config, path='/media/lorenz/Volume/code/msc/pytorch-admm-pruning/logfiles/'):
        self.path = path
        self.logdict = {}
        self.config = config
        self.active = True

    def __make_name(self):
        name = ''
        # for section in self.config:
        for element in self.config['EXPERIMENT']:
            name = name + '_' + str(self.config['EXPERIMENT'][element])
        name = name + '_' + datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
        name += '.png'
        return name

    def visualize_logfile(self, model):
        # Visualize data stored in logifle
        pass

    def visualize_model(self, model):
        sns.set(rc={'figure.figsize':(11.7,5.27)})
        for k, (n, p) in enumerate(model.named_parameters()):
            if 'bias' not in n:
                print(n, p.shape)
                if len(p.shape) == 2:
                    #print(n, p.shape)
                    plt.matshow(p.detach().numpy())
                    plt.grid(None)
                    plt.show()
                if len(p.shape) == 4:
                    #print(n, p.shape)
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
                        #plt.imshow(img)
                            print(i,j,flush=True)
                    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=0.1)
                    plt.savefig(self.path+self.__make_name())