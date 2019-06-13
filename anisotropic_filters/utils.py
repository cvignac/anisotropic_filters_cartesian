import numpy as np
import sys
import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class EarlyStopping(object):
    ''' Based on gist.github.com/stefanonardo/693d96ceb2f531fa05db530f3e21517d
    '''
    def __init__(self, mode='min', min_delta=0, patience=10):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta)

        if patience == 0:
            self.is_better = lambda a, b: True

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if mode == 'min':
            self.is_better = lambda a, best: a < best - min_delta
        if mode == 'max':
            self.is_better = lambda a, best: a > best + min_delta


def load_cifar(path_to_dataset, args, kwargs):
    # Load and transform the dataset
    trans_tr = [transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    transforms_train = transforms.Compose(trans_tr)
    trans_te = [transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    transforms_test = transforms.Compose(trans_te)
    ds_train = datasets.CIFAR10(path_to_dataset, train=True, download=True,
                                transform=transforms_train)
    ds_test = datasets.CIFAR10(path_to_dataset, train=False,
                               transform=transforms_test)
    train_loader = DataLoader(ds_train, args.batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(ds_test, args.test_batch_size, shuffle=True, **kwargs)
    return train_loader, test_loader


def load_mnist(path_to_dataset, args, kwargs):
    ds_train = datasets.MNIST(path_to_dataset, train=True, download=True,
                              transform=transforms.Compose(
                                  [transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                                   ]))

    ds_test = datasets.MNIST(path_to_dataset, train=False,
                             transform=transforms.Compose(
                                 [transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))]))

    train_loader = torch.utils.data.DataLoader(ds_train, args.batch_size,
                                               shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(ds_test, args.test_batch_size,
                                              shuffle=True, **kwargs)
    return train_loader, test_loader


def save_arguments(dataset, model_name):
    path = './saved_models/{}/{}_config.txt'.format(dataset, model_name)
    f = open(path, mode='w')
    for arg in sys.argv:
        f.write(arg)
        f.write('\n')
    f.close()


def test_film_normalizer():
    norma = FilmNormalizer()
    x = np.array([[0, 2],
                  [0, 1],
                  [0, 0]])
    y = np.array([[0, 1],
                  [0, 1],
                  [0, 1]])
    data = {'obs': x, 'tr': y, 'te': y}
    norma.normalize_all(data)


if __name__ == '__main__':
    test_film_normalizer()