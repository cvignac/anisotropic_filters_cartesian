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


class Normalizer:
    def __init__(self, scale=False):
        self.mean = None
        self.std = None
        self.scale = scale

    def fit(self, obs):
        self.mean = (obs[obs > 0]).mean()
        if self.scale:
            self.std = (obs[obs > 0]).std()
            if self.std == 0:
                print("Warning: std==0 in Normalizer.fit")
                self.std == 1

    def __call__(self, x):
        if self.scale:
            x[x != 0] = (x[x !=0] - self.mean) / self.std
        else:
            x[x != 0] = (x[x != 0] - self.mean)
        return x

    def normalize_all(self, obs, tr, te):
        self.fit(obs)
        obs = self.__call__(obs)
        tr = self.__call__(tr)
        te = self.__call__(te)
        return obs, tr, te


class FilmNormalizer:
    def __init__(self):
        self.mean = None

    def fit(self, tr):
        """ Warning: obs can be either a torch.Tensor or a np.ndarray."""
        assert len(tr.shape) == 2, ('Useless dimensions should be removed ' +
                                     '- current shape' + str(tr.shape))
        ratings_sum = tr.sum(0)
        global_mean = (tr[tr > 0]).mean()
        if type(tr) == np.ndarray:
            number_obs = np.sum(tr > 0, axis=0)
        else:
            number_obs = (tr > 0).sum(0).type(torch.float32)
        # Remove nan - normalization for not observed films does not make a difference
        # not observed films are normalized to the global mean
        not_observed_movies = number_obs == 0
        number_obs[not_observed_movies] = 1
        means = ratings_sum / number_obs
        means[not_observed_movies] = global_mean
        self.mean = means

    def __call__(self, x):
        """ Each value u in x is transformed to u - mean if it is not 0
            else it stays 0."""
        assert len(x.shape) == 2, 'Incorrect shape for x' + str(x.shape)
        if type(x) == np.ndarray:
            x = (x - self.mean) * (x != 0)
        else:
            x = (x - self.mean) * (x != 0).type(torch.float)
        return x

    def normalize_all(self, data: dict):
        """ Warning: in this version the training set is used in the normalization"""
        self.fit(data['obs'] + data['tr'])
        for key, val in data.items():
            data[key] = self.__call__(val)


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