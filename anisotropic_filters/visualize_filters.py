import numpy as np
import torch
# import anisotropic_filters.graph_utils as graph_utils
import matplotlib.pyplot as plt
import os
from anisotropic_filters.layers import ChebychevConvolution

d1, d2 = 7, 7


def move_model_to_cpu(model, model_name):
    model_cpu = model.to('cpu')
    save_path = './saved_models/cifar10/{}-cpu.pt'.format(model_name)
    torch.save(model_cpu.state_dict(), save_path)


def visualize_standard_conv(model, model_name):
    """ Visualization of standard CNN layers"""
    print(model)
    weights = model.conv1.weight.data.numpy()   # Shape: units, 3, k, k
    weights = np.transpose(weights, (0, 2, 3, 1))      # units, k, k, 3
    bias = model.conv1.bias.data.numpy()        # Shape: units
    units = weights.shape[0]

    for u in range(units):
        y = weights[u] + bias[u]
        y = (y - np.min(y)) / (np.max(y)) - np.min(y)
        plt.imshow(y)
        plt.title('First layer, unit {}'.format(u))
        folder = './results/visualization/' + model_name
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(folder + '/cnn_unit_{}_{}.eps'.format(model_name, u))


def visualize(model, model_name):
    coefs1 = model.conv1.coefs      # _k, _k, channel, units
    isotropic = len(coefs1.shape) == 3
    if isotropic:
        k, channels, units = coefs1.shape
    else:
        k, channels, units = coefs1.shape[1:]

    A = np.zeros((d1, d1))
    for i in range(d1 - 1):
        A[i, i + 1] = 1
    A = A + A.T

    layer = ChebychevConvolution(A, A, channels, units, k - 1, isotropic)
    layer.coefs = coefs1
    delta1 = torch.from_numpy(np.zeros((1, channels, d1, d1), dtype=np.float32))
    delta2 = torch.from_numpy(np.zeros((1, channels, d1, d1), dtype=np.float32))
    delta3 = torch.from_numpy(np.zeros((1, channels, d1, d1), dtype=np.float32))
    delta1[:, 0, int(d1 / 2), int(d1 / 2)] = 1       # Central node
    delta2[:, 1, int(d1 / 2), int(d1 / 2)] = 1
    delta3[:, 2, int(d1 / 2), int(d1 / 2)] = 1
    y1 = layer.forward(delta1)
    y2 = layer.forward(delta2)
    y3 = layer.forward(delta3)
    y = torch.cat([y1, y2, y3], dim=0).detach().numpy()

    for u in range(units):
        filtered = y[:, u, :, :].transpose([1, 2, 0])
        # Normalize to 0-1 range
        filtered = np.abs(filtered)
        filtered = (filtered - np.min(filtered)) / (np.max(filtered) - np.min(filtered))
        # Plot
        plt.imshow(filtered)
        plt.show()
        folder = './results/visualization/' + model_name
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(folder + '/unit{}.eps'.format(u))
