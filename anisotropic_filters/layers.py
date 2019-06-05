# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import scipy
import graph_utils as graph_utils


class ChebychevConvolution(torch.nn.Module):
    def __init__(self, A1, A2, channels, units, k, isotropic):
        """ Compute Laplacian polynomials where
            the coefficients are trainable parameters.
            A1, A2 (np.array): adjacency matrices of the graphs
        """
        super().__init__()
        self.channels, self.units = channels, units
        self.isotropic = isotropic
        self.k, self._k = k, k + 1
        # Build graphs using the adjacency matrices
        self.G1 = graph_utils.ExtendedGraph(A1, k)
        self.G2 = graph_utils.ExtendedGraph(A2, k)
        self.register_buffer('cheb1', self.G1.chebychev)
        self.register_buffer('cheb2', self.G2.chebychev)

        # Build parameter tensor
        self.powers = self.G1.chebychev.shape[0]
        if self.isotropic:
            shape = [self.powers, self.channels, self.units]
        else:
            shape = [self.powers, self.powers, self.channels, self.units]
        coefs = torch.empty(shape, dtype=torch.float32)
        nn.init.normal_(coefs, mean=0, std=1e-3)
        self.coefs = torch.nn.Parameter(coefs)

    def forward(self, x):
        output = torch.zeros(x.shape[0], self.units, x.shape[2], x.shape[3], device=x.device)
        for i in range(self.channels):
            input = x[:, i].unsqueeze(1)              # bs, 1, d1, d2
            for k2 in range(self.powers):
                cheb2 = self.cheb2[k2]                      # d2 * d2
                filtered2 = input.matmul(cheb2)             # bs, 1, d1, d2
                filtered2 = filtered2.transpose(2, 3)       # bs, 1, d2, d1
                for k1 in range(self.powers):
                    if self.isotropic and k1 + k2 >= self.powers:
                        continue
                    cheb1 = self.cheb1[k1]                  # d1 * d1
                    filtered = filtered2.matmul(cheb1)      # bs, 1, d2, d1
                    filtered = filtered.transpose(2, 3)     # bs, 1, d1, d2
                    if self.isotropic:
                        coef = self.coefs[k1 + k2, i] * scipy.special.binom(k1 + k2, k1)
                    else:
                        coef = self.coefs[k1, k2, i]        # units
                    coef = coef[None, :, None, None]        # 1, units, 1, 1
                    output = output + coef * filtered
        return output

    def __str__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.channels) + ' -> ' \
               + str(self.units) + ')'


class ImageChebychevConvolution(ChebychevConvolution):
    def __init__(self, channels, units, length, k, isotropic, directed, padding):
        """ Specific ChebychevConvolution layer where the graphs are path graphs"""
        # Build the adjacency of path graphs to create a grid
        self.padding = padding
        if padding:
            length = length + 2 * k

        A = np.zeros((length, length))
        for i in range(length - 1):
            A[i, i + 1] = 1
        if not directed:
            A = A + A.T
        super().__init__(A, A, channels, units, k, isotropic)

    def forward(self, x):
        if self.padding:
            new_x = self.pad(x)
            y = super().forward(x)
            new_y = self.unpad(y)
            return new_y

        return super().forward(x)

    def pad(self, x):
        s0, s1, s2, s3 = x.shape
        size = self.size
        new_x = torch.zeros((s0, s1, s2 + 2 * size, s3 + 2 * size), dtype=x.dtype, device=x.device)
        new_x[:, :, size:-size, size:-size] = x
        return new_x

    def unpad(self, x):
        size = self.size
        return x[:, :, size:-size, size:-size]


