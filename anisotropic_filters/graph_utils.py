from pygsp import graphs
import numpy as np
import scipy as sp
import torch


class ExtendedGraph(graphs.Graph):
    def __init__(self, A, k):
        """ A: adjacency matrix of the graph
            k (int): maximum power of the polynomials. """
        self.k, self._k = k, k + 1      # Polynomial of degree k is included
        A_full = A.toarray() if sp.sparse.issparse(A) else A
        self.directed = not np.allclose(A_full, A_full.T)
        if self.directed:
            super().__init__(A, lap_type='combinatorial')
            self.L = self.L.toarray()
        else:
            super().__init__(A, lap_type='normalized')
            self.L = self.L.toarray() - np.eye(self.N)
        # Computation of a normalized Laplacian
        self.L = self.L.astype(np.float32)
        self.L = torch.from_numpy(self.L)
        self.L.requires_grad_(False)
        self.chebychev = self.compute_chebychev_polynomials()

    def compute_chebychev_polynomials(self):
        """ Compute the Chebychev polynomials up to degree k (included). """
        cheb = torch.empty((self._k, self.N, self.N), requires_grad=False,
                           dtype=torch.float)
        cheb[0] = torch.eye(self.N)
        if self._k > 1:
            cheb[1] = self.L
        for i in range(2, self._k):
            cheb[i] = 2 * torch.matmul(self.L, cheb[i - 1]) - cheb[i - 2]
        if not self.directed:
            return cheb

        L = self.L.transpose(0, 1)
        cheb_inv = torch.empty((self._k, self.N, self.N), requires_grad=False,
                               dtype=torch.float)
        cheb_inv[0] = torch.eye(self.N)
        if self._k > 1:
            cheb_inv[1] = L
        for i in range(2, self._k):
            cheb_inv[i] = 2 * torch.matmul(L, cheb_inv[i - 1]) - cheb_inv[i - 2]
        cheb = torch.cat([cheb, cheb_inv[1:]])
        return cheb


def adjacency_line_graph(length:int):
    A = np.zeros((length, length))
    for i in range(length - 1):
        A[i, i + 1] = 1
    A = A + A.T
    return A


if __name__ == '__main__':
    A = np.random.rand(9).reshape(3, 3)
    np.fill_diagonal(A, 0)
    A = A + A.T
    G = ExtendedGraph(A, 3, 'cpu')
    # print(G.W, G.L, G.N)