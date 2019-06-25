import numpy as np
import scipy.sparse as sparse
import torch


class ExtendedGraph():
    def __init__(self, A, k, use_L, use_chebychev):
        """ A: adjacency matrix of the graph
            k (int): maximum power of the polynomials. """
        self.k, self._k = k, k + 1      # Polynomial of degree k is included
        A_full = A.toarray() if sparse.issparse(A) else A
        self.N = A_full.shape[0]
        self.directed = not np.allclose(A_full, A_full.T)

        if use_L:
            # Normalize the Laplacian
            isqrt = []
            for axis in [0, 1]:
                sqrt_d = np.sqrt(np.sum(A_full, axis=axis))
                sqrt_d[sqrt_d == 0] = 1
                isqrt.append(np.diag(1 / sqrt_d))
            # Shifted normalized Laplacian
            self.L = - isqrt[0] @ A_full @ isqrt[1]
        else:
            self.L = A_full

        # Computation of a normalized Laplacian
        self.L = self.L.astype(np.float32)
        self.L = torch.from_numpy(self.L)
        self.L.requires_grad_(False)
        self.chebychev = self.compute_polynomials(use_chebychev)

    def compute_polynomials(self, use_chebychev: bool):
        """ If use_chebychev:
                Compute the Chebychev polynomials up to degree k (included).
            else:
                Compute powers of the Laplacian.
        """
        pol = torch.empty((self._k, self.N, self.N), requires_grad=False,
                           dtype=torch.float)
        pol[0] = torch.eye(self.N)
        if self._k > 1:
            pol[1] = self.L
        for i in range(2, self._k):
            if use_chebychev:
                pol[i] = 2 * torch.matmul(self.L, pol[i - 1]) - pol[i - 2]
            else:
                pol[i] = torch.matmul(self.L, pol[i - 1])
        if not self.directed:
            return pol

        # Handle directed graphs by taking the transpose and computing power
        pol_inv = torch.clone(pol)
        for i in range(self._k):
            pol_inv[i] = pol_inv[i].transpose(0, 1)

        # Beware to remove the power 0 that appears twice
        pol = torch.cat((pol, pol_inv[1:]))
        return pol

