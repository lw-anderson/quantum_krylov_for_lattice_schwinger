from math import log2
from typing import Dict

import numpy as np
from scipy.sparse import csr_matrix

from utils.matrix_operator_helper_functions import sparse_multi_tensor_product
from operators.schwinger_hamiltonian import SchwingerHamiltonian

floattype = np.float64


class SchwingerGaugedHamiltonian(SchwingerHamiltonian):
    def __init__(self, n, mu, x, k):
        super().__init__(n, 1, mu, x, k)

    @property
    def num_qubits(self) -> int:
        return (self.n - 1) * int(log2(self.d)) + self.n

    def _create_sparse_matrices(self) -> Dict[str, csr_matrix]:
        _, _, _, _, _, sigma_minus, sigma_plus, sigma_3, fermionic_Id = self._operator_matrices()
        n = self.n

        for r in range(1, n):
            sparse_h_term = csr_matrix((2 ** n, 2 ** n), dtype=floattype)
            for m in range(1, r + 1):
                sparse_h_term += sparse_multi_tensor_product([fermionic_Id] * (m - 1)
                                                             + [sigma_3 + (-1) ** m * fermionic_Id]
                                                             + [fermionic_Id] * (n - m)
                                                             )
            sparse_h_term = sparse_h_term.multiply(floattype(0.5))
            sparse_h_term = sparse_h_term @ sparse_h_term

            yield {"H_E_" + str(r): sparse_h_term}

        for r in range(1, n):
            sparse_h_term = sparse_multi_tensor_product(
                [fermionic_Id] * (r - 1)
                + [sigma_plus, sigma_minus, ]
                + [fermionic_Id] * (n - r - 1)
            ) * self.x
            yield {"H_I_" + str(r) + "L": sparse_h_term}

            sparse_h_term = sparse_multi_tensor_product(
                [fermionic_Id] * (r - 1)
                + [sigma_minus, sigma_plus, ]
                + [fermionic_Id] * (n - r - 1)
            ) * floattype(self.x)
            yield {"H_I_" + str(r) + "R": sparse_h_term}

        for r in range(1, n + 1):
            sparse_h_term = sparse_multi_tensor_product(
                [fermionic_Id] * (r - 1)
                + [sigma_3, ]
                + [fermionic_Id] * (n - r)
            ).multiply(floattype((self.k + self.mu * (-1) ** r) / 2))
            yield {"H_M_" + str(r): sparse_h_term}

    @property
    def gauss_operators(self) -> Dict[str, csr_matrix]:
        raise NotImplementedError("Gauged Schwinger Hamiltonian does not have Gauss law operators defined.")
