from math import log2
from typing import Dict

import numpy as np
from scipy.sparse import csr_matrix
from sympy import sqrt, eye, diag
from sympy.physics.secondquant import matrix_rep, B, VarBosonicBasis, Bd

from utils.matrix_operator_helper_functions import sparse_multi_tensor_product
from operators.hamiltonian_base_class import HamiltonianBaseClass
from operators.pauli_decomposition import PauliDecomposition, \
    pauli_tensor_product_power


class CoupledQuarticOscillatorsHamiltonian(HamiltonianBaseClass):
    def __init__(self, n: int, d: int, couplings: np.ndarray, anharmonicities: np.ndarray or None):
        """
        Create Hamiltonian of the form Σ_i^n x_i^2 + p_i^2 + α_i x_i^4 + Σ_i,j γ_i,j x_i x_j.
        Truncating local Fock space of oscillators to be of dimension d.

        :param n: Number of oscillators.
        :param d: Number of dimensionality of truncated Fock space
        :param couplings: Coupling matrix containing values of γ_i,j
        :param anharmonicities: Prefactors α_i for x_i^4 anharmonic terms.
        """

        super().__init__(n)

        self.d = d

        if type(d) is not int or d <= 0:
            raise TypeError("d must be positive int.")

        if type(couplings) is not np.ndarray or couplings.shape != (n, n):
            raise TypeError("couplings must be nxn numpy array.")

        anharmonicities = np.zeros(n) if anharmonicities is None else anharmonicities

        if type(anharmonicities) is not np.ndarray or anharmonicities.shape != (n,):
            raise TypeError("anharmonicities must be n length numpy array.")

        self.couplings = couplings
        self.anharmonicities = anharmonicities

        self._paulis = None

    @property
    def num_qubits(self) -> int:
        return self.n * int(log2(self.d))

    def _create_sparse_matrices(self) -> Dict[str, csr_matrix]:

        Id, K, X, X4 = self._operator_matrices()

        for i in range(0, self.n):
            sparse_h_term = sparse_multi_tensor_product([Id, ] * i + [K, ] + [Id, ] * (self.n - i - 1))
            yield {"k" + str(i): sparse_h_term}

        for i in range(0, self.n - 1):
            for j in range(i + 1, self.n):
                sparse_h_term = sparse_multi_tensor_product(
                    [Id, ] * i + [X, ] + [Id, ] * (j - i - 1) + [X, ] + [Id, ] * (self.n - j - 1)
                ).multiply(self.couplings[i, j])
                yield {"x" + str(i) + "x" + str(j): sparse_h_term}

        for i in range(self.n):
            if self.anharmonicities[i]:
                sparse_h_term = sparse_multi_tensor_product(
                    [Id, ] * i + [self.anharmonicities[i] * X4, ] + [Id, ] * (self.n - i - 1))
                yield {"xquad" + str(i): sparse_h_term}

    def _operator_matrices(self) -> (csr_matrix, csr_matrix, csr_matrix, csr_matrix):
        a = matrix_rep(B(0), VarBosonicBasis(self.d))
        aD = matrix_rep(Bd(0), VarBosonicBasis(self.d))
        X = (aD + a) / sqrt(2)
        Id = eye(self.d)
        N = diag(range(self.d), unpack=True)
        K = 2 * N + Id
        K = csr_matrix(np.array(K, dtype=float))
        Id = csr_matrix(np.array(Id, dtype=float))
        X = csr_matrix(np.array(X, dtype=float))
        X2 = X.dot(X)
        X3 = X2.dot(X)
        X4 = X3.dot(X)
        return Id, K, X, X4

    def _pauli_decomposition(self) -> PauliDecomposition:
        if np.log2(self.d) % 1 != 0:
            raise ValueError("d must be power of 2 in order to perform Pauli decomposition [Hamiltonian ∈ GL(2^nd,ℝ)].")

        Id, K, X, X4 = self._operator_matrices()

        Id_paulis = PauliDecomposition.generate_from_matrix(Id)
        K_paulis = PauliDecomposition.generate_from_matrix(K)
        X_paulis = PauliDecomposition.generate_from_matrix(X)
        X4_paulis = PauliDecomposition.generate_from_matrix(X4)

        harmonic_uncoupled_paulis = K_paulis @ pauli_tensor_product_power(Id_paulis, self.n - 1)

        for i in range(1, self.n):
            harmonic_uncoupled_paulis \
                += pauli_tensor_product_power(Id_paulis, i) @ K_paulis @ pauli_tensor_product_power(Id_paulis,
                                                                                                    self.n - 1 - i)

        anharmonic_potentials = X4_paulis @ pauli_tensor_product_power(Id_paulis, self.n - 1) * self.anharmonicities[0]

        for i in range(1, self.n):
            anharmonic_potentials += (pauli_tensor_product_power(Id_paulis, i)
                                      @ X4_paulis
                                      @ pauli_tensor_product_power(Id_paulis, self.n - 1 - i)
                                      * self.anharmonicities[i])

        if self.n == 1:
            interaction_terms = PauliDecomposition({"I" * int(self.n * np.log2(self.d)): 0.})
        else:
            interaction_terms = X_paulis @ X_paulis @ pauli_tensor_product_power(Id_paulis, self.n - 2) * \
                                self.couplings[0, 1]

            for i in list(range(0, self.n, 2)) + list(range(1, self.n, 2)):
                for j in list(range(i + 1, self.n, 2)) + list(range(i + 2, self.n, 2)):
                    if not (i == 0 and j == 1) and self.couplings[i, j] != 0.0:
                        interaction_terms += (pauli_tensor_product_power(Id_paulis, i)
                                              @ X_paulis
                                              @ pauli_tensor_product_power(Id_paulis, j - i - 1)
                                              @ X_paulis
                                              @ pauli_tensor_product_power(Id_paulis, self.n - j - 1)
                                              * self.couplings[i, j])

        return harmonic_uncoupled_paulis + anharmonic_potentials + interaction_terms
