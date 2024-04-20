from math import log2
from typing import Dict

import numpy as np
from scipy.sparse import csr_matrix
from sympy import eye, Matrix

from utils.matrix_operator_helper_functions import sparse_multi_tensor_product
from operators.hamiltonian_base_class import HamiltonianBaseClass
from operators.pauli_decomposition import PauliDecomposition

floattype = np.float64


class SchwingerHamiltonian(HamiltonianBaseClass):
    def __init__(self, n: int, d: int, mu: float, x: float, k: float):
        """
        Create Hamiltonian of the 1+1D Schwinger model. See for example Shaw et al. Quantum 4, 306 (2020) and Banuls
        et al. PRL 118 (2017) [eqn. 1] for form of Hamiltonian.
        Truncating local Fock space of bosonic fields to be of dimension d. For n Fermionic sites, there will be
        n-1 Bosonic chain link fields.
        Following notation by Shaw (with additional chemical potential term), the full Hamiltonian is given by

            H = H_E + H_I + H_M

        where

            H_E = Σ_r E_r^2 (bosonic field)
            H_I = x Σ_r [ψ_r U_r ψ_(r+1)† - ψ_r U_r ψ_(r+1)†] (coupling term)
            H_M = Σ_r (k + mu (-1)^r) ψ_r ψ_r† (mass and chemical potential term)

        Here ψ is the Fermionic ladder operator. E and U are Bosonic number and ladder operators satisfying
            [E_r,U_s] = U_r δ_rs
            [E_r,U_s†] = -U_r† δ_rs
        up to truncation.

        :param n: Number of Fermionic sites.
        :param d: Dimensionality of truncated Bosonic Fock space.
        :param mu: Rescaled mass.
        :param x: Rescaled coupling.
        :param k: Rescaled chemical potential.
        """

        super().__init__(n)

        if type(d) is not int or d <= 0:
            raise TypeError("d must be positive int.")

        self.d = d

        if type(mu) is not float or type(x) is not float or type(k) is not float:
            raise TypeError("mu, x and k must be floats.")

        self.mu = mu
        self.x = x
        self.k = k

        self._paulis = None

        self._gauss_operators = None

        # value used to pad bosonic operators to length 2^integer. Generally val=0 leads to spurious zero energy
        # eigenvalues, val=large removes these but Gauss ops and Hamiltonian stop commuting.
        self._E_dummy_val = 0

    @property
    def num_qubits(self) -> int:
        return (self.n - 1) * int(log2(self.d)) + self.n

    def _create_sparse_matrices(self) -> Dict[str, csr_matrix]:
        U, Ud, bosonic_Id, E, E2, sigma_minus, sigma_plus, sigma_3, fermionic_Id = self._operator_matrices()

        n = self.n

        for r in range(1, n):
            sparse_h_term = sparse_multi_tensor_product(
                [fermionic_Id, bosonic_Id] * (r - 1)
                + [fermionic_Id, E2, fermionic_Id]
                + [bosonic_Id, fermionic_Id] * (n - r - 1)
            )

            yield {"H_E_" + str(r): sparse_h_term}

        for r in range(1, n):
            sparse_h_term = sparse_multi_tensor_product([fermionic_Id, bosonic_Id] * (r - 1)
                                                        + [sigma_plus, Ud, sigma_minus]
                                                        + [bosonic_Id, fermionic_Id] * (n - r - 1)
                                                        )
            yield {"H_I_R_" + str(r): sparse_h_term * self.x}

            sparse_h_term = sparse_multi_tensor_product([fermionic_Id, bosonic_Id] * (r - 1)
                                                        + [sigma_minus, U, sigma_plus]
                                                        + [bosonic_Id, fermionic_Id] * (n - r - 1)
                                                        ) * self.x
            yield {"H_I_L_" + str(r): sparse_h_term}

        for r in range(1, n + 1):
            sparse_h_term = sparse_multi_tensor_product(
                [fermionic_Id, bosonic_Id] * (r - 1)
                + [sigma_3]
                + [bosonic_Id, fermionic_Id] * (n - r)
            ).multiply(floattype((self.k + self.mu * (-1) ** r) / 2))

            yield {"H_M_" + str(r): sparse_h_term}

    @property
    def gauss_operators(self) -> Dict[str, csr_matrix]:
        if self._gauss_operators is None:
            terms = {}
            for key_and_mat in self._create_gauss_operators():
                terms = {**terms, **key_and_mat}
            self._gauss_operators = terms

        return self._gauss_operators

    def _create_gauss_operators(self) -> Dict[str, csr_matrix]:
        U, Ud, bosonic_Id, E, E2, sigma_minus, sigma_plus, sigma_3, fermionic_Id = self._operator_matrices()

        n = self.n

        for r in range(1, n):
            # fermionic term (right most term in above)
            gauss_op = sparse_multi_tensor_product(
                [fermionic_Id, bosonic_Id] * (r - 1)
                + [sigma_3 + (fermionic_Id * ((-1) ** r))]
                + [bosonic_Id, fermionic_Id] * (n - r)
            ).multiply(floattype(-0.5))

            if r != n:  # ensures no E(N) term (since this is not included in Hilbert space and set to zero by boundary condition)
                # E(r) term
                gauss_op += sparse_multi_tensor_product(
                    [fermionic_Id, bosonic_Id, ] * (r - 1)
                    + [fermionic_Id] + [E]
                    + [fermionic_Id]
                    + [bosonic_Id, fermionic_Id] * (n - 1 - r)
                ).multiply(floattype(1.))

            if r != 1:  # ensures no E(0) term (since this is not included in Hilbert space and set to zero by boundary condition)
                # E(r-1) term
                gauss_op += sparse_multi_tensor_product(
                    [fermionic_Id, bosonic_Id, ] * (r - 2)
                    + [fermionic_Id, E, fermionic_Id]
                    + [bosonic_Id, fermionic_Id, ] * (n - r)
                ).multiply(floattype(-1.))

            yield {"G_" + str(r): gauss_op}

    def _operator_matrices(self) -> (Matrix, Matrix, Matrix, Matrix, Matrix, Matrix, Matrix):

        E = np.diag(list(range(1 - self.d // 2, self.d // 2)) + [self._E_dummy_val, ])
        U = np.diag([1., ] * (self.d - 2) + [self._E_dummy_val, ], 1)
        Ud = np.diag([1., ] * (self.d - 2) + [self._E_dummy_val, ], -1)

        E2 = E * E
        bosonic_Id = np.eye(self.d)
        bosonic_Id[self.d - 1, self.d - 1] = self._E_dummy_val

        sigma_minus = Matrix([[0, 0], [1, 0]])  # = |1><0| associated with Fermionic creation operator ψ†
        sigma_plus = Matrix([[0, 1], [0, 0]])  # = |0><1| associated with ψ
        sigma_3 = Matrix([[1, 0], [0, -1]])  # = Pauli Z matrix, ψ†ψ = |1><1| fermionic number operator
        fermionic_Id = eye(2)  # Fermionic identity

        U = csr_matrix(np.array(U), dtype=floattype)  # = e^(-i\theta), lowering [E,U] = -U
        Ud = csr_matrix(np.array(Ud), dtype=floattype)  # = e^(i\theta), raising [E,Ud] = Ud
        E = csr_matrix(np.array(E), dtype=floattype)
        E2 = csr_matrix(np.array(E2), dtype=floattype)
        bosonic_Id = csr_matrix(np.array(bosonic_Id), dtype=floattype)
        sigma_minus = csr_matrix(np.array(sigma_minus), dtype=floattype)
        sigma_plus = csr_matrix(np.array(sigma_plus), dtype=floattype)
        sigma_3 = csr_matrix(np.array(sigma_3), dtype=floattype)
        fermionic_Id = csr_matrix(np.array(fermionic_Id), dtype=floattype)

        return U, Ud, bosonic_Id, E, E2, sigma_minus, sigma_plus, sigma_3, fermionic_Id

    def _pauli_decomposition(self) -> PauliDecomposition:
        raise NotImplementedError
