from copy import copy
from math import log2, isqrt, sqrt
from typing import Dict

import numpy as np
from scipy.sparse import csr_matrix
from sympy import eye
from sympy.physics.secondquant import matrix_rep, VarBosonicBasis, B, Bd

from utils.matrix_operator_helper_functions import sparse_multi_tensor_product
from operators.hamiltonian_base_class import HamiltonianBaseClass
from operators.pauli_decomposition import PauliDecomposition


class Phi4ModelHamiltonian(HamiltonianBaseClass):
    def __init__(self, dims: int, n: int, d: int, a: float, m: float, lam: float):
        """
        Create Hamiltonian for 1+1D or 2+1D interacting scalar field theory. Truncating local Fock space of bosonic
        fields to be of dimension d. Following notation of Eq. (10) of Banuls et al. EPJD 72 (2020), Hamiltonian is
        given by

            H = Σ_x π(x)^2 + ∇φ(x)^2 + m^2φ(x)^2 + λφ(x)^4/12

        in units of a^(D-1)/2, where a is the lattice spacing and D is the spatial dimension (either 1 or 2).
        Discretising the gradient operator on the lattice gives

            D=1: ∇φ(x)^2 = [ φ(x + e1)^2 - 2φ(x)φ(x+e1) + φ(x)^2 ] / (4a^2),

            D=2: ∇φ(x)^2 = [ φ(x + e1)^2 - 2φ(x)φ(x+e1) + φ(x + e2)^2 - 2φ(x)φ(x+e2) + 2φ(x)^2 ] / (4a^2)

        where e1 and e2 are unit vectors in each spatial dimension. We use cyclic boundary conditions in the spatial
        directions.

        :param dims: Number of spatial dimensions (must be 1 or 2).
        :param n: Number of lattice sites. If 2 spatial dimensions, this must be square number.
        :param d: Dimensionality of truncated Bosonic Fock space.
        :param a: Lattice constant.
        :param m: Mass.
        :param lam: Interaction strength.
        """
        super().__init__(n)

        if dims == 1:
            self.dims = 1
        elif dims == 2:
            if isqrt(n) ** 2 == n:
                self.dims = 2
            else:
                raise ValueError("If dims=2, n must be square number.")
        else:
            raise ValueError("dims must be either 1 or 2")

        if type(d) is not int or d <= 0:
            raise TypeError("d must be positive int.")

        self.d = d

        if type(a) is not float:
            raise TypeError("d must be float.")

        self.a = a

        if type(a) is not float or a <= 0:
            raise TypeError("a must be positive float.")

        self.m = m

        if type(lam) is not float:
            raise TypeError("lam must be float.")

        self.lam = lam

    @property
    def num_qubits(self) -> int:
        return self.n * int(log2(self.d))

    def _create_sparse_matrices(self) -> Dict[str, csr_matrix]:
        Id, phi, phi2, phi4, pi2 = self._operator_matrices()

        def rotate(lst, num):
            return copy(lst)[-num:] + copy(lst)[:-num]

        coupling_coeff = 2 / self.a ** 2
        phi_squared_coeff = self.m + coupling_coeff if self.dims == 1 else self.m + 2 * coupling_coeff

        for x in range(self.n):
            sparse_h_term = sparse_multi_tensor_product(
                rotate([pi2, ]
                       + [Id, ] * (self.n - 1),
                       x)
            )
            yield {"pi^2_" + str(x): sparse_h_term}

        for x in range(self.n):
            sparse_h_term = sparse_multi_tensor_product(
                rotate([phi2, ]
                       + [Id, ] * (self.n - 1),
                       x)
            ) * phi_squared_coeff
            yield {"phi^2_" + str(x): sparse_h_term}

        for x in range(self.n):
            if self.n > 1:
                sparse_h_term = sparse_multi_tensor_product(
                    rotate([phi, phi]
                           + [Id, ] * (self.n - 2),
                           x)
                ) * -coupling_coeff
                yield {"phi_" + str(x) + ".phi_" + str(x + 1): sparse_h_term}

        if self.dims == 2:
            nx = int(sqrt(self.n))
            for x in range(self.n):
                if self.n > 1:
                    sparse_h_term = sparse_multi_tensor_product(
                        rotate([phi, ]
                               + [Id, ] * (nx - 1)
                               + [phi, ]
                               + [Id, ] * (self.n - nx - 1),
                               x)
                    ) * -coupling_coeff
                    yield {"phi_" + str(x) + ".phi_" + str(x + nx): sparse_h_term}

        for x in range(self.n):
            sparse_h_term = sparse_multi_tensor_product(
                rotate([phi4, ]
                       + [Id, ] * (self.n - 1),
                       x)
            ) * self.lam
            yield {"phi4_" + str(x): sparse_h_term}

    def _operator_matrices(self) -> (csr_matrix, csr_matrix, csr_matrix, csr_matrix, csr_matrix, csr_matrix):
        a = matrix_rep(B(0), VarBosonicBasis(self.d))
        aD = matrix_rep(Bd(0), VarBosonicBasis(self.d))
        phi = (aD + a) / sqrt(2)
        phi2 = phi * phi
        phi4 = phi2 * phi2
        pi = 1.j * (aD - a) / sqrt(2)
        pi2 = pi * pi
        Id = eye(self.d)

        Id = csr_matrix(np.array(Id, dtype=float))
        phi = csr_matrix(np.array(phi, dtype=float))
        phi2 = csr_matrix(np.array(phi2, dtype=float))
        phi4 = csr_matrix(np.array(phi4, dtype=float))
        pi2 = csr_matrix(np.array(pi2, dtype=float))
        return Id, phi, phi2, phi4, pi2

    def _pauli_decomposition(self) -> PauliDecomposition:
        raise NotImplementedError
