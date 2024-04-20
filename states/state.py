from typing import Union

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import norm

from operators.hamiltonian_base_class import HamiltonianBaseClass
from operators.time_evolution import TimeEvolution


class State:
    def __init__(self, num_qubits: int, vector: Union[np.ndarray, csr_matrix] or None):
        if type(num_qubits) is not int or num_qubits <= 0:
            raise TypeError("num_qubits must be positive int.")

        self.num_qubits = num_qubits

        if type(vector) not in [np.ndarray, csr_matrix] and vector is not None:
            raise TypeError("vector must be numpy array, scipy csr_matrix or None.")

        if vector is None:
            vector = np.array([1., ] + [0., ] * (2 ** num_qubits - 1))

        if type(vector) is csr_matrix:
            self._vector = csr_matrix(vector.reshape((1, 2 ** self.num_qubits)))
        else:
            self._vector = csr_matrix(vector)

        self._check_valid_state(self._vector)

    @property
    def ket(self):
        return self._vector.transpose()

    @property
    def bra(self):
        return self._vector.conjugate()

    def __add__(self, other: "State") -> "State":
        if other.num_qubits != self.num_qubits:
            raise ValueError("states for addition must have same dimensions (n and d).")
        vector = self.ket.transpose() + other.ket.transpose()
        return State(self.num_qubits, vector.toarray().flatten())

    def __mul__(self, other: Union[HamiltonianBaseClass, TimeEvolution, "State"]) -> "State" or float:
        # Multiplication of the form self * other
        if other.num_qubits != self.num_qubits:
            raise ValueError("multiplying objects must have same dimensions (n and d).")

        if isinstance(other, HamiltonianBaseClass):
            vector = other.hamiltonian_sp_matrix.conjugate().transpose() * self.ket
            return State(self.num_qubits, vector.toarray().flatten())
        elif isinstance(other, TimeEvolution):
            vector = other.time_evolution_sp_matrix.conjugate().transpose() * self.ket
            return State(self.num_qubits, vector.toarray().flatten())
        elif type(other) == type(self):
            return self.bra.dot(other.ket)[0, 0]
        else:
            raise TypeError("multiplying objects should be State, Hamiltonian or TimeEvolution type.")

    def __rmul__(self, other: Union[HamiltonianBaseClass, TimeEvolution, float, complex]) -> "State":
        # Multiplication of the form other * self
        if isinstance(other, (float, complex)):
            vector = other * self.ket
            return State(self.num_qubits, vector.toarray().flatten())
        elif isinstance(other, HamiltonianBaseClass):
            if other.num_qubits != self.num_qubits:
                raise ValueError("multiplying objects (states and Hamiltonians) must have same dimensions (n and d).")
            vector = other.hamiltonian_sp_matrix * self.ket
            return State(self.num_qubits, vector.toarray().flatten())
        elif isinstance(other, TimeEvolution):
            if other.num_qubits != self.num_qubits:
                raise ValueError("multiplying objects (states and Hamiltonians) must have same dimensions (n and d).")
            vector = other.time_evolution_sp_matrix * self.ket
            return State(self.num_qubits, vector.toarray().flatten())
        else:
            raise TypeError("multiplying object should be Hamiltonian or TimeEvolution type.")

    def low_mem_ham_mul(self, ham: HamiltonianBaseClass) -> "State":
        """
        Multiplication of the form self * ham. Never creates full sparse Hamiltonian matrix to save on memory
        instead uses generator to iterate through a single individual term at a time.
        """
        if isinstance(ham, HamiltonianBaseClass):
            vector = csr_matrix(self.ket.shape)

            for name_mat in ham._create_sparse_matrices():
                mat = next(iter(name_mat.values()))
                vector += mat.conjugate().transpose() * self.ket

            return State(self.num_qubits, vector.toarray().flatten())

        else:
            raise TypeError("multiplying object should be Hamiltonian.")

    def low_mem_ham_rmul(self, ham: HamiltonianBaseClass) -> "State":
        """
        Multiplication of the form ham * self. Never creates full sparse Hamiltonian matrix to save on memory
        instead uses generator to iterate through a single individual term at a time.
        """
        if isinstance(ham, HamiltonianBaseClass):
            vector = csr_matrix(self.ket.shape)

            for name_mat in ham._create_sparse_matrices():
                mat = next(iter(name_mat.values()))
                vector += mat * self.ket

            return State(self.num_qubits, vector)

        else:
            raise TypeError("multiplying object should be Hamiltonian.")

    def _check_valid_state(self, vector) -> None:
        if vector.shape != (1, 2 ** self.num_qubits):
            raise ValueError(
                f"vector must be of shape (2^num_qubits, 1)=({2 ** self.num_qubits}, 1). Current shape = {self._vector.shape}")
        return

    def normalise(self) -> None:
        self._vector = self._vector / norm(self._vector)
