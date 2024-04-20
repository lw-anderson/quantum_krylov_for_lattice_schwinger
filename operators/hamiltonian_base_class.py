from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
from scipy.sparse import csr_matrix

from operators.pauli_decomposition import PauliDecomposition


class HamiltonianBaseClass(ABC):
    def __init__(self, n: int):

        if type(n) is not int or n <= 0:
            raise TypeError("n must be int.")

        self.n = n

        self._hamiltonian_matrix = None
        self._hamiltonian_sp_matrix = None

        self._ham_terms_matrices = None
        self._ham_terms_sp_matrices = None

        self._paulis = None

    @property
    @abstractmethod
    def num_qubits(self) -> int:
        pass

    @property
    def hamiltonian_matrix(self) -> np.ndarray:
        if self._hamiltonian_matrix is None:
            self._hamiltonian_matrix = self.hamiltonian_sp_matrix.toarray()
        return self._hamiltonian_matrix

    @property
    def hamiltonian_sp_matrix(self) -> csr_matrix:
        if self._hamiltonian_sp_matrix is None:

            if self._ham_terms_sp_matrices is not None:
                first_term_flag = True
                for mat in self._ham_terms_sp_matrices.values():
                    if first_term_flag:
                        self._hamiltonian_sp_matrix = mat
                        first_term_flag = False
                    else:
                        self._hamiltonian_sp_matrix += mat

            else:
                first_term_flag = True
                for name_mat in self._create_sparse_matrices():

                    assert len(name_mat) == 1, "something went wrong!"

                    if first_term_flag:
                        self._hamiltonian_sp_matrix = next(iter(name_mat.values()))
                        first_term_flag = False
                    else:
                        self._hamiltonian_sp_matrix += next(iter(name_mat.values()))
        return self._hamiltonian_sp_matrix

    @property
    def ham_terms_matrices(self) -> Dict[str, np.ndarray]:
        if self._ham_terms_matrices is None:
            self._ham_terms_matrices = {}
            for key, sp_mat in self.ham_terms_sp_matrices.items():
                self._ham_terms_matrices[key] = sp_mat.toarray()
        return self._ham_terms_matrices

    @property
    def ham_terms_sp_matrices(self) -> Dict[str, csr_matrix]:
        if self._ham_terms_sp_matrices is None:
            terms = {}
            for key_and_mat in self._create_sparse_matrices():
                terms = {**terms, **key_and_mat}
            self._ham_terms_sp_matrices = terms

        return self._ham_terms_sp_matrices

    @abstractmethod
    def _create_sparse_matrices(self) -> Dict[str, csr_matrix]:
        pass

    @property
    def paulis(self) -> PauliDecomposition:

        if self._paulis is None:
            self._paulis = self._pauli_decomposition()

        return self._paulis

    @abstractmethod
    def _pauli_decomposition(self) -> PauliDecomposition:
        pass
