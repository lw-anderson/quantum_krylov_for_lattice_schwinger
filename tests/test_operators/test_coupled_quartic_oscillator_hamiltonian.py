from unittest import TestCase

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

from operators.coupled_quartic_oscillators_hamiltonian import CoupledQuarticOscillatorsHamiltonian
from operators.pauli_decomposition import PauliDecomposition


class TestHamiltonian(TestCase):
    def setUp(self):
        self.n = 2
        self.d = 2
        self.couplings = np.array([[0., 1.], [1., 0.]])
        self.anharmonicities = None
        self.ham = CoupledQuarticOscillatorsHamiltonian(self.n, self.d, self.couplings, self.anharmonicities)

    def test_init_correct(self):
        ham = CoupledQuarticOscillatorsHamiltonian(2, 2, np.array([[0, 1], [1, 0]]), None)
        ham = CoupledQuarticOscillatorsHamiltonian(2, 2, np.array([[0, 1], [1, 0]]), np.array([0.1, 0.1]))
        ham = CoupledQuarticOscillatorsHamiltonian(3, 4, np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]), None)

    def test_init_fail(self):
        self.assertRaises(TypeError, CoupledQuarticOscillatorsHamiltonian, 1.0, self.d, self.couplings, np.array([]))
        self.assertRaises(TypeError, CoupledQuarticOscillatorsHamiltonian, self.n, "hello", self.couplings,
                          np.array([]))
        self.assertRaises(TypeError, CoupledQuarticOscillatorsHamiltonian, self.n, self.d, [[0, 1], [1, 0]],
                          np.array([]))
        self.assertRaises(TypeError, CoupledQuarticOscillatorsHamiltonian, self.n, self.d, self.couplings, np.array([]))

    def test_hamiltonian_sp_matrix(self):
        sp_mat = self.ham.hamiltonian_sp_matrix
        assert type(sp_mat) == csr_matrix

    def test_hamiltonian_matrix(self):
        mat = self.ham.hamiltonian_matrix
        assert type(mat) == np.ndarray
        assert mat.shape == (2 ** 2, 2 ** 2)

    def test_ham_terms_sp_matrices(self):
        sp_matrices = self.ham.ham_terms_sp_matrices
        assert list(sp_matrices.keys()) == ["k0", "k1", "x0x1"]
        for key, sp_mat in sp_matrices.items():
            assert type(sp_mat) == csr_matrix
            assert sp_mat.shape == (2 ** 2, 2 ** 2)
        assert len(sp_matrices) == 2 + 2 * (2 - 1) // 2

    def test_ham_terms_matrices(self):
        matrices = self.ham.ham_terms_matrices
        assert list(matrices.keys()) == ["k0", "k1", "x0x1"]
        for key, mat in matrices.items():
            assert type(mat) == np.ndarray
            assert mat.shape == (2 ** 2, 2 ** 2)
        assert len(matrices) == 2 + 2 * (2 - 1) // 2

    def test_paulis(self):
        self.assertIsInstance(self.ham.paulis, PauliDecomposition)

    def test_direct_diagonalisation(self):
        sp_mat = self.ham.hamiltonian_sp_matrix

        eigval, eigvec = eigsh(sp_mat, k=1, which="SA")

        gs_energy_exp = (eigvec.T.conj() @ sp_mat @ eigvec)[0, 0] / (eigvec.T.conj() @ eigvec)[0, 0]
        gs_energy_squared_exp = (eigvec.T.conj() @ sp_mat @ sp_mat @ eigvec)[0, 0] / (eigvec.T.conj() @ eigvec)[0, 0]

        self.assertAlmostEqual(gs_energy_exp, eigval[0], 7, "GS eigenvector does not match GS energy quoted.")
        self.assertAlmostEqual(gs_energy_exp ** 2, gs_energy_squared_exp, 7,
                               "GS energy expectation squared does not match GS energy squared expectation.")
