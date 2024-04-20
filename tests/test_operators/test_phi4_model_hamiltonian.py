from unittest import TestCase

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

from operators.phi4_model_hamiltonian import Phi4ModelHamiltonian


class TestPhi4ModelHamiltonian(TestCase):
    def setUp(self):
        self.n = 9
        self.d = 4
        self.a = 0.1
        self.m = 0.2
        self.lam = 0.3

        self.ham_1d = Phi4ModelHamiltonian(1, self.n, self.d, self.a, self.m, self.lam)
        self.ham_2d = Phi4ModelHamiltonian(2, self.n, self.d, self.a, self.m, self.lam)

    def test_init_fail(self):
        self.assertRaises(ValueError, Phi4ModelHamiltonian, 3, 10, self.d, self.a, self.m, self.lam)
        self.assertRaises(ValueError, Phi4ModelHamiltonian, 2, 10, self.d, self.a, self.m, self.lam)

    def test_1d_hamiltonian_sp_matrix(self):
        sp_mat_1d = self.ham_1d.hamiltonian_sp_matrix
        self.assertIsInstance(sp_mat_1d, csr_matrix)
        expected_dim = self.d ** self.n
        self.assertEqual(sp_mat_1d.shape, (expected_dim, expected_dim))

    def test_1d_ham_terms_sp_matrices(self):
        matrices = self.ham_1d.ham_terms_sp_matrices

        expected_dim = self.d ** self.n

        for mat in matrices.values():
            self.assertIsInstance(mat, csr_matrix)
            self.assertEqual(mat.shape, (expected_dim, expected_dim))

        self.assertEqual(len(matrices), 4 * self.n)

    def test_1d_ham_terms_matrices(self):
        n = 4
        ham_1d_small = Phi4ModelHamiltonian(1, n, self.d, self.a, self.m, self.lam)
        matrices = ham_1d_small.ham_terms_matrices

        expected_dim = self.d ** n

        for mat in matrices.values():
            assert type(mat) == np.ndarray
            assert mat.shape == (expected_dim, expected_dim)

        self.assertEqual(len(matrices), 4 * n)

    def test_2d_ham_terms_sp_matrices(self):
        matrices = self.ham_2d.ham_terms_sp_matrices

        expected_dim = self.d ** self.n

        for mat in matrices.values():
            self.assertIsInstance(mat, csr_matrix)
            self.assertEqual(mat.shape, (expected_dim, expected_dim))

        self.assertEqual(len(matrices), 5 * self.n)

    def test_2d_ham_terms_matrices(self):
        n = 4
        ham_2d_small = Phi4ModelHamiltonian(2, n, self.d, self.a, self.m, self.lam)
        matrices = ham_2d_small.ham_terms_matrices

        expected_dim = self.d ** n

        for mat in matrices.values():
            assert type(mat) == np.ndarray
            assert mat.shape == (expected_dim, expected_dim)

        self.assertEqual(len(matrices), 5 * n)

    def test_paulis(self):
        self.assertRaises(NotImplementedError, self.ham_1d._pauli_decomposition)

    def test_direct_diagonalisation(self):
        sp_mat = self.ham_1d.hamiltonian_sp_matrix

        eigval, eigvec = eigsh(sp_mat, k=1, which="SA")

        gs_energy_exp = (eigvec.T.conj() @ sp_mat @ eigvec)[0, 0] / (eigvec.T.conj() @ eigvec)[0, 0]
        gs_energy_squared_exp = (eigvec.T.conj() @ sp_mat @ sp_mat @ eigvec)[0, 0] / (eigvec.T.conj() @ eigvec)[0, 0]

        self.assertAlmostEqual(gs_energy_exp, eigval[0], 7, "GS eigenvector does not match GS energy quoted.")
        self.assertAlmostEqual(gs_energy_exp ** 2, gs_energy_squared_exp, 7,
                               "GS energy expectation squared does not match GS energy squared expectation.")
