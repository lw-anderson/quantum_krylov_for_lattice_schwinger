from unittest import TestCase

import numpy as np
from numpy.linalg import eig
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

from operators.schwinger_hamiltonian import SchwingerHamiltonian

sm = (2, 4, .5, .1, .3)


class TestSchwingerHamiltonian(TestCase):
    def setUp(self):
        self.n = 2
        self.d = 2
        self.mu = 0.5
        self.x = 0.5
        self.k = 0.

        self.ham = SchwingerHamiltonian(self.n, self.d, self.mu, self.x, self.k)

        self.expected_dim = 2 ** self.n * self.d ** (self.n - 1)

    def test_init_fail(self):
        self.assertRaises(TypeError, SchwingerHamiltonian, 1.2, self.d, self.mu, self.x, self.k)
        self.assertRaises(TypeError, SchwingerHamiltonian, self.n, 4.3, self.mu, self.x, self.k)
        self.assertRaises(TypeError, SchwingerHamiltonian, self.n, self.d, "hello", self.x, self.k)
        self.assertRaises(TypeError, SchwingerHamiltonian, self.n, self.d, self.mu, [1., 2., 3.], self.k)
        self.assertRaises(TypeError, SchwingerHamiltonian, self.n, self.d, self.mu, [1., 2., 3.], 1. + 1.j)

    def test_hamiltonian_sp_matrix(self):
        sp_mat = self.ham.hamiltonian_sp_matrix
        self.assertIsInstance(sp_mat, csr_matrix)
        self.assertEqual(sp_mat.shape, (self.expected_dim, self.expected_dim))

    def test_hamiltonian_matrix(self):
        mat = self.ham.hamiltonian_matrix
        self.assertIsInstance(mat, np.ndarray)
        self.assertEqual(mat.shape, (self.expected_dim, self.expected_dim))

    def test_ham_terms_sp_matrices(self):
        matrices = self.ham.ham_terms_sp_matrices

        self.assertEqual(len(matrices), (self.n - 1) + 2 * (self.n - 1) + self.n)

        for mat in matrices.values():
            self.assertIsInstance(mat, csr_matrix)
            self.assertEqual(mat.shape, (self.expected_dim, self.expected_dim))

    def test_ham_terms_matrices(self):
        matrices = self.ham.ham_terms_matrices

        for mat in matrices.values():
            assert type(mat) == np.ndarray
            assert mat.shape == (self.expected_dim, self.expected_dim)

        self.assertEqual(len(matrices), (self.n - 1) + 2 * (self.n - 1) + self.n)

    def test_paulis(self):
        self.assertRaises(NotImplementedError, self.ham._pauli_decomposition)

    def test_gauss_operators(self):
        ham_mat = self.ham.hamiltonian_sp_matrix
        gauss_mats = self.ham.gauss_operators

        self.assertEqual(len(gauss_mats), self.n - 1)

        for key in gauss_mats:
            gauss_mat = gauss_mats[key]
            np.testing.assert_array_almost_equal((ham_mat @ gauss_mat).toarray(), (gauss_mat @ ham_mat).toarray(),
                                                 decimal=7)

    def test_direct_diagonalisation(self):
        sp_mat = self.ham.hamiltonian_sp_matrix

        eigval, eigvec = eigsh(sp_mat, k=1, which="SA")

        gs_energy_exp = (eigvec.T.conj() @ sp_mat @ eigvec)[0, 0] / (eigvec.T.conj() @ eigvec)[0, 0]
        gs_energy_squared_exp = (eigvec.T.conj() @ sp_mat @ sp_mat @ eigvec)[0, 0] / (eigvec.T.conj() @ eigvec)[0, 0]

        self.assertAlmostEqual(gs_energy_exp, eigval[0], 7, "GS eigenvector does not match GS energy quoted.")
        self.assertAlmostEqual(gs_energy_exp ** 2, gs_energy_squared_exp, 7,
                               "GS energy expectation squared does not match GS energy squared expectation.")

    def test_np_direct_diagonalisation(self):
        mat = self.ham.hamiltonian_sp_matrix.toarray()

        eigvals, eigvecs = eig(mat)

        eigval = eigvals[0]
        eigvec = eigvecs[:, 0]

        gs_energy_exp = (eigvec.T.conj() @ mat @ eigvec) / (eigvec.T.conj() @ eigvec)
        gs_energy_squared_exp = (eigvec.T.conj() @ mat @ mat @ eigvec) / (eigvec.T.conj() @ eigvec)

        self.assertAlmostEqual(gs_energy_exp, eigval, 7, "GS eigenvector does not match GS energy quoted.")
        self.assertAlmostEqual(gs_energy_exp ** 2, gs_energy_squared_exp, 7,
                               "GS energy expectation squared does not match GS energy squared expectation.")
