from unittest import TestCase

import numpy as np
from numpy.linalg import eig
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

from operators.schwinger_gauged_hamiltonian import SchwingerGaugedHamiltonian
from operators.schwinger_hamiltonian import SchwingerHamiltonian


class TestSchwingerGaugedHamiltonian(TestCase):
    def setUp(self):
        self.n = 2
        self.mu = 0.5
        self.x = 0.5
        self.k = 0.

        self.ham = SchwingerGaugedHamiltonian(self.n, self.mu, self.x, self.k)

        self.expected_dim = 2 ** self.n

    def test_init_fail(self):
        self.assertRaises(TypeError, SchwingerGaugedHamiltonian, 1.2, self.mu, self.x, self.k)
        self.assertRaises(TypeError, SchwingerGaugedHamiltonian, self.n, "hello", self.x, self.k)
        self.assertRaises(TypeError, SchwingerGaugedHamiltonian, self.n, self.mu, [1., 2., 3.], self.k)
        self.assertRaises(TypeError, SchwingerGaugedHamiltonian, self.n, self.mu, [1., 2., 3.], 1. + 1.j)

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
        self.assertRaises(NotImplementedError, lambda: self.ham.gauss_operators)

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

    def test_matches_ungauged_model(self):
        n = 2
        mu = 3.
        x = 2.
        k = 1.
        d = 8

        gauged_ham = SchwingerGaugedHamiltonian(n, mu, x, k)
        gauged_mat = gauged_ham.hamiltonian_sp_matrix

        ungauged_ham = SchwingerHamiltonian(n, d, mu, x, k)
        ungauged_mat = ungauged_ham.hamiltonian_sp_matrix

        gauged_eigvals, gauged_eigvecs = eigsh(gauged_mat.toarray(), k=4, which="SA")
        ungauged_eigvals, ungauged_eigvecs = eigsh(ungauged_mat.toarray(), k=32, which="SA")

        eigvals_satisfying_gauss, eigvecs_satisfying_gauss = [], []
        eigvals_not_satisfying_gauss, eigvecs_not_satisfying_gauss = [], []

        for eigvec in ungauged_eigvecs.T:
            gauss_exps = []
            for gauss_op in ungauged_ham.gauss_operators.values():
                gauss_exp = eigvec.T.conj() @ gauss_op @ eigvec
                gauss_exps.append(gauss_exp)

            energy_exp = eigvec.conj().T @ ungauged_mat @ eigvec

            if all(np.isclose(gauss_exp, 0., 1e-10) for gauss_exp in gauss_exps):
                eigvals_satisfying_gauss.append(energy_exp)
            else:
                eigvals_not_satisfying_gauss.append(energy_exp)
        np.testing.assert_array_almost_equal(np.unique(gauged_eigvals), np.unique(eigvals_satisfying_gauss), decimal=6)
