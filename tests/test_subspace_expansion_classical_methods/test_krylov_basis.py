from copy import deepcopy
from unittest import TestCase

import numpy as np
from numpy.linalg import eigh as np_eigh

from operators.coupled_quartic_oscillators_hamiltonian import CoupledQuarticOscillatorsHamiltonian
from states.state import State
from subspace_expansion_classical_methods.krylov_basis import QSEKrylovBasis


class TestKrylovBasis(TestCase):
    def setUp(self):
        self.harmonic_ham = CoupledQuarticOscillatorsHamiltonian(2, 2, couplings=np.array([[0, 0.1], [0.1, 0]]),
                                                                 anharmonicities=None)
        self.state = State(2, np.array([1., .1, .1, .1] / np.linalg.norm([1., .1, .1, .1])))
        self.qse = QSEKrylovBasis(2, 3)
        self.qse.set_hamiltonian(self.harmonic_ham)
        self.qse.set_reference_states(self.state)

    def test_create_basis(self):
        basis = self.qse.basis
        self.assertEqual(3, len(basis))
        for key in basis:
            state = basis[key]
            self.assertIsInstance(state, State)

    def test_solve_gen_eigval_problem_correct_groundstate(self):
        qse_eigval, qse_eigvec = self.qse.solve_gen_eigval_problem()

        ham = self.qse.hamiltonian.hamiltonian_matrix
        direct_diag_eigvals, direct_diag_eigvecs = np_eigh(ham)

        self.assertAlmostEqual(qse_eigval, direct_diag_eigvals[0], 6)

        self.assertAlmostEqual(abs(np.dot(qse_eigvec.bra.todense(), direct_diag_eigvecs[0])[0, 0]), 1., 3)

    def test_solve_gen_eigval_problem_change_basis_size(self):
        qse_eigval, qse_eigvec = self.qse.solve_gen_eigval_problem()

        qse_small_basis = QSEKrylovBasis(2, 2)
        qse_small_basis.set_hamiltonian(self.harmonic_ham)
        qse_small_basis.set_reference_states(self.state)
        qse_small_basis_eigval, qse_small_basis_eigvec = qse_small_basis.solve_gen_eigval_problem()

        self.assertLess(qse_eigval, qse_small_basis_eigval)

    def test_solve_gen_eigval_problem_anharmonic_hamiltonian(self):
        anharmonic_ham = CoupledQuarticOscillatorsHamiltonian(2, 2, couplings=np.array([[0, 0.1], [0.1, 0]]),
                                                              anharmonicities=np.array([0.1, 0.1]))

        qse_anharmonic = QSEKrylovBasis(2, 3)
        qse_anharmonic.set_hamiltonian(anharmonic_ham)
        qse_anharmonic.set_reference_states(self.state)
        qse_anharmonic_eigval, qse_anharmonic_eigvec = qse_anharmonic.solve_gen_eigval_problem()

        anharmonic_ham = qse_anharmonic.hamiltonian.hamiltonian_matrix
        direct_diag_anharmonic_eigvals, direct_diag_anharmonic_eigvecs = np_eigh(anharmonic_ham)

        self.assertAlmostEqual(qse_anharmonic_eigval, direct_diag_anharmonic_eigvals[0], 6)

    def test_solve_gen_eigval_problem_more_reference_states(self):
        anharmonic_ham = CoupledQuarticOscillatorsHamiltonian(2, 2, couplings=np.array([[0, 0.1], [0.1, 0]]),
                                                              anharmonicities=np.array([0.1, 0.1]))

        state_1 = State(2, np.array([1., 0.1, 0.1, 0.1]))
        state_2 = State(2, np.array([0.1, 1., 1., 0.1]))

        qse = QSEKrylovBasis(2, 2)
        qse.set_hamiltonian(anharmonic_ham)

        qse_one_ref_state = deepcopy(qse)
        qse_one_ref_state.set_reference_states(state_1)

        qse_two_ref_states = deepcopy(qse)
        qse_two_ref_states.set_reference_states([state_1, state_2])

        qse_one_ref_state_eigval, qse_one_ref_state_eigvec = qse_one_ref_state.solve_gen_eigval_problem()
        qse_two_ref_states_eigval, qse_two_ref_states_eigvec = qse_two_ref_states.solve_gen_eigval_problem()

        self.assertLess(qse_two_ref_states_eigval, qse_one_ref_state_eigval)

    def test_solve_gen_eigenval_problem_low_mem_mul(self):
        anharmonic_ham = CoupledQuarticOscillatorsHamiltonian(2, 2, couplings=np.array([[0, 0.1], [0.1, 0]]),
                                                              anharmonicities=np.array([0.1, 0.1]))

        qse = QSEKrylovBasis(2, 3)
        qse.set_hamiltonian(anharmonic_ham)
        qse.set_reference_states(self.state)
        qse_eigval, qse_eigvec = qse.solve_gen_eigval_problem()

        qse_low_mem = QSEKrylovBasis(2, 3, low_mem_mul=True)
        qse_low_mem.set_hamiltonian(anharmonic_ham)
        qse_low_mem.set_reference_states(self.state)
        qse_low_mem_eigval, qse_low_mem_eigvec = qse_low_mem.solve_gen_eigval_problem()

        self.assertAlmostEqual(qse_low_mem_eigval, qse_eigval, 6)
