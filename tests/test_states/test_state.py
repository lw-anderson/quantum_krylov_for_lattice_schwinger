from unittest import TestCase

import numpy as np

from operators.coupled_quartic_oscillators_hamiltonian import CoupledQuarticOscillatorsHamiltonian
from operators.time_evolution import TimeEvolution
from states.state import State


class TestState(TestCase):
    def setUp(self):
        self.ham = CoupledQuarticOscillatorsHamiltonian(2, 2, couplings=np.array([[0, 1], [1, 0]]),
                                                        anharmonicities=None)

    def test_inputs(self):
        state_zero = State(2, np.array([1., 0., 0., 0.]))
        state_none = State(2, None)
        self.assertEqual(type(state_zero.ket), type(state_none.ket))
        np.testing.assert_array_equal(state_zero.ket.todense(), state_none.ket.todense())

    def test_rmul_hamiltonian(self):
        state = State(2, np.array([1, 0, 0, 1j]))

        ham_state = self.ham * state
        self.assertIsInstance(ham_state, State)
        self.assertEqual((4, 1), ham_state.ket.shape)

        state_state = state * state
        self.assertIsInstance(state_state, complex)
        self.assertEqual(state_state, 2.)

        state_ham_state = state * self.ham * state
        self.assertIsInstance(state_ham_state, complex)

        state___ham_state = state * ham_state
        self.assertEqual(state___ham_state, state_ham_state)

        groundstate = State(2, np.array([1, 0, 0, 0]))
        eigstate_ham_eigstate = groundstate * self.ham * groundstate
        self.assertEqual(eigstate_ham_eigstate, 2.)

    def test_mul_hamiltonian(self):
        state = State(2, np.array([1, 0, 0, 1j]))

        state_ham = state * self.ham
        state_ham_state = state * self.ham * state
        state_ham___state = state_ham * state

        self.assertEqual(state_ham_state, state_ham___state)

    def test_low_mem_ham_mul(self):
        state = State(2, np.array([1, 0, 0, 1j]))

        state_ham = state.low_mem_ham_rmul(self.ham)
        state_ham_state = state * self.ham * state
        state_ham___state = state * state_ham

        self.assertEqual(state_ham_state, state_ham___state)

    def test_low_mem_ham_rmul(self):
        state = State(2, np.array([1, 0, 0, 1j]))

        ham_state = state.low_mem_ham_mul(self.ham)
        state_ham_state = state * self.ham * state
        state___ham_state = ham_state * state

        self.assertEqual(state_ham_state, state___ham_state)

    def test_mul_time_evolution(self):
        unitary = TimeEvolution(self.ham, 0.1)
        state = State(2, np.array([1, 0, 0, 1j]))

        unitary_state = unitary * state
        self.assertIsInstance(unitary_state, State)
        self.assertEqual((4, 1), unitary_state.ket.shape)

        state_unitary_state = state * unitary * state
        self.assertIsInstance(state_unitary_state, complex)

        state___unitary_state = state * unitary_state
        self.assertAlmostEqual(state___unitary_state, state_unitary_state, 7)

    def test_rmul_time_evolution(self):
        unitary = TimeEvolution(self.ham, 0.1)
        state = State(2, np.array([1, 0, 0, 1j]))

        state_unitary = state * unitary
        state_unitary_state = state * unitary * state
        state_unitary___state = state_unitary * state

        self.assertEqual(state_unitary_state, state_unitary___state)

    def test_mul_rmul_shape_errors(self):
        ham = CoupledQuarticOscillatorsHamiltonian(2, 2, couplings=np.array([[0, 1], [1, 0]]), anharmonicities=None)
        state = State(1, np.array([1, 1j]))

        self.assertRaises(ValueError, state.__mul__, ham)
        self.assertRaises(ValueError, state.__rmul__, ham)
