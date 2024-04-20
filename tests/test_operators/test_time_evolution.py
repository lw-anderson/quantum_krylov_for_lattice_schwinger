from unittest import TestCase

import numpy as np

from operators.coupled_quartic_oscillators_hamiltonian import CoupledQuarticOscillatorsHamiltonian
from operators.time_evolution import TimeEvolution
from states.state import State


class TestTimeEvolution(TestCase):
    def setUp(self):
        self.n = 2
        self.d = 2
        self.num_qubits = 2
        self.couplings = np.array([[0., 1.], [1., 0.]])
        self.anharmonicities = None
        self.ham = CoupledQuarticOscillatorsHamiltonian(self.n, self.d, self.couplings, self.anharmonicities)
        self.time_evol = TimeEvolution(self.ham, 0.1)

    def test_init_correct(self):
        time_evol = TimeEvolution(hamiltonian=self.ham, time_step=0.1)
        self.assertEqual(time_evol.n, self.ham.n)
        self.assertEqual(time_evol.d, self.ham.d)

    def test_init_fail(self):
        self.assertRaises(TypeError, TimeEvolution, hamiltonian=np.zeros((2, 2)), time_step=0.1)
        self.assertRaises(TypeError, TimeEvolution, hamiltonian=self.ham, time_step=1)

    def test_eigenstate(self):
        zero_couplings = np.zeros((2, 2))
        uncoupled_ham = CoupledQuarticOscillatorsHamiltonian(self.n, self.d, zero_couplings, None)
        uncoupled_time_evol = TimeEvolution(uncoupled_ham, 0.1)
        groundstate = State(self.num_qubits, np.array([1., 0., 0., 0.]))

        evolved_state = uncoupled_time_evol * groundstate
        expected_phase = np.exp(-0.1j * 2)

        self.assertTrue(np.all((expected_phase * groundstate.ket == evolved_state.ket).data))

    def test_time_reversal(self):
        state = State(self.num_qubits, np.array([1., 0., 0., 0.]))
        backwards_time_evol = TimeEvolution(self.ham, -0.1)

        evolved_state = backwards_time_evol * (self.time_evol * state)

        self.assertTrue(np.all((evolved_state.ket == state.ket).data))

    def test_cyclic(self):
        state = State(self.num_qubits, np.array([1., 0., 0., 0.]))
        whole_period_time_evol = TimeEvolution(self.ham, 2 * np.pi)

        evolved_state = whole_period_time_evol * state

        self.assertTrue(np.all((evolved_state.ket == state.ket).data))
