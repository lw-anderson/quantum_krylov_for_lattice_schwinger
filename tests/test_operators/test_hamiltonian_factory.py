from unittest import TestCase

import numpy as np

from operators.coupled_quartic_oscillators_hamiltonian import CoupledQuarticOscillatorsHamiltonian
from operators.hamiltonian_factory import HamiltonianFactory
from operators.phi4_model_hamiltonian import Phi4ModelHamiltonian
from operators.schwinger_gauged_hamiltonian import SchwingerGaugedHamiltonian
from operators.schwinger_hamiltonian import SchwingerHamiltonian


class TestHamiltonianFactory(TestCase):
    def test_get_coupled_quartic_oscillators_hamiltonian(self):
        fact = HamiltonianFactory(type="oscillators", n=2, d=4, couplings=np.array([[0, 0.1], [0.1, 0]]),
                                  anharmonicities=np.array([0.1, 0.2]))
        ham = fact.get()
        self.assertIsInstance(ham, CoupledQuarticOscillatorsHamiltonian)

    def test_get_phi4_model_hamiltonian(self):
        fact = HamiltonianFactory(type="phi4-model", dims=1, n=2, d=4, a=0.1, m=0.2, lam=0.1)
        ham = fact.get()
        self.assertIsInstance(ham, Phi4ModelHamiltonian)

    def test_get_schwinger_model_hamiltonian(self):
        fact = HamiltonianFactory(type="schwinger-model", n=2, d=4, mu=0.3, x=0.1, k=0.2)
        ham = fact.get()
        self.assertIsInstance(ham, SchwingerHamiltonian)

    def test_get_scheinger_model_gauged_hamiltonian(self):
        fact = HamiltonianFactory(type="schwinger-model-gauged", n=2, mu=0.3, x=0.1, k=0.2)
        ham = fact.get()
        self.assertIsInstance(ham, SchwingerGaugedHamiltonian)
