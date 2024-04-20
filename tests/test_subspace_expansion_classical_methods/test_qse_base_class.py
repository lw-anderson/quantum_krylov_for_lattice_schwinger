from typing import Dict
from unittest import TestCase

import numpy as np

from states.state import State
from subspace_expansion_classical_methods.qse_base_class import QSEBaseClass


class TestQSEBaseClass(TestCase):

    def setUp(self):
        class NonAbstractClass(QSEBaseClass):
            def _create_basis(self) -> Dict[str, State]:
                pass

        self.qse = NonAbstractClass(2)

    def test_set_basis_states_correct(self):
        state_1 = State(2, np.array([1., 0., 0., 0.]))
        state_2 = State(2, np.array([0., 1., 0., 0.]))
        self.qse.set_reference_states(state_1)
        self.qse.set_reference_states([state_1, state_2])

    def test_set_basis_states_fail(self):
        state_1 = State(2, np.array([1., 0., 0., 0.]))
        state_wrong = State(1, np.array([1., 0.]))
        self.assertRaises(TypeError, self.qse.set_reference_states, "string")
        self.assertRaises(TypeError, self.qse.set_reference_states, [state_1, "string"])
        self.assertRaises(ValueError, self.qse.set_reference_states, [state_1, state_wrong])
