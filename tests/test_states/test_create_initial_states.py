from unittest import TestCase

import numpy as np

from states.create_initial_states import create_initial_state_and_unpert_ham


class TestCreateInitialStates(TestCase):
    def setUp(self):
        self.oscillators_kwargs = {"type": "oscillators", "n": 2, "d": 2,
                                   "anharmonicities": np.array([.1, .1]),
                                   "couplings": np.array([[0., .2], [.2, 0.]])}

        self.phi4_model_kwargs = {"type": "phi4-model", "n": 2, "d": 2, "dims": 1, "a": .1, "m": .1, "lam": .1}

        self.schwinger_model_kwargs = {"type": "schwinger-model", "n": 2, "d": 2, "mu": .1, "x": .1, "k": .1}

    def test_create_initial_state_oscillators_gaussian(self):
        state, unpert_ham = create_initial_state_and_unpert_ham("gaussian", **self.oscillators_kwargs)

    def test_create_initial_state_oscillators_anharmonic(self):
        state, unpert_ham = create_initial_state_and_unpert_ham("anharmonic", **self.oscillators_kwargs)

    def test_create_initial_state_oscillators_zero(self):
        state, unpert_ham = create_initial_state_and_unpert_ham("zero", **self.oscillators_kwargs)

    def test_create_initial_state_oscillators_hadamard(self):
        state, unpert_ham = create_initial_state_and_unpert_ham("hadamard", **self.oscillators_kwargs)

    def test_create_initial_state_phi4_model_gaussian(self):
        state, unpert_ham = create_initial_state_and_unpert_ham("gaussian", **self.phi4_model_kwargs)

    def test_create_initial_state_phi4_model_anharmonic(self):
        create_initial_state_and_unpert_ham("anharmonic", **self.phi4_model_kwargs)

    def test_create_initial_state_phi4_model_zero(self):
        state, unpert_ham = create_initial_state_and_unpert_ham("zero", **self.phi4_model_kwargs)

    def test_create_initial_state_phi4_model_hadamard(self):
        state, unpert_ham = create_initial_state_and_unpert_ham("hadamard", **self.phi4_model_kwargs)

    def test_create_initial_state_schwinger_model_gaussian(self):
        self.assertRaises(ValueError, create_initial_state_and_unpert_ham, "gaussian", **self.schwinger_model_kwargs)

    def test_create_initial_state_schwinger_model_anharmonic(self):
        self.assertRaises(ValueError, create_initial_state_and_unpert_ham, "anharmonic", **self.schwinger_model_kwargs)

    def test_create_initial_state_schwinger_model_zero(self):
        state, unpert_ham = create_initial_state_and_unpert_ham("zero", **self.schwinger_model_kwargs)

    def test_create_initial_state_schwinger_model_hadamard(self):
        state, unpert_ham = create_initial_state_and_unpert_ham("hadamard", **self.schwinger_model_kwargs)

    def test_create_initial_state_schwinger_model_uncoupled(self):
        state, unpert_ham = create_initial_state_and_unpert_ham("uncoupled", **self.schwinger_model_kwargs)

    def test_create_initial_state_schwinger_model_gauged_gaussian(self):
        self.assertRaises(ValueError, create_initial_state_and_unpert_ham, "gaussian", **self.schwinger_model_kwargs)

    def test_create_initial_state_schwinger_model_gauged_anharmonic(self):
        self.assertRaises(ValueError, create_initial_state_and_unpert_ham, "anharmonic", **self.schwinger_model_kwargs)

    def test_create_initial_state_schwinger_model_gauged_zero(self):
        state, unpert_ham = create_initial_state_and_unpert_ham("zero", **self.schwinger_model_kwargs)

    def test_create_initial_state_schwinger_model_gauged_hadamard(self):
        state, unpert_ham = create_initial_state_and_unpert_ham("hadamard", **self.schwinger_model_kwargs)

    def test_create_initial_state_schwinger_model_gauged_uncoupled(self):
        state, unpert_ham = create_initial_state_and_unpert_ham("uncoupled", **self.schwinger_model_kwargs)
