import logging
from copy import deepcopy
from itertools import product
from typing import List, Dict, Union

import numpy as np
from scipy.sparse import csr_matrix
from sympy import re

from utils.matrix_operator_helper_functions import sparse_multi_tensor_product


class PauliDecomposition:
    def __init__(self, paulis_and_coeffs: Dict[str, float]):

        if type(paulis_and_coeffs) is not dict:
            raise TypeError("coeefs_and_paulis must be dict of (Pauli) strings and float (coefficients) ")

        self._num_qubits = len(list(paulis_and_coeffs.keys())[0])

        self._check_valid_paulis(paulis_and_coeffs)

        self._paulis_and_coeffs = dict(paulis_and_coeffs)

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def paulis_and_coeffs(self) -> Dict[str, float]:
        return self._paulis_and_coeffs

    def _check_valid_paulis(self, paulis_and_coeffs) -> None:
        list_tup_str_float_type_error = TypeError(
            "coeefs_and_paulis must be dict of (Pauli) strings and float (coefficients)")

        if not isinstance(paulis_and_coeffs, dict):
            raise list_tup_str_float_type_error

        for pauli_string, coeff in paulis_and_coeffs.items():
            if not isinstance(coeff, float) or not isinstance(pauli_string, str):
                raise list_tup_str_float_type_error

            if len(pauli_string) != self._num_qubits:
                raise ValueError("All Pauli strings must be the same length")

            if not all(s in ["I", "X", "Y", "Z"] for s in pauli_string):
                raise ValueError("Pauli string must contain only characters I,X,Y,Z")

        if len([pauli_string for pauli_string in paulis_and_coeffs.keys()]) != len(
                set([pauli_string for pauli_string in paulis_and_coeffs.keys()])):
            raise ValueError("All Pauli strings must be unique in decomposition.")

        return

    def __add__(self, other: "PauliDecomposition") -> "PauliDecomposition":

        assert self._num_qubits == other.num_qubits, ValueError(
            "Pauli decompositions must have same number of qubits")

        new_paulis_and_coeffs = deepcopy(self.paulis_and_coeffs)
        other_paulis_and_coeffs = deepcopy(other.paulis_and_coeffs)

        for pauli_string in other_paulis_and_coeffs:
            if pauli_string in new_paulis_and_coeffs:
                new_paulis_and_coeffs[pauli_string] = new_paulis_and_coeffs[pauli_string] + other_paulis_and_coeffs[
                    pauli_string]
            else:
                new_paulis_and_coeffs[pauli_string] = other_paulis_and_coeffs[pauli_string]

        return PauliDecomposition(new_paulis_and_coeffs)

    def __mul__(self, other: Union[float, np.float]) -> "PauliDecomposition":

        assert type(other) in [float, np.float64], TypeError(
            "Pauli decompositions can only be multiplied by floats.")

        new_coeffs_and_paulis = deepcopy(self.paulis_and_coeffs)

        for pauli_string in new_coeffs_and_paulis:
            new_coeffs_and_paulis[pauli_string] = new_coeffs_and_paulis[pauli_string] * other

        self._check_valid_paulis(new_coeffs_and_paulis)

        return PauliDecomposition(new_coeffs_and_paulis)

    def __matmul__(self, other: "PauliDecomposition") -> "PauliDecomposition":

        assert type(other) == type(self), TypeError(
            "Tensor procuct (@) only works between two PauliDecomposition objects.")

        new_coeffs_and_paulis = {}

        for pauli_string_1, coeff_1 in self.paulis_and_coeffs.items():
            for pauli_string_2, coeff_2 in other.paulis_and_coeffs.items():
                new_pauli_string = pauli_string_1 + pauli_string_2
                new_coeff = coeff_1 * coeff_2
                new_coeffs_and_paulis[new_pauli_string] = new_coeff

        return PauliDecomposition(new_coeffs_and_paulis)

    @staticmethod
    def generate_from_matrix(matrix: Union[csr_matrix, np.ndarray]) -> "PauliDecomposition":
        if type(matrix) == csr_matrix:
            paulis_and_coeffs = get_paulis_from_matrix(matrix)
        elif type(matrix) == np.ndarray:
            paulis_and_coeffs = get_paulis_from_matrix(csr_matrix(matrix))
        else:
            raise TypeError("matrix must be scipy sparse array or numpy ndarray.")

        return PauliDecomposition(paulis_and_coeffs)


def get_paulis_from_matrix(matrix: csr_matrix) -> Dict[str, float]:
    """
    Given a matrix M, outputs all pauli strings (σ_1 ⊗ σ_2 ⊗ σ_3 ⊗ ...) ∈ {"I","X","Y","Z"}^n with non-zero
    coefficient as well as corresponding coefficient for each one.
    """
    num_qubits = int(np.log2(matrix.shape[0]))

    all_possible_pauli_strings = [''.join(s) for s in product(["I", "X", "Y", "Z"], repeat=num_qubits)]

    paulis_and_coeffs = {}

    for i, pauli_string in enumerate(all_possible_pauli_strings):
        coeff = calculate_coefficient(matrix, pauli_string)
        if coeff != 0:
            if abs(np.imag(coeff)) > 1e-15:
                logging.warning(f'complex coeff calculated, value = {coeff}')

            paulis_and_coeffs[pauli_string] = float(np.real(coeff))

    return paulis_and_coeffs


def calculate_coefficient(matrix: csr_matrix, pauli_string: str) -> float:
    """
    Given a matrix M and pauli string (σ_1 ⊗ σ_2 ⊗ σ_3 ⊗ ...) ∈ {"I","X","Y","Z"}^n calculates the corresponding
    coefficient by symbolically calculating the trace Tr[M (σ_1 ⊗ σ_2 ⊗ σ_3 ⊗ ...)].
    """
    if len(pauli_string) != int(np.log2(matrix.shape[0])):
        raise ValueError(
            f"matrix should be size 2^n x 2^n where n = pauli_string length = {int(np.log2(matrix.shape[0]))}")

    pauli = {"I": csr_matrix([[.5, 0], [0, .5]]),
             "X": csr_matrix([[0, .5], [.5, 0]]),
             "Y": csr_matrix([[0, -.5j], [.5j, 0]]),
             "Z": csr_matrix([[.5, 0], [0, -.5]])}

    basis_state = sparse_multi_tensor_product([pauli[s] for s in pauli_string])

    coeff = re((matrix * basis_state).trace())

    if abs(coeff) < 1e-15:
        coeff = 0.0

    return coeff


def multi_pauli_tensor_product(pauli_decomps: List[PauliDecomposition]) -> PauliDecomposition:
    assert type(pauli_decomps) == list, TypeError("pauli_decomps should be list of PauliDecomposition objects.")
    prod = pauli_decomps[0]

    for pauli_decomp in pauli_decomps[1:]:
        prod = prod @ pauli_decomp

    return prod


def pauli_tensor_product_power(pauli_decomp: PauliDecomposition, exponent: int) -> PauliDecomposition:
    assert type(exponent) == int, TypeError("exponent should be int.")

    if exponent == 0:
        return PauliDecomposition({"": 1.})

    prod = deepcopy(pauli_decomp)
    for _ in range(1, exponent):
        prod = prod @ deepcopy(pauli_decomp)

    return prod
