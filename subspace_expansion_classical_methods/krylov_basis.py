from copy import deepcopy
from typing import Dict, Tuple

import numpy as np

from states.state import State
from subspace_expansion_classical_methods.qse_base_class import QSEBaseClass


class QSEKrylovBasis(QSEBaseClass):
    def __init__(self, num_qubits: int, max_power: int, low_mem_mul: float = False):
        super().__init__(num_qubits)

        if type(max_power) is not int or max_power <= 0:
            raise TypeError("max_power much be positive int.")

        self.max_power = max_power
        self.low_mem_mul = low_mem_mul

    def _create_matrix_elements(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create matrix elements of the from
            H_(α,i),(β,j) = <ψ_α|H^i.H.H^j|ψ_β>
            S_(α,i),(β,j) = <ψ_α|H^i.H^j|ψ_β>
        """
        num_ref_states = len(self.reference_state)
        basis_size = self.max_power * num_ref_states
        M = np.empty((basis_size, basis_size))
        S = np.empty((basis_size, basis_size))
        kets = self._create_hamiltonian_kets(2 * self.max_power + 1)
        for alpha in range(0, num_ref_states):
            for beta in range(0, num_ref_states):
                for i in range(0, self.max_power):
                    for j in range(i, self.max_power):
                        H_ij1_bra_alpha = kets["H^" + str(i + j + 1) + ".psi_" + str(alpha)]
                        H_ij_bra_alpha = kets["H^" + str(i + j) + ".psi_" + str(alpha)]
                        ket_beta = kets["H^0.psi_" + str(beta)]

                        M_i_alpha_j_beta = H_ij1_bra_alpha * ket_beta
                        S_i_alpha_j_beta = H_ij_bra_alpha * ket_beta

                        M[num_ref_states * alpha + i, num_ref_states * beta + j] = deepcopy(M_i_alpha_j_beta.real)
                        M[num_ref_states * beta + j, num_ref_states * alpha + i] = deepcopy(M_i_alpha_j_beta.real)
                        S[num_ref_states * alpha + i, num_ref_states * beta + j] = deepcopy(S_i_alpha_j_beta.real)
                        S[num_ref_states * beta + j, num_ref_states * alpha + i] = deepcopy(S_i_alpha_j_beta.real)

        return M, S

    def _create_hamiltonian_kets(self, max_power) -> Dict[str, State]:
        if self._hamiltonian is None or self._reference_states is None:
            raise Exception("hamiltonian or reference_state not set. Use QSEHamiltonianPowerBasis.set_hamiltonian"
                            "and QSEHamiltonianPowerBasis.set_reference_state to set these values.")

        basis = {}
        for alpha, ref_state in enumerate(self._reference_states):
            for i in range(max_power):
                state = ref_state
                for _ in range(i):
                    if self.low_mem_mul:
                        state = state.low_mem_ham_rmul(self.hamiltonian)
                    else:
                        state = self.hamiltonian * state
                basis["H^" + str(i) + ".psi_" + str(alpha)] = state

        return basis

    def _create_basis(self) -> Dict[str, State]:
        return self._create_hamiltonian_kets(self.max_power)
