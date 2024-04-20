import time
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Dict, Union, List, Tuple

import numpy as np
from numpy.linalg import LinAlgError
from scipy.linalg import eigh

from operators.coupled_quartic_oscillators_hamiltonian import CoupledQuarticOscillatorsHamiltonian
from states.state import State


class QSEBaseClass(ABC):
    def __init__(self, num_qubits):
        if type(num_qubits) is not int or num_qubits <= 0:
            raise TypeError("num_qubits must be positive int.")

        self.num_qubits = num_qubits

        self._hamiltonian = None
        self._reference_states = None
        self._basis = None

    def set_hamiltonian(self, hamiltonian: CoupledQuarticOscillatorsHamiltonian) -> None:
        if hamiltonian.num_qubits != self.num_qubits:
            raise ValueError(f"hamiltonian must have same num_qubit as QSE object (num_qubits={self.num_qubits}).")
        self._hamiltonian = hamiltonian

    @property
    def hamiltonian(self) -> CoupledQuarticOscillatorsHamiltonian:
        return self._hamiltonian

    def set_reference_states(self, states: Union[State, List[State]]) -> None:
        if type(states) is State:
            states = [states]
        elif type(states) is list:
            for state in states:
                if type(state) is not State:
                    raise TypeError("state must be State object.")
        else:
            raise TypeError("states must be single State object or list of State objects.")

        for state in states:
            if state.num_qubits != self.num_qubits:
                raise ValueError(f"state must have same num_qubits as QSE object (num_qubits={self.num_qubits}).")

        self._reference_states = states

    @property
    def reference_state(self) -> Union[State, List[State]]:
        return self._reference_states

    @property
    def basis(self):
        if self._basis is None:
            self._basis = self._create_basis()
        return self._basis

    @abstractmethod
    def _create_basis(self) -> Dict[str, State]:
        pass

    def _create_matrix_elements(self) -> Tuple[np.ndarray, np.ndarray]:
        basis_size = len(self.basis)
        list_basis = list(self.basis.values())
        H = np.empty((basis_size, basis_size))
        S = np.empty((basis_size, basis_size))
        for alpha in range(0, basis_size):
            for beta in range(alpha, basis_size):
                psi_alpha = list_basis[alpha]
                psi_beta = list_basis[beta]
                H_alpha_beta = psi_alpha * self.hamiltonian * psi_beta
                S_alpha_beta = psi_alpha * psi_beta

                H[alpha, beta] = deepcopy(H_alpha_beta)
                H[beta, alpha] = deepcopy(H_alpha_beta)
                S[alpha, beta] = deepcopy(S_alpha_beta)
                S[beta, alpha] = deepcopy(S_alpha_beta)
        return H, S

    def solve_gen_eigval_problem(self, return_on_failure: bool = False, print_status: bool = False) -> [np.ndarray,
                                                                                                        np.ndarray]:
        """
        Solves generalised eigenvalue equation of form Hc = λSc for non-orthoganal basis {|ψ_α>}, where λ is
        (generalised eigenvalue), H_α,β = <ψ_α|H|ψ_β> and S_α,β = <ψ_α|ψ_β>.
        :returns: eigvals (ndarray), eigvecs (ndarray): lists of eigenvalues and eigenvectors
        """

        t0 = time.time()

        H, S = self._create_matrix_elements()

        t1 = time.time()

        if print_status:
            print("time to create basis = ", t1 - t0)

        if return_on_failure:
            try:
                eigvals, eigvecs = eigh(H, S)
            except LinAlgError as e:
                print(e)
                return None, None
        else:
            eigvals, eigvecs = eigh(H, S)

        t2 = time.time()

        if print_status:
            print("time for generalised eigval problem = ", t2 - t1)

        eigvals, eigvecs = (list(t) for t in zip(*sorted(zip(eigvals, eigvecs.T))))

        list_basis = list(self.basis.values())
        groundstate_coeffs = eigvecs[0]
        groundstate = groundstate_coeffs[0] * list_basis[0]
        for (coeff, basis_state) in zip(groundstate_coeffs[1:], list_basis[1:]):
            groundstate += coeff * basis_state

        groundstate.normalise()

        return eigvals[0], groundstate
