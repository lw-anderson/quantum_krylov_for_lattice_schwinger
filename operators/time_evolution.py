import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm

from operators.coupled_quartic_oscillators_hamiltonian import CoupledQuarticOscillatorsHamiltonian


class TimeEvolution:
    def __init__(self, hamiltonian: CoupledQuarticOscillatorsHamiltonian, time_step: float):

        if type(hamiltonian) is not CoupledQuarticOscillatorsHamiltonian:
            raise TypeError("hamiltonian must be of Hamiltonian type.")

        if not isinstance(time_step, (np.floating, float)):
            raise TypeError("time_step must be float.")

        self.n = hamiltonian.n
        self.d = hamiltonian.d
        self.time_step = time_step
        self.hamiltonian = hamiltonian
        self.num_qubits = hamiltonian.num_qubits

        self._time_evolution_sp_matrix = None
        self._time_evolution_matrix = None

    @property
    def time_evolution_sp_matrix(self) -> csr_matrix:

        if self._time_evolution_sp_matrix is None:
            self._time_evolution_sp_matrix = expm(-1.j * self.time_step * self.hamiltonian.hamiltonian_sp_matrix)

        return self._time_evolution_sp_matrix

    @property
    def time_evolution_matrix(self) -> np.ndarray:

        if self._time_evolution_matrix is None:
            self._time_evolution_matrix = self.time_evolution_sp_matrix.toarray()

        return self._time_evolution_matrix
