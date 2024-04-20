import numpy as np
from scipy.sparse import kron, csr_matrix
from scipy.sparse.linalg import eigsh

from operators.coupled_quartic_oscillators_hamiltonian import CoupledQuarticOscillatorsHamiltonian
from operators.hamiltonian_factory import HamiltonianFactory
from states.state import State

floattype = np.float64


def create_initial_state_and_unpert_ham(initial_state_type: str, **ham_kwargs) \
        -> [State, CoupledQuarticOscillatorsHamiltonian]:
    if ham_kwargs["type"] in ["oscillators", "phi4-model"] and initial_state_type == "gaussian":
        ham_kwargs["anharmonicites"] = None
        ham_kwargs["lam"] = 0.
        unperturbed_ham = HamiltonianFactory(**ham_kwargs).get()
        vec = eigsh(unperturbed_ham.hamiltonian_sp_matrix, which="SA", k=1)[1][:, 0]
    elif ham_kwargs["type"] in ["oscillators", "phi4-model"] and initial_state_type == "anharmonic":
        ham_kwargs["couplings"] = np.zeros((ham_kwargs["n"], ham_kwargs["n"]))
        ham_kwargs["a"] = np.inf
        unperturbed_ham = HamiltonianFactory(**ham_kwargs).get()
        vec = eigsh(unperturbed_ham.hamiltonian_sp_matrix, which="SA", k=1)[1][:, 0]
    elif ham_kwargs["type"] in ["oscillators", "phi4-model", "schwinger-model",
                                "schwinger-model-gauged"] and initial_state_type == "zero":
        unperturbed_ham = HamiltonianFactory(**ham_kwargs).get()
        vec = np.array([1., ] + [0., ] * (2 ** unperturbed_ham.num_qubits - 1))
    elif ham_kwargs["type"] in ["oscillators", "phi4-model", "schwinger-model",
                                "schwinger-model-gauged"] and initial_state_type == "hadamard":
        unperturbed_ham = HamiltonianFactory(**ham_kwargs).get()
        vec = np.array([1. / np.sqrt(2 ** unperturbed_ham.num_qubits), ] * (2 ** unperturbed_ham.num_qubits))
    elif ham_kwargs["type"] == "schwinger-model" and initial_state_type == "uncoupled":
        ham_kwargs["x"] = 0.
        unperturbed_ham = HamiltonianFactory(**ham_kwargs).get()
        vec = csr_matrix(np.array(1))
        for r in range(ham_kwargs["n"]):
            if r % 2 == 0:
                vec = kron(vec, csr_matrix(np.array([1, 0])), format="csr")
            else:
                vec = kron(vec, csr_matrix(np.array([0, 1])), format="csr")

            if r < ham_kwargs["n"] - 1:
                vec = kron(vec, csr_matrix(([1], [ham_kwargs["d"] // 2 - 1], [0, 1]), shape=(1, ham_kwargs["d"])),
                           format="csr")

    elif ham_kwargs["type"] == "schwinger-model-gauged" and initial_state_type == "uncoupled":
        ham_kwargs["x"] = 0.
        unperturbed_ham = HamiltonianFactory(**ham_kwargs).get()
        vec = csr_matrix(np.array(1))
        for r in range(ham_kwargs["n"]):
            if r % 2 == 0:
                vec = kron(vec, csr_matrix(np.array([1, 0])), format="csr")
            else:
                vec = kron(vec, csr_matrix(np.array([0, 1])), format="csr")

    else:
        raise ValueError("Invalid hamiltonian and initial state types.")

    return State(unperturbed_ham.num_qubits, csr_matrix(vec.transpose(), dtype=floattype)), unperturbed_ham
