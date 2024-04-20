import os
import sys
from datetime import datetime

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

sys.path.append("/data/phys-tnt-library/kebl6296/qse-project/")

from states.create_initial_states import create_initial_state_and_unpert_ham

from operators.hamiltonian_factory import HamiltonianFactory

from experiments.subspace_expansion.generate_overlaps import OverlapsParser

parser = OverlapsParser()


def main():
    args = parser.parse_args()

    print(args)
    n = args.n
    d = args.d
    gamma = args.gamma
    anharmonicity = args.anharmonicity

    couplings = np.zeros((n, n))
    np.fill_diagonal(couplings[1:], [gamma, ] * (n - 1))
    np.fill_diagonal(couplings[:, 1:], [gamma, ] * (n - 1))

    anharmonicities = np.array([anharmonicity, ] * n)

    ham = HamiltonianFactory(couplings=couplings, anharmonicities=anharmonicities, **vars(args)).get()

    eigvals, eigvecs = eigsh(ham.hamiltonian_sp_matrix, k=2, which="SA")

    output_prefix = os.getcwd() + f"/experiments/results/groundstate_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}_"
    if args.type == "oscillators":
        output_suffix = f" n={n}, d={d}, gamma={gamma}, anharm={anharmonicity}"
    elif args.type == "phi4-model":
        output_suffix = f" n={n}, d={d}, a={args.a}, m={args.m}, lam={args.lam}"
    elif args.type == "schwinger-model":
        output_suffix = f" n={n}, d={d}, mu={args.mu}, x={args.x}, k={args.k}"
    elif args.type == "schwinger-model-gauged":
        output_suffix = f"no field, n={n}, d={d}, mu={args.mu}, x={args.x}, k={args.k}"
    else:
        raise ValueError("Unrecognised Hamiltonian type.")

    np.save(output_prefix + output_suffix, [vars(args), eigvals, eigvecs, None],
            allow_pickle=True)

    initial_state, _ = create_initial_state_and_unpert_ham(args.initial_state, couplings=couplings,
                                                           anharmonicities=anharmonicities, **vars(args))

    gs_eigvec = eigvecs[:, 0]

    initial_state_overlap = (gs_eigvec.conj().T @ initial_state.ket)[0]

    np.save(output_prefix + output_suffix, [vars(args), eigvals, eigvecs.flatten(), initial_state_overlap],
            allow_pickle=True)

    if args.type == "schwinger-model":
        sp_matrix_terms = ham.ham_terms_sp_matrices

        field_exp_values = []
        field_exp_sites = []
        ferm_exp_values = []
        ferm_exp_sites = []
        for key in sp_matrix_terms:
            if key[:4] == "H_E_":
                inner_prod = csr_matrix(gs_eigvec.T.conj()) * sp_matrix_terms[key] * csr_matrix(gs_eigvec)
                field_exp_values.append(inner_prod[0, 0])
                field_exp_sites.append(int(key[4:]))

            if key[:4] == "H_M_":
                site = int(key[4:])
                inner_prod = csr_matrix(gs_eigvec.T.conj()) * sp_matrix_terms[key] * csr_matrix(gs_eigvec)
                ferm_exp_values.append(inner_prod[0, 0] / (args.k + (-1) ** site * args.mu))
                ferm_exp_sites.append(site)

        gauss_operators = ham.gauss_operators

        gauss_op_exp_values = []
        gauss_op_sites = []
        for key in gauss_operators:
            site = int(key[2:])
            inner_prod = csr_matrix(gs_eigvec.T.conj()) * gauss_operators[key] * csr_matrix(gs_eigvec)
            gauss_op_exp_values.append(inner_prod[0, 0])
            gauss_op_sites.append(site)

        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        # ax[0].plot(ferm_exp_sites, ferm_exp_values, "bo", label="Ferm. lattice site occ.")
        # ax[0].plot(np.array(field_exp_sites) + 0.5, field_exp_values, "ro", label="Gauge field value")
        # ax[0].set_yscale("log")
        # ax[0].legend()
        #
        # ax[1].plot(gauss_op_sites, gauss_op_exp_values, "go", label="Gauss operators expectation")
        # ax[1].legend()
        #
        # fig.show()


if __name__ == '__main__':
    main()
