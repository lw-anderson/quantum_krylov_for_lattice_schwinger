import os
import sys
import time
from copy import deepcopy
from datetime import datetime

import numpy as np
from scipy.sparse.linalg import eigsh

from states.create_initial_states import create_initial_state_and_unpert_ham

sys.path.append("/data/phys-tnt-library/kebl6296/qse-project/")

from operators.hamiltonian_factory import HamiltonianFactory

from experiments.subspace_expansion.generate_overlaps import OverlapsParser

parser = OverlapsParser()


def main():
    # CALCULATING OVERLAPS

    starttime = datetime.now().strftime("%Y-%m-%d_%H%M%S")

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

    gs_eigval, gs_eigvec = eigsh(ham.hamiltonian_sp_matrix, k=1, which="SA")

    gs_prefix = os.getcwd() + f"/experiments/results/exact_groundstate_{starttime}"

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

    np.save(gs_prefix + output_suffix, [vars(args), gs_eigval, gs_eigvec],
            allow_pickle=True)

    # PERFORM OVERLAPS FOR KRYLOV

    overlaps_prefix = os.getcwd() + f"/experiments/results/gs_init_overlaps_{starttime}_"

    initial_state, _ = create_initial_state_and_unpert_ham(args.initial_state,
                                                           couplings=couplings, anharmonicities=anharmonicities,
                                                           **vars(args))
    current_state = deepcopy(initial_state)
    powers = [0, ]
    initial_state_overlaps = [1., ]
    ground_state_overlaps = [gs_eigvec.conj().T @ initial_state.ket]

    max_num_basis_state = args.max_num_basis_states

    t0 = time.time()
    for power in range(1, max_num_basis_state + 1):
        if args.low_mem:
            current_state = current_state.low_mem_ham_rmul(ham)
        else:
            current_state = ham * current_state

        initial_state_overlap = initial_state * current_state
        ground_state_overlap = (gs_eigvec.conj().T @ current_state.ket)[0, 0]

        print(power, initial_state_overlap, ground_state_overlap, time.time() - t0)

        initial_state_overlaps.append(initial_state_overlap)
        ground_state_overlaps.append(ground_state_overlap)
        powers.append(power)

        np.save(overlaps_prefix + output_suffix, [vars(args), powers, initial_state_overlaps, ground_state_overlaps],
                allow_pickle=True)


if __name__ == '__main__':
    main()
