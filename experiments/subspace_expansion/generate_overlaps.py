import argparse
import os
import sys
import time
from copy import deepcopy
from datetime import datetime

import numpy as np

sys.path.append("/data/phys-tnt-library/kebl6296/qse-project/")

from operators.hamiltonian_factory import HamiltonianFactory

from states.create_initial_states import create_initial_state_and_unpert_ham


class OverlapsParser(argparse.ArgumentParser):

    def __init__(self):
        super().__init__(description='Execute optimisation of chosen cost function.')

        self.add_argument('--n', default=6, type=int)
        self.add_argument('-d', default=4, type=int)
        self.add_argument('--type', type=str, default="schwinger-model-gauged",
                          choices=["oscillators", "phi4-model", "schwinger-model", "schwinger-model-gauged"])
        self.add_argument('--max_num_basis_states', default=100, type=int)
        self.add_argument('--initial_state', type=str, default='uncoupled')
        self.add_argument('--low_mem', type=bool, default=True)
        self.add_argument('--gamma', default=0.1, type=float)
        self.add_argument('--anharmonicity', type=float, default=0.1)
        self.add_argument('--dims', type=int, default=2, choices=[1, 2], help="Spatial dimentions for phi^4 model.")
        self.add_argument('--a', type=float, default=0.1, help="Lattice spacing for phi^4 model.")
        self.add_argument('--m', type=float, default=0.2, help="Mass for phi^4 model.")
        self.add_argument('--lam', type=float, default=0.1, help="Interaction strength (lambda) for phi^4 model.")
        self.add_argument('--mu', type=float, default=1.5, help="Mass for Schwinger model.")
        self.add_argument('--x', type=float, default=0.5, help="Coupling for Schwinger model.")
        self.add_argument('--k', type=float, default=0., help="Chemical potential for Schwinger model.")


parser = OverlapsParser()


def calculate_overlaps(args):
    print(args)
    n = args.n
    d = args.d
    gamma = args.gamma
    max_num_basis_state = args.max_num_basis_states
    anharmonicity = args.anharmonicity
    initial_state_type = args.initial_state

    couplings = np.zeros((n, n))
    np.fill_diagonal(couplings[1:], [gamma, ] * (n - 1))
    np.fill_diagonal(couplings[:, 1:], [gamma, ] * (n - 1))

    anharmonicities = np.array([anharmonicity, ] * n)

    ham = HamiltonianFactory(couplings=couplings, anharmonicities=anharmonicities, **vars(args)).get()

    initial_state, _ = create_initial_state_and_unpert_ham(initial_state_type,
                                                           couplings=couplings, anharmonicities=anharmonicities,
                                                           **vars(args))

    output_prefix = os.getcwd() + f"/experiments/results/overlaps_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}_"
    if args.type == "oscillators":
        output_suffix = f" n={n}, d={d}, gamma={gamma}, anharm={anharmonicity}, {initial_state_type}"
    elif args.type == "phi4-model":
        output_suffix = f" n={n}, d={d}, a={args.a}, m={args.m}, lam={args.lam}, {initial_state_type}"
    elif args.type == "schwinger-model":
        output_suffix = f" n={n}, d={d}, mu={args.mu}, x={args.x}, k={args.k}, {initial_state_type}"
    elif args.type == "schwinger-model-gauged":
        output_suffix = f"no field, n={n}, d={d}, mu={args.mu}, x={args.x}, k={args.k}, {initial_state_type}"
    else:
        raise ValueError("Unrecognised Hamiltonian type.")

    current_state = deepcopy(initial_state)
    powers = [0, ]
    overlaps = [1., ]

    t0 = time.time()

    for power in range(1, max_num_basis_state + 1):
        if args.low_mem:
            current_state = current_state.low_mem_ham_rmul(ham)
        else:
            current_state = ham * current_state
        overlap = initial_state * current_state

        print(power, overlap, time.time() - t0)

        overlaps.append(overlap)
        powers.append(power)

        np.save(output_prefix + output_suffix, [vars(args), powers, overlaps], allow_pickle=True)


def main():
    args = parser.parse_args()
    calculate_overlaps(args)


if __name__ == '__main__':
    main()
