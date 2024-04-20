import argparse

import numpy as np
from matplotlib import pyplot as plt

from costing.shots.shot_estimates_utils import calc_h_rescale_factor
from costing.shots.zhang_shot_estimation import calc_min_for_eta_range
from experiments.subspace_expansion.generate_and_use_overlaps.load_data_from_directory import load_stdev_over_exp, \
    load_groundstate_data, load_overlap_data

parser = argparse.ArgumentParser("loading overlaps")
parser.add_argument("--directory",
                    default="/home/lewis/git/qse-for-oscillators/experiments/results/zero_chem_pot_fixed_coupling",
                    type=str, help="Directory to load files from.")

args = parser.parse_args()
directory = args.directory

all_stdev_data, all_stdev_over_exp_data = load_stdev_over_exp(directory)
overlap_data = load_overlap_data(directory)
groundstate_data = load_groundstate_data(directory)

etas = np.geomspace(1e-14, 1e8, 100)

eps_results = {}

for n in all_stdev_data:
    vars = np.array(all_stdev_data[n]) ** 2

    matched_overlap_data = next(filter(lambda dat: dat[0]["n"] == n, overlap_data), None)

    single_n_eps_results = {}

    if matched_overlap_data is not None:
        (info_dict, overlap_orders, overlaps_vals) = matched_overlap_data
        matched_gs_result = next(filter(lambda dat: dat[0]["n"] == n, groundstate_data), None)
        rescale_factor = calc_h_rescale_factor(info_dict["n"], int(np.ceil(np.log2(n) + 1)), n, info_dict["x"])

        if matched_gs_result is not None:
            rescaled_matched_gs_energy = matched_gs_result[1] / rescale_factor
        else:
            rescaled_matched_gs_energy = None
            print(f"no matched gs energy for n={n}")

        max_order = max(overlap_orders)
        big_rescaled_M, big_rescaled_S = (np.zeros(((max_order - 1) // 2, (max_order - 1) // 2)),
                                          np.zeros(((max_order - 1) // 2, (max_order - 1) // 2)))
        for i in range((max_order - 1) // 2):
            for j in range((max_order - 1) // 2):
                big_rescaled_M[i, j] = overlaps_vals[i + j + 1] / rescale_factor ** (i + j + 1)
                big_rescaled_S[i, j] = overlaps_vals[i + j] / rescale_factor ** (i + j)

        for order in range(2, (max(overlap_orders) - 1) // 2):
            reduced_rescaled_M = big_rescaled_M[:order, :order]
            reduced_rescaled_S = big_rescaled_S[:order, :order]

            Eg_plus_eps = calc_min_for_eta_range(etas, reduced_rescaled_M, reduced_rescaled_S, max(vars[:max_order]),
                                                 max(vars[:max_order - 1]), Eg=rescaled_matched_gs_energy,
                                                 tol=1e-8, print_in_progress=False)

            if rescaled_matched_gs_energy is not None:
                eps = Eg_plus_eps - rescaled_matched_gs_energy
                # print(n, order, rescaled_matched_gs_energy)
                # print(eps)
                single_n_eps_results[order] = abs(eps / rescaled_matched_gs_energy)

        eps_results[n] = single_n_eps_results

    else:
        matched_gs_result = None
        print(f"no matched overlaps for n={n}")

for n in eps_results:
    for k in eps_results[n]:
        color = "C" + str(n)
        if k == 4:
            plt.plot(eps_results[n][k], etas, color=color, fillstyle="none",
                     label=f"n={n}")
        else:
            plt.plot(eps_results[n][k], etas, color=color, fillstyle="none")
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.xlabel("eps")
plt.ylabel("eta")
plt.xlim(1e-2, 1.)
plt.show()
