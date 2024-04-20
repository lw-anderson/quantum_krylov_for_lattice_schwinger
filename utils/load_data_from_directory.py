import os

import numpy as np

from costing.shots.shot_estimates_utils import calc_h_rescale_factor, calc_chebyshev_with_array


def load_overlap_data(directory):
    overlaps = []
    for file in os.listdir(directory):
        overlap_data = np.load(os.path.join(directory, file), allow_pickle=True)
        if file[:8] == "overlaps":
            overlaps.append(overlap_data)
    return sorted(overlaps, key=lambda d: d[0]["n"])


def load_groundstate_data(directory):
    groundstates = []
    for file in os.listdir(directory):
        overlap_data = np.load(os.path.join(directory, file), allow_pickle=True)
        if file[:11] == "groundstate":
            groundstates.append(overlap_data)
    return sorted(groundstates, key=lambda d: d[0]["n"])


def load_stdev_over_exp(directory):
    all_overlap_data = []

    for file in os.listdir(directory):
        overlap_data = np.load(os.path.join(directory, file), allow_pickle=True)
        if file[:8] == "overlaps":
            all_overlap_data.append(overlap_data)

    all_overlap_data = sorted(all_overlap_data, key=lambda d: d[0]["n"])

    mu = all_overlap_data[0][0]["mu"]
    x = all_overlap_data[0][0]["x"]

    all_stdev_over_exp_data = {}
    all_stdev_data = {}

    for overlap_data in all_overlap_data:

        (info_dict, overlap_orders, overlaps_vals) = overlap_data

        n = info_dict["n"]

        stdev_over_exps = []
        stdevs = []
        for k in range(len(overlaps_vals)):
            if k == 0:
                stdev_over_exps.append(0.)
            else:
                try:
                    rescale_factor = calc_h_rescale_factor(n, int(np.ceil(np.log2(n))), mu, x)
                    exp_cheb_val, exp_cheb_squared_val = calc_chebyshev_with_array(k, overlaps_vals, rescale_factor)
                    var = exp_cheb_squared_val - exp_cheb_val ** 2

                    stdev_over_exp = np.sqrt(var) / abs(exp_cheb_val)

                    if type(stdev_over_exp) not in [float, np.float64]:
                        breakpoint()
                    stdevs.append(np.sqrt(var))
                    stdev_over_exps.append(stdev_over_exp)
                except:
                    break
        all_stdev_over_exp_data[n] = stdev_over_exps
        all_stdev_data[n] = stdevs
    return all_stdev_data, all_stdev_over_exp_data
