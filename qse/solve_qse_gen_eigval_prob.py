from copy import copy
from typing import Dict, Callable

import numpy as np
from scipy.linalg import eigh, eig
from scipy.special import eval_chebyt
from scipy.stats import linregress
from scipy.stats._stats_mstats_common import LinregressResult

from costing.shots.kirby_shot_estimation.spectral_norm_of_random_toeplitz_matrices import spec_norm
from thresholded_qse.thresholded_qse import project_onto_subspace, \
    projector_threshold_space


def remove_sudden_jump(input_list):
    output_list = []
    for i in range(len(input_list)):
        if input_list[i] == 0.:
            break
        elif i == 0 or input_list[i] < input_list[i - 1]:
            output_list.append(input_list[i])
        elif input_list[i] > input_list[i - 1]:
            break
        else:
            break
    return output_list


def remove_smallest_and_nans(lst: list, percentage: float, remove_largest: bool = False):
    sorting_key = lambda x: (np.inf if np.isnan(x) else -np.inf) if remove_largest else (np.inf if np.isnan(x) else x)
    try:
        new_lst = sorted(lst, key=sorting_key, reverse=remove_largest)[int(len(lst) * percentage):]
        if new_lst[0] == np.nan:
            print("Fails greater than allowed rate (too many nans).")
        return new_lst
    except TypeError as e:
        breakpoint()


def append_to_list_in_dict(dic, key, val):
    new_dic = copy(dic)
    if key in new_dic.keys():
        new_dic[key].append(val)
    else:
        new_dic[key] = [val]
    return new_dic


def solve_gen_eig_prob(M: np.ndarray, S: np.ndarray) -> (np.ndarray, np.ndarray, float):
    # eigvals, eigvecs = eigh(M, S)
    eigvals, eigvecs = eigh(M, S)
    eigvals, eigvecs = (list(t) for t in zip(*sorted(zip(eigvals, eigvecs.T))))
    # sorting eigvals and vecs. eigvec coeffs are changed to rows rather than columns
    return eigvals, eigvecs


def calculate_fidelity(eigvec: np.ndarray, gs_init_state_overlap: float, matched_gs_energy: float,
                       s_matrix: np.ndarray, epsilon: float) -> float:
    # TODO: If using projector_threshold_space as defined by Kirby, this gives fidelities greater than 1
    projector = projector_threshold_space(s_matrix, epsilon)
    Ti_groundstate = gs_init_state_overlap * np.array(
        [eval_chebyt(i, matched_gs_energy) for i in range(projector.shape[0])])
    fidelity = Ti_groundstate @ projector @ eigvec
    return fidelity


def calculate_energy_from_eigvec(eigvec: np.ndarray, h_matrix: np.ndarray, s_matrix: np.ndarray) -> float:
    total = 0.
    norm_factor = 0.
    for (i, coeff_i) in enumerate(eigvec):
        for (j, coeff_j) in enumerate(eigvec):
            norm_factor += np.conj(coeff_i) * coeff_j * s_matrix[i, j]
            total += np.conj(coeff_i) * coeff_j * h_matrix[i, j]
    energy = total / norm_factor
    return energy


def perform_qse(overlap_data: dict,
                groundstates: list[Dict],
                rescale_factor: float,
                value_of_merit: str,
                noise_level: float,
                number_noise_instances,
                noise_seed=None,
                failure_tolerance: float = 0.,
                threshold_func: Callable[[int], float] = lambda order: 0.) \
        -> (list, list, list, list, LinregressResult, bool, float, Dict[int, list], Dict[int, list],
            Dict[int, list], Dict[int, list]):
    (info_dict, overlap_orders, overlaps_vals) = overlap_data

    rand = np.random.RandomState(noise_seed)

    n = info_dict["n"]
    # mu = info_dict["mu"]
    # x = info_dict["x"]

    matched_gs_result = next(filter(lambda dat: dat[0]["n"] == n, groundstates), None)
    if matched_gs_result is not None:
        _, matched_energies, _, gs_init_state_overlap = matched_gs_result

        matched_gs_energy = matched_energies[0] / rescale_factor

        qse_min_energy_out = {}
        qse_fidelities_out = {}
        krylov_basis_sizes = {}
        spec_s_vals = {}
        kirby_mu_vals = {}
        all_eigenvals = {}
        all_s_eigvals = {}

        for noise_instance_no in range(number_noise_instances):

            rescaled_overlaps_vals = np.array([exp / rescale_factor ** i for i, exp in enumerate(overlaps_vals)])

            overlap_vals_vars = [rescaled_overlaps_vals[2 * k] - rescaled_overlaps_vals[k] ** 2 for k in
                                 range(len(rescaled_overlaps_vals) // 2)]

            noisy_overlaps_vals = (rescaled_overlaps_vals[:len(overlap_vals_vars)]
                                   + rand.normal(0, noise_level, len(overlap_vals_vars)) * np.sqrt(overlap_vals_vars))

            max_order = len(noisy_overlaps_vals) // 2 - 1
            big_M = np.empty((max_order, max_order))
            big_S = np.empty((max_order, max_order))
            for i in range(max_order):
                for j in range(max_order):
                    big_M[i, j] = noisy_overlaps_vals[i + j + 1]
                    big_S[i, j] = noisy_overlaps_vals[i + j]

            for order in range(1, max_order):
                red_M = big_M[:order, :order]
                red_S = big_S[:order, :order]

                try:
                    threshold_value = threshold_func(order)
                    red_thr_S, red_thr_M = project_onto_subspace(red_S, red_M, threshold_value)

                    eigvals, eigvecs, gs_energy = solve_gen_eig_prob(red_thr_M, red_thr_S)
                    spec_norm_s = spec_norm(red_thr_S)
                    kirby_mu = max(np.abs(eigvals))

                    fidelities = []
                    for eigvec in eigvecs:
                        fidelity = calculate_fidelity(eigvec, gs_init_state_overlap, matched_gs_energy, red_S,
                                                      threshold_value)
                        fidelities.append(fidelity)

                    s_eigvals, _ = np.linalg.eigh(red_thr_S)


                except Exception as e:
                    print(f"n={n}, krylov_dim={order}:", e)
                    gs_energy = np.nan
                    fidelity = np.nan
                    spec_norm_s = np.nan
                    kirby_mu = np.nan
                    eigvals = [np.nan]
                    fidelities = [np.nan]
                    s_eigvals = [np.nan]

                qse_min_energy_out = append_to_list_in_dict(qse_min_energy_out, order, gs_energy)
                qse_fidelities_out = append_to_list_in_dict(qse_fidelities_out, order, abs(fidelities[0]))
                krylov_basis_sizes = append_to_list_in_dict(krylov_basis_sizes, order, order)
                spec_s_vals = append_to_list_in_dict(spec_s_vals, order, spec_norm_s)
                kirby_mu_vals = append_to_list_in_dict(kirby_mu_vals, order, kirby_mu)
                all_eigenvals = append_to_list_in_dict(all_eigenvals, order, eigvals)
                all_s_eigvals = append_to_list_in_dict(all_s_eigvals, order, s_eigvals)

            if noise_instance_no % 5000 == 0:
                print(f"QSE done for n={n}, noise instance {noise_instance_no}")

        for key, value in qse_min_energy_out.items():
            assert len(value) == number_noise_instances
            qse_min_energy_out[key] = remove_smallest_and_nans(value, failure_tolerance, remove_largest=True)
        for key, value in qse_fidelities_out.items():
            assert len(value) == number_noise_instances
            failure_rate = np.isnan(qse_fidelities_out[key]).sum() / len(qse_fidelities_out[key])
            print(f"QSE failure rate for n={n}, k={key} is {failure_rate:.3g}")
            qse_fidelities_out[key] = remove_smallest_and_nans(value, failure_tolerance, remove_largest=True)

        qse_min_energy_av = [np.mean(value) for key, value in qse_min_energy_out.items()]
        qse_min_energy_stdev = [np.std(value) for key, value in qse_min_energy_out.items()]
        qse_fidelities_av = [np.mean(value) for key, value in qse_fidelities_out.items()]
        qse_fidelities_stdev = [np.std(value) for key, value in qse_fidelities_out.items()]
        spec_s_vals_av = [np.mean(value) for key, value in spec_s_vals.items()]
        kirby_mu_vals_av = [np.mean(value) for key, value in kirby_mu_vals.items()]
        all_eigenvals_av = [np.mean(values, 0) for key, values in all_eigenvals.items()]

        fractional_energy_error = abs(np.array(qse_min_energy_av) - matched_gs_energy) / abs(matched_gs_energy)

        try:
            infidelities = 1 - np.array(qse_fidelities_av)

            def round_negative_to_zero(arr, threshold):
                return np.where((arr < 0) & (arr > -threshold), 0, arr)

            infidelities = round_negative_to_zero(infidelities, 1e-10)
        except:
            print(f"No infidelities data for n={n}")
            infidelities = [None, ] * len(qse_fidelities_av)

        if value_of_merit == "energy":
            error_of_interest_av = fractional_energy_error
        elif value_of_merit == "fidelity":
            error_of_interest_av = infidelities
        else:
            raise ValueError("invalid name for value_of_merit")

        error_of_interest_reduced = remove_sudden_jump(error_of_interest_av)
        krylov_basis_sizes = [key for key in krylov_basis_sizes]
        krylov_basis_sizes_reduced = krylov_basis_sizes[:len(error_of_interest_reduced)]

        try:
            lin_fit = linregress(krylov_basis_sizes_reduced,
                                 np.log(error_of_interest_reduced))
        except ValueError as e:
            print(f"n={n}, krylov order={order}: Fit failed")
            lin_fit = None

        success = True

    else:
        (krylov_basis_sizes_reduced, error_of_interest_reduced, qse_min_energy_av, qse_fidelities_av, lin_fit,
         gs_init_state_overlap, spec_s_vals_av, kirby_mu_vals_av, all_eigenvals_av) \
            = (None, None, None, None, None, None, None, None, None)

        matched_gs_energy = np.nan
        success = False
        print(f"No matched GS data for n={n}")

    return (
        krylov_basis_sizes_reduced, error_of_interest_reduced, qse_min_energy_av, qse_fidelities_av, lin_fit, success,
        matched_gs_energy, gs_init_state_overlap, spec_s_vals_av, kirby_mu_vals_av, all_eigenvals_av)
