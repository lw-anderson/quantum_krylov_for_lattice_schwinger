import itertools

import numpy as np
from scipy.linalg import eig, eigh, LinAlgError


def run_pqse(kbo_final,
             unique_H_elements_noisy,
             exact_gs_energy):
    # kbo_final is the largest power you want to have in your Krylov basis. This does not include additional powers
    # needed for computing things like variance.

    # unique_H_elements_noisy:
    #   - all of the possible unique quantum expectation values needed
    #   - should have unique_H_elements_noisy[power] = <ref|H^power|ref>
    #   - noise should have been applied to elements
    #   - includes <state|H^0|state>  element at 0th idx, unique elements on first row and final column of H, where H
    #     is dimension d x d
    #     d = (kbo_final + 1) + 1, the extra +1 in d is for computing the variance

    # how to understand this condition:
    # with extra elements for variance computation, qse H has 2 * d - 1 unique elements
    # d = (kbo_final + 1) + 1
    # the above +1 is to take into account variance computation.
    # if we include <ref|ref> in the list of unique elements we need  (2 * d - 1) + 1 in total.
    assert len(unique_H_elements_noisy) == 2 * kbo_final + 3

    # exact_gs_energy : exact ground state energy for your problem instance.

    # this is a dictionary which will store PQSE data, feel free to adapt.
    exp_data = get_exp_data()

    kbo_current = 0

    optimal_coeffs = []
    coeff_cache = {0: 1}

    counter = 0
    best_soln = None
    curr_min_abs_var = None

    while kbo_current < kbo_final:
        trial_optimal_solns = []  # (coeffs, energy, H, S)

        max_size_basis = (kbo_final - kbo_current) + 1

        if len(optimal_coeffs) > 0:
            coeff_cache = update_coeff_cache(coeff_cache, optimal_coeffs[-1], kbo_current)

        for size_basis in range(1, max_size_basis + 1):
            # IMPORTANT: for our purposes we expect S and H to be real, regardless of noise.
            # This may not hold for other choices for QSE basis i.e. not a Krylov basis.
            H = np.zeros((size_basis, size_basis), dtype=np.float64)
            S = np.zeros((size_basis, size_basis), dtype=np.float64)

            for idx in range(size_basis):
                for idy in range(size_basis):
                    H[idx][idy] = compute_partitioned_state_power_H_cached(unique_H_elements_noisy, coeff_cache,
                                                                           idx + idy + 1)

                    S[idx][idy] = compute_partitioned_state_power_H_cached(unique_H_elements_noisy, coeff_cache,
                                                                           idx + idy)

            evals, evecs = generalised_eval_prob_solver(H, S)

            sorted_evals = np.sort(evals)
            evals_coeffs = {evals[i]: evecs[:, i] for i in range(len(evals))}
            gs_evec = evals_coeffs[sorted_evals[0]]

            if sorted_evals[0].imag < 1e-10:
                #  we can remove imaginary part of energy here as we check if it is negligible above
                trial_optimal_solns.append((gs_evec, sorted_evals[0].real, H, S))

        #  update dependent variables
        if len(trial_optimal_solns) > 0:
            new_best_soln, new_min_abs_var, all_vars = find_best_trial_soln_var_running_cached(best_soln,
                                                                                               curr_min_abs_var,
                                                                                               coeff_cache,
                                                                                               unique_H_elements_noisy,
                                                                                               kbo_current,
                                                                                               trial_optimal_solns)

            if best_soln is None or new_best_soln[1] != best_soln[1]:  # compare energy estimates
                best_soln = new_best_soln
                curr_min_abs_var = new_min_abs_var

                optimal_coeffs.append(best_soln[0])
                kbo_current = kbo_current + (len(optimal_coeffs[-1]) - 1)

                if kbo_current == kbo_final:
                    exp_data['final_iteration'].append(True)
                else:
                    exp_data['final_iteration'].append(False)
            else:  # no update, terminate
                exp_data['final_iteration'].append(True)

            exp_data['size_basis_current'].append(len(optimal_coeffs[-1]))
            exp_data['iteration_current'].append(counter)
            exp_data['kbo_current'].append(kbo_current)

            temp_coeff_cache = dict(coeff_cache)
            temp_coeff_cache = update_coeff_cache(temp_coeff_cache,
                                                  optimal_coeffs[-1],
                                                  kbo_current)

            state_norm = compute_partitioned_state_power_H_cached(unique_H_elements_noisy,
                                                                  temp_coeff_cache,
                                                                  0)

            energy = best_soln[1]

            exp_data['energy'].append(energy)
            exp_data['energy_target_H'].append(compute_partitioned_state_power_H_cached(unique_H_elements_noisy,
                                                                                        temp_coeff_cache,
                                                                                        1))
            exp_data['energy_err_rel'].append(np.abs(energy - exact_gs_energy) / np.abs(exact_gs_energy))

            # temporary fields
            exp_data['min_abs_var'].append(curr_min_abs_var)
            exp_data['vars'].append(all_vars)

            exp_data['state_norm'].append(state_norm)

            if exp_data['final_iteration'][-1]:
                break

            counter += 1
        else:
            exp_data['size_basis_current'].append(exp_data['size_basis_current'][-1]) if len(
                exp_data['size_basis_current']) > 0 else size_basis

            exp_data['iteration_current'].append(counter)
            exp_data['final_iteration'].append(True)
            exp_data['kbo_current'].append(kbo_current)

            exp_data['energy'].append(exp_data['energy'][-1])
            exp_data['energy_target_H'].append(exp_data['energy_target_H'][-1])
            exp_data['energy_err_rel'].append(exp_data['energy_err_rel'][-1])

            # temporary fields
            exp_data['min_abs_var'].append(exp_data['min_abs_var'][-1])
            exp_data['vars'].append(exp_data['vars'][-1])

            exp_data['state_norm'].append(exp_data['state_norm'][-1])

            print('\nFAILED')
            print('Could not continue iteration')
            print('FAILED\n')
            break

    return exp_data


def generalised_eval_prob_solver(H, S=None, hermitian=False):
    if hermitian:
        try:
            evals, evecs = eigh(H, b=S)
        except LinAlgError:
            print('scipy.eig eigenvalue computation did not converge.')
            quit()
    else:
        try:
            evals, evecs = eig(H, b=S, right=True)
        except LinAlgError:
            print('scipy.eig eigenvalue computation did not converge.')
            quit()

    return evals, evecs


def compute_partitioned_state_power_H_cached(unique_H_elements, coeff_cache, power):
    """
        Unlike compute_partitioned_state_power_H, which figures out how to map from current iteration inner product
        power to unique H elements inner product power, this function takes the unique H elements power as input.
    """
    # matrix element is a weighted sum of inner products and coefficients
    # the inner product Hamiltonian power changes with matrix element indices, but
    # the coefficients are the same for every matrix element at a given
    # partitioned qse iteration
    return sum(coeff_cache[_] * unique_H_elements[power + _] for _ in coeff_cache)


def update_coeff_cache(coeff_cache, new_opt_coeffs, kbo_current):
    new_coeffs_prods = {}

    for new_coeff_idxs in itertools.product(*[range(len(new_opt_coeffs)), range(len(new_opt_coeffs))]):
        coeff_idx_sum = sum(new_coeff_idxs)

        if coeff_idx_sum in new_coeffs_prods:
            new_coeffs_prods[coeff_idx_sum] += np.conj(new_opt_coeffs[new_coeff_idxs[0]]) * new_opt_coeffs[
                new_coeff_idxs[1]]
        else:
            new_coeffs_prods[coeff_idx_sum] = np.conj(new_opt_coeffs[new_coeff_idxs[0]]) * new_opt_coeffs[
                new_coeff_idxs[1]]

    return {m: compute_recombined_coeff(coeff_cache, new_coeffs_prods, m) for m in range(0, (2 * kbo_current) + 1)}


def compute_recombined_coeff(coeff_cache, new_coeffs_prods, m):
    new_coeff = 0

    for j in range(m + 1):
        if m - j in coeff_cache and j in new_coeffs_prods:
            new_coeff += new_coeffs_prods[j] * coeff_cache[m - j]

    return new_coeff


def find_best_trial_soln_var_running_cached(curr_best_soln,
                                            curr_min_abs_var,
                                            coeff_cache,
                                            unique_H_elements,
                                            kbo_current,
                                            trial_optimal_solns):
    vars = []

    for (coeffs, energy, stored_H, stored_S) in trial_optimal_solns:
        temp_coeff_cache = dict(coeff_cache)

        temp_coeff_cache = update_coeff_cache(temp_coeff_cache,
                                              coeffs,
                                              kbo_current + (len(coeffs) - 1))

        H_10 = compute_partitioned_state_power_H_cached(unique_H_elements,
                                                        temp_coeff_cache,
                                                        2)

        H_00 = compute_partitioned_state_power_H_cached(unique_H_elements,
                                                        temp_coeff_cache,
                                                        1)

        S_00 = compute_partitioned_state_power_H_cached(unique_H_elements,
                                                        temp_coeff_cache,
                                                        0)

        var = compute_var(H_10, H_00, S_00)

        vars.append(var)

        if curr_min_abs_var is None or abs(var) < curr_min_abs_var:
            curr_min_abs_var = abs(var)
            curr_best_soln = (coeffs, energy, stored_H, stored_S)

    sorted_vars = np.array(sorted(vars))

    return curr_best_soln, curr_min_abs_var, sorted_vars


def compute_var(exp_H_sq, exp_H, exp_H_0):
    return (exp_H_sq / exp_H_0) - (exp_H / exp_H_0) ** 2


def get_exp_data():
    return {
        'size_basis_current': [],
        'iteration_current': [],
        'final_iteration': [],
        'kbo_current': [],
        'energy': [],
        'energy_target_H': [],
        'energy_err_rel': [],
        'min_abs_var': [],
        'vars': [],
        'state_norm': []
    }
