from typing import List

import numpy as np
from scipy.optimize import minimize
from scipy.stats import linregress
from scipy.stats._stats_mstats_common import LinregressResult

from costing.shots.kirby_shot_estimation.spectral_norm_of_random_toeplitz_matrices import \
    spec_norm
from costing.shots.shot_estimates_utils import calc_h_rescale_factor


def spectral_norms_of_s(maximum_size_s: np.ndarray) -> List[float]:
    spectral_norms = []
    for n in range(1, maximum_size_s.shape[0] + 1):
        s = maximum_size_s[:n, :n]
        spectral_norms.append(spec_norm(s))
    return spectral_norms


def spectral_norms_of_rescaled_s_from_overlap(overlap_data: List) -> (List[float], LinregressResult):
    (info_dict, overlap_orders, overlaps_vals) = overlap_data
    n = info_dict["n"]
    m = int(np.ceil(np.log2(n)))
    mu = info_dict["mu"]
    k = info_dict["k"]

    rescale_factor = calc_h_rescale_factor(n, m, mu, k)

    max_order = max(overlap_orders)

    big_S = np.empty((max_order // 2, max_order // 2))
    for i in range(max_order // 2):
        for j in range(max_order // 2):
            big_S[i, j] = overlaps_vals[i + j] / rescale_factor ** (i + j)

    spec_norms = spectral_norms_of_s(big_S)

    lin_fit = linregress(np.arange(1, len(spec_norms) + 1), np.log(spec_norms))

    return spec_norms, lin_fit


def calc_eta(energy_error: float, krylov_order: int, mu: float = 1., spec_norm_of_s: float = 1.,
             delta_prime: float = 1., rho: float = 1) -> float:
    eta = (energy_error ** (5 / 4)) / (3 * (2 + mu) * krylov_order ** 4
                                       * (1 + 1 / rho)
                                       * spec_norm_of_s ** (1 / 4)
                                       * delta_prime
                                       * krylov_order ** 4)
    return eta


def calc_epsilon_lem_b1(krylov_order: int, eta: float, mu: float, rho: float, spec_norm_of_s: float,
                        delta_prime: float) -> float:
    epsilon = ((krylov_order ** 4 * 3 * (2 + mu) * (1 + 1 / rho) * spec_norm_of_s ** (1 / 4) * eta / delta_prime)
               ** (4 / 5))
    return epsilon


def calc_epsilon_thrm_2(thr_error: float, krylov_order: int, gamma_0: float):
    """
    solving
        4D(E_0' - E_0)ε^2 + [D-4(E_0' - E_0)√D]ε + [1+|γ_0|(E_0' - E_0)] ≤ 0,
    where (E_0' - E_0) is error due to thresholding, D is Krylov basis size and γ_0 is overlap between Krylov output
    state and true ground state. This equation is derived by simplifying inequality (91) in Kirby et al. Theorem 2.
    Use quadratic equation aε^2+bε+c=0.
    """

    a = 4 * krylov_order * thr_error
    b = -4 * krylov_order * 0. - 4 * gamma_0 * np.sqrt(krylov_order) * thr_error
    c = gamma_0 ** 2 * thr_error - 1

    eps_plus = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    eps_minus = (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)

    return eps_minus, eps_plus


def calc_eta_lem_b1(epsilon: float, rho: float, delta: float, krylov_order: int, mu: float = 1., spec_norm_s: float = 1,
                    alpha: float = 0.25):
    """
    calculating eta from
        η ≤ Δ'ε^(1+α) / D^4 3(2+μ)(1+1/ρ)||S||^α
    found by calculating prefactors and rearranging eqn. (89) in Kirby et al. Lemma 1B.1.
    """
    return delta * epsilon ** (1 + alpha) / (krylov_order ** 4 * 3 * (2 + mu) * (1 + 1 / rho) * spec_norm_s ** alpha)


def calc_rho(eigenvalues: List[float], eps: float):
    eigvals_greater_than_eps = [eigval for eigval in eigenvalues if eigval > eps]

    if eigvals_greater_than_eps:
        return (min(eigvals_greater_than_eps) / eps) - 1
    else:
        return None


def calc_energy_error_bound(delta: float, epsilon: float, krylov_order: int, s_eigvals: List[float],
                            initial_state_overlap: float) -> float:
    where_s_eigval_less_than_epslion = [i for i, val in enumerate(s_eigvals) if val < epsilon]
    eps_tot = sum(s_eigvals[i] for i in where_s_eigval_less_than_epslion)

    gamma_0 = initial_state_overlap

    k = krylov_order - 1
    # print(f"1- gamma square = {1 - gamma ** 2}")
    if abs(gamma_0) < np.sqrt(2 * krylov_order) * epsilon:
        # raise Exception("bound only applies when |gamma_0| > sqrt(2D) epsilon")
        print(gamma_0, krylov_order, epsilon)
        return np.inf

    # TODO: Currently setting to lower bound of gamma_0. Should really calculate this from the overlaps of all
    #  Hamiltonian eigenvectors.
    gamma = gamma_0

    assert gamma ** 2 < 1 + 4 * eps_tot, "something went wrong."

    numerator = np.sqrt(delta) * eps_tot + (1 - gamma_0 ** 2 + 4 * eps_tot) * (1 + delta / 2) ** (-2 * np.floor(k / 2))
    denominator = (abs(gamma_0) - 2 * np.sqrt((k + 1) * epsilon)) ** 2

    threshold_error_upper_bound = delta + 8 * numerator / denominator

    assert threshold_error_upper_bound >= 0., "Something went wrong."

    return threshold_error_upper_bound


def find_optimal_energy_error_lower_bound(epsilon: float, krylov_order: int, energy_eigvals: List[float],
                                          s_eigvals: List[float], initial_state_overlap: float) -> float:
    result = minimize(calc_energy_error_bound, x0=0.5,
                      args=(epsilon, krylov_order, s_eigvals, initial_state_overlap),
                      bounds=[(0., 1.)])

    return result.fun
