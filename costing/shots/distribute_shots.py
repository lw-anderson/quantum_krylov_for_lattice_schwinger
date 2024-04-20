from typing import List, Union

import numpy as np


def calc_variances(num_expecation_values: int, h_exps: List[float]) -> List[float]:
    # for D dim krylov basis. Max power H in basis is k_max = D-1. Max exp power needed for var measurement
    # is 2k_max + 2 = 2D
    assert len(h_exps) >= 2 * num_expecation_values, "Not enough expectation values supplied"
    variances = []
    for k in range(num_expecation_values):
        var_k = h_exps[2 * k] - h_exps[k] ** 2
        variances.append(var_k)
    return variances


def shot_distribution(max_calls: Union[int, float], num_expecation_values: int, h_exps: List[float],
                      exact_variances: List[float]) \
        -> List[Union[float, int]]:
    norm_factor = sum([2 * k * exact_variances[k] / abs(h_exps[k]) for k in range(1, num_expecation_values)])
    shots = [max_calls * exact_variances[k] / (abs(h_exps[k]) * norm_factor) for k in range(num_expecation_values)]
    return shots


def calc_measurement_variances(max_calls: Union[int, float], num_expecation_values: int,
                               h_exps: Union[List[float], np.ndarray]) \
        -> List[float]:
    exact_variances = calc_variances(num_expecation_values, h_exps)
    shots = shot_distribution(max_calls, num_expecation_values, h_exps, exact_variances)
    measurement_variances = [0. if k == 0 else exact_variances[k] / shots[k] for k in range(num_expecation_values)]
    return measurement_variances
