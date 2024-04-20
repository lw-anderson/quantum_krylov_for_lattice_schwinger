import numpy as np
from scipy.optimize import minimize, bisect


def cost_function_to_minimise(vec: np.ndarray, H_plus_pert: np.ndarray, S_plus_pert: np.ndarray):
    shape = H_plus_pert.shape[0]
    assert H_plus_pert.shape == (shape, shape), "Incorrect shape."
    assert S_plus_pert.shape == (shape, shape), "Incorrect shape."
    assert vec.shape == (shape,), "Incorrect shape."

    x = np.expand_dims(vec, 1)
    xT = x.T
    return (xT @ H_plus_pert @ x)[0] / (xT @ S_plus_pert @ x)[0]


def solve_jacobian_for_vector(H_plus_pert: np.ndarray, S_plus_pert: np.ndarray, Eg_plus_eps, tol):
    jac = H_plus_pert - S_plus_pert * Eg_plus_eps

    eigenvalues, eigenvectors = np.linalg.eig(jac)

    zero_eigenvalue_indices = np.where(np.abs(eigenvalues) < tol)

    zero_eigenvectors = eigenvectors[:, zero_eigenvalue_indices].T
    zero_eigenvectors = np.squeeze(zero_eigenvectors, 1)

    return zero_eigenvectors


def calc_min_for_cost_function_given_eta(eta: float, H: np.ndarray, S: np.ndarray, C_H: float, C_S: float,
                                         Eg: float, eps: float, tol: float, method: str):
    shape = H.shape[0]
    H_plus_pert = H + 2 * eta * C_H * np.identity(shape)

    S_plus_pert = S + 2 * eta * C_S * np.identity(shape)

    if method == "solve_jac":
        if type(Eg) is not float:
            raise ValueError("Eg must be float.")

        Eg_plus_eps = Eg + eps

        zero_eigenvectors = solve_jacobian_for_vector(H_plus_pert, S_plus_pert, Eg_plus_eps, tol)

        cost_function_values = [cost_function_to_minimise(vec, H_plus_pert, S_plus_pert) for vec in zero_eigenvectors]

        if len(cost_function_values) > 0:
            return min(cost_function_values)
        else:
            return np.nan

    elif method == "minimise_cost":

        f = lambda x_red: cost_function_to_minimise(np.append(1, x_red), H_plus_pert, S_plus_pert)
        min_output = minimize(f, x0=np.zeros((shape - 1,)))
        if min_output.success:
            return min_output.fun
        else:
            print(f"Minimisation failed: eta={eta}")
            return np.nan
    else:
        raise ValueError("Invalid method type")


def binary_search_for_eta(H: np.ndarray, S: np.ndarray, C_H: float, C_S: float,
                          Eg_plus_eps: float, tol: float = 1e-8, method: str = "solve_jac"):
    f = lambda eta: calc_min_for_cost_function_given_eta(eta, H, S, C_H, C_S, Eg_plus_eps, tol, method)
    bisect_out = bisect(f, 0., 1., )
    return bisect_out


def calc_min_for_eta_range(eta_range: np.ndarray, H: np.ndarray, S: np.ndarray, C_H: float, C_S: float, Eg, tol: float,
                           print_in_progress: bool = True):
    mins = []
    for eta in eta_range:
        minimum = calc_min_for_cost_function_given_eta(eta, H, S, C_H, C_S, Eg, 0.0, tol, "minimise_cost")
        mins.append(minimum)
        if print_in_progress:
            print(eta, minimum)

    return mins

# def evaluate_hessian(vec: np.ndarray, H_plus_pert: np.ndarray, S_plus_pert: np.ndarray, Eg_plus_eps: float):
#     shape = H_plus_pert.shape[0]
#     assert H_plus_pert.shape == (shape, shape), "Incorrect shape."
#     assert S_plus_pert.shape == (shape, shape), "Incorrect shape."
#     assert vec.shape == (shape,), "Incorrect shape."
#
#     x = np.expand_dims(vec, 1)
#     xT = x.T
#
#     xT_H_x = (xT @ H_plus_pert @ x)[0]
#     xT_S_x = (xT @ S_plus_pert @ x)[0]
#     H_x_xT_S = np.outer(H_plus_pert @ x, xT @ S_plus_pert)
#     S_x_xT_H = np.outer(H_plus_pert @ x, xT @ S_plus_pert)
#     assert np.array_equal(H_x_xT_S.T, S_x_xT_H)
