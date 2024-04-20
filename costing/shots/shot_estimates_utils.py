import numpy as np
import sympy as sp
from sympy import chebyshevt, expand


def get_S(basis_size, overlaps) -> np.ndarray:
    if len(overlaps) < 2 * basis_size - 1:
        raise ValueError("too few overlaps provided")
    S = np.empty((basis_size, basis_size))
    for i in range(basis_size):
        for j in range(basis_size):
            S[i, j] = overlaps[i + j]
    return S


def get_M(basis_size, overlaps) -> np.ndarray:
    if len(overlaps) < 2 * basis_size:
        raise ValueError("too few overlaps provided")
    M = np.empty((basis_size, basis_size))
    for i in range(basis_size):
        for j in range(basis_size):
            M[i, j] = overlaps[i + j + 1]
    return M


# Define the symbolic variable
x = sp.Symbol('x')


def get_string_chebyshev(k):
    return str(chebyshevt(k, x))


def get_string_chebyshev_squared(k):
    return str(expand(chebyshevt(k, x) ** 2))


def replace_x_to_ar(string: str, array_name: str):
    # Initialize the result string as the input string
    result = string

    # Iterate from 100 down to 0
    for n in range(100, -1, -1):
        # Construct the pattern "x**n"
        pattern = f'x**{n}'

        # Replace occurrences of "x**n" with "ar[n]"
        result = result.replace(pattern, f'{array_name}[{n}]')

    result = result.replace("x", f"{array_name}[1]")

    return result


def calc_chebyshev_with_array(k, h_values, rescale_factor):
    h_values_rescaled = np.array(h_values) / np.array(list(rescale_factor ** n for n in range(len(h_values))))
    string_cheb = get_string_chebyshev(k)
    string_cheb_to_eval = replace_x_to_ar(string_cheb, "h_values_rescaled")
    exp_cheb_val = eval(string_cheb_to_eval)

    string_cheb_squared = get_string_chebyshev_squared(k)
    string_cheb_squared_to_eval = replace_x_to_ar(string_cheb_squared, "h_values_rescaled")
    exp_cheb_squared_val = eval(string_cheb_squared_to_eval)
    assert abs(exp_cheb_squared_val) < 1., "Something went wrong, Cheb polys should be bounded."
    return exp_cheb_val, exp_cheb_squared_val


def calc_Ti_H_Tj_with_array(i, j, rescaled_h_values):
    sympy_eqn = expand(chebyshevt(i, x) * x * chebyshevt(j, x))
    sympy_const = sympy_eqn.subs(x, 0)
    sympy_eqn_no_const = sympy_eqn - sympy_const
    str_expression_no_const = str(sympy_eqn_no_const)
    str_to_evalute_no_const = replace_x_to_ar(str_expression_no_const, "rescaled_h_values")
    exp_Ti_H_Tj = eval(str_to_evalute_no_const) + float(sympy_const)
    return exp_Ti_H_Tj


def calc_Ti_Tj_with_array(i, j, rescaled_h_values):
    sympy_eqn = expand(chebyshevt(i, x) * chebyshevt(j, x))
    sympy_const = sympy_eqn.subs(x, 0)
    sympy_eqn_no_const = sympy_eqn - sympy_const
    str_expression_no_const = str(sympy_eqn_no_const)
    str_to_evalute_no_const = replace_x_to_ar(str_expression_no_const, "rescaled_h_values")
    exp_Ti_Tj = eval(str_to_evalute_no_const) + float(sympy_const)
    return exp_Ti_Tj


def calc_h_rescale_factor(n, m, mu, x):
    tot_alpha = 0.

    # field
    for l in range(2 ** m):
        bin_l = format(l, '0' + str(m) + 'b')
        prod = 1.
        for j in range(m):
            prod *= 1 + (-1) ** int((bin_l[j]))
        tot_alpha += (n - 1) * abs(prod) * (l - 2 ** (m - 1)) ** 2

    # interactions
    tot_alpha += abs(x) * (n - 1) * 2 ** (m + 2)

    # spins
    tot_alpha += abs(mu) * n
    return tot_alpha
