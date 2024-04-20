import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

from utils.plotting_utils import set_plot_style, figsize

MIN_SIGMA = 1e-8
MAX_SIGMA = 1.0
MIN_N = 2
MAX_N = 100

set_plot_style(10)


def create_toeplitz_matrix(n: int, sigma: float) -> np.ndarray:
    """
    Create an n x n Toeplitz matrix with elements drawn from a normal distribution
    with standard deviation 'sigma'.

    Parameters:
        n (int): Size of the Toeplitz matrix (n x n).
        sigma (float): Standard deviation of the normal distribution.

    Returns:
        np.ndarray: The Toeplitz matrix.
    """
    random_values = np.random.normal(0, sigma, n)
    toeplitz_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i >= j:
                toeplitz_matrix[i, j] = random_values[i - j]
            else:
                toeplitz_matrix[i, j] = random_values[j - i]
    return toeplitz_matrix


def spec_norm(matrix: np.ndarray) -> float:
    return np.linalg.norm(matrix, ord=2)


def calculate_toeplitz_spectral_norm_prefactor(quartile: float = 0.9, plot: bool = True) -> float:
    num_combinations = 10000

    sigma_values = np.empty(num_combinations)
    n_values = np.empty(num_combinations)
    spectral_norms = np.empty(num_combinations)

    for i in range(num_combinations):
        # sigma = np.random.uniform(MIN_SIGMA, MAX_SIGMA)  # Random 'sigma' within the specified range
        sigma = np.exp(np.random.uniform(np.log(MIN_SIGMA), np.log(MAX_SIGMA)))
        n = np.random.randint(MIN_N, MAX_N + 1)  # Random 'n' within the specified range
        toeplitz_matrix = create_toeplitz_matrix(n, sigma)
        spectral_norm = spec_norm(toeplitz_matrix)

        sigma_values[i] = sigma
        n_values[i] = n
        spectral_norms[i] = spectral_norm

    x_data = sigma_values * np.sqrt(n_values * np.log(n_values))
    y_data = spectral_norms

    if quartile is None:
        c_estimate = max(y_data / x_data)
    else:
        quant_reg = sm.QuantReg(y_data, x_data)
        quantile_fit = quant_reg.fit(q=quartile, f_custom=lambda x, m: m * x)
        c_estimate = quantile_fit.params[0]

    # params, covariance = curve_fit(lambda x, m: m * x, x_data, y_data)
    # c_estimate = params[0]

    if plot:
        @np.vectorize
        def marker_size_func(n):
            return 15 * n / MAX_N + 0.5

        @np.vectorize
        def marker_color(sigma):
            return (np.log(sigma) - np.log(MIN_SIGMA)) / (np.log(MAX_SIGMA) - np.log(MIN_SIGMA))

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        ax.scatter(
            x_data, y_data,
            c=marker_color(sigma_values), cmap='viridis', alpha=0.7, label='Data',
            s=marker_size_func(n_values), edgecolor='black', linewidth=0.05
        )
        x_fit = np.linspace(min(x_data), max(x_data), 100)
        y_fit = c_estimate * x_fit
        ax.plot(x_fit, y_fit, 'r-', label=r'$' + f'{c_estimate:.3f}' + r'\cdot\sigma \sqrt{n\log n}$')
        ax.set_xlabel(r'$\sigma \sqrt{n\log n}$')
        ax.set_ylabel(r"$||\mathbf{M}||$")
        ax.set_xscale("log")
        ax.set_yscale("log")

        leg = ax.legend()
        leg.get_frame().set_facecolor('none')
        leg.set_frame_on(False)
        fig.tight_layout()
        fig.savefig("toeplitz_spectral_norm.pdf")
        fig.show()

    print(f"Estimated c: {c_estimate:.4f}")
    return c_estimate


def main():
    calculate_toeplitz_spectral_norm_prefactor(quartile=None)


if __name__ == "__main__":
    main()
