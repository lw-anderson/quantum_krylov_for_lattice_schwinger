import multiprocessing
import os
import sys
import warnings
from datetime import datetime
from typing import List, Union, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from scipy.linalg._decomp_update import LinAlgError
from scipy.optimize import curve_fit
from scipy.stats import linregress

sys.path.append(os.getcwd())

from costing.gate_costs.cost import CostTotal
from costing.shots.distribute_shots import calc_measurement_variances
from costing.shots.kirby_shot_estimation.spectral_norm_of_random_toeplitz_matrices import \
    spec_norm
from costing.shots.shot_estimates_utils import get_S, get_M
from experiments.subspace_expansion.qse_from_file_parser import QSEFromFileParser
from pqse.pqse import run_pqse
from qse.solve_qse_gen_eigval_prob import solve_gen_eig_prob
from thresholded_qse.thresholded_qse import project_onto_subspace
from utils.load_data_from_directory import load_overlap_data, load_groundstate_data
from utils.plotting_utils import set_plot_style, figsize, colors

parser = QSEFromFileParser()
args = parser.parse_args()

warnings.filterwarnings("ignore", category=np.ComplexWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

if args.load_output:
    output_df = pd.read_csv(args.load_output)
    method_str = "_pqse_" if "_pqse_" in args.load_output else "_thqse_" if "_thqse_" in args.load_output else "_"

else:
    if args.thresholding and args.partitioning:
        raise ValueError("Cannot apply both thresholding and partitioning.")

    dir = args.directory

    total_calls_to_qubitisation_procedure = [1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15,
                                             np.inf]
    overlaps = load_overlap_data(dir)
    groundstates = load_groundstate_data(dir)


    def process(overlap_data):
        single_n_output_df = pd.DataFrame(columns=["n", "calls", "order", "threshold-order", "energy-error"])

        (info_dict, overlap_orders, overlaps_vals) = overlap_data

        n = info_dict["n"]
        m = int(np.ceil(np.log2(n)))
        mu = info_dict["mu"]
        x = info_dict["x"]
        k = info_dict["k"]

        overlaps_vals = np.array(overlaps_vals) / np.array([n ** k for k in range(len(overlaps_vals))])

        max_order = min(len(overlaps_vals) // 4 - 1, 36)

        matched_gs_result = next(filter(lambda dat: dat[0]["n"] == n, groundstates), None)

        if matched_gs_result is not None:
            _, matched_energies, _, gs_init_state_overlap = matched_gs_result

            gs_energy = matched_energies[0]
            uncoupled_gs_energy = (- mu * n / 2)

            for calls in total_calls_to_qubitisation_procedure:
                for order in range(2, max_order):
                    print(f"N={n}, calls={calls}, order={order}")

                    num_expectation_values = 2 * order + 1 if args.partitioning else 2 * order

                    rand = np.random.RandomState(args.noise_seed)

                    if np.isinf(calls):
                        sigma = 0.
                        stdevs = np.zeros(num_expectation_values)
                    else:
                        variances = calc_measurement_variances(calls, num_expectation_values, overlaps_vals)
                        stdevs = np.sqrt(np.array(variances))

                        assert len(variances) == num_expectation_values
                        sigma = np.sqrt(max(variances))

                    S = get_S(order, overlaps_vals)
                    M = get_M(order, overlaps_vals)

                    for noise_instance in range(args.number_noise_instances):

                        perturbations = [rand.normal(0., stdev) for stdev in stdevs]

                        noisy_overlaps_vals = np.array(overlaps_vals[:len(perturbations)]) + np.array(perturbations)

                        if args.partitioning:
                            pqse_out = run_pqse(order - 1, noisy_overlaps_vals, gs_energy)
                            # Recalling, H_>H/n for overlaps. Energies resulting from overlaps need to be rescaled by n.
                            fractional_energy_error = (n * pqse_out["energy"][0] - gs_energy) / (
                                    uncoupled_gs_energy - gs_energy)
                            threshold_order = None

                        elif args.thresholding:
                            noisy_S = get_S(order, noisy_overlaps_vals)
                            noisy_M = get_M(order, noisy_overlaps_vals)

                            delta_S = noisy_S - S

                            epsilon = spec_norm(delta_S)

                            th_S, th_M = project_onto_subspace(noisy_S, noisy_M, epsilon)

                            try:
                                if th_M.shape == (0, 0):
                                    fractional_energy_error = np.nan
                                else:
                                    eigvals, eigvecs = solve_gen_eig_prob(th_M, th_S)
                                    fractional_energy_error = (n * eigvals[0] - gs_energy) / (
                                            uncoupled_gs_energy - gs_energy)
                                    print(n * eigvals[0], gs_energy, uncoupled_gs_energy, fractional_energy_error)
                            except LinAlgError as e:
                                print(e)
                                fractional_energy_error = np.nan
                            except ValueError as e:
                                print(e)
                                fractional_energy_error = np.nan

                            threshold_order = th_M.shape[0]

                        else:
                            noisy_S = get_S(order, noisy_overlaps_vals)
                            noisy_M = get_M(order, noisy_overlaps_vals)

                            eigvals, eigvecs = solve_gen_eig_prob(noisy_M, noisy_S)
                            fractional_energy_error = (n * eigvals[0] - gs_energy) / (
                                    uncoupled_gs_energy - gs_energy)
                            threshold_order = None

                        single_n_output_df.loc[len(single_n_output_df)] = {"n": n, "calls": calls, "order": order,
                                                                           "threshold-order": threshold_order,
                                                                           "energy-error": fractional_energy_error}
        return single_n_output_df


    num_processes = multiprocessing.cpu_count()

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(process, overlaps)

    pool.close()
    pool.join()

    print(results)

    output_df = pd.concat(results, ignore_index=True)

    print(output_df)
    if args.partitioning:
        method_str = "_pqse_"
    elif args.thresholding:
        method_str = "_thqse_"
    else:
        method_str = "_"
    output_df.to_csv(f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}{method_str}outputs.csv")

nan_fractions = output_df.groupby(['n', 'calls', 'order'])['energy-error'].apply(
    lambda x: x.isna().mean()).reset_index()
nan_fractions.rename(columns={'energy-error': 'nan_count'}, inplace=True)

output_df['energy-error'].fillna(np.inf, inplace=True)

quartiles_and_median = output_df.groupby(['n', 'calls', 'order'])['energy-error'].agg(
    lower_quartile=lambda x: x.abs().quantile(0.25, interpolation='linear'),
    median=lambda x: x.abs().median(),
    upper_quartile=lambda x: x.abs().quantile(0.75, interpolation='linear')
).reset_index()

averaged_outputs_df = pd.merge(quartiles_and_median, nan_fractions, on=['n', 'calls', 'order'])

unique_n_values = averaged_outputs_df['n'].unique()

set_plot_style()

nrows = 3
ncols = 4
fig_order, axs_order = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(6.75, 4.5))

for n_val, ax in zip(unique_n_values, axs_order.flatten()):
    colors_for_plot = colors
    ax.set_yscale('log')

    for idx, calls_val in enumerate(averaged_outputs_df[averaged_outputs_df['n'] == n_val]['calls'].unique()):
        subset = averaged_outputs_df[(averaged_outputs_df['n'] == n_val) & (averaged_outputs_df['calls'] == calls_val)]
        marker = "none"
        if np.isinf(calls_val):
            label = "$\infty$"
            color = "black"
        else:
            label = r"$10^{" + str(int(np.log10(calls_val))) + "}$"
            color = colors_for_plot[0]

        # Commented code plots fewer lines to avoid clutter
        if True:  # calls_val in [1e3, 1e6, 1e9, 1e12, 1e15, 1e18, np.inf]:
            ax.plot(subset['order'], subset['median'], label=label,
                    marker=marker, linestyle='-', color=color, linewidth=.5)
            if not np.isinf(calls_val):
                ax.plot(subset['order'], subset['lower_quartile'],
                        marker=marker, linestyle='-', color=color, linewidth=.5)
                ax.plot(subset['order'], subset['upper_quartile'],
                        marker=marker, linestyle='-', color=color, linewidth=.5)
                ax.fill_between(subset['order'], subset['lower_quartile'], subset['upper_quartile'],
                                color=color, alpha=0.1, edgecolor="none")

            if n_val == 26:
                ax.annotate(label, xy=(30, subset['median'].values[30]),
                            xytext=(35 * 1.3, 10 * subset['median'].values[-1]), color=color,
                            arrowprops=dict(arrowstyle='->', color=color, linewidth=0.5),
                            va="center", ha="center")
            colors_for_plot = colors_for_plot[1:] + [colors_for_plot[0]]

    ax.text(x=.5, y=.12, s=f'$N={n_val}$', ha='left', va='center', transform=ax.transAxes)

    ax.set_xlim(0, 35)
    ax.set_xticks(range(0, 36, 10))
    ax.set_ylim(1e-9, 1e-1)

    if n_val == 26:
        ax.text(x=1.05, y=1.25, s=r"\underline{$\Pi_\varphi U$ calls}", transform=ax.transAxes)

fig_order.text(0.03, 0.5, 'Fractional energy error', va='center', rotation='vertical')
fig_order.text(0.5, 0.025, 'Maximum Krylov order', ha='center')

fig_order.savefig(f"noisy_error_vs_calls{method_str}.pdf")
fig_order.show()

fig_calls = plt.figure(figsize=(6.75, 3.5))  # Adjust the figure size as needed
gs = fig_calls.add_gridspec(2, 2, width_ratios=[2, 1])
ax_calls = plt.subplot(gs[:, 0])
ax_slopes = plt.subplot(gs[0, 1])
ax_intercepts = plt.subplot(gs[1, 1])

min_median_df = averaged_outputs_df.groupby(['n', 'calls'])['median'].min().reset_index()

intercepts, intercepts_stderrs, slopes, stderrs = [], [], [], []

for (idx, n_value) in enumerate(min_median_df['n'].unique()):
    color = colors[idx % len(colors)]

    subset = min_median_df[min_median_df['n'] == n_value]

    calls = subset['calls'].values
    min_medians = subset['median'].values

    filtered_lists = [(x, y) for x, y in zip(calls, min_medians) if not np.isinf(x)]
    calls_non_inf, min_medians_non_inf = zip(*filtered_lists)
    medians_inf_calls = [y for x, y in zip(calls, min_medians) if np.isinf(x)]

    ax_calls.plot(calls, min_medians, linestyle="-", marker="o", fillstyle="none",
                  label=f"$N={n_value}$", markersize=0, linewidth=1, color=color)

    lin = linregress(np.log10(calls_non_inf), np.log10(min_medians_non_inf))

    intercepts.append(lin.intercept)
    intercepts_stderrs.append(lin.intercept_stderr)
    slopes.append(lin.slope)
    stderrs.append(lin.stderr)

    calls_for_fit = np.geomspace(1e5, 1e16, 1000)
    fitted_line = 10 ** (lin.slope * np.log10(calls_for_fit) + lin.intercept)

ax_calls.set_xscale("log")
ax_calls.set_yscale("log")
ax_calls.set_xticks([x for x in calls if x != np.inf])
leg_calls = ax_calls.legend(ncols=2)
leg_calls.set_frame_on(False)
ax_calls.set_xlabel(r"$\Pi_\varphi U$ calls")
ax_calls.set_ylabel("Fractional energy error")
ax_calls.text(s=r"\textbf{(a)}", x=0.9, y=0.9, transform=ax_calls.transAxes)

ax_slopes.plot(min_median_df['n'].unique(), slopes, linestyle="-", marker="o", markersize=4, linewidth=1, color="k",
               fillstyle="none")
ax_slopes.plot(min_median_df['n'].unique(), np.array(slopes) + np.array(stderrs), linestyle="-", linewidth=0.5,
               color="k", alpha=0.5)
ax_slopes.plot(min_median_df['n'].unique(), np.array(slopes) - np.array(stderrs), linestyle="-", linewidth=0.5,
               color="k", alpha=0.5)
ax_slopes.fill_between(min_median_df['n'].unique(), np.array(slopes) + np.array(stderrs),
                       np.array(slopes) - np.array(stderrs), color="k", alpha=0.2, edgecolor="none")
ax_intercepts.plot(min_median_df['n'].unique(), intercepts, linestyle="-", marker="o", markersize=4, linewidth=1,
                   color="k", fillstyle="none")
ax_intercepts.plot(min_median_df['n'].unique(), np.array(intercepts) + np.array(intercepts_stderrs), linestyle="-",
                   linewidth=0.5,
                   color="k", alpha=0.5)
ax_intercepts.plot(min_median_df['n'].unique(), np.array(intercepts) - np.array(intercepts_stderrs), linestyle="-",
                   linewidth=0.5,
                   color="k", alpha=0.5)
ax_intercepts.fill_between(min_median_df['n'].unique(), np.array(intercepts) + np.array(intercepts_stderrs),
                           np.array(intercepts) - np.array(intercepts_stderrs), color="k", alpha=0.2, edgecolor="none")

ax_intercepts.set_xlabel("$N$")
ax_slopes.set_ylabel("Gradient")
ax_intercepts.set_ylabel("Intercept")
ax_slopes.text(s=r"\textbf{(b)}", x=0.05, y=0.88, transform=ax_slopes.transAxes)
ax_intercepts.text(s=r"\textbf{(c)}", x=0.05, y=0.88, transform=ax_intercepts.transAxes)

n_tick_vals = range(min(min_median_df['n'].unique()), max(min_median_df['n'].unique()) + 2, 2)
ax_slopes.set_xticks(n_tick_vals)
ax_intercepts.set_xticks(n_tick_vals)
ax_slopes.set_xticklabels([])

fig_calls.tight_layout()

fig_calls.savefig(f"noisy_error_vs_calls{method_str}fits.pdf")

fig_calls.show()

fig_predict_calls, ax_predict_calls = plt.subplots(1, 1, figsize=figsize)


def fit_to_predict_calls(desired_energy_error: float,
                         n_fit: List = np.linspace(0, 30, 10000),
                         return_fig: bool = False) -> Union[Tuple[List, List, List], Tuple[List, List, List, Figure]]:
    ns = []
    predicted_calls_list = []
    uncertainty_calls_list = []

    for n, m, m_stderr, c, c_stderr in zip(min_median_df['n'].unique(), slopes, stderrs, intercepts,
                                           intercepts_stderrs):
        log_desired_energy = np.log(desired_energy_error)
        predicted_log_calls = (log_desired_energy - c) / m
        uncertainty_log_calls = ((c_stderr ** 2) + ((log_desired_energy / m) ** 2 * (m_stderr ** 2))) ** 0.5
        predicted_calls = np.exp(predicted_log_calls)
        uncertainty_calls = predicted_calls * uncertainty_log_calls * np.log(10)
        ns.append(n)
        predicted_calls_list.append(predicted_calls)
        uncertainty_calls_list.append(uncertainty_calls)

    ax_predict_calls.errorbar(ns, predicted_calls_list, uncertainty_calls_list, marker="o", markersize=3, capsize=2,
                              color="k", linestyle="none", linewidth=1)

    popt, pcov = curve_fit(lambda x, m, c: np.exp(m * x + c), ns, predicted_calls_list, sigma=uncertainty_calls_list)
    m = popt[0]
    c = popt[1]
    m_std = np.sqrt(pcov[0, 0])
    c_std = np.sqrt(pcov[1, 1])

    log_y_fit = m * n_fit + c
    y_fit = np.exp(log_y_fit)
    y_upper = np.exp((m + m_std) * n_fit + (c + c_std))
    y_lower = np.exp((m - m_std) * n_fit + (c - c_std))

    ax_predict_calls.plot(n_fit, y_fit, color="red", linewidth=.5,
                          label=r'' + f'$10^{{{np.log10(np.exp(c)):.3f}}} \cdot {np.exp(m):.3f}' + '^N$',
                          zorder=0)
    ax_predict_calls.plot(n_fit, y_lower, color="red", linewidth=.2, alpha=0.5, zorder=0)
    ax_predict_calls.plot(n_fit, y_upper, color="red", linewidth=.2, alpha=0.5, zorder=0)
    ax_predict_calls.fill_between(n_fit, y_lower, y_upper, color="red", alpha=0.2, zorder=0, edgecolor="none")

    ax_predict_calls.set_xlabel("$N$")
    ax_predict_calls.set_ylabel(
        r"$\Pi_\varphi U$ calls for $\Delta E/E = 10^{" + f"{int(np.log10(desired_energy_error)):d}" + r"}$")
    ax_predict_calls.set_yscale("log")
    ax_predict_calls.set_xticks(n_tick_vals)
    ax_predict_calls.set_xlim(0, 30)
    leg_predicted_calls = ax_predict_calls.legend()
    leg_predicted_calls.set_frame_on(False)
    fig_predict_calls.tight_layout()

    if return_fig:
        return y_fit, y_upper, y_lower, fig_predict_calls
    else:
        return y_fit, y_upper, y_lower


_, _, _, fig_predict_calls = fit_to_predict_calls(1e-4, return_fig=True)
fig_predict_calls.savefig(rf"noisy_error_vs_calls{method_str}extrapolation.pdf")
fig_predict_calls.show()

fig_total_resources, ax_total_resources = plt.subplots(1, 1, figsize=figsize)

# ax_total_resources_inset = inset_axes(ax_total_resources, width="40%", height="35%", loc="upper left")
ax_total_resources_inset = fig_total_resources.add_axes([0.31, 0.62, 0.25, 0.25])
mark_inset(ax_total_resources, ax_total_resources_inset, loc1=3, loc2=4, fc="none", ec="0.5")

ns = np.geomspace(2, 1000, 1000)
errors = [1e-2, 1e-4, 1e-6]

for idx, energy_error in enumerate(errors):
    t_gates_k_equals_1 = []
    for n in ns:
        t_gates_k_equals_1.append(CostTotal(np.ceil(np.log2(n)), n, 1, False, False, False).t_gates_inc_rz)
    calls_to_qubitisation_procedure, calls_upper, calls_lower = fit_to_predict_calls(energy_error, ns, False)
    total_t_gates = np.array(t_gates_k_equals_1) * np.array(calls_to_qubitisation_procedure)
    total_t_gates_lower = np.array(t_gates_k_equals_1) * np.array(calls_lower)
    total_t_gates_upper = np.array(t_gates_k_equals_1) * np.array(calls_upper)

    label = f"$10^{{{int(np.log10(energy_error)):d}}}$"

    for ax in [ax_total_resources, ax_total_resources_inset]:
        ax.plot(ns, total_t_gates, color=colors[idx % len(errors)], linewidth=.5, label=label)
        ax.plot(ns, total_t_gates_lower, color=colors[idx % len(errors)], linewidth=.2, alpha=0.5, zorder=0)
        ax.plot(ns, total_t_gates_upper, color=colors[idx % len(errors)], linewidth=.2, alpha=0.5, zorder=0)
        ax.fill_between(ns, total_t_gates_lower, total_t_gates_upper, color=colors[idx % len(errors)], alpha=0.2,
                        zorder=0, edgecolor="none")

ax_total_resources.set_xlabel("$N$")
ax_total_resources.set_ylabel("\# T gates")
ax_total_resources.set_xscale("log")
ax_total_resources.set_yscale("log")

ax_total_resources.set_yticks([1e20, 1e40, 1e60])

ax_total_resources_inset.set_yscale("log")
ax_total_resources_inset.set_xlim(2, 100)
ax_total_resources_inset.set_xticks([25, 50, 75])
ax_total_resources_inset.set_yticks([1e12, 1e14, 1e16, 1e18, 1e20])
ax_total_resources_inset.set_ylim(1e12, 1e20)

leg_total_resources = ax_total_resources.legend(title=r"$\underline{\Delta E/E}$", loc=(0.55, 0.55))
leg_total_resources.set_frame_on(False)
fig_total_resources.tight_layout()
fig_total_resources.savefig(f"total_t_gate_cost{method_str}.pdf")
fig_total_resources.show()
