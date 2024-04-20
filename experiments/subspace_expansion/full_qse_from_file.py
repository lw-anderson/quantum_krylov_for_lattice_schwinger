import multiprocessing
import os
import sys
import warnings
from typing import Union, List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pandas import Series
from scipy.linalg._decomp_update import LinAlgError
from scipy.optimize import curve_fit
from scipy.stats import linregress

sys.path.append(os.getcwd())

from costing.shots.shot_estimates_utils import get_S, get_M
from experiments.subspace_expansion.qse_from_file_parser import QSEFromFileParser
from pqse.pqse import run_pqse
from qse.solve_qse_gen_eigval_prob import solve_gen_eig_prob
from thresholded_qse.thresholded_qse import project_onto_subspace
from utils.load_data_from_directory import load_overlap_data, load_groundstate_data
from utils.plotting_utils import colors, figsize

parser = QSEFromFileParser()
args = parser.parse_args()

warnings.filterwarnings("ignore", category=np.ComplexWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

if args.thresholding and args.partitioning:
    raise ValueError("Cannot apply both thresholding and partitioning.")

if args.load_output:
    output_df = pd.read_csv(args.load_output)
    method_str = "_pqse_" if "_pqse_" in args.load_output else "_thqse_" if "_thqse_" in args.load_output else "_"

else:
    if args.thresholding and args.partitioning:
        raise ValueError("Cannot apply both thresholding and partitioning.")

    dir = args.directory

    overlaps = load_overlap_data(dir)
    groundstates = load_groundstate_data(dir)


    def process(overlap_data):
        single_n_output_df = pd.DataFrame(columns=["n", "calls", "order", "threshold-order", "energy-error",
                                                   "init-state-overlap"])
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
            exact_gs_energy = matched_energies[0]
            uncoupled_gs_energy = (- mu * n / 2)

            for order in range(1, max_order):
                print(f"N={n}, order={order}")
                S = get_S(order, overlaps_vals)
                M = get_M(order, overlaps_vals)

                if args.partitioning:
                    if order == 1:
                        pqse_out = {"energy": [overlaps_vals[1] / overlaps_vals[0]]}
                    else:
                        num_expectation_values = 2 * order + 1 if args.partitioning else 2 * order
                        pqse_out = run_pqse(order - 1, overlaps_vals[:num_expectation_values], exact_gs_energy)
                    fractional_energy_error = (n * pqse_out["energy"][0] - exact_gs_energy) / (
                            uncoupled_gs_energy - exact_gs_energy)
                    threshold_order = None

                elif args.thresholding:

                    epsilon = 0

                    th_S, th_M = project_onto_subspace(S, M, epsilon)

                    try:
                        if th_M.shape == (0, 0):
                            fractional_energy_error = np.nan
                        else:
                            eigvals, eigvecs = solve_gen_eig_prob(th_M, th_S)
                            fractional_energy_error = (n * eigvals[0] - exact_gs_energy) / (
                                    uncoupled_gs_energy - exact_gs_energy)
                    except LinAlgError as e:
                        print(e)
                        fractional_energy_error = np.nan
                    except ValueError as e:
                        print(e)
                        fractional_energy_error = np.nan

                    threshold_order = th_M.shape[0]

                else:
                    try:
                        eigvals, eigvecs, gs_energy = solve_gen_eig_prob(M, S)
                        fractional_energy_error = (n * eigvals[0] - exact_gs_energy) / (
                                uncoupled_gs_energy - exact_gs_energy)
                        threshold_order = None
                    except LinAlgError as e:
                        # print(e)
                        fractional_energy_error = np.nan

                single_n_output_df.loc[len(single_n_output_df)] = {"n": n, "calls": np.inf, "order": order,
                                                                   "threshold-order": threshold_order,
                                                                   "energy-error": fractional_energy_error,
                                                                   "init-state-overlap": gs_init_state_overlap}

        return single_n_output_df


    num_processes = multiprocessing.cpu_count()

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(process, overlaps)

    pool.close()
    pool.join()
    output_df = pd.concat(results, ignore_index=True)

    # for overlap in overlaps:
    #     process(overlap)

    if args.partitioning:
        method_str = "_pqse_"
    elif args.thresholding:
        method_str = "_thqse_"
    else:
        method_str = "_"
    # output_df.to_csv(f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}{method_str}outputs_noiseless.csv")


def truncate_lists_to_decreasing_terms(orders: Union[List[float], Series], errors: Union[List[float], Series]) \
        -> (List[int], List[float]):
    orders = list(orders)
    errors = list(errors)
    first_nan_index = next((i for i, error in enumerate(errors) if np.isnan(error)), None)
    first_error_increase_index = next((i for i in range(len(errors) - 1) if errors[i] < errors[i + 1]), None)
    first_order_greater_than_10 = next((order for order in orders if order > 10), None)

    min_value = min(filter(None, [first_nan_index, first_error_increase_index, first_order_greater_than_10]))
    truncated_errors = errors[:min_value + 1]
    truncated_orders = orders[:min_value + 1]
    return list(truncated_orders), list(truncated_errors)


output_df = output_df[output_df['calls'] == np.inf]

# fig_calls, ax_calls = plt.subplots(1, 1, figsize=figsize)
fig_order = plt.figure(figsize=(6.75, 3.5))  # Adjust the figure size as needed
gs = fig_order.add_gridspec(2, 2, width_ratios=[2, 1])
ax_energy = plt.subplot(gs[:, 0])
ax_slopes = plt.subplot(gs[0, 1])
ax_intercepts = plt.subplot(gs[1, 1])

intercepts, intercepts_stderrs, slopes, stderrs = [], [], [], []
ns = np.array(output_df["n"].unique(), dtype=int)
for (idx, n_value) in enumerate(ns):
    n_group = output_df[output_df["n"] == n_value]
    ax_energy.plot(n_group['order'], abs(n_group['energy-error']), linestyle="-", linewidth=1, label=f'${n_value}$',
                   color=colors[idx % len(colors)])

    orders, energy_errors = truncate_lists_to_decreasing_terms(n_group['order'], abs(n_group['energy-error']))

    lin = linregress(orders, np.log10(energy_errors))

    intercepts.append(lin.intercept)
    intercepts_stderrs.append(lin.intercept_stderr)
    slopes.append(lin.slope)
    stderrs.append(lin.stderr)

n_tick_vals = range(min(ns), max(ns) + 2, 4)

ax_energy.set_xticks(range(max(output_df["order"])))
ax_energy.set_xlim(1, 15)
ax_energy.set_ylim(1e-16, 1e0)
ax_energy.set_yscale("log")
ax_energy.set_xlabel("$D$")
ax_energy.set_ylabel(r"$\Delta E/E_\mathrm{int}$")
leg_order = ax_energy.legend(ncols=2, loc="lower left", title=r"\underline{$N$}", fancybox=False)
leg_order.get_frame().set_edgecolor('none')  # Remove the outline
# leg_order.get_frame().set_facecolor('white')  # Set background color to white
leg_order.get_frame().set_alpha(0.8)

ax_overlaps = inset_axes(ax_energy, width="40%", height="35%", loc="upper right")

unique_pairs = output_df.drop_duplicates(subset=["n", "init-state-overlap"])
ax_overlaps.plot(unique_pairs['n'], abs(unique_pairs['init-state-overlap']), marker='o', linestyle='-',
                 label=f'N={n_value}', markersize=4, linewidth=1, fillstyle="none", color="k")
ax_overlaps.set_xlabel(r"$N$")
ax_overlaps.set_ylabel("Initial\noverlap")
ax_overlaps.set_xticks(n_tick_vals)

ax_slopes.plot(ns, slopes, linestyle="-", marker="o", markersize=4, linewidth=1, color="k",
               fillstyle="none")
ax_slopes.plot(ns, np.array(slopes) + np.array(stderrs), linestyle="-", linewidth=0.5,
               color="k", alpha=0.5)
ax_slopes.plot(ns, np.array(slopes) - np.array(stderrs), linestyle="-", linewidth=0.5,
               color="k", alpha=0.5)
ax_slopes.fill_between(ns, np.array(slopes) + np.array(stderrs),
                       np.array(slopes) - np.array(stderrs), color="k", alpha=0.2, edgecolor="none")
ax_intercepts.plot(ns, intercepts, linestyle="-", marker="o", markersize=4, linewidth=1,
                   color="k", fillstyle="none")
ax_intercepts.plot(ns, np.array(intercepts) + np.array(intercepts_stderrs), linestyle="-",
                   linewidth=0.5,
                   color="k", alpha=0.5)
ax_intercepts.plot(ns, np.array(intercepts) - np.array(intercepts_stderrs), linestyle="-",
                   linewidth=0.5,
                   color="k", alpha=0.5)
ax_intercepts.fill_between(ns, np.array(intercepts) + np.array(intercepts_stderrs),
                           np.array(intercepts) - np.array(intercepts_stderrs), color="k", alpha=0.2, edgecolor="none")

ax_intercepts.set_xlabel("$N$")
ax_slopes.set_ylabel("Gradient")
ax_intercepts.set_ylabel("Intercept")

ax_energy.text(s=r"\textbf{(a)}", x=0.15, y=0.92, transform=ax_energy.transAxes)
ax_overlaps.text(s=r"\textbf{(b)}", x=0.9, y=0.9, transform=ax_energy.transAxes)
ax_slopes.text(s=r"\textbf{(c)}", x=0.06, y=0.84, transform=ax_slopes.transAxes)
ax_intercepts.text(s=r"\textbf{(d)}", x=0.82, y=0.84, transform=ax_intercepts.transAxes)

ax_slopes.set_xticks(n_tick_vals)
ax_intercepts.set_xticks(n_tick_vals)
ax_slopes.set_xticklabels([])
ax_slopes.set_ylim(-1.5, -.6)
ax_intercepts.set_ylim(0, 1)

fig_order.tight_layout()
fig_order.savefig(f"noiseless_error_vs_order{method_str}.pdf")
fig_order.show()

desired_energy_error = 1e-4

fig_predict_order, ax_predict_order = plt.subplots(1, 1, figsize=figsize)

predicted_order_list, uncertainty_order_list = [], []

for n, m, m_stderr, c, c_stderr in zip(ns, slopes, stderrs, intercepts, intercepts_stderrs):
    log_desired_energy = np.log10(desired_energy_error)
    predicted_order = (log_desired_energy - c) / m
    uncertainty_order = ((c_stderr ** 2) + ((log_desired_energy / m) ** 2 * (m_stderr ** 2))) ** 0.5
    predicted_order_list.append(predicted_order)
    uncertainty_order_list.append(uncertainty_order)

ax_predict_order.errorbar(ns, predicted_order_list, uncertainty_order_list, marker="o", markersize=3, capsize=2,
                          color="k", linestyle="none", linewidth=1)

popt, pcov = curve_fit(lambda x, m, c: m * x + c, ns[-2:], predicted_order_list[-2:], sigma=uncertainty_order_list[-2:])
m = popt[0]
c = popt[1]
m_std = np.sqrt(pcov[0, 0])
c_std = np.sqrt(pcov[1, 1])

x_fit = np.linspace(0, 1000, 10000)
y_fit = m * x_fit + c
y_upper = (m + m_std) * x_fit + (c + c_std)
y_lower = (m - m_std) * x_fit + (c - c_std)

ax_predict_order.plot(x_fit, y_fit, color="red", linewidth=1,
                      label=f'${m:.3f}N + {c:.3f}$', zorder=0)
ax_predict_order.plot(x_fit, y_lower, color="red", linewidth=.2, alpha=0.5, zorder=0)
ax_predict_order.plot(x_fit, y_upper, color="red", linewidth=.2, alpha=0.5, zorder=0)
ax_predict_order.fill_between(x_fit, y_lower, y_upper, color="red", alpha=0.2, zorder=0, edgecolor="none")

ax_predict_order.set_xlabel("$N$")
ax_predict_order.set_ylabel(r"$D$")
ax_predict_order.set_xticks(n_tick_vals)
ax_predict_order.set_xlim(5, 28)
ax_predict_order.set_ylim(2.5, 6.5)
leg_predicted_order = ax_predict_order.legend()
leg_predicted_order.set_frame_on(False)

fig_predict_order.tight_layout()

fig_predict_order.savefig(f"noiseless_error_vs_order{method_str}extrapolation.pdf")
fig_predict_order.show()
