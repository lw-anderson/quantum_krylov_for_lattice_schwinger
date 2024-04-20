import os
import sys

import numpy as np

sys.path.append(os.getcwd())

from costing.gate_costs.cost import CostTotal
from utils.plotting_utils import set_plot_style, colors

set_plot_style()

ns = range(4, 10002, 2)
t_gates = []
cnot_gates = []
rz_gates = []
t_gates_inc_rz = []
num_qubits = []

g_t_gates = []
g_cnot_gates = []
g_rz_gates = []

u_t_gates = []
u_cnot_gates = []
u_rz_gates = []

pi_t_gates = []
pi_cnot_gates = []
pi_rz_gates = []

sub_opt_pi_t_gates = []
sub_opt_pi_cnot_gates = []
sub_opt_pi_rz_gates = []

sub_opt_u_t_gates = []
sub_opt_u_cnot_gates = []
sub_opt_u_rz_gates = []

k = 1

for n in ns:
    m = np.ceil(np.log2(n))
    cost_total = CostTotal(m, n, k, include_gs=False, suboptimal_g=False, suboptimal_u=False)
    t_gates.append(cost_total.t_gates)
    cnot_gates.append(cost_total.cnot_gates)
    rz_gates.append(cost_total.rz_gates)
    t_gates_inc_rz.append(cost_total.t_gates_inc_rz)

    if n in [1e2, 1e3, 1e4]:
        print(f"n={n}, cnots={cnot_gates[-1]}, tgates in rz={t_gates_inc_rz[-1]}")

    num_qubits.append(cost_total.num_qubits)

    # g_t_gates.append(cost_total._cost_g.t_gates)
    # g_cnot_gates.append(cost_total._cost_g.cnot_gates)
    # g_rz_gates.append(cost_total._cost_g.rz_gates)

    u_t_gates.append(k * cost_total._cost_u.t_gates)
    u_cnot_gates.append(k * cost_total._cost_u.cnot_gates)
    u_rz_gates.append(k * cost_total._cost_u.rz_gates)

    pi_t_gates.append(k * cost_total._cost_pi.t_gates_inc_rz)
    pi_cnot_gates.append(k * cost_total._cost_pi.cnot_gates)
    pi_rz_gates.append(k * cost_total._cost_pi.rz_gates)

    sub_opt_cost_total = CostTotal(m, n, k, include_gs=False, suboptimal_g=True, suboptimal_u=True)

    sub_opt_pi_t_gates.append(sub_opt_cost_total._cost_pi.t_gates_inc_rz)
    sub_opt_pi_cnot_gates.append(sub_opt_cost_total._cost_pi.cnot_gates)
    sub_opt_pi_rz_gates.append(sub_opt_cost_total._cost_pi.rz_gates)

    sub_opt_u_t_gates.append(sub_opt_cost_total._cost_u.t_gates_inc_rz)
    sub_opt_u_cnot_gates.append(sub_opt_cost_total._cost_u.cnot_gates)
    sub_opt_u_rz_gates.append(sub_opt_cost_total._cost_u.rz_gates)

from matplotlib import pyplot as plt, gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

fig = plt.figure(figsize=(6.75, 6))
gs = gridspec.GridSpec(2, 2, height_ratios=[1.5, 1])

ax1 = plt.subplot(gs[0, :])
ax2 = plt.subplot(gs[1, 0])
ax3 = plt.subplot(gs[1, 1])

# Add a subplot to ax1
axins = inset_axes(ax1, width="28%", height="38%", loc="upper left")
axins2 = axins.twinx()
axins2.set_yscale('log')
axins2.plot(ns, num_qubits, color="k", linewidth=1.)
axins2.set_xscale('log')
axins2.set_yscale('log')
axins.set_xlabel(r"$N$")
axins2.set_ylabel(r"\# qubits")
axins.set_yticks([])
axins2.set_yticks([1e2, 1e3, 1e4, 1e5])
axins2.set_xticks([1e1, 1e2, 1e3, 1e4])

ax1.plot(ns, t_gates, color=colors[0], linewidth=1., label=r"T gates")
ax1.plot(ns, rz_gates, color=colors[1], linewidth=1., label=r"$R_Z$ gates")
ax1.plot(ns, t_gates_inc_rz, color=colors[2], linewidth=1., label=r"T gates inc. rotations")
ax1.set_xlabel(r"$N$")
ax1.set_ylabel(r"\# gates")
ax1.set_ylim(5e2, 1e8)
leg1 = ax1.legend(loc="lower right", fancybox=False, edgecolor="k")
leg1.set_frame_on(False)
ax1.set_xscale('log')
ax1.set_yscale('log')

ax2.plot(ns, sub_opt_u_t_gates, color=colors[0], linewidth=1., linestyle="--")  # , label="$U$ (Kirby)")
ax2.plot(ns, u_t_gates, color=colors[0], linewidth=1., label="$U$")  # (improved)")
ax2.plot(ns, sub_opt_pi_t_gates, color=colors[2], linestyle="--", linewidth=1.)  # , label=r"$\Pi_\varphi$ (Kirby)")
ax2.plot(ns, pi_t_gates, color=colors[2], linewidth=1., label=r"$\Pi_\varphi$")  # (improved)")

ax2.set_xlabel(r"$N$")
ax2.set_ylabel(r"\# T gates inc. rotations")
leg2 = ax2.legend(fancybox=False, edgecolor="k")
leg2.set_frame_on(False)
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_yticks([1e2, 1e4, 1e6, 1e8, 1e10])

ax3.plot(ns, sub_opt_u_cnot_gates, color=colors[0], linestyle="--", linewidth=1.)  # , label="$U$ (Kirby)")
ax3.plot(ns, u_cnot_gates, color=colors[0], linewidth=1., label="$U$")  # (improved)")
ax3.plot(ns, sub_opt_pi_cnot_gates, color=colors[2], linestyle="--", linewidth=1.)  # , label=r"$\Pi_\varphi$ (Kirby)")
ax3.plot(ns, pi_cnot_gates, color=colors[2], linewidth=1., label=r"$\Pi_\varphi$")  # (improved)")
ax3.set_xlabel(r"$N$")
ax3.set_ylabel(r"\# CNOT gates")
leg3 = ax3.legend(fancybox=False, edgecolor="k")
leg3.set_frame_on(False)
ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.set_yticks([1e2, 1e4, 1e6, 1e8, 1e10])

ax1.text(s=r"\textbf{(a)}", x=0.5, y=0.9, transform=ax1.transAxes)
axins.text(s=r"\textbf{(b)}", x=0.03, y=0.9, transform=ax1.transAxes)
ax2.text(s=r"\textbf{(c)}", x=0.5, y=0.87, transform=ax2.transAxes, ha="center")
ax3.text(s=r"\textbf{(d)}", x=0.5, y=0.87, transform=ax3.transAxes, ha="center")

fig.tight_layout()
fig.savefig("gate_cost_single_k.pdf")
fig.show()
