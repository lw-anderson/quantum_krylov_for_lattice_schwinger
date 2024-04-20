from abc import ABC, abstractmethod

import numpy as np
from scipy.constants import golden


def is_even(n):
    return bool((n + 1) % 2)


def fib(n):
    if n % 2 == 0:
        return (golden ** n - golden ** (-n)) / np.sqrt(5)
    else:
        return (golden ** n + golden ** (-n)) / np.sqrt(5)


def t_cost_of_rz(m, n):
    return 3 * np.log((n - 1) * 2 ** m + 2 ** (m + 1) * n + n) + 3 * np.log(12 * m + 48)


class Cost(ABC):
    def __init__(self, m: int, n: int):
        self.m = m
        self.n = n
        self.t_gates = self._calc_t_gates()
        self.cnot_gates = self._calc_cnot_gates()
        self.rz_gates = self._calc_rz_gates()
        self.t_gates_inc_rz = self._calc_t_gate_inc_rz()

    @abstractmethod
    def _calc_t_gates(self):
        pass

    @abstractmethod
    def _calc_cnot_gates(self):
        pass

    @abstractmethod
    def _calc_rz_gates(self):
        pass

    def _calc_t_gate_inc_rz(self):
        m = self.m
        n = self.n
        return self.t_gates + self.rz_gates * t_cost_of_rz(m, n)


class CostG(Cost):
    def __init__(self, m: int, n: int, suboptimal_g: bool = False, suboptimal_u: bool = False):
        self.suboptimal_g = suboptimal_g
        self.suboptimal_u = suboptimal_u
        super().__init__(m, n)

    def _calc_t_gates(self):
        m = self.m
        n = self.n

        tgates = 0

        if self.suboptimal_g:
            tgates += (n - 1) * (8 * m * fib(m + 6) - 8 * fib(m + 6) + 64
                                 + 4 * m * 2 ** m - 16 * 2 ** m)
        else:
            tgates += (56 - 16 * m - 8 * fib(m + 7) + 8 * m * fib(m + 5)
                       - 16 * 2 ** m + 4 * n + 4 * m * 2 ** m + 8 * n * m)

        # if self.suboptimal_u:
        #     pass
        #
        # else:
        #     tgates += (4 * m * 2 ** m - 15 * 2 ** m
        #                + 8 * m * fib(m + 5) - 8 * fib(m + 6) - 7 * fib(m + 5) + 61)

        return tgates

    def _calc_cnot_gates(self):
        m = self.m
        n = self.n

        cnots = 0

        if self.suboptimal_g:
            cnots += (n - 1) * (8 * m * fib(m + 6) - 6 * fib(m + 5) + 58
                                + 4 * m * 2 ** m - 14 * 2 ** m
                                + 2)
        else:
            cnots += (54 - 22 * m - 6 * fib(m + 5) - 8 * fib(m + 6)
                      + 8 * m * fib(m + 5) - 14 * 2 ** m + 25 * n + 4 * m * 2 ** m + 11 * n * m)

        # if self.suboptimal_u:
        #     pass
        #
        # else:
        #     cnots += (4 * m * 2 ** m - 12 * 2 ** m +
        #               8 * m * fib(m + 5) - 8 * fib(m + 6) - 4 * fib(m + 5) + 52)

        return cnots

    def _calc_rz_gates(self):
        m = self.m
        n = self.n

        rzs = 0

        if self.suboptimal_g:
            rzs += (n - 1) * (12 * fib(m + 3) - 36
                              + 12 * 2 ** m
                              + 6)
        else:
            rzs += -48 + 12 * fib(m + 5) + 12 * 2 ** m + 6 * n

        return rzs


class CostU(Cost):
    def __init__(self, m: int, n: int, suboptimal_u: bool = False):
        self.suboptimal_u = suboptimal_u
        super().__init__(m, n)

    def _calc_t_gates(self):
        m = self.m
        n = self.n

        tgates = + 6 * n

        if self.suboptimal_u:
            tgates += (n - 1) * (32 + 6 * m + 4 * m * fib(m + 5) - 4 * fib(m + 7)
                                 - 8 * 2 ** m + 2 * m * 2 ** m)

        return tgates

    def _calc_cnot_gates(self):
        m = self.m
        n = self.n

        cnots = 8 * n

        if self.suboptimal_u:
            cnots += (n - 1) * (29 + 8 * m - 4 * fib(m + 6) - 3 * fib(m + 5)
                                + 4 * m * fib(m + 5) - 7 * 2 ** m + 2 * m * 2 ** m)

        return cnots

    def _calc_rz_gates(self):
        return 0


class CostPi(Cost):
    def __init__(self, m, n, g_cost: CostG):
        self._g_cost = g_cost
        super().__init__(m, n)

    def _calc_t_gates(self):
        m = self.m
        n = self.n
        return 2 * self._g_cost.t_gates + 128 * m * n + 128 * n - 128 * m - 192

    def _calc_cnot_gates(self):
        m = self.m
        n = self.n
        return 2 * self._g_cost.t_gates + 96 * m * n + 96 * n - 96 * m - 144

    def _calc_rz_gates(self):
        return 2 * self._g_cost.rz_gates + 1


class CostTotal(Cost):
    def __init__(self, m, n, k, include_gs: bool = True, suboptimal_g: bool = False, suboptimal_u: bool = False):
        self._cost_g = CostG(m, n, suboptimal_g, suboptimal_u)
        self._cost_u = CostU(m, n, suboptimal_u)
        self._cost_pi = CostPi(m, n, self._cost_g)
        self._k = k
        self._include_gs = include_gs
        super().__init__(m, n)
        self.num_qubits = self._calc_num_qubits()

    def _calc_t_gates(self):
        return self._k * (self._cost_u.t_gates + self._cost_pi.t_gates)

    def _calc_cnot_gates(self):
        return self._k * (self._cost_u.cnot_gates + self._cost_pi.cnot_gates)

    def _calc_rz_gates(self):
        return self._k * (self._cost_u.rz_gates + self._cost_pi.rz_gates)

    def _calc_num_qubits(self):
        m = self.m
        n = self.n
        return 3 * (m * (n - 1) + n) + (m - 1)


def num_terms(N):
    m = int(np.ceil(np.log2(N) + 1))
    return (N - 1) * N + (N - 1) * (fib(m + 5) - 3) + N
