# CBOE Data: S&P 500

# This file mimics the structure and content of the MATLAB file CBOE_data.m
# It defines market parameters and example implied volatility data
# to be used with the implied density surface reconstruction function.

import numpy as np

from plot_results import plot
from problem import get_problem_formulation
from solvers.classical_solver import breeden_litzenberger_solve_system, numpy_solve_system
from solvers.dwave_solver import solve_qubo

# Initial stock price
S0 = 10
# Risk-free interest rate
r = 0.03
# Implied volatilities (descending from 0.3 to 0.21)
implied_vols = np.arange(0.3, 0.20, -0.01)  # NOTE: np.arange excludes the stop value
# Strike prices (from 6 to 15 inclusive)
K = np.arange(6, 16, 1)
# Strike price increment
delta_K = 0.5
# Time to maturity in years
T = 3 / 12  # 3 months
# Maturity increment
delta_T = 1 / 12  # 1 month

Ki, S_T, A, C = get_problem_formulation(S0, r, implied_vols, K, T, delta_K)

g = breeden_litzenberger_solve_system(Ki, delta_K, r, T, C)
f = numpy_solve_system(A, C)
q_f = solve_qubo(len(f), 7, A, C, f, 100)

plot(S_T, g, f, q_f)

if all([x > 0 for x in f]):
    print('The implied density vectory is strictly positive: no arbitrage.')
else:
    print('The implied density vectory is NOT strictly positive: arbitrage is possible.')
