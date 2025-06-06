import numpy as np


def breeden_litzenberger_solve_system(Ki, delta_K, r, T, C):
    g = np.zeros(len(Ki) - 2)
    for i in range(1, len(Ki) - 1):
        g[i - 1] = (np.exp(r * T) *
                    (C[i - 1] - 2 * C[i] + C[i + 1]) / (delta_K ** 2))

    # Bias correction for non-negative densities
    g[g < 0] = 0

    return g


def numpy_solve_system(A, C):
    f = np.linalg.lstsq(A, C)[0]
    return f
