import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import norm
from numpy.linalg import inv


def black_scholes_call(S, K, r, T, sigma):
    """Black-Scholes Call option price."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call


def positive_part(x):
    return x if x > 0 else 0


def implied_density_function(S0, r, implied_vols, K, T, delta_K):
    # Check that 1/delta_K is an integer
    if not np.isclose(1 / delta_K, round(1 / delta_K)):
        raise ValueError("The variable delta_K must be such that 1/delta_K is an integer.")

    # Interpolated strike prices
    Ki = np.arange(K[0], K[-1] + delta_K, delta_K)

    # Interpolated implied volatilities
    interp_vol = interp1d(K, implied_vols, kind='linear', fill_value="extrapolate")
    interpolated_IV = interp_vol(Ki)

    # Black-Scholes prices
    call_prices = black_scholes_call(S0, Ki, r, T, interpolated_IV)

    # Apply Breeden-Litzenberger formula to approximate second derivative
    g = np.zeros(len(Ki) - 2)
    for i in range(1, len(Ki) - 1):
        g[i - 1] = (np.exp(r * T) *
                    (call_prices[i - 1] - 2 * call_prices[i] + call_prices[i + 1]) / (delta_K ** 2))

    # Bias correction for non-negative densities
    g[g < 0] = 0

    # Adjusted strike grid for density
    S_T = Ki[1:-1]

    # Approximation to linear system
    k = 3
    sigma = max(interpolated_IV)
    s_min = S0 * math.exp(-k * sigma * math.sqrt(T))
    s_max = S0 * math.exp(k * sigma * math.sqrt(T))
    # ensures that A is square (M=len(K))
    N = len(Ki)
    delta_s = (s_max - s_min) / (N - 1)
    s_i = [s_min + (i - 1) * delta_s for i in range(N)]
    A = [[positive_part(float(s_i[i] - Ki[j]) * delta_s) for i in range(len(s_i))] for j in range(len(Ki))]
    f = np.linalg.lstsq(A, call_prices)[0]

    # Plot result
    plt.figure(figsize=(10, 5))
    plt.plot(S_T, g, label='Implied Risk-Neutral Density', color='blue')
    plt.plot(S_T, f[1:-1], label='Approx. Risk-Neutral Density', color='red')
    plt.xlabel("Stock Price")
    plt.ylabel("Density")
    plt.title("Implied Risk-Neutral Density Function")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("implied_density_function.png")
    plt.show()

    return f
