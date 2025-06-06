import matplotlib.pyplot as plt


def plot(S_T, g, f):
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
