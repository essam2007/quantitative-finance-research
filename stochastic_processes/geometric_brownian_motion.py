import numpy as np
import matplotlib.pyplot as plt


def simulate_geometric_brownian_motion(
    S0=100.0,
    mu=0.05,
    sigma=0.2,
    T=1.0,
    N=1000,
    paths=1,
    seed=None,
):
    """
    Simulate geometric Brownian motion (GBM) for asset price S_t.

    The process follows the SDE: dS_t = μ S_t dt + σ S_t dW_t
    with closed-form solution: S_t = S_0 exp((μ - σ²/2)t + σ W_t).

    Parameters
    ----------
    S0 : float
        Initial asset price at t=0
    mu : float
        Drift (expected return per unit time)
    sigma : float
        Volatility (standard deviation of returns per unit time)
    T : float
        Time horizon
    N : int
        Number of time steps
    paths : int
        Number of independent sample paths
    seed : int or None
        Random seed for reproducibility

    Returns
    -------
    t : ndarray
        Time grid of shape (N+1,)
    S : ndarray
        Simulated asset price paths of shape (paths, N+1)
    """

    if seed is not None:
        np.random.seed(seed)

    dt = T / N

    # Brownian increments ~ Normal(0, dt)
    dW = np.sqrt(dt) * np.random.randn(paths, N)

    # Cumulative Brownian motion W_t at each step
    W = np.cumsum(dW, axis=1)
    W = np.hstack((np.zeros((paths, 1)), W))

    # GBM: S_t = S_0 exp((μ - σ²/2)t + σ W_t)
    t = np.linspace(0, T, N + 1)
    drift = (mu - 0.5 * sigma**2) * t
    diffusion = sigma * W
    S = S0 * np.exp(drift + diffusion)

    return t, S


if __name__ == "__main__":

    # Simulation parameters (e.g. stock with 5% drift, 20% vol)
    S0 = 100.0
    mu = 0.05
    sigma = 0.2
    T = 1.0
    N = 1000
    paths = 10000

    t, S = simulate_geometric_brownian_motion(
        S0=S0, mu=mu, sigma=sigma, T=T, N=N, paths=paths, seed=42
    )

    S_T = S[:, -1]
    log_returns = np.log(S_T / S0)

    # Theoretical moments for GBM at T
    E_ST = S0 * np.exp(mu * T)
    Var_ST = S0**2 * np.exp(2 * mu * T) * (np.exp(sigma**2 * T) - 1)

    print("Geometric Brownian Motion simulation")
    print("  S0 = {}, μ = {}, σ = {}, T = {}".format(S0, mu, sigma, T))
    print("  E[S(T)]  (theory): {:.4f}".format(E_ST))
    print("  E[S(T)]  (sample): {:.4f}".format(np.mean(S_T)))
    print("  Var[S(T)] (theory): {:.4f}".format(Var_ST))
    print("  Var[S(T)] (sample): {:.4f}".format(np.var(S_T)))

    # -------- Plot 1: Sample price paths --------
    plt.figure()
    for i in range(5):
        plt.plot(t, S[i])
    plt.axhline(S0, color="gray", linestyle="--", alpha=0.7)
    plt.title("Simulated Geometric Brownian Motion (sample paths)")
    plt.xlabel("Time")
    plt.ylabel("S(t)")

    # -------- Plot 2: Distribution of S(T) --------
    plt.figure()
    plt.hist(S_T, bins=50, density=True, alpha=0.6, label="Simulated")
    x = np.linspace(S_T.min(), S_T.max(), 300)
    # Log-normal density for S_T (derived from GBM)
    log_mean = np.log(S0) + (mu - 0.5 * sigma**2) * T
    log_var = sigma**2 * T
    density = (1 / (x * np.sqrt(2 * np.pi * log_var))) * np.exp(
        -(np.log(x) - log_mean) ** 2 / (2 * log_var)
    )
    plt.plot(x, density, label="Theoretical (log-normal)")
    plt.title("Distribution of S(T)")
    plt.xlabel("S(T)")
    plt.ylabel("Density")
    plt.legend()

    plt.show()
