import numpy as np
import matplotlib.pyplot as plt


def simulate_brownian_motion(T=1.0, N=1000, paths=1, seed=None):
    """
    Simulate standard Brownian motion W_t on the interval [0, T].

    Parameters
    ----------
    T : float
        Final time
    N : int
        Number of time steps
    paths : int
        Number of independent sample paths
    seed : int or None
        Random seed for reproducibility

    Returns
    -------
    t : ndarray
        Time grid
    W : ndarray
        Simulated Brownian paths of shape (paths, N+1)
    """

    if seed is not None:
        np.random.seed(seed)

    dt = T / N

    # Brownian increments ~ Normal(0, dt)
    dW = np.sqrt(dt) * np.random.randn(paths, N)

    # Construct Brownian paths via cumulative sum
    W = np.cumsum(dW, axis=1)

    # Insert W(0) = 0
    W = np.hstack((np.zeros((paths, 1)), W))

    t = np.linspace(0, T, N + 1)

    return t, W


if __name__ == "__main__":

    # Simulation parameters
    T = 1.0
    N = 1000
    paths = 10000

    # Simulate Brownian motion
    t, W = simulate_brownian_motion(T=T, N=N, paths=paths, seed=42)

    # Extract W(T)
    W_T = W[:, -1]

    # Theoretical properties
    empirical_mean = np.mean(W_T)
    empirical_variance = np.var(W_T)

    print("Empirical mean at time T:", empirical_mean)
    print("Empirical variance at time T:", empirical_variance)
    print("Theoretical variance at time T:", T)

    # -------- Plot 1: Sample Paths --------
    plt.figure()
    for i in range(5):
        plt.plot(t, W[i])
    plt.title("Simulated Standard Brownian Motion")
    plt.xlabel("Time")
    plt.ylabel("W(t)")

    # -------- Plot 2: Distribution at T --------
    plt.figure()
    plt.hist(W_T, bins=50, density=True, alpha=0.6)

    x = np.linspace(min(W_T), max(W_T), 300)
    theoretical_density = (1 / np.sqrt(2 * np.pi * T)) * np.exp(-x**2 / (2 * T))
    plt.plot(x, theoretical_density)

    plt.title("Distribution of W(T)")
    plt.xlabel("Value")
    plt.ylabel("Density")

    # Show both figures
    plt.show()