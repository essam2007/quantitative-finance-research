"""
Heston Stochastic Volatility Model.

dS_t = μ S_t dt + √v_t S_t dW_t
dv_t = κ(θ − v_t)dt + σ√v_t dZ_t   with  E[dW dZ] = ρ dt

Simulate paths; compare volatility with Black-Scholes (constant vol).
"""

import numpy as np

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


# -----------------------------------------------------------------------------
# Heston simulation (Euler with reflection for v)
# -----------------------------------------------------------------------------

def simulate_heston(S0, v0, mu, kappa, theta, sigma_vol, rho, T, n_steps, n_paths=1, seed=None):
    """
    Euler-Maruyama for Heston. v is variance; we reflect v at 0 to keep non-negative.
    Returns t, S paths, v paths.
    """
    if seed is not None:
        np.random.seed(seed)
    dt = T / n_steps
    t = np.linspace(0, T, n_steps + 1)
    S = np.zeros((n_paths, n_steps + 1))
    v = np.zeros((n_paths, n_steps + 1))
    S[:, 0] = S0
    v[:, 0] = v0
    sqrt_dt = np.sqrt(dt)
    for i in range(n_steps):
        Z1 = np.random.standard_normal(n_paths)
        Z2 = np.random.standard_normal(n_paths)
        W = Z1
        Z = rho * Z1 + np.sqrt(1 - rho**2) * Z2
        sqrt_v = np.sqrt(np.maximum(v[:, i], 1e-10))
        S[:, i + 1] = S[:, i] * (1 + mu * dt + sqrt_v * sqrt_dt * W)
        v[:, i + 1] = np.maximum(v[:, i] + kappa * (theta - v[:, i]) * dt + sigma_vol * sqrt_v * sqrt_dt * Z, 1e-10)
    return t, S, v


# -----------------------------------------------------------------------------
# Compare with Black-Scholes (constant vol)
# -----------------------------------------------------------------------------

def simulate_bs(S0, mu, sigma, T, n_steps, n_paths=1, seed=None):
    """GBM with constant vol."""
    if seed is not None:
        np.random.seed(seed)
    dt = T / n_steps
    t = np.linspace(0, T, n_steps + 1)
    Z = np.random.standard_normal((n_paths, n_steps))
    log_S = np.log(S0) + np.cumsum((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z, axis=1)
    S = np.column_stack([np.full(n_paths, S0), np.exp(log_S)])
    return t, S


# -----------------------------------------------------------------------------
# Realized vol (annualized) from path
# -----------------------------------------------------------------------------

def realized_vol(log_returns, dt):
    """Annualized realized vol from log returns."""
    return np.std(log_returns, ddof=1) / np.sqrt(dt)


# -----------------------------------------------------------------------------
# Demo: Heston vs BS paths and vol
# -----------------------------------------------------------------------------

def run_demo(S0=100, v0=0.04, mu=0.05, kappa=2, theta=0.04, sigma_vol=0.3, rho=-0.5, T=1, n_steps=252, n_paths=5, seed=42):
    t, S_h, v_h = simulate_heston(S0, v0, mu, kappa, theta, sigma_vol, rho, T, n_steps, n_paths, seed=seed)
    sigma_bs = np.sqrt(theta)
    t_bs, S_bs = simulate_bs(S0, mu, sigma_bs, T, n_steps, n_paths, seed=seed + 1)
    print("Heston vs Black-Scholes")
    print("  Heston: κ={} θ={} σ_vol={} ρ={}".format(kappa, theta, sigma_vol, rho))
    print("  BS vol (√θ): {:.4f}".format(sigma_bs))
    if _HAS_MPL:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        for i in range(n_paths):
            axes[0, 0].plot(t, S_h[i], alpha=0.7)
            axes[0, 1].plot(t_bs, S_bs[i], alpha=0.7)
        axes[0, 0].set_title("Heston: S_t"); axes[0, 0].set_ylabel("Price"); axes[0, 0].grid(True, alpha=0.3)
        axes[0, 1].set_title("Black-Scholes: S_t"); axes[0, 1].grid(True, alpha=0.3)
        for i in range(n_paths):
            axes[1, 0].plot(t, v_h[i], alpha=0.7)
        axes[1, 0].axhline(theta, color="gray", ls="--", label="θ")
        axes[1, 0].set_title("Heston: variance v_t"); axes[1, 0].set_ylabel("v"); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)
        # Realized vol along one path
        dt = T / n_steps
        log_ret_h = np.diff(np.log(S_h[0])) 
        axes[1, 1].plot(t[1:], np.sqrt(v_h[0, 1:]), label="√v_t (Heston)")
        axes[1, 1].axhline(sigma_bs, color="green", ls="--", label="BS σ")
        axes[1, 1].set_title("Volatility"); axes[1, 1].set_xlabel("Time"); axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show(block=True)
    return {"t": t, "S_heston": S_h, "v": v_h, "S_bs": S_bs}


if __name__ == "__main__":
    run_demo(S0=100, v0=0.04, mu=0.05, kappa=2, theta=0.04, sigma_vol=0.3, rho=-0.5, T=1, n_steps=252, n_paths=5, seed=42)
