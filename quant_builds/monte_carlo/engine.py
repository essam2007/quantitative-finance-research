"""
Monte Carlo Option Pricing Engine.

Price derivatives by simulation:
  C = e^{-rT} E[ payoff(S_T) ]

- Simulate price paths (GBM)
- Compute payoff (e.g. European call: max(S_T - K, 0))
- Discount expected value

Variance reduction: antithetic sampling, control variates.
"""

import math
import numpy as np

try:
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False


def _norm_cdf(x):
    """Standard normal CDF (for Black-Scholes control variate)."""
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


def black_scholes_call(S, K, T, r, sigma, q=0.0):
    """European call price (used as control variate reference)."""
    if T <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * math.exp(-q * T) * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


# -----------------------------------------------------------------------------
# Path simulation (GBM)
# -----------------------------------------------------------------------------

def simulate_gbm_paths(S0, r, sigma, T, n_paths, n_steps=1, seed=None):
    """
    Simulate terminal spot prices S_T under GBM (and optionally full paths).

    dS_t = r S_t dt + sigma S_t dW_t  =>  S_T = S0 exp((r - sigma^2/2)T + sigma W_T).

    Parameters
    ----------
    S0 : float
        Initial spot
    r : float
        Risk-free rate (continuous)
    sigma : float
        Volatility
    T : float
        Time to maturity (years)
    n_paths : int
        Number of Monte Carlo paths
    n_steps : int
        Number of steps (1 => single jump to T; >1 for path-dependent use)
    seed : int or None
        Random seed

    Returns
    -------
    S_T : ndarray (n_paths,)
        Terminal spot at T for each path
    paths : ndarray (n_paths, n_steps+1) or None
        Full paths if n_steps > 1; else None
    """
    if seed is not None:
        np.random.seed(seed)
    dt = T / n_steps
    # (n_paths, n_steps) standard normals
    Z = np.random.standard_normal((n_paths, n_steps))
    # W_T = sum of sqrt(dt)*Z over steps
    W_T = np.sum(np.sqrt(dt) * Z, axis=1)
    drift = (r - 0.5 * sigma**2) * T
    S_T = S0 * np.exp(drift + sigma * W_T)
    if n_steps > 1:
        # Build paths: W at each step
        W = np.cumsum(np.sqrt(dt) * Z, axis=1)
        t = np.linspace(0, T, n_steps + 1)
        drift_t = (r - 0.5 * sigma**2) * t
        paths = S0 * np.exp(np.hstack([np.zeros((n_paths, 1)), drift_t[1:] + sigma * W]))
        return S_T, paths
    return S_T, None


# -----------------------------------------------------------------------------
# Payoffs and pricing
# -----------------------------------------------------------------------------

def european_call_payoff(S_T, K):
    """Payoff of European call: max(S_T - K, 0)."""
    return np.maximum(S_T - K, 0.0)


def european_put_payoff(S_T, K):
    """Payoff of European put: max(K - S_T, 0)."""
    return np.maximum(K - S_T, 0.0)


def price_european_option(
    S0, K, T, r, sigma,
    payoff_fn=european_call_payoff,
    n_paths=100_000,
    n_steps=1,
    seed=None,
    use_antithetic=False,
    use_control_variate=False,
    q=0.0,
):
    """
    Price a European option by Monte Carlo.

    C = e^{-rT} E[ payoff(S_T) ]

    Parameters
    ----------
    S0, K, T, r, sigma : float
        Spot, strike, maturity, rate, volatility
    payoff_fn : callable
        payoff_fn(S_T) -> array; e.g. european_call_payoff(..., K)
    n_paths : int
        Number of simulation paths
    n_steps : int
        Time steps per path (1 sufficient for European)
    seed : int or None
        Random seed
    use_antithetic : bool
        If True, use antithetic variates (half the draws, paired with -Z)
    use_control_variate : bool
        If True, use control variate with known E[S_T] = S0*exp(r*T) and optional BS
    q : float
        Dividend yield (for control variate BS)

    Returns
    -------
    price : float
        Option price estimate
    std_err : float
        Standard error of the estimate (optional; can extend)
    """
    discount = np.exp(-r * T)

    if use_antithetic:
        n_draws = (n_paths + 1) // 2
        if seed is not None:
            np.random.seed(seed)
        Z = np.random.standard_normal(n_draws)
        W_T = np.sqrt(T) * Z
        W_T_anti = -np.sqrt(T) * Z
        drift = (r - 0.5 * sigma**2) * T
        S_T = S0 * np.exp(drift + sigma * W_T)
        S_T_anti = S0 * np.exp(drift + sigma * W_T_anti)
        payoffs = 0.5 * (payoff_fn(S_T, K) + payoff_fn(S_T_anti, K))
        # For control variate use average terminal spot (same expectation S0*exp(r*T))
        S_T_control = 0.5 * (S_T + S_T_anti)
        n_used = n_draws
    else:
        S_T, _ = simulate_gbm_paths(S0, r, sigma, T, n_paths, n_steps=n_steps, seed=seed)
        payoffs = payoff_fn(S_T, K)
        S_T_control = S_T
        n_used = n_paths

    raw_estimate = discount * np.mean(payoffs)
    std_err = discount * np.std(payoffs, ddof=1) / np.sqrt(n_used)

    if use_control_variate:
        # Control variate: Y = S_T, E[Y] = S0 * exp(r*T). Adjust estimate by c*(E[Y]-Y_bar).
        mean_Y = np.mean(S_T_control)
        known_EY = S0 * np.exp(r * T)
        cov_XY = np.cov(payoffs, S_T_control)[0, 1]
        var_Y = np.var(S_T_control, ddof=1)
        c = cov_XY / var_Y if var_Y > 1e-20 else 0.0
        if var_Y > 1e-20:
            raw_estimate = discount * (np.mean(payoffs) + c * (known_EY - mean_Y))
        residuals = payoffs + c * (known_EY - S_T_control)
        std_err = discount * np.std(residuals, ddof=1) / np.sqrt(n_used)

    return raw_estimate, std_err


def price_european_call(
    S0, K, T, r, sigma,
    n_paths=100_000,
    seed=None,
    use_antithetic=False,
    use_control_variate=False,
    q=0.0,
):
    """
    Price European call: C = e^{-rT} E[max(S_T - K, 0)].
    """
    return price_european_option(
        S0, K, T, r, sigma,
        payoff_fn=lambda S_T, K_: european_call_payoff(S_T, K_),
        n_paths=n_paths,
        seed=seed,
        use_antithetic=use_antithetic,
        use_control_variate=use_control_variate,
        q=q,
    )


def price_european_put(
    S0, K, T, r, sigma,
    n_paths=100_000,
    seed=None,
    use_antithetic=False,
    use_control_variate=False,
    q=0.0,
):
    """Price European put: e^{-rT} E[max(K - S_T, 0)]."""
    return price_european_option(
        S0, K, T, r, sigma,
        payoff_fn=lambda S_T, K_: european_put_payoff(S_T, K_),
        n_paths=n_paths,
        seed=seed,
        use_antithetic=use_antithetic,
        use_control_variate=use_control_variate,
        q=q,
    )


# -----------------------------------------------------------------------------
# Path visualization (opens a window on your PC)
# -----------------------------------------------------------------------------

def run_path_demo(S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2, n_paths=500, n_steps=252, seed=42):
    """
    Generate full GBM paths, price the option, and open a window with:
      - Left: sample price paths S(t) over time
      - Right: histogram of terminal spot S_T and payoff
    """
    if not _HAS_MATPLOTLIB:
        print("Install matplotlib to see the path plot: pip install matplotlib")
        return
    # Generate full paths (n_steps >= 2 so we get a path array)
    n_steps = max(2, int(n_steps))
    S_T, paths = simulate_gbm_paths(S0, r, sigma, T, n_paths, n_steps=n_steps, seed=seed)
    t = np.linspace(0, T, paths.shape[1])
    payoffs = european_call_payoff(S_T, K)
    discount = np.exp(-r * T)
    price_est = discount * np.mean(payoffs)
    bs = black_scholes_call(S0, K, T, r, sigma)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    # Left: sample paths
    n_show = min(80, paths.shape[0])
    ax1.plot(t, paths[:n_show].T, alpha=0.6, color="steelblue", lw=0.8)
    ax1.axhline(K, color="red", linestyle="--", label=f"Strike K={K}")
    ax1.axhline(S0, color="gray", linestyle=":", alpha=0.8)
    ax1.set_xlabel("Time (years)")
    ax1.set_ylabel("Spot S(t)")
    ax1.set_title("Simulated GBM price paths")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Right: histogram of S_T and payoffs
    ax2.hist(S_T, bins=50, alpha=0.6, color="steelblue", density=True, label="S_T")
    ax2.axvline(K, color="red", linestyle="--", label=f"K={K}")
    ax2.axvline(S0 * np.exp(r * T), color="green", linestyle=":", label="E[S_T]")
    ax2.set_xlabel("Terminal spot S_T")
    ax2.set_ylabel("Density")
    ax2.set_title("Distribution of S_T")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        f"Monte Carlo European call  |  S0={S0}  K={K}  T={T}  r={r}  σ={sigma}  |  "
        f"Paths={n_paths}  Steps={n_steps}  |  MC price={price_est:.4f}  BS={bs:.4f}",
        fontsize=10,
    )
    plt.tight_layout()
    plt.show(block=True)


# -----------------------------------------------------------------------------
# Example / CLI
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    n_paths = 100_000
    seed = 42

    bs = black_scholes_call(S0, K, T, r, sigma)
    mc, se = price_european_call(S0, K, T, r, sigma, n_paths=n_paths, seed=seed)
    mc_anti, se_anti = price_european_call(
        S0, K, T, r, sigma, n_paths=n_paths, seed=seed, use_antithetic=True
    )
    mc_cv, se_cv = price_european_call(
        S0, K, T, r, sigma, n_paths=n_paths, seed=seed, use_control_variate=True
    )

    print("European call: S0={}, K={}, T={}, r={}, sigma={}".format(S0, K, T, r, sigma))
    print("  Black-Scholes:     {:.6f}".format(bs))
    print("  Monte Carlo:       {:.6f}  (std err {:.6f})".format(mc, se))
    print("  MC + Antithetic:   {:.6f}  (std err {:.6f})".format(mc_anti, se_anti))
    print("  MC + Control Var:  {:.6f}  (std err {:.6f})".format(mc_cv, se_cv))
    print()
    print("Opening path demo window (500 paths, 252 steps)...")
    run_path_demo(S0=S0, K=K, T=T, r=r, sigma=sigma, n_paths=500, n_steps=252, seed=seed)
