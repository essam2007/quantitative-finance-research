"""
Ornstein-Uhlenbeck Mean Reversion Strategy.

Model: dX_t = θ(μ − X_t)dt + σ dW_t

Used in statistical arbitrage: pairs trading, spread trading.
- Estimate OU parameters (θ, μ, σ)
- Detect deviations from equilibrium
- Trade mean reversion; backtest with Sharpe ratio.
"""

import numpy as np

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


# -----------------------------------------------------------------------------
# OU process simulation
# -----------------------------------------------------------------------------

def simulate_ou(T, dt, theta, mu, sigma, X0=None, seed=None):
    """
    Simulate OU process: dX_t = θ(μ − X_t)dt + σ dW_t.
    Returns time grid and path X.
    """
    if seed is not None:
        np.random.seed(seed)
    n_steps = int(round(T / dt))
    t = np.linspace(0, T, n_steps + 1)
    X = np.zeros(n_steps + 1)
    X[0] = X0 if X0 is not None else mu
    dW = np.sqrt(dt) * np.random.standard_normal(n_steps)
    for i in range(n_steps):
        X[i + 1] = X[i] + theta * (mu - X[i]) * dt + sigma * dW[i]
    return t, X


# -----------------------------------------------------------------------------
# Parameter estimation (AR(1) / OLS)
# -----------------------------------------------------------------------------

def estimate_ou_params(x, dt):
    """
    Estimate (θ, μ, σ) from discrete observations via AR(1) mapping.
    x[t+1] = a + b*x[t] + ε  =>  θ = -log(b)/dt, μ = a/(1-b), σ = σ_ε * sqrt(2θ/(1-b²)).
    Returns theta, mu, sigma (annualized where applicable).
    """
    x = np.asarray(x)
    y = x[1:]
    z = x[:-1]
    n = len(z)
    b = (np.sum(z * y) - np.sum(z) * np.mean(y)) / (np.sum(z**2) - n * np.mean(z)**2)
    a = np.mean(y) - b * np.mean(z)
    resid = y - (a + b * z)
    sigma_eps = np.std(resid, ddof=2)
    if b >= 1 or b <= 0:
        return np.nan, np.nan, np.nan
    theta = -np.log(b) / dt
    mu = a / (1 - b)
    sigma = sigma_eps * np.sqrt(2 * theta / (1 - b**2)) / np.sqrt(dt)
    return theta, mu, sigma


# -----------------------------------------------------------------------------
# Deviation and signals
# -----------------------------------------------------------------------------

def zscore(x, mu=None, sigma=None):
    """Z-score of x; if mu/sigma None use sample mean/std."""
    x = np.asarray(x)
    if mu is None:
        mu = np.mean(x)
    if sigma is None:
        sigma = np.std(x, ddof=1)
    if sigma == 0:
        return np.zeros_like(x)
    return (x - mu) / sigma


def mean_reversion_signal(spread, mu=None, sigma=None, entry_z=2.0, exit_z=0.5):
    """
    Signal: short spread when z > entry_z, long when z < -entry_z; exit when |z| < exit_z.
    Returns position: +1 long spread, -1 short spread, 0 flat.
    """
    z = zscore(spread, mu, sigma)
    if mu is None:
        mu = np.mean(spread)
    if sigma is None:
        sigma = np.std(spread, ddof=1)
    z = zscore(spread, mu, sigma)
    position = np.zeros(len(spread))
    pos = 0
    for i in range(len(spread)):
        if pos == 0:
            if z[i] > entry_z:
                pos = -1
            elif z[i] < -entry_z:
                pos = 1
        else:
            if (pos == 1 and z[i] >= -exit_z) or (pos == -1 and z[i] <= exit_z):
                pos = 0
        position[i] = pos
    return position


# -----------------------------------------------------------------------------
# Backtest and Sharpe
# -----------------------------------------------------------------------------

def backtest_spread(spread, position, spread_returns=None):
    """
    P&L from spread trading: position[i] held over period i -> i+1; profit = position[i] * (spread[i+1]-spread[i]).
    If spread_returns None, use spread diff.
    """
    spread = np.asarray(spread)
    position = np.asarray(position)
    if spread_returns is None:
        spread_returns = np.diff(spread)
    # position[i] applied to return from i to i+1 => len(pnl) = len(spread_returns)
    pos_held = position[: len(spread_returns)]
    if len(pos_held) != len(spread_returns):
        spread_returns = np.diff(spread)
        pos_held = position[: len(spread_returns)]
    pnl = pos_held * spread_returns
    return pnl


def sharpe_ratio(returns, rf=0.0, periods_per_year=252):
    """Annualized Sharpe. returns: 1d array."""
    excess = np.asarray(returns) - rf / periods_per_year
    if len(excess) < 2:
        return 0.0
    return np.sqrt(periods_per_year) * np.mean(excess) / (np.std(excess, ddof=1) + 1e-12)


# -----------------------------------------------------------------------------
# Demo: synthetic OU spread + backtest
# -----------------------------------------------------------------------------

def run_demo(T=2.0, dt=1/252, theta=2.0, mu=0.0, sigma=0.5, entry_z=2.0, seed=42):
    """Simulate OU spread, estimate params, generate signals, backtest, report Sharpe."""
    t, X = simulate_ou(T, dt, theta, mu, sigma, X0=mu, seed=seed)
    theta_est, mu_est, sigma_est = estimate_ou_params(X, dt)
    position = mean_reversion_signal(X, mu=mu_est, sigma=sigma_est or np.std(X), entry_z=entry_z, exit_z=0.5)
    pnl = backtest_spread(X, position)
    sharpe = sharpe_ratio(pnl, periods_per_year=1/dt)
    out = {
        "t": t, "X": X, "position": position, "pnl": pnl,
        "theta_est": theta_est, "mu_est": mu_est, "sigma_est": sigma_est,
        "sharpe": sharpe,
    }
    if _HAS_MPL:
        fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=False)
        axes[0].plot(t, X, label="Spread (OU)")
        axes[0].axhline(mu_est, color="gray", ls="--", label=f"μ_est={mu_est:.4f}")
        axes[0].set_ylabel("Spread"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
        axes[1].plot(t, position, drawstyle="steps-post", label="Position")
        axes[1].set_ylabel("Position"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
        t_pnl = t[1:]  # pnl aligns with intervals
        axes[2].plot(t_pnl, np.cumsum(pnl), label="Cumulative P&L")
        axes[2].set_ylabel("Cum P&L"); axes[2].set_xlabel("Time"); axes[2].legend(); axes[2].grid(True, alpha=0.3)
        fig.suptitle(f"OU Mean Reversion  |  θ={theta_est:.3f} μ={mu_est:.3f} σ={sigma_est:.3f}  |  Sharpe={sharpe:.3f}")
        plt.tight_layout()
        plt.show(block=True)
    return out


if __name__ == "__main__":
    print("Ornstein-Uhlenbeck mean reversion demo")
    res = run_demo(T=2.0, dt=1/252, theta=2.0, mu=0.0, sigma=0.5, entry_z=2.0, seed=42)
    print("  θ_est={:.4f} μ_est={:.4f} σ_est={:.4f}".format(res["theta_est"], res["mu_est"], res["sigma_est"]))
    print("  Sharpe = {:.4f}".format(res["sharpe"]))
