"""
Portfolio Optimization Engine.

Markowitz: min w'Σw  subject to  w'μ = r  (and sum w = 1, w >= 0).
Plot efficient frontier. Extensions: transaction costs, risk parity, Black-Litterman.
"""

import numpy as np

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


def _solve_min_var(mu, Sigma, target_return):
    """Solve min w'Σw s.t. μ'w = r, 1'w = 1, w >= 0. Requires scipy."""
    try:
        from scipy.optimize import minimize
    except ImportError:
        return None
    n = len(mu)
    def obj(w):
        return 0.5 * (w @ Sigma @ w)
    A_eq = np.vstack([mu, np.ones(n)])
    b_eq = np.array([target_return, 1.0])
    res = minimize(obj, np.ones(n) / n, method="SLSQP", bounds=[(0, 1)] * n,
                  constraints={"type": "eq", "fun": lambda w: A_eq @ w - b_eq})
    if res.success:
        return res.x
    return None


def efficient_frontier(mu, Sigma, n_points=50, min_ret=None, max_ret=None):
    """
    For a grid of target returns, solve min w'Σw s.t. w'μ = r, 1'w = 1, w >= 0.
    Returns (returns, volatilities, weights). Uses scipy if available.
    """
    mu = np.asarray(mu)
    Sigma = np.asarray(Sigma)
    n = len(mu)
    if min_ret is None:
        min_ret = mu.min()
    if max_ret is None:
        max_ret = mu.max()
    targets = np.linspace(min_ret, max_ret, n_points)
    vols = []
    weights = []
    for r in targets:
        w = _solve_min_var(mu, Sigma, r)
        if w is not None:
            vols.append(np.sqrt(w @ Sigma @ w))
            weights.append(w)
        else:
            vols.append(np.nan)
            weights.append(np.full(n, np.nan))
    return targets, np.array(vols), np.array(weights)


def efficient_frontier_no_short(mu, Sigma, n_points=50):
    """Efficient frontier with w >= 0 (no shorting)."""
    return efficient_frontier(mu, Sigma, n_points=n_points)


def risk_parity_weights(Sigma):
    """Risk parity: inverse-vol weighting then scale so sum = 1."""
    vol = np.sqrt(np.diag(Sigma))
    w = 1.0 / (vol + 1e-12)
    return w / w.sum()


def run_demo(n_assets=5, n_obs=252, seed=42):
    """Synthetic μ, Σ; compute frontier and risk parity; plot. Requires scipy for frontier."""
    if seed is not None:
        np.random.seed(seed)
    A = np.random.standard_normal((n_assets, n_assets))
    Sigma = A @ A.T + 0.1 * np.eye(n_assets)
    mu = 0.05 + 0.1 * np.random.standard_normal(n_assets)
    rets, vols, weights = efficient_frontier_no_short(mu, Sigma, n_points=40)
    valid = ~np.isnan(vols)
    rets, vols = rets[valid], vols[valid]
    w_rp = risk_parity_weights(Sigma)
    ret_rp = mu @ w_rp
    vol_rp = np.sqrt(w_rp @ Sigma @ w_rp)
    print("Portfolio optimization demo")
    print("  Efficient frontier: {} points (install scipy if empty)".format(len(rets)))
    print("  Risk parity: return={:.4f} vol={:.4f}".format(ret_rp, vol_rp))
    if _HAS_MPL:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        if len(rets) > 0:
            ax.plot(vols, rets, "b-", label="Efficient frontier")
        ax.scatter([vol_rp], [ret_rp], color="red", s=80, label="Risk parity", zorder=5)
        ax.scatter(np.sqrt(np.diag(Sigma)), mu, color="gray", alpha=0.7, label="Assets")
        ax.set_xlabel("Volatility"); ax.set_ylabel("Return")
        ax.legend(); ax.grid(True, alpha=0.3)
        ax.set_title("Efficient frontier (min variance for target return)")
        plt.tight_layout()
        plt.show(block=True)
    return {"mu": mu, "Sigma": Sigma, "rets": rets, "vols": vols, "weights": weights, "w_rp": w_rp}


if __name__ == "__main__":
    run_demo(n_assets=5, seed=42)
