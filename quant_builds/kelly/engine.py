"""
Kelly Criterion Portfolio Simulator.

Optimal fraction: f* = [p(b+1) − 1] / b  (win prob p, payoff odds b).
Simulate: full Kelly, fractional Kelly, fixed allocation.
Analyze: drawdowns, long-run wealth growth, ruin probability.
"""

import numpy as np

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


# -----------------------------------------------------------------------------
# Kelly formula
# -----------------------------------------------------------------------------

def kelly_fraction(p, b):
    """
    p = P(win), b = odds (win pays b per unit staked).
    f* = [p(b+1) − 1] / b. Return 0 if f* <= 0.
    """
    if b <= 0:
        return 0.0
    f = (p * (b + 1) - 1) / b
    return max(0.0, min(1.0, f))


# -----------------------------------------------------------------------------
# Wealth path simulation
# -----------------------------------------------------------------------------

def simulate_betting(n_trials, f, p, b, W0=1.0, seed=None):
    """
    Each trial: with prob p win (wealth *= 1 + f*b), else lose (wealth *= 1 - f).
    Returns wealth path (length n_trials+1).
    """
    if seed is not None:
        np.random.seed(seed)
    W = np.zeros(n_trials + 1)
    W[0] = W0
    wins = np.random.rand(n_trials) < p
    for i in range(n_trials):
        if wins[i]:
            W[i + 1] = W[i] * (1 + f * b)
        else:
            W[i + 1] = W[i] * (1 - f)
    return W


def drawdown(path):
    """Peak-to-trough drawdown at each time."""
    peak = np.maximum.accumulate(path)
    return (peak - path) / (peak + 1e-12)


def ruin_probability(n_trials, f, p, b, n_sims=5000, ruin_level=0.01, seed=None):
    """Fraction of paths that fall below ruin_level * W0."""
    if seed is not None:
        np.random.seed(seed)
    ruined = 0
    for _ in range(n_sims):
        W = simulate_betting(n_trials, f, p, b, W0=1.0, seed=None)
        if np.min(W) < ruin_level:
            ruined += 1
    return ruined / n_sims


# -----------------------------------------------------------------------------
# Compare full Kelly, fractional, fixed
# -----------------------------------------------------------------------------

def run_demo(p=0.55, b=1.0, n_trials=500, n_sims=2000, seed=42):
    """Compare full Kelly, half Kelly, quarter Kelly, fixed 10%."""
    f_star = kelly_fraction(p, b)
    strategies = [
        ("Full Kelly", f_star),
        ("Half Kelly", 0.5 * f_star),
        ("Quarter Kelly", 0.25 * f_star),
        ("Fixed 10%", 0.10),
    ]
    if seed is not None:
        np.random.seed(seed)
    results = {}
    for name, f in strategies:
        paths = np.array([simulate_betting(n_trials, f, p, b, seed=None) for _ in range(n_sims)])
        results[name] = {
            "f": f,
            "mean_final": np.mean(paths[:, -1]),
            "median_final": np.median(paths[:, -1]),
            "max_dd_mean": np.mean(np.max(drawdown(paths), axis=1)),
            "ruin": ruin_probability(n_trials, f, p, b, n_sims=n_sims, seed=None),
            "paths": paths,
        }
    print("Kelly Criterion simulator (p={}, b={}, f*={:.4f})".format(p, b, f_star))
    for name, r in results.items():
        print("  {}: f={:.4f}  E[W_end]={:.4f}  mean max DD={:.2%}  ruin P={:.2%}".format(
            name, r["f"], r["mean_final"], r["max_dd_mean"], r["ruin"]))
    if _HAS_MPL:
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        for name, r in results.items():
            mean_path = np.mean(r["paths"], axis=0)
            axes[0].plot(mean_path, label=name)
        axes[0].set_ylabel("Mean wealth"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
        for name, r in results.items():
            dd = drawdown(r["paths"])
            axes[1].plot(np.mean(dd, axis=0), label=name)
        axes[1].set_ylabel("Mean drawdown"); axes[1].set_xlabel("Trial"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
        plt.suptitle("Kelly: wealth growth and drawdowns")
        plt.tight_layout()
        plt.show(block=True)
    return results


if __name__ == "__main__":
    run_demo(p=0.55, b=1.0, n_trials=500, n_sims=2000, seed=42)
