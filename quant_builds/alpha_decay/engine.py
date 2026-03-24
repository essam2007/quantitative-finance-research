"""
Alpha Decay Research: measure strategy decay after publication.

Steps (conceptual):
- Collect factor strategies; define pre- and post-publication windows
- Backtest pre-publication → Sharpe_prev
- Backtest post-publication → Sharpe_post
- Measure Sharpe decay (e.g. Sharpe_post / Sharpe_prev or difference)

This module provides synthetic backtests and decay metrics for an academic-style study.
"""

import numpy as np

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


def sharpe(returns, ann=252):
    if len(returns) < 2:
        return 0.0
    return np.sqrt(ann) * np.mean(returns) / (np.std(returns, ddof=1) + 1e-12)


def simulate_factor_returns(n_obs, sharpe_target=0.5, decay_after=None, decay_slope=0.5, seed=42):
    """
    Synthetic factor returns: before decay_after index we have one vol/sharpe;
    after, Sharpe decays (e.g. lower mean return).
    """
    if seed is not None:
        np.random.seed(seed)
    vol = 0.15 / np.sqrt(252)
    ret = vol * np.random.standard_normal(n_obs)
    if decay_after is None:
        decay_after = n_obs // 2
    # Before: add drift for sharpe_target
    ret[:decay_after] += sharpe_target * vol / np.sqrt(252)
    # After: drift decays
    for i in range(decay_after, n_obs):
        decay = decay_slope ** ((i - decay_after) / (n_obs - decay_after) * 5)
        ret[i] += sharpe_target * vol * decay / np.sqrt(252)
    return ret


def alpha_decay_report(pre_returns, post_returns):
    """Compute Sharpe pre/post and decay metrics."""
    sh_pre = sharpe(pre_returns)
    sh_post = sharpe(post_returns)
    ratio = sh_post / sh_pre if abs(sh_pre) > 1e-10 else np.nan
    diff = sh_post - sh_pre
    return {"Sharpe_pre": sh_pre, "Sharpe_post": sh_post, "Sharpe_ratio_post_pre": ratio, "Sharpe_diff": diff}


def run_demo(n_pre=252*5, n_post=252*5, sharpe_target=0.6, decay_slope=0.4, seed=42):
    """Synthetic alpha decay: one strategy pre/post 'publication'."""
    pre = simulate_factor_returns(n_pre, sharpe_target=sharpe_target, decay_after=None, seed=seed)
    post = simulate_factor_returns(n_post, sharpe_target=sharpe_target, decay_after=int(n_post * 0.1), decay_slope=decay_slope, seed=seed + 1)
    report = alpha_decay_report(pre, post)
    print("Alpha decay research (synthetic)")
    print("  Sharpe pre-publication:  {:.4f}".format(report["Sharpe_pre"]))
    print("  Sharpe post-publication: {:.4f}".format(report["Sharpe_post"]))
    print("  Ratio (post/pre):        {:.4f}".format(report["Sharpe_ratio_post_pre"]))
    print("  Difference:             {:.4f}".format(report["Sharpe_diff"]))
    if _HAS_MPL:
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.plot(np.cumsum(pre), label="Pre-publication (cum return)")
        ax.plot(np.arange(len(pre), len(pre) + len(post)), np.cumsum(post), label="Post-publication")
        ax.axvline(len(pre), color="gray", ls="--", label="Publication")
        ax.set_xlabel("Time"); ax.legend(); ax.grid(True, alpha=0.3)
        ax.set_title("Alpha decay: cumulative returns pre vs post")
        plt.tight_layout()
        plt.show(block=True)
    return report


if __name__ == "__main__":
    run_demo(n_pre=252*5, n_post=252*5, seed=42)
