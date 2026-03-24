"""
Ornstein-Uhlenbeck mean reversion strategy (pairs/spread trading).
"""

from .engine import (
    simulate_ou,
    estimate_ou_params,
    zscore,
    mean_reversion_signal,
    backtest_spread,
    sharpe_ratio,
    run_demo,
)

__all__ = [
    "simulate_ou",
    "estimate_ou_params",
    "zscore",
    "mean_reversion_signal",
    "backtest_spread",
    "sharpe_ratio",
    "run_demo",
]
