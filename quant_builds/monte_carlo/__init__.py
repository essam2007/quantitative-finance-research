"""
Monte Carlo option pricing engine.

Price derivatives by simulation: C = e^{-rT} E[payoff(S_T)].
Includes variance reduction: antithetic sampling, control variates.
"""

from .engine import (
    simulate_gbm_paths,
    european_call_payoff,
    european_put_payoff,
    price_european_option,
    price_european_call,
    price_european_put,
    black_scholes_call,
    run_path_demo,
)

__all__ = [
    "simulate_gbm_paths",
    "european_call_payoff",
    "european_put_payoff",
    "price_european_option",
    "price_european_call",
    "price_european_put",
    "black_scholes_call",
    "run_path_demo",
]
