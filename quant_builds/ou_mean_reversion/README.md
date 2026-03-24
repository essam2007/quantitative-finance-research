# Ornstein-Uhlenbeck Mean Reversion Strategy

Used in **statistical arbitrage**: pairs trading, spread trading.

## Model

dX_t = θ(μ − X_t)dt + σ dW_t

- θ: mean reversion speed  
- μ: long-run mean  
- σ: volatility  

## Features

- **Parameter estimation**: (θ, μ, σ) via AR(1) regression on discrete observations  
- **Deviation detection**: z-score of spread vs estimated μ, σ  
- **Trading**: long spread when z < -entry_z, short when z > entry_z; exit when |z| < exit_z  
- **Backtesting**: P&L from spread returns × position  
- **Sharpe ratio**: annualized  

## Usage

```python
from quant_builds.ou_mean_reversion import simulate_ou, estimate_ou_params, mean_reversion_signal, backtest_spread, sharpe_ratio

t, X = simulate_ou(T=2, dt=1/252, theta=2, mu=0, sigma=0.5, seed=42)
theta, mu, sigma = estimate_ou_params(X, 1/252)
position = mean_reversion_signal(X, mu=mu, sigma=sigma, entry_z=2.0)
pnl = backtest_spread(X, position)
print("Sharpe", sharpe_ratio(pnl, periods_per_year=252))
```

Run demo (synthetic OU + plot):  
`python quant_builds/ou_mean_reversion/engine.py`
