# Quant Builds

Quantitative finance building blocks (chronological order).

| # | Project | Description |
|---|---------|-------------|
| 1 | [monte_carlo](monte_carlo/) | Monte Carlo option pricing (European, antithetic, control variates) |
| 4 | [ou_mean_reversion](ou_mean_reversion/) | Ornstein-Uhlenbeck mean reversion (pairs/spread trading, backtest, Sharpe) |
| 5 | [fama_french](fama_french/) | Fama-French factor replication (Market, SMB, HML, Momentum; regressions) |
| 6 | [order_book](order_book/) | Limit order book simulator (bid/ask, market/limit orders, price impact) |
| 7 | [kelly](kelly/) | Kelly criterion (full/fractional/fixed; drawdowns, ruin probability) |
| 8 | [heston](heston/) | Heston stochastic volatility (simulate vs Black-Scholes) |
| 9 | [alpha_decay](alpha_decay/) | Alpha decay research (pre/post publication Sharpe; notebook) |
| 10 | [portfolio_optimization](portfolio_optimization/) | Markowitz efficient frontier, risk parity |

Run each from repo root, e.g.:

```bash
python quant_builds/monte_carlo/engine.py
python quant_builds/ou_mean_reversion/engine.py
python quant_builds/fama_french/engine.py
python quant_builds/order_book/engine.py
python quant_builds/kelly/engine.py
python quant_builds/heston/engine.py
python quant_builds/alpha_decay/engine.py
python quant_builds/portfolio_optimization/engine.py   # requires scipy
```
