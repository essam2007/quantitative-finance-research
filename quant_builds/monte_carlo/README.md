# Monte Carlo Option Pricing Engine

Price derivatives by simulation:

**European call:**  
C = e^{-rT} E[max(S_T − K, 0)]

## Implementation

1. **Simulate price paths** — GBM: S_T = S_0 exp((r − σ²/2)T + σ W_T)
2. **Compute payoff** — e.g. max(S_T − K, 0) for call
3. **Discount expected value** — e^{-rT} × sample mean of payoffs

## Variance reduction

- **Antithetic sampling:** pair each normal draw Z with −Z; average payoffs of the two paths to reduce variance.
- **Control variates:** use known E[S_T] = S_0 e^{rT}; adjust the estimator with the covariance between payoff and S_T to lower variance.

## Usage

```python
from quant_builds.monte_carlo import price_european_call, price_european_put, black_scholes_call

S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2

# Plain MC
price, std_err = price_european_call(S0, K, T, r, sigma, n_paths=100_000, seed=42)

# With antithetic variates
price_anti, se_anti = price_european_call(..., use_antithetic=True)

# With control variate (S_T)
price_cv, se_cv = price_european_call(..., use_control_variate=True)

# Compare to Black-Scholes
bs = black_scholes_call(S0, K, T, r, sigma)
```

Run from repo root (prints prices, then **opens a window** with simulated paths and S_T distribution):

```bash
python quant_builds/monte_carlo/engine.py
```

To only open the path plot from code:

```python
from quant_builds.monte_carlo import run_path_demo
run_path_demo(S0=100, K=100, T=1, r=0.05, sigma=0.2, n_paths=500, n_steps=252)
```
