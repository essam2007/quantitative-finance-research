# Heston Stochastic Volatility Model

dS_t = μ S_t dt + √v_t S_t dW_t  
dv_t = κ(θ − v_t)dt + σ√v_t dZ_t  with  corr(W,Z) = ρ

## Features

- **Simulation**: Euler-Maruyama for (S, v); variance reflected at 0
- **Comparison**: side-by-side with Black-Scholes (constant vol √θ)
- **Volatility**: plot √v_t vs constant σ

## Usage

```python
from quant_builds.heston import simulate_heston, simulate_bs

t, S, v = simulate_heston(S0=100, v0=0.04, mu=0.05, kappa=2, theta=0.04, sigma_vol=0.3, rho=-0.5, T=1, n_steps=252, n_paths=1000, seed=42)
t_bs, S_bs = simulate_bs(100, 0.05, 0.2, 1, 252, 1000, seed=42)
```

Run: `python quant_builds/heston/engine.py`
