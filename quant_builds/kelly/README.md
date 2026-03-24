# Kelly Criterion Portfolio Simulator

Optimal bet fraction: **f* = [p(b+1) − 1] / b** (p = win prob, b = odds).

## Features

- **Full / fractional / fixed** allocation
- **Drawdowns**: peak-to-trough
- **Long-term wealth** paths (simulated)
- **Ruin probability**: fraction of paths below threshold

## Usage

```python
from quant_builds.kelly import kelly_fraction, simulate_betting, drawdown, ruin_probability

f_star = kelly_fraction(p=0.55, b=1.0)
W = simulate_betting(500, f=0.5*f_star, p=0.55, b=1.0, seed=42)
dd = drawdown(W)
ruin = ruin_probability(500, f_star, 0.55, 1.0, n_sims=5000)
```

Run: `python quant_builds/kelly/engine.py`
