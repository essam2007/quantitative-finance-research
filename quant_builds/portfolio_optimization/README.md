# Portfolio Optimization Engine

**Markowitz:** min w'Σw  subject to  w'μ = r,  1'w = 1,  w ≥ 0.

- **Efficient frontier**: solve for each target return; plot volatility vs return.
- **Risk parity**: weights ∝ 1/vol (per asset); plot on same figure.

Extensions (in code): transaction costs, Black-Litterman can be added.

## Usage

```python
from quant_builds.portfolio_optimization import efficient_frontier, risk_parity_weights

import numpy as np
mu = np.array([0.08, 0.10, 0.06])
Sigma = np.array([[0.04, 0.01, 0.01], [0.01, 0.09, 0.02], [0.01, 0.02, 0.04]])
rets, vols, weights = efficient_frontier(mu, Sigma, n_points=30)
w_rp = risk_parity_weights(Sigma)
```

Requires **scipy** for the frontier. Run: `python quant_builds/portfolio_optimization/engine.py`
