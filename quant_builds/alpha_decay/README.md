# Alpha Decay Research Project

Measure **alpha decay** after publication (academic-style).

## Steps

1. Collect factor strategies; define pre- and post-publication windows
2. Backtest pre-publication → Sharpe_prev
3. Backtest post-publication → Sharpe_post
4. Report: Sharpe ratio decay (post/pre or difference)

## Usage

```python
from quant_builds.alpha_decay import simulate_factor_returns, alpha_decay_report, run_demo

pre = simulate_factor_returns(252*5, sharpe_target=0.5, seed=42)
post = simulate_factor_returns(252*5, sharpe_target=0.5, decay_after=100, decay_slope=0.5, seed=43)
report = alpha_decay_report(pre, post)
```

Run: `python quant_builds/alpha_decay/engine.py`

For full research: plug in real factor returns and event dates (e.g. paper publication) to compute pre/post Sharpes and produce an academic-style notebook.
