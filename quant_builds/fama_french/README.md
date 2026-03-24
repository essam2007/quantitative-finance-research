# Fama-French Factor Model Replication

Recreate classic factors: **Market**, **SMB** (size), **HML** (value), **Momentum**.

## Steps

1. Stock returns (synthetic or load CSV / yfinance)
2. Sort into portfolios by size, value, momentum
3. Compute factor returns (e.g. SMB = small minus big)
4. Run regression: **R_i − R_f = α + β F + ε**

## Usage

```python
from quant_builds.fama_french import generate_synthetic_returns, factor_regression, compute_smb_hml_mom_simple

R, Rf, Rm, SMB, HML, MOM = generate_synthetic_returns(n_obs=252, n_stocks=100, seed=42)
SMB_p, HML_p, MOM_p = compute_smb_hml_mom_simple(R, Rm, Rf)
alpha, betas, R2, resid = factor_regression(R[:, 0], Rf, Rm, SMB_p, HML_p, MOM_p)
```

Run: `python quant_builds/fama_french/engine.py`

For real data: replace `generate_synthetic_returns` with your loader (e.g. pandas + yfinance) and pass returns + rankings into `compute_smb_hml_mom_simple` and `factor_regression`.
