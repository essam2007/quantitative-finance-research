"""
Fama-French Factor Model Replication.

Goal: Recreate factors — Market, SMB (size), HML (value), Momentum.
Steps: (synthetic or loaded) returns → sort into portfolios → factor returns → regressions.
Output: R_i − R_f = α + β F + ε (econometrics + portfolio theory).
"""

import numpy as np

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


# -----------------------------------------------------------------------------
# Synthetic dataset (when no CSV/yfinance)
# -----------------------------------------------------------------------------

def generate_synthetic_returns(n_obs=252, n_stocks=100, market_beta=1.0, size_load=0.3, value_load=0.2, mom_load=0.1, seed=42):
    """
    Generate panel of stock returns and factor returns for demo.
    Returns: (stock_returns (n_obs, n_stocks), Rf, Rm, SMB, HML, MOM).
    """
    if seed is not None:
        np.random.seed(seed)
    T, N = n_obs, n_stocks
    Rf = 0.02 / 252
    # Factor returns (daily)
    Rm = 0.05 / 252 + 0.01 * np.random.standard_normal(T)
    SMB = 0.01 * np.random.standard_normal(T)
    HML = 0.008 * np.random.standard_normal(T)
    MOM = 0.006 * np.random.standard_normal(T)
    # Loadings: random per stock
    betas_m = market_beta + 0.3 * np.random.standard_normal(N)
    betas_s = size_load + 0.2 * np.random.standard_normal(N)
    betas_v = value_load + 0.2 * np.random.standard_normal(N)
    betas_mom = mom_load + 0.15 * np.random.standard_normal(N)
    # Idiosyncratic
    eps = 0.01 * np.random.standard_normal((T, N))
    R = Rf + np.outer(Rm - Rf, betas_m) + np.outer(SMB, betas_s) + np.outer(HML, betas_v) + np.outer(MOM, betas_mom) + eps
    return R, Rf, Rm, SMB, HML, MOM


# -----------------------------------------------------------------------------
# Factor returns from sorted portfolios (simplified)
# -----------------------------------------------------------------------------

def compute_smb_hml_mom_simple(R, Rm, Rf, size_rank=None, value_rank=None, mom_rank=None):
    """
    Placeholder: given returns R (T x N), compute SMB/HML/MOM as portfolio returns.
    In practice: sort stocks by size → SMB = small minus big; by B/M → HML; by past return → MOM.
    Here we return synthetic factors if rank arrays not provided.
    """
    T, N = R.shape
    if size_rank is None:
        size_rank = np.random.permutation(N)  # random size
    if value_rank is None:
        value_rank = np.random.permutation(N)
    if mom_rank is None:
        mom_rank = np.random.permutation(N)
    n_small = max(1, N // 3)
    n_big = max(1, N - n_small)
    # SMB: small minus big (by size_rank)
    small_idx = np.argsort(size_rank)[:n_small]
    big_idx = np.argsort(size_rank)[-n_big:]
    SMB = (R[:, small_idx].mean(axis=1) - R[:, big_idx].mean(axis=1))
    # HML: high B/M minus low (by value_rank)
    high_idx = np.argsort(value_rank)[-n_small:]
    low_idx = np.argsort(value_rank)[:n_small]
    HML = (R[:, high_idx].mean(axis=1) - R[:, low_idx].mean(axis=1))
    # MOM: winners minus losers (by mom_rank)
    win_idx = np.argsort(mom_rank)[-n_small:]
    lose_idx = np.argsort(mom_rank)[:n_small]
    MOM = (R[:, win_idx].mean(axis=1) - R[:, lose_idx].mean(axis=1))
    return SMB, HML, MOM


# -----------------------------------------------------------------------------
# Regression: R_i - R_f = α + β_m (R_m - R_f) + β_s SMB + β_h HML + β_mom MOM + ε
# -----------------------------------------------------------------------------

def factor_regression(R_i, Rf, Rm, SMB, HML, MOM=None):
    """
    OLS: excess return on factors. R_i (T,), Rf/Rm/SMB/HML/MOM (T,).
    Returns alpha, betas (dict), R2, residuals.
    """
    y = np.asarray(R_i) - np.asarray(Rf)
    X = np.column_stack([
        np.asarray(Rm) - np.asarray(Rf),
        np.asarray(SMB),
        np.asarray(HML),
    ])
    if MOM is not None:
        X = np.column_stack([X, np.asarray(MOM)])
    X = np.column_stack([np.ones(len(y)), X])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    alpha = b[0]
    betas = {"Mkt-Rf": b[1], "SMB": b[2], "HML": b[3]}
    if MOM is not None:
        betas["MOM"] = b[4]
    yhat = X @ b
    resid = y - yhat
    ss_tot = np.sum((y - np.mean(y))**2)
    ss_res = np.sum(resid**2)
    R2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    return alpha, betas, R2, resid


# -----------------------------------------------------------------------------
# Run full pipeline and report
# -----------------------------------------------------------------------------

def run_demo(n_obs=252, n_stocks=50, seed=42):
    """Generate data, run factor regression on a few stocks, print and optionally plot."""
    R, Rf, Rm, SMB, HML, MOM = generate_synthetic_returns(n_obs=n_obs, n_stocks=n_stocks, seed=seed)
    SMB_p, HML_p, MOM_p = compute_smb_hml_mom_simple(R, Rm, Rf)
    # Regress first 5 stocks
    results = []
    for j in range(min(5, n_stocks)):
        alpha, betas, R2, _ = factor_regression(R[:, j], Rf, Rm, SMB_p, HML_p, MOM_p)
        results.append({"alpha": alpha, "betas": betas, "R2": R2})
    if _HAS_MPL:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        names = ["Mkt-Rf", "SMB", "HML", "MOM"]
        for i, r in enumerate(results):
            b = [r["betas"].get(k, 0) for k in names]
            ax.bar(np.arange(4) + i * 0.2, b, width=0.2, label=f"Stock {i+1} R2={r['R2']:.2f}")
        ax.set_xticks(np.arange(4) + 0.3); ax.set_xticklabels(names)
        ax.axhline(0, color="gray", ls="-")
        ax.legend(); ax.set_ylabel("β"); ax.set_title("Fama-French factor loadings (synthetic)")
        plt.tight_layout()
        plt.show(block=True)
    return {"R": R, "Rf": Rf, "Rm": Rm, "SMB": SMB_p, "HML": HML_p, "MOM": MOM_p, "results": results}


if __name__ == "__main__":
    print("Fama-French factor replication (synthetic data)")
    out = run_demo(n_obs=252, n_stocks=50, seed=42)
    for i, r in enumerate(out["results"]):
        print("  Stock {}: α={:.6f}  β={}  R2={:.4f}".format(i + 1, r["alpha"], r["betas"], r["R2"]))
