"""
Fama-French factor model replication (Market, SMB, HML, Momentum).
"""

from .engine import (
    generate_synthetic_returns,
    compute_smb_hml_mom_simple,
    factor_regression,
    run_demo,
)

__all__ = [
    "generate_synthetic_returns",
    "compute_smb_hml_mom_simple",
    "factor_regression",
    "run_demo",
]
