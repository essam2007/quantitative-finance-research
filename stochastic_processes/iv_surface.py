"""
Implied volatility surface simulation using the Black-Scholes formula.

Builds a 3D implied volatility surface (strike K, time-to-maturity T, implied vol σ)
by generating synthetic European option prices from a parametric volatility model,
then inverting the Black-Scholes formula to recover implied volatility at each (K, T).
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D


def _norm_cdf(x):
    """Standard normal CDF using math.erf (no scipy)."""
    x = np.asarray(x)
    scalar = x.ndim == 0
    x = np.atleast_1d(x)
    out = 0.5 * (1.0 + np.vectorize(lambda z: math.erf(z / math.sqrt(2)))(x))
    return out.item() if scalar else out


# -----------------------------------------------------------------------------
# Black-Scholes formula
# -----------------------------------------------------------------------------

def black_scholes_call(S, K, T, r, sigma, q=0.0):
    """
    European call option price under Black-Scholes.

    Parameters
    ----------
    S : float or ndarray
        Spot price
    K : float or ndarray
        Strike price
    T : float or ndarray
        Time to maturity (in years)
    r : float
        Risk-free rate (continuous)
    sigma : float or ndarray
        Volatility (annualized)
    q : float
        Continuous dividend yield (default 0)

    Returns
    -------
    float or ndarray
        Call option price
    """
    T = np.asarray(T, dtype=float)
    intrinsic = np.maximum(S - K, 0.0)
    safe_T = np.maximum(T, 1e-10)
    sqrtT = np.sqrt(safe_T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    price = S * np.exp(-q * T) * _norm_cdf(d1) - K * np.exp(-r * T) * _norm_cdf(d2)
    result = np.where(T <= 0, intrinsic, price)
    return result if result.ndim > 0 else float(result)


def black_scholes_put(S, K, T, r, sigma, q=0.0):
    """European put option price under Black-Scholes (put-call parity)."""
    if T <= 0:
        return np.maximum(K - S, 0.0)
    return K * np.exp(-r * T) - S * np.exp(-q * T) + black_scholes_call(S, K, T, r, sigma, q)


def implied_volatility_call(price, S, K, T, r, q=0.0, tol=1e-6, max_iter=100):
    """
    Implied volatility of a European call from its market price.

    Solves BS_call(S, K, T, r, sigma) = price for sigma via bisection.

    Parameters
    ----------
    price : float
        Observed call option price
    S, K, T, r, q : float
        Spot, strike, time to maturity, rate, dividend yield
    tol, max_iter : optional
        Solver tolerance and iteration cap

    Returns
    -------
    float
        Implied volatility (annualized), or np.nan if no solution.
    """
    if T <= 0 or price <= 0:
        return np.nan
    intrinsic = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    if price < intrinsic - 1e-10:
        return np.nan
    sigma_low, sigma_high = 1e-6, 5.0
    if black_scholes_call(S, K, T, r, sigma_high, q) < price:
        return np.nan
    for _ in range(max_iter):
        sigma_mid = (sigma_low + sigma_high) * 0.5
        if (sigma_high - sigma_low) < tol:
            return sigma_mid
        p_mid = black_scholes_call(S, K, T, r, sigma_mid, q)
        if p_mid < price:
            sigma_low = sigma_mid
        else:
            sigma_high = sigma_mid
    return (sigma_low + sigma_high) * 0.5


# -----------------------------------------------------------------------------
# Simulated volatility surface (parametric)
# -----------------------------------------------------------------------------

def simulated_vol_surface(K, T, S, r, vol_atm=0.2, smile_strength=0.1, term_slope=0.01):
    """
    Parametric volatility surface: smile in strike + term structure in T.

    σ(K,T) = base(T) * (1 + smile_strength * (K/S - 1)^2)
    base(T) = vol_atm + term_slope * T  (simple upward term structure).

    Parameters
    ----------
    K, T : ndarray
        Strike and time-to-maturity grids (broadcastable)
    S, r : float
        Spot and risk-free rate
    vol_atm : float
        At-the-money volatility at T≈0
    smile_strength : float
        Strength of smile (strike effect)
    term_slope : float
        Slope of vol vs T

    Returns
    -------
    ndarray
        Volatility at each (K, T)
    """
    base_vol = vol_atm + term_slope * T
    moneyness = K / S
    smile = 1.0 + smile_strength * (moneyness - 1.0) ** 2
    return base_vol * smile


# -----------------------------------------------------------------------------
# Build (K, T) grid and compute implied volatilities
# -----------------------------------------------------------------------------

def build_iv_surface(S=100.0, r=0.05, q=0.0, n_strikes=25, n_maturities=15,
                    vol_atm=0.2, smile_strength=0.08, term_slope=0.01,
                    strike_min_ratio=0.7, strike_max_ratio=1.3, T_min=0.1, T_max=2.0,
                    seed=None):
    """
    Build a 3D implied volatility surface using simulated option prices.

    For each (K, T) we set a "true" vol from a parametric surface, compute
    the Black-Scholes call price, then invert to get implied vol (so the
    resulting IV surface matches the parametric design, up to numerical error).

    Parameters
    ----------
    S, r, q : float
        Spot, risk-free rate, dividend yield
    n_strikes, n_maturities : int
        Grid size for strikes and maturities
    vol_atm, smile_strength, term_slope : float
        Parametric surface parameters
    strike_min_ratio, strike_max_ratio : float
        Strike range as multiple of spot (K in [S*strike_min_ratio, S*strike_max_ratio])
    T_min, T_max : float
        Maturity range in years
    seed : int or None
        Random seed (unused; for future extensions)

    Returns
    -------
    K_grid : ndarray (n_strikes, n_maturities)
        Strike grid
    T_grid : ndarray (n_strikes, n_maturities)
        Time-to-maturity grid
    IV_grid : ndarray (n_strikes, n_maturities)
        Implied volatility at each (K, T)
    """
    if seed is not None:
        np.random.seed(seed)
    strikes = np.linspace(S * strike_min_ratio, S * strike_max_ratio, n_strikes)
    maturities = np.linspace(T_min, T_max, n_maturities)
    K_grid, T_grid = np.meshgrid(strikes, maturities, indexing="ij")
    # True vol from parametric surface
    sigma_true = simulated_vol_surface(K_grid, T_grid, S, r, vol_atm, smile_strength, term_slope)
    # Synthetic market price = BS price with true vol
    call_prices = black_scholes_call(S, K_grid, T_grid, r, sigma_true, q)
    # Invert to get implied vol at each (K, T)
    IV_grid = np.empty_like(K_grid)
    for i in range(K_grid.shape[0]):
        for j in range(K_grid.shape[1]):
            iv = implied_volatility_call(
                call_prices[i, j], S, K_grid[i, j], T_grid[i, j], r, q
            )
            IV_grid[i, j] = iv if not np.isnan(iv) else sigma_true[i, j]
    return K_grid, T_grid, IV_grid


# -----------------------------------------------------------------------------
# 3D plot
# -----------------------------------------------------------------------------

def plot_iv_surface_3d(K_grid, T_grid, IV_grid, title="Implied Volatility Surface"):
    """
    Plot the IV surface as a 3D mesh.

    Parameters
    ----------
    K_grid, T_grid, IV_grid : ndarray
        From build_iv_surface
    title : str
        Plot title
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(K_grid, T_grid, IV_grid, cmap="viridis", alpha=0.9, edgecolor="none")
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Time to maturity T (years)")
    ax.set_zlabel("Implied volatility σ")
    ax.set_title(title)
    plt.tight_layout()
    return fig, ax


# -----------------------------------------------------------------------------
# Interactive 3D view (Bloomberg-style: drag to rotate, scroll to zoom)
# -----------------------------------------------------------------------------

def run_interactive():
    """
    Launch an interactive IV surface: rotate/drag the 3D view and adjust
    all parameters via sliders. Use mouse: drag to rotate, scroll to zoom,
    right-drag to pan (platform-dependent).
    """
    fig = plt.figure(figsize=(14, 8))
    # 3D axes: leave room on right for sliders
    ax_3d = fig.add_axes([0.05, 0.15, 0.6, 0.8], projection="3d")
    ax_3d.set_xlabel("Strike K")
    ax_3d.set_ylabel("Time to maturity T (years)")
    ax_3d.set_zlabel("Implied volatility σ")

    # Slider layout: vertical stack on the right
    slider_left = 0.72
    slider_width = 0.24
    slider_height = 0.02
    slider_bottom_start = 0.88
    slider_spacing = 0.032

    def make_slider(bottom, label, vmin, vmax, valinit, valfmt="%.2f"):
        ax = fig.add_axes([slider_left, bottom, slider_width, slider_height])
        s = Slider(ax, label, vmin, vmax, valinit=valinit, valfmt=valfmt)
        return s

    # Sliders: S, r, q, vol_atm, smile_strength, term_slope, strike_lo, strike_hi, T_min, T_max, n_K, n_T
    sliders = {}
    row = slider_bottom_start
    sliders["S"] = make_slider(row, "Spot S", 50, 200, 100); row -= slider_spacing
    sliders["r"] = make_slider(row, "Rate r", 0, 0.15, 0.05); row -= slider_spacing
    sliders["q"] = make_slider(row, "Div yield q", 0, 0.1, 0); row -= slider_spacing
    sliders["vol_atm"] = make_slider(row, "ATM vol", 0.05, 0.5, 0.2); row -= slider_spacing
    sliders["smile"] = make_slider(row, "Smile strength", 0, 0.3, 0.08); row -= slider_spacing
    sliders["term_slope"] = make_slider(row, "Term slope", -0.05, 0.05, 0.01); row -= slider_spacing
    sliders["K_lo"] = make_slider(row, "K min / S", 0.5, 1.0, 0.7); row -= slider_spacing
    sliders["K_hi"] = make_slider(row, "K max / S", 1.0, 1.6, 1.3); row -= slider_spacing
    sliders["T_min"] = make_slider(row, "T min (y)", 0.02, 1.0, 0.1); row -= slider_spacing
    sliders["T_max"] = make_slider(row, "T max (y)", 0.5, 4.0, 2.0); row -= slider_spacing
    sliders["n_K"] = make_slider(row, "Strikes", 8, 45, 22, "%.0f"); row -= slider_spacing
    sliders["n_T"] = make_slider(row, "Maturities", 5, 25, 12, "%.0f")

    surface_art = [None]  # mutable so update can replace

    def update(_=None):
        S = sliders["S"].val
        r = sliders["r"].val
        q = sliders["q"].val
        vol_atm = sliders["vol_atm"].val
        smile = sliders["smile"].val
        term_slope = sliders["term_slope"].val
        K_lo = sliders["K_lo"].val
        K_hi = sliders["K_hi"].val
        T_min = sliders["T_min"].val
        T_max = sliders["T_max"].val
        n_K = int(sliders["n_K"].val)
        n_T = int(sliders["n_T"].val)
        if K_lo >= K_hi:
            K_hi = K_lo + 0.05
        if T_min >= T_max:
            T_max = T_min + 0.1
        K_grid, T_grid, IV_grid = build_iv_surface(
            S=S, r=r, q=q,
            n_strikes=n_K, n_maturities=n_T,
            vol_atm=vol_atm, smile_strength=smile, term_slope=term_slope,
            strike_min_ratio=K_lo, strike_max_ratio=K_hi, T_min=T_min, T_max=T_max,
        )
        # Replace surface (keep 3D view angle)
        if surface_art[0] is not None:
            surface_art[0].remove()
        surface_art[0] = ax_3d.plot_surface(
            K_grid, T_grid, IV_grid, cmap="viridis", alpha=0.9, edgecolor="none"
        )
        ax_3d.set_title("Implied Volatility Surface — drag to rotate, scroll to zoom")
        fig.canvas.draw_idle()

    for s in sliders.values():
        s.on_changed(update)

    # Initial draw
    update()
    plt.show()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--static":
        # Static plot and save (e.g. for CI or screenshots)
        S, r, q = 100.0, 0.05, 0.0
        K_grid, T_grid, IV_grid = build_iv_surface(
            S=S, r=r, q=q,
            n_strikes=25, n_maturities=15,
            vol_atm=0.20, smile_strength=0.08, term_slope=0.01,
            seed=42,
        )
        print("IV surface shape:", IV_grid.shape)
        fig, ax = plot_iv_surface_3d(K_grid, T_grid, IV_grid)
        import os
        _dir = os.path.dirname(os.path.abspath(__file__))
        out_path = os.path.join(_dir, "iv_surface_3d.png")
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        print("Saved:", out_path)
        plt.show()
    else:
        # Interactive: drag to rotate, scroll to zoom, sliders to adjust all parameters
        run_interactive()
