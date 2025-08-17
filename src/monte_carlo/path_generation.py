from __future__ import annotations

__all__ = ['gbm_path_generation', 'mc_price', 'plot_paths_density', 'MCParams']

from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np
import matplotlib.pyplot as plt

Array = np.ndarray

@dataclass
class MCParams:
    S0: float              # starting price
    sigma: float           # volatility (annualized)
    T: float               # time to maturity in years
    M: int                 # number of paths
    steps: int = 252       # time steps
    r: float = 0.0         # risk-free rate
    antithetic: bool = False
    seed: Optional[int] = None
    numgen: str = "normal"


def gbm_path_generation(cfg: MCParams) -> Array:
    """
    Generate GBM paths under risk-neutral measure:
        dS_t = r S_t dt + sigma S_t dW_t
    Discretization (log-Euler):
        S_{t+dt} = S_t * exp((r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)

    Parameters
    ----------
    cfg: MCParams
        Monte Carlo parameters

    Return
    ------
    paths: np.ndarray
        shape (M, steps+1) including S0 at column 0.
    """
    if cfg.S0 <= 0:
        raise ValueError("S0 must be positive.")
    if cfg.sigma < 0:
        raise ValueError("sigma must be non-negative.")
    if cfg.T < 0:
        raise ValueError("T must be non-negative.")
    if cfg.M <= 0 or cfg.steps <= 0:
        raise ValueError("M and steps must be positive integers.")

    dt = cfg.T / cfg.steps
    drift = (cfg.r - 0.5 * cfg.sigma**2) * dt
    vol_step = cfg.sigma * np.sqrt(dt)

    rng = np.random.default_rng(cfg.seed)

    M_eff = cfg.M // 2 if cfg.antithetic else cfg.M
    if cfg.antithetic and cfg.M % 2 != 0:
        raise ValueError("For antithetic=True, use an even M.")

    Z_base = rng.standard_normal((M_eff, cfg.steps))

    if cfg.antithetic:
        Z = np.vstack([Z_base, -Z_base])  # antithetic pairs
    else:
        Z = Z_base

    # build paths
    increments = drift + vol_step * Z
    log_increments = increments.cumsum(axis=1)
    S_paths = np.empty((cfg.M, cfg.steps + 1), dtype=float)
    S_paths[:, 0] = cfg.S0
    S_paths[:, 1:] = cfg.S0 * np.exp(log_increments)

    return S_paths



def mc_price(paths: Array, payoff: Callable[[Array], Array], r: float, T: float) -> float:
    """
    Calculates Option Price with Monte Carlo.

    Parameters
    ----------
    paths : np.ndarray
        Simulated paths, shape (M, steps+1).
    payoff : Callable[[Array], Array]
        Payoff function. Should accept either terminal prices (shape (M,)) or full paths (shape (M, steps+1))
    r : float
        Risk-free rate (continuous compounding)
    T : float
        time to maturity

    Return
    ------
    Price - discounted expectation of payoff.
    """
    try:
        pay = payoff(paths[:, -1])  # terminal-only payoff
    except Exception:
        pay = payoff(paths)         # path-dependent payoff
    pay = np.asarray(pay, dtype=float)
    if pay.shape[0] != paths.shape[0]:
        raise ValueError("Payoff must return one value per path.")
    disc = np.exp(-r * T)
    return float(disc * pay.mean())



def plot_paths_density(paths: Array, n_paths: int = 200, title: str = "Simulated Paths",
                       base_color: str = "blue", alpha: float = 0.1, T: Optional[float] = None) -> None:
    """
    Plot Monte Carlo paths with density effect.

    Parameters
    ----------
    paths : np.ndarray
        Simulated paths, shape (M, steps+1).
    n_paths : int, optional
        Number of paths to plot (default: 200).
    title : str, optional
        Plot title.
    base_color : str, optional
        Color for all paths (default: "blue").
    alpha : float, optional
        Transparency for density effect.
    T : float, optional
        If provided, time axis will be [0, T] instead of normalized [0, 1].
    """
    M, steps_plus1 = paths.shape
    n_paths = min(n_paths, M)
    idx = np.random.choice(M, size=n_paths, replace=False)

    if T is None:
        time_grid = np.linspace(0, 1, steps_plus1)
    else:
        time_grid = np.linspace(0, T, steps_plus1)

    plt.figure(figsize=(10, 6))
    for i in idx:
        plt.plot(time_grid, paths[i], color=base_color, alpha=alpha, linewidth=1)

    plt.xlabel("Time (years)" if T is not None else "Normalized Time")
    plt.ylabel("Underlying Price")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()