from __future__ import annotations

__all__ = ['bs_pricer']

import numpy as np
from typing import Literal
import math

Array = np.ndarray

def _norm_cdf(x: np.ndarray | float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / np.sqrt(2.0)))

def bs_pricer(S: Array, K: Array, r: Array, T: Array, sigma: Array, option_type: Literal['call', 'put'] = 'call') -> Callable[[Array], Array]:
    """
    Black Scholes pricer for european call/put option .

    Parameters
    ----------
    S : ndarray
      Spot price(s)
    K : ndarray
      Strike price(s)
    r : ndarray
      Risk-free rate(s)
    T : ndarray
      Time(s) to maturity
    sigma : ndarray
      Volatility parameter
    option_type : {'call', 'put'}
      Type of option

    Return
    ------
    np.ndarray
      Price of option(s)
    """

    option_type = option_type.strip().lower()
    if option_type not in {'call', 'put'}:
        raise ValueError('option_type must be "call" or "put"')

    d1 = ((np.log(S) - np.log(K)) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S * _norm_cdf(d1) - K * np.exp(-r * T) * _norm_cdf(d2)
        return price

    elif option_type == 'put':
        price = K * np.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)
        return price