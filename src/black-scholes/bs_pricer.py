from __future__ import annotations

__all__ = []

import numpy as np
from typing import Literal

Array = np.ndarray

def _norm_cdf(x: Array) -> Array:
    return (1.0 + np.erf(x / np.sqrt(2.0))) / 2.0

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