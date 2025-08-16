__all__ = ['european_payoff', 'digital_payoff']

import numpy as np
from typing import Callable, Literal

Array = np.ndarray

def _as_terminal(S: Array) -> Array:
    """Accept (M,) or (M, steps+1); return terminal (M,)."""
    S = np.asarray(S, dtype=float)
    return S if S.ndim == 1 else S[:, -1]

"""--------------------------------- PAYOFF FUNCTIONS ---------------------------------"""
def european_payoff(K: float, option_type: Literal['call', 'put'] = 'call') -> Callable[[Array], Array]:
    """
        European call/put payoff on terminal price.
          call: max(S_T - K, 0)
          put : max(K - S_T, 0)
        """
    t = option_type.strip().lower()
    if t not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'.")

    def payoff(S: Array) -> Array:
        S_T = _as_terminal(S)
        return np.maximum(S_T - K, 0.0) if t == "call" else np.maximum(K - S_T, 0.0)

    return payoff


def digital_payoff(K: float, option_type: Literal['call', 'put'] = 'call', cash: float = 1.0) -> Callable[[Array], Array]:
    """
        Digital call/put payoff on terminal price.
          call: 1{S_T > K} * cash
          put: 1{S_T < K} * cash
    """
    t = option_type.strip().lower()
    if t not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'.")

    def payoff(S: Array) -> Array:
        S_T = _as_terminal(S)
        return (S_T > K).astype(float) * cash if t == "call" else (S_T < K).astype(float) * cash

    return payoff