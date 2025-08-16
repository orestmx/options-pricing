__all__ = ['year_fraction']

from datetime import date
import pandas as pd

def year_fraction(today: date, expiry: date, convention: str = "ACT/365") -> float:
    """
    Compute year fraction between two dates according to day count convention.

    Supported conventions:
      - "ACT/365" : actual calendar days / 365
      - "ACT/252" : business (trading) days / 252
      - "ACT/360" : actual calendar days / 360

    Parameters
    ----------
    today : date
        Valuation date.
    expiry : date
        Expiry date.
    convention : str
        Day count convention ("ACT/365", "ACT/252", "ACT/360").

    Returns
    -------
    float
        Year fraction according to the specified convention.
    """
    if expiry <= today:
        return 0.0

    delta_days = (expiry - today).days

    if convention.upper() == "ACT/365":
        return delta_days / 365.0

    elif convention.upper() == "ACT/360":
        return delta_days / 360.0

    elif convention.upper() == "ACT/252":
        # count only business days
        bdays = pd.bdate_range(start=today, end=expiry, freq="B")
        return len(bdays) / 252.0

    else:
        raise ValueError(f"Unsupported convention: {convention}")