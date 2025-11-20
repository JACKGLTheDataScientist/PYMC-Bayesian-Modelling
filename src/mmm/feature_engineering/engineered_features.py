import numpy as np
import pandas as pd



# Trend
# Linear Trend
def add_trend(df: pd.DataFrame, time_col: str = "week", trend_col: str = "Trend") -> pd.DataFrame:
    """
    Add a linear trend variable to panel data.
    Captures steady growth or decline

    Parameters
    ----------
    df : pd.DataFrame
        Input panel dataframe with a time column.
    time_col : str, default="week"
        Column representing time index (e.g., weeks).
    trend_col : str, default="Trend"
        Name of the new trend column.

    Returns
    -------
    pd.DataFrame
        DataFrame with trend column added.
    """
    df = df.copy()
    df[trend_col] = df[time_col]
    return df

    
# Polynominal Trend/s
def add_polynomial_trend(
    df: pd.DataFrame, 
    time_col: str = "week", 
    degree: int = 2, 
    prefix: str = "trend"
) -> pd.DataFrame:
    """
    Add polynomial trend terms (non-linear trend) to a panel dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input panel dataframe with a time column.
    time_col : str, default="week"
        Column representing time index (e.g., weeks).
    degree : int, default=2
        Maximum degree of polynomial (e.g., 2 = quadratic, 3 = cubic).
    prefix : str, default="trend"
        Prefix for new feature column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with polynomial trend features added.
        e.g., trend_1 (linear), trend_2 (quadratic), ..., trend_d.
    """
    df = df.copy()
    for d in range(1, degree + 1): # Doesn't include right number 
        df[f"{prefix}_{d}"] = df[time_col] ** d
    return df

# Seasonality
def add_fourier_terms(df: pd.DataFrame, time_col: str = "week", period: int = 52, K: int = 2) -> pd.DataFrame:
    """
    Add Fourier series terms for seasonality.

    Parameters
    ----------
    df : pd.DataFrame
        Input panel dataframe with a time column.
    time_col : str, default="week"
        Column representing time index (e.g., weeks).
    period : int, default=52
        Seasonality period (52 weeks for yearly seasonality).
    K : int, default=2
        Number of harmonics. Higher K allows more complex seasonal patterns.

    Returns
    -------
    pd.DataFrame
        DataFrame with added Fourier terms (sin_k, cos_k).
    """
    df = df.copy()
    for k in range(1, K+1):
        df[f"sin_{k}"] = np.sin(2 * np.pi * k * df[time_col] / period)
        df[f"cos_{k}"] = np.cos(2 * np.pi * k * df[time_col] / period)
    return df


def add_date_range_dummy(df, date_col, start_date, end_date, dummy_col="dummy_event"):
    """
    Add a dummy variable (0/1) for rows where date_col falls within a given range.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    date_col : str
        Column name in df containing dates.
    start_date : str, datetime, or Timestamp
        Start date of the range (inclusive).
    end_date : str, datetime, or Timestamp
        End date of the range (inclusive).
    dummy_col : str, default="dummy_event"
        Name of the dummy column to create.

    Returns
    -------
    pd.DataFrame
        DataFrame with a new dummy column (0/1).

    Raises
    ------
    KeyError
        If `date_col` is not in the DataFrame.
    ValueError
        If dates are invalid or start_date > end_date.
    """

    if date_col not in df.columns:
        raise KeyError(f"Column '{date_col}' not found in DataFrame.")

    try:
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
    except Exception as e:
        raise ValueError(f"Could not parse dates: {e}")

    if start > end:
        raise ValueError("start_date must be before or equal to end_date.")

    # Assign dummy
    df[dummy_col] = ((df[date_col] >= start) & (df[date_col] <= end)).astype(int)

    return df

