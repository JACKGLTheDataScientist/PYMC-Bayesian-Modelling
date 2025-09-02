import numpy as np
import pandas as pd

def add_trend(df: pd.DataFrame, time_col: str = "week", trend_col: str = "Trend") -> pd.DataFrame:
    """
    Add a linear trend variable to panel data.

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
    for d in range(1, degree + 1):
        df[f"{prefix}_{d}"] = df[time_col] ** d
    return df



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
