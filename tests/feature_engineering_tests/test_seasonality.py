import pandas as pd
import numpy as np
import pytest
from mmm.feature_engineering.engineered_features import add_fourier_terms, add_polynomial_trend, add_trend, add_date_range_dummy


def test_add_trend_creates_column():
    df = pd.DataFrame({"week": [0, 1, 2, 3]})
    result = add_trend(df, time_col="week", trend_col="Trend")
    
    # New column created
    assert "Trend" in result.columns
    
    # Values should equal week
    assert all(result["Trend"] == df["week"])

def test_polynomial_trend_correctness():
    df = pd.DataFrame({"week": [0, 1, 2, 3]})
    result = add_polynomial_trend(df, time_col="week", degree=2)
    
    assert "trend_1" in result.columns
    assert "trend_2" in result.columns
    assert result.loc[2, "trend_1"] == 2
    assert result.loc[2, "trend_2"] == 4

def test_fourier_terms_shape_and_columns():
    df = pd.DataFrame({"week": [0, 13, 26, 39]})
    result = add_fourier_terms(df, time_col="week", period=52, K=2)
    
    # Columns added
    assert "sin_1" in result.columns
    assert "cos_1" in result.columns
    assert "sin_2" in result.columns
    assert "cos_2" in result.columns
    
    # Values should be numeric
    assert np.isclose(result.loc[0, "sin_1"], 0.0)
    assert np.isclose(result.loc[0, "cos_1"], 1.0)

def test_missing_time_column():
    df = pd.DataFrame({"day": [1, 2, 3]})
    with pytest.raises(KeyError):
        add_polynomial_trend(df, time_col="week", degree=2)

def test_add_date_range_dummy():
    df = pd.DataFrame({
        "date": ["2025-01-01", "2025-01-08", "2025-01-15", "2025-01-22"]
    })

    # Apply function
    result = add_date_range_dummy(
        df,
        date_col="date",
        start_date="2025-01-08",
        end_date="2025-01-15",
        dummy_col="promo_period"
    )

    # New column exists
    assert "promo_period" in result.columns

    # Expected values: 0,1,1,0
    expected = [0, 1, 1, 0]
    assert result["promo_period"].tolist() == expected


def test_invalid_date_column():
    df = pd.DataFrame({"day": ["2025-01-01", "2025-01-08"]})

    with pytest.raises(KeyError):
        add_date_range_dummy(df, date_col="date", start_date="2025-01-01", end_date="2025-01-08")


def test_invalid_date_order():
    df = pd.DataFrame({"date": ["2025-01-01", "2025-01-08"]})

    with pytest.raises(ValueError):
        add_date_range_dummy(df, date_col="date", start_date="2025-01-10", end_date="2025-01-05")

