import pandas as pd
import numpy as np
from mmm.data.make_synthetic_panel import make_synthetic_panel

def test_column_dtypes():
    df = make_synthetic_panel(n_products=1, n_weeks=5, seed=1)
    assert pd.api.types.is_numeric_dtype(df["product_id"])
    assert pd.api.types.is_numeric_dtype(df["week"])
    assert pd.api.types.is_numeric_dtype(df["spend_tv"])

def test_output_shape_and_columns():
    df = make_synthetic_panel(n_products=2, n_weeks=10, seed=123)
    # 2 products × 10 weeks
    assert df.shape[0] == 20
    # expected columns exist
    expected = {"product_id", "week", "spend_tv", "spend_search",
                "spend_social", "price", "gdp_index", "kpi_sales"}
    assert expected.issubset(df.columns)

def test_values_are_reasonable():
    df = make_synthetic_panel(n_products=2, n_weeks=10, seed=1)
    assert (df["spend_tv"] >= 0).all()
    assert (df["spend_search"] >= 0).all()
    assert (df["spend_social"] >= 0).all()
    assert not df["kpi_sales"].isnull().any()

def test_reproducibility_with_seed():
    df1 = make_synthetic_panel(n_products=2, n_weeks=10, seed=123)
    df2 = make_synthetic_panel(n_products=2, n_weeks=10, seed=123)
    pd.testing.assert_frame_equal(df1, df2)

def test_different_products_have_different_spends():
    df = make_synthetic_panel(n_products=2, n_weeks=10, seed=123)
    df0 = df[df["product_id"] == 0]
    df1 = df[df["product_id"] == 1]
    assert not df0["spend_tv"].equals(df1["spend_tv"])
