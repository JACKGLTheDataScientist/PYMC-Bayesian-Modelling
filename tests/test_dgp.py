import pandas as pd
from mmm.make_synthetic_panel import make_synthetic_panel

def test_column_dtypes():
    df = make_synthetic_panel(n_products=1, n_weeks=5, seed=1)
    assert df["product_id"].dtype == "int64"
    assert df["week"].dtype == "int64"
    assert df["kpi_sales"].dtype in ("float64", "float32")

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
