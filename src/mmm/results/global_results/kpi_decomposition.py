###############################
# KPI Decomposition - global or Product x Market
###############################
import pandas as pd
import numpy as np
import xarray as xr

def decompose_model_outputs(
    idata,
    data,
    market=None,
    product=None,
    include_vars_prefix="contrib_",
    add_total=True,
):
    """
    Extract posterior contribution components (contrib_* variables)
    and compute their posterior means ONLY for the selected market × product.
    """

    # -----------------------------------------
    # 1. Build observation mask (market/product)
    # -----------------------------------------
    if market is None:
        market_mask = np.ones(len(data), dtype=bool)
    else:
        market_mask = data["market"] == market

    if product is None:
        product_mask = np.ones(len(data), dtype=bool)
    else:
        product_mask = data["product_id"] == product

    mask = market_mask & product_mask
    obs_idx = np.where(mask)[0]

    if len(obs_idx) == 0:
        raise ValueError(f"No observations found for market={market}, product={product}")


    # -----------------------------------------
    # 2. Identify deterministic contrib variables
    # -----------------------------------------
    contrib_vars = [
        v for v in idata.posterior.data_vars
        if v.startswith(include_vars_prefix)
    ]
    if not contrib_vars:
        raise ValueError(f"No posterior variables found with prefix {include_vars_prefix}")

    all_rows = []

    # -----------------------------------------
    # 3. Extract posterior mean per variable (subsetted)
    # -----------------------------------------
    for var in contrib_vars:

        da = idata.posterior[var].isel(obs=obs_idx).mean(dim=["chain", "draw"])
        df = da.to_dataframe().reset_index()

        # The value column is always the variable name
        value_col = da.name

        # Other columns besides obs and value_col become part of component name
        extra_dims = [c for c in df.columns if c not in ["obs", value_col]]

        df["component"] = df.apply(
            lambda r: var + "".join([f"__{dim}={r[dim]}" for dim in extra_dims]),
            axis=1
        )

        df = df[["obs", "component", value_col]]
        df = df.rename(columns={value_col: "value"})

        all_rows.append(df)

    # -----------------------------------------
    # 4. Combine long format
    # -----------------------------------------
    long_df = pd.concat(all_rows, ignore_index=True)

    # -----------------------------------------
    # 5. Pivot to wide
    # -----------------------------------------
    wide_df = long_df.pivot_table(
        index="obs",
        columns="component",
        values="value",
        aggfunc="mean"
    ).reset_index()

    # -----------------------------------------
    # 6. Total modelled KPI
    # -----------------------------------------
    if add_total:
        comp_cols = [c for c in wide_df.columns if c not in ["obs"]]
        wide_df["modelled_kpi"] = wide_df[comp_cols].sum(axis=1)

    return {"long": long_df, "wide": wide_df}

