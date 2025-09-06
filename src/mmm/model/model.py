import pymc as pm
import numpy as np
import pandas as pd
import pytensor.tensor as at
from mmm.transformation.modelling_transformations import adstock_geometric, hill_saturation

def build_mmm(panel, model_cfg):
    """
    Build hierarchical Bayesian MMM in PyMC (no sampling - sampler.py used for this).
    Exposes deterministic components for decomposition/ROI outputs. 
    """
    # ---- Coords
    channels = list(model_cfg["channels"])
    products = np.sort(panel["product_id"].unique())
    obs_idx  = np.arange(len(panel))

    coords = {"channels": channels, "product": products, "obs": obs_idx}

    # Map product_id -> contiguous index 0..P-1
    prod_map = {pid: i for i, pid in enumerate(products)}
    prod_idx_arr = panel["product_id"].map(prod_map).to_numpy(int)

    # Prepare inputs before converted to pytensor
    week_arr  = panel["week"].to_numpy(float)
    price_arr = panel["price"].to_numpy(float)
    gdp_arr   = panel["gdp_index"].to_numpy(float)
    X_spend_arr = panel[["spend_tv","spend_search","spend_social"]].to_numpy(float)
    y_obs_arr   = panel["kpi_sales"].to_numpy(float)

    with pm.Model(coords=coords) as mmm:
        # ---- Data nodes (mutable for OOS/counterfactuals)
        # Converting numpy arrays into PyTensor vectors for diff math compatibility 
        prod_idx = pm.Data("prod_idx", prod_idx_arr, dims="obs")
        week     = pm.Data("week",     week_arr,     dims="obs")
        price    = pm.Data("price",    price_arr,    dims="obs")
        gdp      = pm.Data("gdp",      gdp_arr,      dims="obs")
        X_spend  = pm.Data("X_spend",  X_spend_arr,  dims=("obs","channels"))
        y_data   = pm.Data("y_data",    y_obs_arr,    dims = "obs")

        # ---- Hierarchical priors for media effects
        mu_beta    = pm.TruncatedNormal("mu_beta", lower=0,
                                        mu=model_cfg["mu_beta"],
                                        sigma=model_cfg["sigma_beta"],
                                        dims="channels")
        
        beta_sigma = pm.HalfNormal("beta_sigma", sigma=model_cfg["beta_sigma"], dims="channels")
        z          = pm.Normal("z", 0, 1, dims=("product","channels")) # Reparameterising
        beta       = pm.Deterministic("beta", mu_beta + beta_sigma * z, dims=("product","channels"))

        # ---- Controls
        price_beta = pm.TruncatedNormal("beta_price", upper=0, mu=model_cfg["price_mu"], sigma=model_cfg["price_sigma"])
        gdp_beta   = pm.Normal("beta_gdp", mu=model_cfg["gdp_mu"], sigma=model_cfg["gdp_sigma"])

        # ---- Seasonality & trend
        beta_sin1  = pm.Normal("beta_sin1", 0, model_cfg["season_sigma"])
        beta_cos1  = pm.Normal("beta_cos1", 0, model_cfg["season_sigma"])
        beta_trend = pm.Normal("beta_trend", 0, model_cfg["trend_sigma"])

        # ---- Baseline & noise
        baseline = pm.Normal("beta_baseline", mu=model_cfg["baseline_mu"],
                             sigma=model_cfg["baseline_sigma"], dims="product")
        sigma = pm.HalfNormal("sigma", sigma=model_cfg["noise_sigma"])

        # ---- Media transformation priors
        adstock = pm.Beta("adstock", alpha=2, beta=2, dims="channels")
        theta   = pm.HalfNormal("theta", sigma=model_cfg["theta_sigma"], dims="channels")
        slope   = pm.TruncatedNormal("slope", mu=model_cfg["slope_mu"],
                                     sigma=model_cfg["slope_sigma"], lower=0.5, upper=5, dims="channels")

        # ---- Transform media (use pm.Data -> slice per channel)
        tv_raw, search_raw, social_raw = X_spend[:,0], X_spend[:,1], X_spend[:,2]

        tv_ad     = adstock_geometric(tv_raw,     adstock[0])
        search_ad = adstock_geometric(search_raw, adstock[1])
        social_ad = adstock_geometric(social_raw, adstock[2])

        tv_trans     = hill_saturation(tv_ad,     theta[0],  slope[0])
        search_trans = hill_saturation(search_ad, theta[1],  slope[1])
        social_trans = hill_saturation(social_ad, theta[2],  slope[2])

        X_media = at.stack([tv_trans, search_trans, social_trans], axis=1)  # dims ("obs","channels")

        # ---- Mean structure
        media_contrib_obs_ch = beta[prod_idx, :] * X_media                      # (obs, channels)
        media_contrib_obs    = media_contrib_obs_ch.sum(axis=1)                 # (obs,)  Total media contribution per observation 

        season = beta_sin1 * at.sin(2 * np.pi * week / 52.0) + \
                 beta_cos1 * at.cos(2 * np.pi * week / 52.0)

        mu = baseline[prod_idx] + media_contrib_obs + price_beta * price + gdp_beta * gdp + season + beta_trend * week

        # ---- Deterministics for decomposition/ROI
        pm.Deterministic("contrib_media",        media_contrib_obs_ch, dims=("obs","channels")) # Contribution of each channel per observation
        pm.Deterministic("contrib_media_total",  media_contrib_obs_ch.sum(axis=0), dims=("channels",)) # Total contribution per channel over period
        pm.Deterministic("contrib_baseline",     baseline[prod_idx], dims="obs")
        pm.Deterministic("contrib_price",        price_beta * price,  dims="obs")
        pm.Deterministic("contrib_gdp",          gdp_beta * gdp,      dims="obs")
        pm.Deterministic("contrib_season",       season,               dims="obs")
        pm.Deterministic("contrib_trend",        beta_trend * week,    dims="obs")
        pm.Deterministic("mu",                   mu,                   dims="obs")
        pm.Deterministic("roi_channel",
                         media_contrib_obs_ch.sum(axis=0) / (X_spend.sum(axis=0) + 1e-8),
                         dims="channels")

        # ---- Robust likelihood
        nu_raw = pm.Exponential("nu_raw", 1/10)
        nu     = pm.Deterministic("nu", nu_raw + 2)  # ensures nu>2
        pm.StudentT("y_obs", mu=mu, sigma=sigma, nu=nu, observed=y_data)

    return mmm

