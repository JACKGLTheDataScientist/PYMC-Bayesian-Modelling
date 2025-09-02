import pymc as pm
import numpy as np
import pytensor.tensor as at
from mmm.modelling_transformations import adstock_geometric, hill_saturation

def build_mmm(panel, draws=1000, tune=1000, target_accept=0.9):
    """
    Build and fit a hierarchical Bayesian MMM using PyMC.

    Parameters
    ----------
    panel : pd.DataFrame
        DataFrame with product_id, week, kpi_sales, spend_tv, spend_search,
        spend_social, price, gdp_index columns.
    draws : int
        Number of posterior draws.
    tune : int
        Number of tuning iterations.
    target_accept : float
        Target acceptance rate for NUTS sampler.

    Returns
    -------
    idata : arviz.InferenceData
        Posterior samples and predictive checks.
    """

    coords = {
        'channels': ['tv', 'search', 'social'], 
        'product': panel['product_id'].unique()
    }

    with pm.Model(coords=coords) as mmm: 
        
        # Global channel means
        mu_beta = pm.TruncatedNormal(
            "mu_beta", lower=0,
            mu=[5000, 7000, 3000],  # expected incremental contribution
            sigma=[2000, 2000, 1000],
            dims="channels"
        )

        
        # Variation across products
        beta_sigma = pm.HalfNormal(
            'beta_sigma', sigma=[2000, 2000, 2000], dims='channels'
        )
        
        # Product × channel random effects
        z = pm.Normal("z", mu=0, sigma=1, dims=("product", "channels"))
        beta = pm.Deterministic("beta", mu_beta + beta_sigma * z)
        
        # Controls
        price_beta = pm.TruncatedNormal("beta_price", upper = 0, mu=-100, sigma=2000)
        gdp_beta   = pm.Normal("beta_gdp", mu=0, sigma=200)

        
        # Seasonality + trend
        beta_sin1  = pm.Normal('beta_sin1', mu=0, sigma=1000)
        beta_cos1  = pm.Normal('beta_cos1', mu=0, sigma=1000)
        beta_trend = pm.Normal('beta_trend', mu=0, sigma=1000)
        
        # Product-specific baseline
        baseline = pm.Normal("beta_baseline", mu=20000, sigma=5000, dims="product")

        # Noise
        sigma = pm.HalfNormal("sigma", sigma=5000)

        
        # Channel-specific priors
        adstock = pm.Beta("adstock", alpha=2, beta=2, dims="channels")
        theta   = pm.HalfNormal("theta", sigma=2000, dims="channels")
        slope   = pm.TruncatedNormal(
            "slope", mu=2, sigma=1, lower=0.5, upper=5, dims="channels"
        )

        # Stack spends
        X_spend = at.stack([
            panel["spend_tv"].values.astype("float64"),
            panel["spend_search"].values.astype("float64"),
            panel["spend_social"].values.astype("float64"),
        ], axis=1) 

        # Apply adstock (PyTensor version)
        tv_ad     = adstock_geometric(X_spend[:,0], adstock[0])
        search_ad = adstock_geometric(X_spend[:,1], adstock[1])
        social_ad = adstock_geometric(X_spend[:,2], adstock[2])

        # Apply Hill saturation
        tv_trans     = (tv_ad ** slope[0]) / (theta[0] ** slope[0] + tv_ad ** slope[0] + 1e-8)
        search_trans = (search_ad ** slope[1]) / (theta[1] ** slope[1] + search_ad ** slope[1] + 1e-8)
        social_trans = (social_ad ** slope[2]) / (theta[2] ** slope[2] + social_ad ** slope[2] + 1e-8)

        X_media = at.stack([tv_trans, search_trans, social_trans], axis=1)

        # Contribution per product
        product_ids = panel["product_id"].values.astype(int)
        media_contrib = (beta[product_ids, :] * X_media).sum(axis=1)

        # Controls + seasonality + trend
        mu = (
            baseline[product_ids]
            + media_contrib
            + price_beta * panel["price"].values
            + gdp_beta   * panel["gdp_index"].values
            + beta_sin1 * np.sin(2*np.pi*panel["week"].values/52)
            + beta_cos1 * np.cos(2*np.pi*panel["week"].values/52)
            + beta_trend * panel["week"].values
        )
        
        # Likelihood (StudentT for robustness)
        y_obs = pm.StudentT(
            'y_obs', mu=mu, sigma=sigma, nu=5,
            observed=panel['kpi_sales'].values
        )

        # Sample
        idata = pm.sample(
            draws=draws, tune=tune, target_accept=target_accept,
            return_inferencedata=True, idata_kwargs={"log_likelihood": True}
        )

    return idata
