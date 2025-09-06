# tests/conftest.py
import pytest
import numpy as np
import pymc as pm
import arviz as az
import pandas as pd
import matplotlib
matplotlib.use("Agg", force=True)


@pytest.fixture(scope="session")
def simple_model():
    # minimal example model used by the tests
    x = np.linspace(0, 1, 50)
    y = 1.0 + 2.0 * x + np.random.normal(0, 0.1, size=len(x))

    with pm.Model() as m:
        beta0 = pm.Normal("beta0", 0, 10)
        beta1 = pm.Normal("beta1", 0, 10)
        sigma = pm.HalfNormal("sigma", 1)
        x_data = pm.Data("x", x)
        pm.Normal("y_obs", beta0 + beta1 * x_data, sigma, observed=y)
    return m

@pytest.fixture(scope="session")
def idata_no_ppc(simple_model) -> az.InferenceData:
    with simple_model:
        idata = pm.sample(
            draws=30, tune=30, chains=2, cores=1, target_accept=0.8, 
            idata_kwargs={"log_likelihood": True}, progressbar=False, return_inferencedata=True)
    return idata


@pytest.fixture(scope="session")
def idata_with_ppc(idata_no_ppc: az.InferenceData, simple_model) -> az.InferenceData:
    """Extend the posterior with PPC; reuse the session posterior."""
    idata = idata_no_ppc.copy()
    with simple_model:
        ppc = pm.sample_posterior_predictive(
            idata.posterior,            
            var_names=["y_obs"],
            random_seed=2025,
            progressbar=False,
            return_inferencedata=True,   
        )
    idata.extend(ppc)                   
    return idata


@pytest.fixture(scope="session")
def idata_small():
    """
    Fit a tiny PyMC model end-to-end and return InferenceData with:
      - posterior
      - sample_stats (energy, diverging)
      - observed_data -> y_obs
      - posterior_predictive -> y_obs
    Kept tiny so tests run quickly and work on Windows (cores=1).
    """
    rng = np.random.default_rng(123)
    n = 40
    x = rng.normal(size=n)
    y = 2.0 + 0.5 * x + rng.normal(scale=0.5, size=n)  # ground truth

    coords = {"obs": np.arange(n)}
    with pm.Model(coords=coords) as model:
        x_data = pm.Data("x", x, dims="obs")
        beta0 = pm.Normal("beta0", 0, 10)
        beta1 = pm.Normal("beta1", 0, 10)
        sigma = pm.HalfNormal("sigma", 1.0)

        mu = beta0 + beta1 * x_data
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y, dims="obs")

        # Small, quick run; cores=1 for Windows friendliness
        idata = pm.sample(
            draws=50,
            tune=50,
            chains=2,
            cores=1,
            target_accept=0.9,
            random_seed=2024,
            return_inferencedata=True,
            idata_kwargs={"log_likelihood": True},
            progressbar=False,
        )
        # Posterior predictive for y_obs, keeping chain/draw dims
        ppc = pm.sample_posterior_predictive(
            idata,
            var_names=["y_obs"],
            random_seed=2025,
            progressbar=False,
        )
        idata.extend(ppc)  # attach posterior_predictive

    return idata


@pytest.fixture
def heldout_small():
    """
    Optional simple holdout 'observed' for OOS tests (same name 'y_obs').
    Your oos function can accept this as a mapping {observed_var: array}.
    """
    rng = np.random.default_rng(456)
    n = 20
    return {"y_obs": 2.0 + 0.5 * rng.normal(size=n) + rng.normal(scale=0.6, size=n)}


@pytest.fixture
def panel_small():
    # 4 rows, 2 products, 3 channels
    return pd.DataFrame({
        "product_id": [0, 0, 101, 101],  # exercise non-contiguous mapping
        "week": [1, 2, 1, 2],
        "kpi_sales": [100., 120., 90., 110.],
        "spend_tv": [10., 20., 5., 15.],
        "spend_search": [5., 10., 2., 8.],
        "spend_social": [3., 5., 1., 4.],
        "price": [50., 50., 55., 55.],
        "gdp_index": [100., 101., 100., 101.],
    })

@pytest.fixture
def model_cfg():
    # Matches your current builder expectations
    return {
        "channels": ["tv","search","social"],
        "mu_beta": [5000,7000,3000],
        "sigma_beta": [2000,2000,1000],
        "beta_sigma": [2000,2000,2000],
        "price_mu": -100, "price_sigma": 2000,
        "gdp_mu": 0, "gdp_sigma": 200,
        "season_sigma": 1000, "trend_sigma": 1000,
        "baseline_mu": 20000, "baseline_sigma": 5000,
        "noise_sigma": 5000,
        "theta_sigma": [2000,2000,2000],
        "slope_mu": 2, "slope_sigma": 1,
    }

# A very small dummy model for fast tests
@pytest.fixture
def tiny_model():
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0, sigma=1)
        sigma = pm.HalfNormal("sigma", sigma=1)
        pm.Normal("y", mu=mu, sigma=sigma, observed=np.random.randn(5))
    return model
